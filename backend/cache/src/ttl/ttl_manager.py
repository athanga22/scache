"""
TTL Manager Implementation
Handles time-to-live expiration for cache entries with background cleanup.
"""

import time
import threading
from typing import Dict, Optional, Tuple
from collections import defaultdict


class TTLManager:
    """Manages TTL expiration for cache entries with background cleanup."""
    
    def __init__(self, config, storage_engine=None, cache_instance=None):
        """
        Initialize TTL manager.
        
        Args:
            config: Cache configuration
            storage_engine: Storage engine reference (for cleanup)
            cache_instance: Cache instance reference (for cleanup)
        """
        self.config = config
        # key -> (expiry_time, level, ttl_value)
        self.ttl_data: Dict[str, Tuple[float, str, int]] = {}
        self.cleanup_thread = None
        self.running = False
        self.lock = threading.RLock()
        self.storage_engine = storage_engine
        self.cache_instance = cache_instance
        
        print(f"TTL Manager initialized with {config.cleanup_interval}s cleanup interval")
    
    def set_ttl(self, key: str, ttl: int, level: str = "auto"):
        """
        Set TTL for a key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            level: Cache level
        """
        with self.lock:
            expiry_time = time.time() + ttl
            self.ttl_data[key] = (expiry_time, level, ttl)
    
    def is_expired(self, key: str) -> bool:
        """
        Check if a key is expired (lazy expiration checking).
        
        Args:
            key: Cache key to check
            
        Returns:
            True if expired, False otherwise
        """
        with self.lock:
            if key not in self.ttl_data:
                return False
            
            expiry_time, _, _ = self.ttl_data[key]
            return time.time() > expiry_time
    
    def extend_ttl(self, key: str):
        """
        Extend TTL for a key on cache hit.
        
        Args:
            key: Cache key to extend
        """
        with self.lock:
            if key in self.ttl_data:
                current_ttl = self.ttl_data[key][2]  # Get original TTL value
                new_expiry = time.time() + current_ttl
                level = self.ttl_data[key][1]
                self.ttl_data[key] = (new_expiry, level, current_ttl)
    
    def remove_ttl(self, key: str):
        """
        Remove TTL for a key.
        
        Args:
            key: Cache key to remove TTL for
        """
        with self.lock:
            if key in self.ttl_data:
                del self.ttl_data[key]
    
    def clear_all(self):
        """Clear all TTL data."""
        with self.lock:
            self.ttl_data.clear()
    
    def clear_level(self, level: str):
        """
        Clear TTL data for a specific level.
        
        Args:
            level: Cache level to clear
        """
        with self.lock:
            keys_to_remove = [
                key for key, (_, key_level, _) in self.ttl_data.items() 
                if key_level == level
            ]
            for key in keys_to_remove:
                del self.ttl_data[key]
    
    def get_expired_keys(self) -> list:
        """
        Get list of expired keys for cleanup.
        
        Returns:
            List of expired key names
        """
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, (expiry_time, _, _) in self.ttl_data.items()
                if current_time > expiry_time
            ]
            return expired_keys
    
    def get_ttl_info(self, key: str) -> Optional[dict]:
        """
        Get TTL information for a key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL info dict or None if not found
        """
        with self.lock:
            if key not in self.ttl_data:
                return None
            
            expiry_time, level, ttl_value = self.ttl_data[key]
            current_time = time.time()
            remaining_time = max(0, expiry_time - current_time)
            
            return {
                'expiry_time': expiry_time,
                'level': level,
                'ttl_value': ttl_value,
                'remaining_time': remaining_time,
                'is_expired': remaining_time <= 0
            }
    
    def start_cleanup_thread(self):
        """Start background cleanup thread."""
        if not self.running:
            self.running = True
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop, 
                daemon=True,
                name="TTLCleanupThread"
            )
            self.cleanup_thread.start()
            print(f"TTL cleanup thread started (interval: {self.config.cleanup_interval}s)")
    
    def stop_cleanup_thread(self):
        """Stop background cleanup thread."""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
            print("TTL cleanup thread stopped")
    
    def _cleanup_loop(self):
        """Background cleanup loop for expired entries."""
        while self.running:
            try:
                # Actually delete expired entries if we have storage engine reference
                if self.storage_engine and self.cache_instance:
                    deleted_count = self.cleanup_expired_entries(self.storage_engine, self.cache_instance)
                    if deleted_count > 0:
                        # Also clear from similarity engine if cache instance available
                        if hasattr(self.cache_instance, 'similarity_engine'):
                            # Remove from similarity engine (lazy - will be cleaned on next access)
                            pass
                else:
                    # Fallback: just report expired keys
                    expired_keys = self.get_expired_keys()
                    if expired_keys:
                        print(f"TTL cleanup: found {len(expired_keys)} expired keys (storage engine not available)")
                
                # Sleep until next cleanup
                time.sleep(self.config.cleanup_interval)
                
            except Exception as e:
                print(f"TTL cleanup error: {e}")
                time.sleep(1)  # Short sleep on error
    
    def cleanup_expired_entries(self, storage_engine, cache_instance):
        """
        Actually delete expired entries from storage.
        
        Args:
            storage_engine: Storage engine to delete from
            cache_instance: Cache instance for cleanup operations
            
        Returns:
            Number of entries deleted
        """
        with self.lock:
            expired_keys = self.get_expired_keys()
            deleted_count = 0
            
            for key in expired_keys:
                # Get the level from TTL data
                if key in self.ttl_data:
                    _, level, _ = self.ttl_data[key]
                    # Delete from storage
                    if storage_engine.exists(key, level):
                        storage_engine.delete(key, level)
                        deleted_count += 1
                    
                    # Also remove from eviction policy if available
                    if hasattr(cache_instance, 'eviction_policy'):
                        cache_instance.eviction_policy.remove_key(key)
                    
                    # Remove from TTL tracking
                    del self.ttl_data[key]
            
            if deleted_count > 0:
                print(f"TTL cleanup: deleted {deleted_count} expired entries")
            
            return deleted_count
    
    def get_stats(self) -> dict:
        """
        Get TTL statistics.
        
        Returns:
            Dictionary with TTL statistics
        """
        with self.lock:
            current_time = time.time()
            total_keys = len(self.ttl_data)
            expired_keys = len(self.get_expired_keys())
            active_keys = total_keys - expired_keys
            
            # Calculate average TTL
            total_ttl = sum(ttl_value for _, _, ttl_value in self.ttl_data.values())
            avg_ttl = total_ttl / total_keys if total_keys > 0 else 0
            
            return {
                'total_keys': total_keys,
                'active_keys': active_keys,
                'expired_keys': expired_keys,
                'average_ttl': round(avg_ttl, 2),
                'cleanup_interval': self.config.cleanup_interval,
                'thread_running': self.running
            }

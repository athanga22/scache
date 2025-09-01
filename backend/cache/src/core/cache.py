"""
Core Cache Implementation
Main cache class that orchestrates all caching functionality.
"""

import time
import threading
from typing import Any, Optional, Dict, List
from collections import OrderedDict

from ..storage.storage_engine import StorageEngine
from ..ttl.ttl_manager import TTLManager
from ..eviction.eviction_policy import EvictionPolicy
from ..threading.thread_safety import ThreadSafety
from ..utils.config import CacheConfig


class Cache:
    """
    Main cache class that provides high-level caching functionality.
    
    This class orchestrates:
    - Storage engine for data persistence
    - TTL management for expiration
    - Eviction policies for memory management
    - Thread safety for concurrent access
    """
    
    def __init__(self, config):
        """
        Initialize the cache with configuration.
        
        Args:
            config: Cache configuration object
        """
        self.config = config
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
        
        # Initialize components
        self.storage = StorageEngine(config)
        self.ttl_manager = TTLManager(config)
        self.eviction_policy = EvictionPolicy(config)
        self.thread_safety = ThreadSafety(
            max_readers=config.max_threads,
            lock_timeout=config.lock_timeout
        )
        
        # Start background processes
        self._start_background_processes()
        
        print(f"✅ Cache initialized with {config.memory_limit} memory limit")
    
    def _start_background_processes(self):
        """Start background TTL cleanup and monitoring."""
        try:
            self.ttl_manager.start_cleanup_thread()
            self.eviction_policy.start_monitoring_thread()
            print("✅ Background processes started")
        except Exception as e:
            print(f"⚠️ Warning: Could not start background processes: {e}")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, level: str = "auto") -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for default)
            level: Cache level (query, embedding, result, context, auto)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.thread_safety.write_lock():
                # Determine cache level if auto
                if level == "auto":
                    level = self._determine_cache_level(value)
                
                # Check memory limits before setting
                if not self.eviction_policy.check_memory_available(value):
                    self.eviction_policy.evict_entries(self.storage)
                
                # Store the value
                success = self.storage.set(key, value, level)
                if success:
                    # Set TTL
                    if ttl is None:
                        ttl = self.config.get_ttl_for_level(level)
                    self.ttl_manager.set_ttl(key, ttl, level)
                    
                    # Update statistics
                    self.stats['sets'] += 1
                    
                    # Update eviction policy
                    self.eviction_policy.record_access(key)
                    
                    return True
                return False
                
        except Exception as e:
            print(f"❌ Error setting cache key {key}: {e}")
            return False
    
    def get(self, key: str, level: str = "auto") -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            level: Cache level to search in
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            with self.thread_safety.read_lock():
                # If auto level, search across all levels
                if level == "auto":
                    for cache_level in ["query", "embedding", "context", "result"]:
                        if self.storage.exists(key, cache_level):
                            # Check TTL
                            if self.ttl_manager.is_expired(key):
                                self.delete(key, cache_level)
                                self.stats['misses'] += 1
                                return None
                            
                            # Get the value
                            value = self.storage.get(key, cache_level)
                            if value is not None:
                                # Update statistics
                                self.stats['hits'] += 1
                                
                                # Update eviction policy
                                self.eviction_policy.record_access(key)
                                
                                # Extend TTL if configured
                                if self.config.ttl_extension_on_hit:
                                    self.ttl_manager.extend_ttl(key)
                                
                                return value
                    
                    # Key not found in any level
                    self.stats['misses'] += 1
                    return None
                else:
                    # Specific level search
                    if not self.storage.exists(key, level):
                        self.stats['misses'] += 1
                        return None
                    
                    # Check TTL
                    if self.ttl_manager.is_expired(key):
                        self.delete(key, level)
                        self.stats['misses'] += 1
                        return None
                    
                    # Get the value
                    value = self.storage.get(key, level)
                    if value is not None:
                        # Update statistics
                        self.stats['hits'] += 1
                        
                        # Update eviction policy
                        self.eviction_policy.record_access(key)
                        
                        # Extend TTL if configured
                        if self.config.ttl_extension_on_hit:
                            self.ttl_manager.extend_ttl(key)
                        
                        return value
                    
                    self.stats['misses'] += 1
                    return None
                
        except Exception as e:
            print(f"❌ Error getting cache key {key}: {e}")
            self.stats['misses'] += 1
            return None
    
    def delete(self, key: str, level: str = "auto") -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key to delete
            level: Cache level
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            with self.thread_safety.write_lock():
                if level == "auto":
                    # Delete from all levels
                    deleted = False
                    for cache_level in ["query", "embedding", "context", "result"]:
                        if self.storage.exists(key, cache_level):
                            success = self.storage.delete(key, cache_level)
                            if success:
                                self.ttl_manager.remove_ttl(key)
                                self.eviction_policy.remove_key(key)
                                self.stats['deletes'] += 1
                                deleted = True
                    return deleted
                else:
                    # Delete from specific level
                    success = self.storage.delete(key, level)
                    if success:
                        self.ttl_manager.remove_ttl(key)
                        self.eviction_policy.remove_key(key)
                        self.stats['deletes'] += 1
                    return success
        except Exception as e:
            print(f"❌ Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str, level: str = "auto") -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: Cache key to check
            level: Cache level
            
        Returns:
            True if exists and not expired, False otherwise
        """
        try:
            with self.thread_safety.read_lock():
                if level == "auto":
                    # Check across all levels
                    for cache_level in ["query", "embedding", "context", "result"]:
                        if self.storage.exists(key, cache_level):
                            return not self.ttl_manager.is_expired(key)
                    return False
                else:
                    # Check specific level
                    if not self.storage.exists(key, level):
                        return False
                    return not self.ttl_manager.is_expired(key)
        except Exception as e:
            print(f"❌ Error checking existence of key {key}: {e}")
            return False
    
    def clear(self, level: Optional[str] = None) -> bool:
        """
        Clear all entries from specified level or all levels.
        
        Args:
            level: Specific level to clear, or None for all levels
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.thread_safety.write_lock():
                success = self.storage.clear(level)
                if success:
                    if level is None:
                        self.ttl_manager.clear_all()
                        self.eviction_policy.clear_all()
                    else:
                        self.ttl_manager.clear_level(level)
                        self.eviction_policy.clear_level(level)
                return success
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        # Get storage stats
        storage_stats = self.storage.get_stats()
        
        # Get component stats
        ttl_stats = self.ttl_manager.get_stats()
        eviction_stats = self.eviction_policy.get_stats()
        thread_stats = self.thread_safety.get_stats()
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'deletes': self.stats['deletes'],
            'evictions': self.stats['evictions'],
            'hit_rate': round(hit_rate, 2),
            'total_requests': total_requests,
            'memory_usage': self.storage.get_memory_usage(),
            'memory_limit': self.config.memory_limit,
            'ttl_enabled': self.config.ttl_enabled,
            'eviction_policy': self.config.eviction_policy,
            # Include storage stats
            'total_entries': storage_stats.get('total_entries', 0),
            'entries_by_level': storage_stats.get('entries_by_level', {}),
            'memory_pressure': storage_stats.get('memory_pressure', 'low'),
            # Include component stats
            'ttl_stats': ttl_stats,
            'eviction_stats': eviction_stats,
            'thread_stats': thread_stats
        }
    
    def _determine_cache_level(self, value: Any) -> str:
        """
        Automatically determine the appropriate cache level for a value.
        
        Args:
            value: Value to be cached
            
        Returns:
            Cache level string
        """
        # This is a simple heuristic - can be enhanced later
        if isinstance(value, str) and len(value) < 1000:
            return "query"
        elif hasattr(value, 'shape') and hasattr(value, 'dtype'):  # numpy array
            return "embedding"
        elif isinstance(value, dict) and 'answer' in value:
            return "result"
        else:
            return "context"
    
    def shutdown(self):
        """Gracefully shutdown the cache system."""
        print("🔄 Shutting down cache system...")
        try:
            self.ttl_manager.stop_cleanup_thread()
            self.eviction_policy.stop_monitoring_thread()
            print("✅ Cache system shutdown complete")
        except Exception as e:
            print(f"⚠️ Warning during shutdown: {e}")


# Convenience function for quick cache creation
def create_cache(memory_limit: str = "25%", ttl_enabled: bool = True, 
                eviction_policy: str = "lru") -> Cache:
    """
    Create a cache instance with default configuration.
    
    Args:
        memory_limit: Memory limit (e.g., "25%", "2GB")
        ttl_enabled: Whether TTL is enabled
        eviction_policy: Eviction policy to use
        
    Returns:
        Configured Cache instance
    """
    from ..utils.config import CacheConfig
    
    config = CacheConfig(
        memory_limit=memory_limit,
        ttl_enabled=ttl_enabled,
        eviction_policy=eviction_policy
    )
    
    return Cache(config)

"""
Eviction Policy Implementation
Handles cache eviction when memory limits are reached using LRU strategy.
"""

import time
import threading
import sys
from typing import Dict, Any, List
from collections import OrderedDict


class EvictionPolicy:
    """Manages cache eviction policies with LRU implementation."""
    
    def __init__(self, config):
        """Initialize eviction policy."""
        self.config = config
        
        # LRU tracking: OrderedDict maintains access order
        self.access_order = OrderedDict()  # key -> last_access_time
        
        # Access frequency tracking for potential LFU implementation
        self.access_counts = {}  # key -> access_count
        
        # Memory monitoring
        self.memory_thresholds = config.memory_thresholds
        self.eviction_batch_size = config.eviction_batch_size
        
        # Threading
        self.monitoring_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        print(f"Eviction Policy initialized: {config.eviction_policy} with {config.eviction_batch_size} batch size")
        print(f"Max entries: {config.max_entries}, Eviction threshold: {config.eviction_threshold}")
    
    def record_access(self, key: str):
        """
        Record access to a key for LRU tracking.
        
        Args:
            key: Cache key that was accessed
        """
        with self.lock:
            current_time = time.time()
            
            # Update access order (LRU)
            if key in self.access_order:
                # Remove and re-add to move to end (most recently used)
                del self.access_order[key]
            self.access_order[key] = current_time
            
            # Update access count (for potential LFU)
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
    
    def remove_key(self, key: str):
        """
        Remove tracking for a key.
        
        Args:
            key: Cache key to remove
        """
        with self.lock:
            if key in self.access_order:
                del self.access_order[key]
            if key in self.access_counts:
                del self.access_counts[key]
    
    def clear_all(self):
        """Clear all tracking data."""
        with self.lock:
            self.access_order.clear()
            self.access_counts.clear()
    
    def clear_level(self, level: str):
        """
        Clear tracking data for a specific level.
        Note: This would need level information from storage engine.
        """
        # For now, just clear all since we don't have level-specific tracking
        # In a real implementation, this would clear only keys for the specific level
        with self.lock:
            self.access_order.clear()
            self.access_counts.clear()
    
    def check_memory_available(self, value: Any) -> bool:
        """
        Check if memory is available for a value.
        
        Args:
            value: Value to be cached
            
        Returns:
            True if memory is available, False otherwise
        """
        # This is a placeholder - actual memory checking is done in storage engine
        # In a real implementation, this would check current memory usage vs limits
        return True
    
    def should_evict(self, current_memory_usage: float, memory_limit: int) -> bool:
        """
        Check if eviction should be triggered.
        
        Args:
            current_memory_usage: Current memory usage in bytes
            memory_limit: Memory limit in bytes
            
        Returns:
            True if eviction should be triggered
        """
        current_entries = len(self.access_order)
        max_entries = self.config.max_entries
        eviction_threshold = self.config.eviction_threshold
        
        # Proactive eviction based on entry count (primary trigger)
        if current_entries >= max_entries:
            return True  # Hard limit reached - immediate eviction
        elif current_entries >= max_entries * eviction_threshold:
            return True  # Proactive eviction at 80% of max entries
        
        # Memory-based eviction (secondary trigger)
        usage_percentage = current_memory_usage / memory_limit
        if usage_percentage >= self.memory_thresholds["critical"]:
            return True  # Critical - immediate eviction
        elif usage_percentage >= self.memory_thresholds["eviction"]:
            return True  # Eviction threshold reached
        
        return False
    
    def evict_entries(self, storage_engine, target_memory_reduction: int = None) -> int:
        """
        Evict entries to free up memory or reduce entry count.
        
        Args:
            storage_engine: Storage engine to evict from
            target_memory_reduction: Target memory to free in bytes
            
        Returns:
            Number of entries evicted
        """
        with self.lock:
            if not self.access_order:
                return 0
            
            evicted_count = 0
            current_entries = len(self.access_order)
            max_entries = self.config.max_entries
            
            # Determine how many entries to evict
            if current_entries >= max_entries:
                # Evict enough to get below max_entries
                entries_to_evict = current_entries - max_entries + 1  # +1 to be safe
            else:
                # Evict based on memory if specified
                if target_memory_reduction is None:
                    return 0
                entries_to_evict = min(self.eviction_batch_size, 5)  # Default batch
            
            # Limit to batch size
            entries_to_evict = min(entries_to_evict, self.eviction_batch_size)
            
            # Evict oldest entries first (LRU)
            keys_to_evict = list(self.access_order.keys())[:entries_to_evict]
            
            # Perform eviction
            for key in keys_to_evict:
                # Try to delete from all levels
                for level in ["query", "embedding", "context", "result"]:
                    if storage_engine.exists(key, level):
                        if storage_engine.delete(key, level):
                            # Remove from tracking
                            self.remove_key(key)
                            evicted_count += 1
                            break
            
            if evicted_count > 0:
                print(f"Evicted {evicted_count} entries, cache now has {len(self.access_order)} entries (max: {self.config.max_entries})")
            
            return evicted_count
    
    def _estimate_entry_size(self, key: str, storage_engine) -> int:
        """
        Estimate memory size of a cache entry.
        
        Args:
            key: Cache key
            storage_engine: Storage engine to get actual data
            
        Returns:
            Estimated size in bytes
        """
        try:
            # Get actual data from storage engine
            total_size = sys.getsizeof(key)
            
            # Estimate size for each cache level
            for level in ["query", "embedding", "context", "result"]:
                if storage_engine.exists(key, level):
                    value = storage_engine.get(key, level)
                    if value is not None:
                        total_size += sys.getsizeof(value)
            
            # Minimum size for RAG responses (query + embedding + context + result)
            # Typical RAG response: 2-10KB
            return max(total_size, 2048)  # 2KB minimum
            
        except Exception:
            # Fallback to realistic estimate for RAG responses
            return 5120  # 5KB average for RAG responses
    
    def get_lru_order(self) -> List[str]:
        """
        Get keys in LRU order (oldest first).
        
        Returns:
            List of keys in LRU order
        """
        with self.lock:
            return list(self.access_order.keys())
    
    def get_mru_order(self) -> List[str]:
        """
        Get keys in MRU order (newest first).
        
        Returns:
            List of keys in MRU order
        """
        with self.lock:
            return list(reversed(self.access_order.keys()))
    
    def start_monitoring_thread(self):
        """Start background monitoring thread."""
        if not self.running:
            self.running = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="EvictionMonitoringThread"
            )
            self.monitoring_thread.start()
            print("Eviction monitoring thread started")
    
    def stop_monitoring_thread(self):
        """Stop background monitoring thread."""
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
            print("Eviction monitoring thread stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop for memory pressure."""
        while self.running:
            try:
                # This would integrate with storage engine for actual memory monitoring
                # For now, just sleep
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Eviction monitoring error: {e}")
                time.sleep(5)
    
    def get_stats(self) -> dict:
        """
        Get eviction policy statistics.
        
        Returns:
            Dictionary with eviction statistics
        """
        with self.lock:
            return {
                'policy': self.config.eviction_policy,
                'total_tracked_keys': len(self.access_order),
                'batch_size': self.eviction_batch_size,
                'memory_thresholds': self.memory_thresholds,
                'thread_running': self.running,
                'lru_order_sample': list(self.access_order.keys())[:10]  # First 10 keys
            }

"""
Storage Engine Implementation
Provides hash table-based storage for cache data with memory management.
"""

import sys
import time
from typing import Any, Dict, Optional, Union
from collections import defaultdict


class StorageEngine:
    """
    Storage engine using hash tables for O(1) access times.
    
    Supports multiple cache levels and comprehensive memory usage tracking.
    """
    
    def __init__(self, config):
        """
        Initialize storage engine.
        
        Args:
            config: Cache configuration object
        """
        self.config = config
        
        # Multi-level storage: level -> {key -> value}
        self.storage = {
            'query': {},
            'embedding': {},
            'context': {},
            'result': {}
        }
        
        # Memory usage tracking per level
        self.memory_usage = {
            'query': 0,
            'embedding': 0,
            'context': 0,
            'result': 0
        }
        
        # Total memory usage across all levels
        self.total_memory = 0
        
        # Access tracking for LRU (last access timestamp)
        self.access_times = {}
        
        print(f"Storage engine initialized with {config.memory_limit} memory limit")
    
    def set(self, key: str, value: Any, level: str = "auto") -> bool:
        """
        Store a value in the specified cache level.
        
        Args:
            key: Cache key
            value: Value to store
            level: Cache level (query, embedding, context, result)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate level
            if level not in self.storage:
                print(f"Invalid cache level: {level}")
                return False
            
            # Calculate memory usage
            old_size = self._get_object_size(self.storage[level].get(key, None))
            new_size = self._get_object_size(value)
            size_delta = new_size - old_size
            
            # Check if we have enough memory
            if not self._check_memory_available(size_delta):
                print(f"Insufficient memory for key: {key} (need {size_delta} bytes)")
                return False
            
            # Store the value
            self.storage[level][key] = value
            
            # Update memory usage
            self.memory_usage[level] += size_delta
            self.total_memory += size_delta
            
            # Update access time
            self.access_times[key] = time.time()
            
            return True
            
        except Exception as e:
            print(f"Error storing key {key}: {e}")
            return False
    
    def get(self, key: str, level: str = "auto") -> Optional[Any]:
        """
        Retrieve a value from the specified cache level.
        
        Args:
            key: Cache key
            level: Cache level
            
        Returns:
            Stored value or None if not found
        """
        try:
            if level not in self.storage:
                print(f"Invalid cache level: {level}")
                return None
            
            # Check if key exists
            if key not in self.storage[level]:
                return None
            
            # Update access time for LRU
            self.access_times[key] = time.time()
            
            return self.storage[level][key]
            
        except Exception as e:
            print(f"Error retrieving key {key}: {e}")
            return None
    
    def delete(self, key: str, level: str = "auto") -> bool:
        """
        Delete a key from the specified cache level.
        
        Args:
            key: Cache key to delete
            level: Cache level
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            if level not in self.storage:
                print(f"Invalid cache level: {level}")
                return False
            
            if key in self.storage[level]:
                # Update memory usage
                old_size = self._get_object_size(self.storage[level][key])
                self.memory_usage[level] -= old_size
                self.total_memory -= old_size
                
                # Delete the key
                del self.storage[level][key]
                
                # Remove access time tracking
                if key in self.access_times:
                    del self.access_times[key]
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error deleting key {key}: {e}")
            return False
    
    def exists(self, key: str, level: str = "auto") -> bool:
        """
        Check if a key exists in the specified cache level.
        
        Args:
            key: Cache key to check
            level: Cache level
            
        Returns:
            True if exists, False otherwise
        """
        try:
            if level not in self.storage:
                return False
            
            return key in self.storage[level]
            
        except Exception as e:
            print(f"Error checking existence of key {key}: {e}")
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
            if level is None:
                # Clear all levels
                for level_name in self.storage:
                    self.storage[level_name].clear()
                    self.memory_usage[level_name] = 0
                self.total_memory = 0
                self.access_times.clear()
            else:
                # Clear specific level
                if level in self.storage:
                    # Remove access times for keys in this level
                    keys_to_remove = list(self.storage[level].keys())
                    for key in keys_to_remove:
                        if key in self.access_times:
                            del self.access_times[key]
                    
                    self.storage[level].clear()
                    self.total_memory -= self.memory_usage[level]
                    self.memory_usage[level] = 0
                else:
                    print(f"Invalid cache level: {level}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def get_memory_usage(self) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive memory usage information.
        
        Returns:
            Dictionary with memory usage details
        """
        memory_limit_bytes = self.config.get_memory_limit_bytes()
        memory_limit_mb = memory_limit_bytes / (1024 ** 2)
        memory_limit_gb = memory_limit_bytes / (1024 ** 3)
        
        return {
            'total_bytes': self.total_memory,
            'total_mb': round(self.total_memory / (1024 ** 2), 2),
            'total_gb': round(self.total_memory / (1024 ** 3), 3),
            'limit_bytes': memory_limit_bytes,
            'limit_mb': round(memory_limit_mb, 2),
            'limit_gb': round(memory_limit_gb, 3),
            'usage_percentage': round((self.total_memory / memory_limit_bytes) * 100, 2),
            'by_level': {
                level: {
                    'bytes': usage,
                    'mb': round(usage / (1024 ** 2), 2),
                    'percentage': round((usage / memory_limit_bytes) * 100, 2)
                }
                for level, usage in self.memory_usage.items()
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        memory_info = self.get_memory_usage()
        
        return {
            'total_entries': sum(len(level_storage) for level_storage in self.storage.values()),
            'entries_by_level': {level: len(storage) for level, storage in self.storage.items()},
            'memory_usage': memory_info,
            'memory_pressure': self._get_memory_pressure_level(),
            'oldest_access': min(self.access_times.values()) if self.access_times else None,
            'newest_access': max(self.access_times.values()) if self.access_times else None
        }
    
    def _get_object_size(self, obj: Any) -> int:
        """
        Estimate the memory size of an object.
        
        Args:
            obj: Object to measure
            
        Returns:
            Estimated size in bytes
        """
        try:
            if obj is None:
                return 0
            
            # Basic size estimation
            size = sys.getsizeof(obj)
            
            # Handle common data types
            if isinstance(obj, str):
                size += len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                size += sum(self._get_object_size(item) for item in obj)
            elif isinstance(obj, dict):
                size += sum(self._get_object_size(k) + self._get_object_size(v) 
                           for k, v in obj.items())
            elif hasattr(obj, 'shape') and hasattr(obj, 'dtype'):  # numpy array
                size += obj.nbytes
            elif hasattr(obj, 'nbytes'):  # objects with nbytes attribute
                size += obj.nbytes
            
            return size
            
        except Exception:
            # Fallback to basic size
            return sys.getsizeof(obj)
    
    def _check_memory_available(self, additional_size: int) -> bool:
        """
        Check if we have enough memory for additional data.
        
        Args:
            additional_size: Size of additional data in bytes
            
        Returns:
            True if memory is available, False otherwise
        """
        if additional_size <= 0:
            return True
        
        memory_limit = self.config.get_memory_limit_bytes()
        return (self.total_memory + additional_size) <= memory_limit
    
    def _get_memory_pressure_level(self) -> str:
        """
        Get current memory pressure level.
        
        Returns:
            Memory pressure level: 'low', 'medium', 'high', 'critical'
        """
        if self.total_memory == 0:
            return 'low'
        
        usage_percentage = self.total_memory / self.config.get_memory_limit_bytes()
        
        if usage_percentage < 0.5:
            return 'low'
        elif usage_percentage < 0.8:
            return 'medium'
        elif usage_percentage < 0.95:
            return 'high'
        else:
            return 'critical'
    
    def get_keys_by_level(self, level: str) -> list:
        """
        Get all keys for a specific cache level.
        
        Args:
            level: Cache level
            
        Returns:
            List of keys in the specified level
        """
        if level not in self.storage:
            return []
        
        return list(self.storage[level].keys())
    
    def get_level_info(self, level: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific cache level.
        
        Args:
            level: Cache level
            
        Returns:
            Dictionary with level information
        """
        if level not in self.storage:
            return {}
        
        return {
            'entry_count': len(self.storage[level]),
            'memory_usage_bytes': self.memory_usage[level],
            'memory_usage_mb': round(self.memory_usage[level] / (1024 ** 2), 2),
            'keys': list(self.storage[level].keys())
        }

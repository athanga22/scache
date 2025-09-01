"""
Cache API Implementation
Provides high-level API for cache operations with validation and error handling.
"""

import time
import logging
from typing import Any, Optional, Dict, List, Union
from dataclasses import dataclass


@dataclass
class CacheOperation:
    """Represents a cache operation for logging."""
    operation: str
    key: str
    level: str
    timestamp: float
    success: bool
    duration: float
    error: Optional[str] = None


class CacheAPI:
    """High-level API for cache operations with validation and logging."""
    
    def __init__(self, cache, enable_logging: bool = True):
        """
        Initialize cache API.
        
        Args:
            cache: Cache instance
            enable_logging: Whether to enable operation logging
        """
        self.cache = cache
        self.enable_logging = enable_logging
        
        # Setup logging
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger('CacheAPI')
        else:
            self.logger = None
        
        # Operation tracking
        self.operation_history: List[CacheOperation] = []
        self.max_history = 1000
        
        print("✅ Cache API initialized with validation and logging")
    
    def _log_operation(self, operation: str, key: str, level: str, 
                       success: bool, duration: float, error: str = None):
        """Log a cache operation."""
        if not self.enable_logging:
            return
        
        # Create operation record
        op_record = CacheOperation(
            operation=operation,
            key=key,
            level=level,
            timestamp=time.time(),
            success=success,
            duration=duration,
            error=error
        )
        
        # Add to history
        self.operation_history.append(op_record)
        if len(self.operation_history) > self.max_history:
            self.operation_history.pop(0)
        
        # Log to logger
        if self.logger:
            if success:
                self.logger.info(f"{operation.upper()}: {key} (level: {level}) - {duration:.3f}s")
            else:
                self.logger.error(f"{operation.upper()} FAILED: {key} (level: {level}) - {error}")
    
    def _validate_key(self, key: str) -> bool:
        """
        Validate cache key.
        
        Args:
            key: Cache key to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(key, str):
            return False
        if not key.strip():
            return False
        if len(key) > 1000:  # Reasonable key length limit
            return False
        return True
    
    def _validate_value(self, value: Any) -> bool:
        """
        Validate cache value.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        if value is None:
            return False
        return True
    
    def _validate_level(self, level: str) -> bool:
        """
        Validate cache level.
        
        Args:
            level: Cache level to validate
            
        Returns:
            True if valid, False otherwise
        """
        valid_levels = ["query", "embedding", "context", "result", "auto"]
        return level in valid_levels
    
    def _validate_ttl(self, ttl: Optional[int]) -> bool:
        """
        Validate TTL value.
        
        Args:
            ttl: TTL value to validate
            
        Returns:
            True if valid, False otherwise
        """
        if ttl is None:
            return True
        if not isinstance(ttl, int):
            return False
        if ttl < 0:
            return False
        if ttl > 31536000:  # Max 1 year
            return False
        return True
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            level: str = "auto") -> bool:
        """
        Set a value in the cache with validation.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for default)
            level: Cache level (query, embedding, result, context, auto)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If validation fails
        """
        start_time = time.time()
        success = False
        error = None
        
        try:
            # Validate inputs
            if not self._validate_key(key):
                raise ValueError(f"Invalid key: {key}")
            if not self._validate_value(value):
                raise ValueError(f"Invalid value for key: {key}")
            if not self._validate_level(level):
                raise ValueError(f"Invalid level: {level}")
            if not self._validate_ttl(ttl):
                raise ValueError(f"Invalid TTL: {ttl}")
            
            # Perform operation
            success = self.cache.set(key, value, ttl, level)
            
        except Exception as e:
            error = str(e)
            success = False
        
        finally:
            # Log operation
            duration = time.time() - start_time
            self._log_operation("set", key, level, success, duration, error)
        
        return success
    
    def get(self, key: str, level: str = "auto") -> Optional[Any]:
        """
        Get a value from the cache with validation.
        
        Args:
            key: Cache key
            level: Cache level to search in
            
        Returns:
            Cached value or None if not found/expired
            
        Raises:
            ValueError: If validation fails
        """
        start_time = time.time()
        success = False
        error = None
        result = None
        
        try:
            # Validate inputs
            if not self._validate_key(key):
                raise ValueError(f"Invalid key: {key}")
            if not self._validate_level(level):
                raise ValueError(f"Invalid level: {level}")
            
            # Perform operation
            result = self.cache.get(key, level)
            success = result is not None
            
        except Exception as e:
            error = str(e)
            success = False
            result = None
        
        finally:
            # Log operation
            duration = time.time() - start_time
            self._log_operation("get", key, level, success, duration, error)
        
        return result
    
    def delete(self, key: str, level: str = "auto") -> bool:
        """
        Delete a key from the cache with validation.
        
        Args:
            key: Cache key to delete
            level: Cache level
            
        Returns:
            True if deleted, False otherwise
            
        Raises:
            ValueError: If validation fails
        """
        start_time = time.time()
        success = False
        error = None
        
        try:
            # Validate inputs
            if not self._validate_key(key):
                raise ValueError(f"Invalid key: {key}")
            if not self._validate_level(level):
                raise ValueError(f"Invalid level: {level}")
            
            # Perform operation
            success = self.cache.delete(key, level)
            
        except Exception as e:
            error = str(e)
            success = False
        
        finally:
            # Log operation
            duration = time.time() - start_time
            self._log_operation("delete", key, level, success, duration, error)
        
        return success
    
    def exists(self, key: str, level: str = "auto") -> bool:
        """
        Check if a key exists in the cache with validation.
        
        Args:
            key: Cache key to check
            level: Cache level
            
        Returns:
            True if exists, False otherwise
            
        Raises:
            ValueError: If validation fails
        """
        start_time = time.time()
        success = False
        error = None
        result = False
        
        try:
            # Validate inputs
            if not self._validate_key(key):
                raise ValueError(f"Invalid key: {key}")
            if not self._validate_level(level):
                raise ValueError(f"Invalid level: {level}")
            
            # Perform operation
            result = self.cache.exists(key, level)
            success = True
            
        except Exception as e:
            error = str(e)
            success = False
            result = False
        
        finally:
            # Log operation
            duration = time.time() - start_time
            self._log_operation("exists", key, level, success, duration, error)
        
        return result
    
    def clear(self, level: Optional[str] = None) -> bool:
        """
        Clear all entries from specified level or all levels with validation.
        
        Args:
            level: Specific level to clear, or None for all levels
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If validation fails
        """
        start_time = time.time()
        success = False
        error = None
        
        try:
            # Validate level if specified
            if level is not None and not self._validate_level(level):
                raise ValueError(f"Invalid level: {level}")
            
            # Perform operation
            success = self.cache.clear(level)
            
        except Exception as e:
            error = str(e)
            success = False
        
        finally:
            # Log operation
            duration = time.time() - start_time
            operation = f"clear_{level}" if level else "clear_all"
            self._log_operation(operation, "N/A", level or "all", success, duration, error)
        
        return success
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()
    
    def get_operation_history(self, limit: int = 100) -> List[Dict]:
        """
        Get recent operation history.
        
        Args:
            limit: Maximum number of operations to return
            
        Returns:
            List of recent operations
        """
        recent_ops = self.operation_history[-limit:] if self.operation_history else []
        return [
            {
                'operation': op.operation,
                'key': op.key,
                'level': op.level,
                'timestamp': op.timestamp,
                'success': op.success,
                'duration': op.duration,
                'error': op.error
            }
            for op in recent_ops
        ]
    
    def get_performance_summary(self) -> dict:
        """
        Get performance summary from operation history.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.operation_history:
            return {}
        
        total_ops = len(self.operation_history)
        successful_ops = sum(1 for op in self.operation_history if op.success)
        failed_ops = total_ops - successful_ops
        
        # Calculate average durations by operation type
        operation_durations = {}
        for op in self.operation_history:
            if op.operation not in operation_durations:
                operation_durations[op.operation] = []
            operation_durations[op.operation].append(op.duration)
        
        avg_durations = {}
        for op_type, durations in operation_durations.items():
            avg_durations[op_type] = sum(durations) / len(durations)
        
        return {
            'total_operations': total_ops,
            'successful_operations': successful_ops,
            'failed_operations': failed_ops,
            'success_rate': (successful_ops / total_ops * 100) if total_ops > 0 else 0,
            'average_durations': avg_durations,
            'recent_operations': len(self.operation_history[-100:]) if self.operation_history else 0
        }

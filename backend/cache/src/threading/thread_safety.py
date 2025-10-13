"""
Thread Safety Implementation
Provides thread-safe access to cache operations with reader-writer locks.
"""

import threading
import time
from typing import Any, Optional
from contextlib import contextmanager


class ThreadSafety:
    """Provides thread safety for cache operations with reader-writer locks."""
    
    def __init__(self, max_readers: int = 10, lock_timeout: int = 30):
        """
        Initialize thread safety.
        
        Args:
            max_readers: Maximum concurrent readers
            lock_timeout: Lock acquisition timeout in seconds
        """
        self.max_readers = max_readers
        self.lock_timeout = lock_timeout
        
        # Main lock for write operations
        self._write_lock = threading.RLock()
        
        # Semaphore for limiting concurrent readers
        self._read_semaphore = threading.Semaphore(max_readers)
        
        # Reader count tracking
        self._reader_count = 0
        self._reader_count_lock = threading.Lock()
        
        # Performance tracking
        self._lock_contention = 0
        self._read_locks = 0
        self._write_locks = 0
        
        print(f"Thread Safety initialized: max_readers={max_readers}, timeout={lock_timeout}s")
    
    @contextmanager
    def read_lock(self):
        """
        Get read lock context manager.
        Multiple readers can access simultaneously.
        
        Yields:
            Lock context for read operations
        """
        start_time = time.time()
        
        try:
            # Acquire read semaphore
            if not self._read_semaphore.acquire(timeout=self.lock_timeout):
                raise TimeoutError(f"Read lock acquisition timeout after {self.lock_timeout}s")
            
            # Increment reader count
            with self._reader_count_lock:
                self._reader_count += 1
                self._read_locks += 1
            
            # Wait for any active write to complete
            self._write_lock.acquire()
            self._write_lock.release()
            
            yield self
            
        except Exception as e:
            self._lock_contention += 1
            raise e
        finally:
            # Decrement reader count
            with self._reader_count_lock:
                self._reader_count -= 1
            
            # Release read semaphore
            self._read_semaphore.release()
            
            # Track performance
            lock_time = time.time() - start_time
            if lock_time > 0.1:  # Track slow locks
                self._lock_contention += 1
    
    @contextmanager
    def write_lock(self):
        """
        Get write lock context manager.
        Exclusive access - no other reads or writes allowed.
        
        Yields:
            Lock context for write operations
        """
        start_time = time.time()
        
        try:
            # Acquire exclusive write lock
            if not self._write_lock.acquire(timeout=self.lock_timeout):
                raise TimeoutError(f"Write lock acquisition timeout after {self.lock_timeout}s")
            
            # Wait for all readers to complete
            while True:
                with self._reader_count_lock:
                    if self._reader_count == 0:
                        break
                time.sleep(0.001)  # Small sleep to avoid busy waiting
            
            self._write_locks += 1
            yield self
            
        except Exception as e:
            self._lock_contention += 1
            raise e
        finally:
            # Release write lock
            self._write_lock.release()
            
            # Track performance
            lock_time = time.time() - start_time
            if lock_time > 0.1:  # Track slow locks
                self._lock_contention += 1
    
    def get_stats(self) -> dict:
        """
        Get thread safety statistics.
        
        Returns:
            Dictionary with thread safety statistics
        """
        with self._reader_count_lock:
            return {
                'max_readers': self.max_readers,
                'current_readers': self._reader_count,
                'read_locks': self._read_locks,
                'write_locks': self._write_locks,
                'lock_contention': self._lock_contention,
                'lock_timeout': self.lock_timeout,
                'available_read_slots': self._read_semaphore._value
            }
    
    def is_deadlock_safe(self) -> bool:
        """
        Check if the current state is deadlock-safe.
        
        Returns:
            True if safe, False if potential deadlock
        """
        with self._reader_count_lock:
            # Check for potential deadlock conditions
            if self._reader_count > self.max_readers:
                return False
            
            # Check if write lock is held while readers are waiting
            if self._write_lock._owner == threading.current_thread() and self._reader_count > 0:
                return False
            
            return True
    
    def reset_stats(self):
        """Reset performance statistics."""
        with self._reader_count_lock:
            self._lock_contention = 0
            self._read_locks = 0
            self._write_locks = 0
    
    def __enter__(self):
        """Enter lock context (defaults to read lock)."""
        return self.read_lock().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit lock context."""
        pass  # Handled by context managers


class LockTimeoutError(TimeoutError):
    """Custom exception for lock timeouts."""
    pass

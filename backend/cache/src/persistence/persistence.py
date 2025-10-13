"""
Persistence Manager Implementation
Handles snapshot-based persistence and recovery for the cache system.
"""

import os
import json
import pickle
import time
import threading
import gzip
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import OrderedDict


class PersistenceManager:
    """
    Manages cache persistence with snapshots and append-only logging.
    
    Features:
    - Snapshot-based persistence (full cache state)
    - Append-only logging for all operations
    - Automatic recovery on startup
    - Background persistence processing
    - Data compression for efficiency
    """
    
    def __init__(self, config, storage_engine, ttl_manager, eviction_policy):
        """
        Initialize persistence manager.
        
        Args:
            config: Cache configuration
            storage_engine: Storage engine instance
            ttl_manager: TTL manager instance
            eviction_policy: Eviction policy instance
        """
        self.config = config
        self.storage_engine = storage_engine
        self.ttl_manager = ttl_manager
        self.eviction_policy = eviction_policy
        
        # Persistence settings
        self.persistence_dir = Path("cache_persistence")
        self.snapshot_dir = self.persistence_dir / "snapshots"
        self.log_dir = self.persistence_dir / "logs"
        
        # Create directories
        self._create_directories()
        
        # Background processing
        self.snapshot_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'snapshots_created': 0,
            'snapshots_loaded': 0,
            'operations_logged': 0,
            'recovery_time': 0,
            'last_snapshot': None,
            'last_log_cleanup': None
        }
        
        print(f"Persistence Manager initialized")
        print(f"   📁 Snapshot dir: {self.snapshot_dir}")
        print(f"   📁 Log dir: {self.log_dir}")
    
    def _create_directories(self):
        """Create persistence directories if they don't exist."""
        self.persistence_dir.mkdir(exist_ok=True)
        self.snapshot_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
    
    def start_background_persistence(self):
        """Start background snapshot creation."""
        if not self.running and self.config.persistence_enabled:
            self.running = True
            self.snapshot_thread = threading.Thread(
                target=self._snapshot_loop,
                daemon=True,
                name="PersistenceSnapshotThread"
            )
            self.snapshot_thread.start()
            print(f"Background persistence started (interval: {self.config.snapshot_interval}s)")
    
    def stop_background_persistence(self):
        """Stop background persistence."""
        self.running = False
        if self.snapshot_thread and self.snapshot_thread.is_alive():
            self.snapshot_thread.join(timeout=5)
            print("Background persistence stopped")
    
    def create_snapshot(self, force: bool = False) -> bool:
        """
        Create a snapshot of the current cache state.
        
        Args:
            force: Force snapshot creation even if recent snapshot exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.lock:
                # Check if we need a new snapshot
                if not force and self._should_skip_snapshot():
                    return True
                
                # Create snapshot data
                snapshot_data = self._collect_snapshot_data()
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_file = self.snapshot_dir / f"snapshot_{timestamp}.pkl.gz"
                
                # Save snapshot with compression
                with gzip.open(snapshot_file, 'wb') as f:
                    pickle.dump(snapshot_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Update statistics
                self.stats['snapshots_created'] += 1
                self.stats['last_snapshot'] = timestamp
                
                # Cleanup old snapshots
                self._cleanup_old_snapshots()
                
                print(f"📸 Snapshot created: {snapshot_file.name}")
                return True
                
        except Exception as e:
            print(f"Error creating snapshot: {e}")
            return False
    
    def load_snapshot(self, snapshot_file: Optional[str] = None) -> bool:
        """
        Load cache state from a snapshot.
        
        Args:
            snapshot_file: Specific snapshot file to load, or None for latest
            
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = time.time()
            
            with self.lock:
                # Find snapshot file
                if snapshot_file is None:
                    snapshot_file = self._find_latest_snapshot()
                
                if not snapshot_file or not os.path.exists(snapshot_file):
                    print("No snapshot found to load")
                    return False
                
                # Load snapshot data
                with gzip.open(snapshot_file, 'rb') as f:
                    snapshot_data = pickle.load(f)
                
                # Restore cache state
                self._restore_from_snapshot(snapshot_data)
                
                # Update statistics
                self.stats['snapshots_loaded'] += 1
                self.stats['recovery_time'] = time.time() - start_time
                
                print(f"Snapshot loaded: {Path(snapshot_file).name}")
                print(f"   Recovery time: {self.stats['recovery_time']:.2f}s")
                return True
                
        except Exception as e:
            print(f"Error loading snapshot: {e}")
            return False
    
    def log_operation(self, operation: str, key: str, level: str, 
                     value: Any = None, ttl: Optional[int] = None):
        """
        Log a cache operation to append-only log.
        
        Args:
            operation: Operation type (set, get, delete, clear)
            key: Cache key
            level: Cache level
            value: Value (for set operations)
            ttl: TTL value (for set operations)
        """
        try:
            if not self.config.persistence_enabled:
                return
            
            # Create log entry
            log_entry = {
                'timestamp': time.time(),
                'operation': operation,
                'key': key,
                'level': level,
                'value': value if operation == 'set' else None,
                'ttl': ttl if operation == 'set' else None
            }
            
            # Write to log file
            log_file = self.log_dir / f"operations_{datetime.now().strftime('%Y%m%d')}.log"
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Update statistics
            self.stats['operations_logged'] += 1
            
        except Exception as e:
            print(f"Error logging operation: {e}")
    
    def recover_from_logs(self, since_timestamp: Optional[float] = None) -> int:
        """
        Recover cache state from operation logs.
        
        Args:
            since_timestamp: Only replay operations since this timestamp
            
        Returns:
            Number of operations replayed
        """
        try:
            replayed_ops = 0
            
            # Find log files
            log_files = sorted(self.log_dir.glob("operations_*.log"))
            
            for log_file in log_files:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            
                            # Check timestamp filter
                            if since_timestamp and log_entry['timestamp'] <= since_timestamp:
                                continue
                            
                            # Replay operation
                            self._replay_operation(log_entry)
                            replayed_ops += 1
                            
                        except json.JSONDecodeError:
                            continue  # Skip malformed log entries
            
            print(f"Replayed {replayed_ops} operations from logs")
            return replayed_ops
            
        except Exception as e:
            print(f"Error recovering from logs: {e}")
            return 0
    
    def _collect_snapshot_data(self) -> Dict[str, Any]:
        """Collect all cache data for snapshot."""
        return {
            'timestamp': time.time(),
            'storage_data': {
                'storage': self.storage_engine.storage,
                'memory_usage': self.storage_engine.memory_usage,
                'total_memory': self.storage_engine.total_memory,
                'access_times': self.storage_engine.access_times
            },
            'ttl_data': self.ttl_manager.ttl_data,
            'eviction_data': {
                'access_order': dict(self.eviction_policy.access_order),
                'access_counts': self.eviction_policy.access_counts
            },
            'config': self.config.to_dict()
        }
    
    def _restore_from_snapshot(self, snapshot_data: Dict[str, Any]):
        """Restore cache state from snapshot data."""
        try:
            # Restore storage engine
            storage_data = snapshot_data['storage_data']
            self.storage_engine.storage = storage_data['storage']
            self.storage_engine.memory_usage = storage_data['memory_usage']
            self.storage_engine.total_memory = storage_data['total_memory']
            self.storage_engine.access_times = storage_data['access_times']
            
            # Restore TTL manager
            self.ttl_manager.ttl_data = snapshot_data['ttl_data']
            
            # Restore eviction policy
            eviction_data = snapshot_data['eviction_data']
            self.eviction_policy.access_order = OrderedDict(eviction_data['access_order'])
            self.eviction_policy.access_counts = eviction_data['access_counts']
            
            print(f"Restored {len(self.storage_engine.storage)} cache levels from snapshot")
            
        except Exception as e:
            print(f"Error restoring from snapshot: {e}")
            # Clear everything on error
            self.storage_engine.storage = {
                'query': {},
                'embedding': {},
                'context': {},
                'result': {}
            }
            self.storage_engine.memory_usage = {
                'query': 0,
                'embedding': 0,
                'context': 0,
                'result': 0
            }
            self.storage_engine.total_memory = 0
            self.storage_engine.access_times = {}
            self.ttl_manager.ttl_data = {}
            self.eviction_policy.access_order = OrderedDict()
            self.eviction_policy.access_counts = {}
    
    def _replay_operation(self, log_entry: Dict[str, Any]):
        """Replay a logged operation."""
        operation = log_entry['operation']
        key = log_entry['key']
        level = log_entry['level']
        
        if operation == 'set':
            value = log_entry['value']
            ttl = log_entry['ttl']
            self.storage_engine.set(key, value, level)
            if ttl:
                self.ttl_manager.set_ttl(key, ttl, level)
        elif operation == 'delete':
            self.storage_engine.delete(key, level)
            self.ttl_manager.remove_ttl(key)
        elif operation == 'clear':
            self.storage_engine.clear(level)
            if level:
                self.ttl_manager.clear_level(level)
            else:
                self.ttl_manager.clear_all()
    
    def _find_latest_snapshot(self) -> Optional[str]:
        """Find the most recent snapshot file."""
        snapshot_files = list(self.snapshot_dir.glob("snapshot_*.pkl.gz"))
        if not snapshot_files:
            return None
        
        # Sort by modification time (newest first)
        latest_file = max(snapshot_files, key=os.path.getmtime)
        return str(latest_file)
    
    def _should_skip_snapshot(self) -> bool:
        """Check if we should skip creating a new snapshot."""
        if not self.stats['last_snapshot']:
            return False
        
        # Check if enough time has passed
        last_snapshot_time = datetime.strptime(self.stats['last_snapshot'], "%Y%m%d_%H%M%S")
        time_since_last = datetime.now() - last_snapshot_time
        
        return time_since_last.total_seconds() < self.config.snapshot_interval
    
    def _cleanup_old_snapshots(self):
        """Cleanup old snapshot files."""
        try:
            snapshot_files = list(self.snapshot_dir.glob("snapshot_*.pkl.gz"))
            
            # Keep only the last 5 snapshots
            if len(snapshot_files) > 5:
                # Sort by modification time (oldest first)
                old_files = sorted(snapshot_files, key=os.path.getmtime)[:-5]
                
                for old_file in old_files:
                    old_file.unlink()
                    print(f"Cleaned up old snapshot: {old_file.name}")
                    
        except Exception as e:
            print(f"Error cleaning up snapshots: {e}")
    
    def _snapshot_loop(self):
        """Background loop for creating snapshots."""
        while self.running:
            try:
                self.create_snapshot()
                time.sleep(self.config.snapshot_interval)
            except Exception as e:
                print(f"Snapshot loop error: {e}")
                time.sleep(60)  # Wait a minute on error
    
    def cleanup_old_logs(self):
        """Cleanup old log files."""
        try:
            cutoff_date = datetime.now() - timedelta(seconds=self.config.log_retention)
            
            for log_file in self.log_dir.glob("operations_*.log"):
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                    log_file.unlink()
                    print(f"Cleaned up old log: {log_file.name}")
            
            self.stats['last_log_cleanup'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        except Exception as e:
            print(f"Error cleaning up logs: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        return {
            'snapshots_created': self.stats['snapshots_created'],
            'snapshots_loaded': self.stats['snapshots_loaded'],
            'operations_logged': self.stats['operations_logged'],
            'recovery_time': self.stats['recovery_time'],
            'last_snapshot': self.stats['last_snapshot'],
            'last_log_cleanup': self.stats['last_log_cleanup'],
            'persistence_enabled': self.config.persistence_enabled,
            'snapshot_interval': self.config.snapshot_interval,
            'log_retention': self.config.log_retention,
            'snapshot_dir': str(self.snapshot_dir),
            'log_dir': str(self.log_dir)
        }
    
    def get_snapshot_info(self) -> List[Dict[str, Any]]:
        """Get information about available snapshots."""
        snapshots = []
        
        for snapshot_file in self.snapshot_dir.glob("snapshot_*.pkl.gz"):
            stat = snapshot_file.stat()
            snapshots.append({
                'filename': snapshot_file.name,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created': datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Sort by creation time (newest first)
        snapshots.sort(key=lambda x: x['created'], reverse=True)
        return snapshots

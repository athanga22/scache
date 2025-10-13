"""
Adaptive Invalidation System
Handles smart cache invalidation based on data changes, model updates, and semantic drift.
"""

import time
import hashlib
import threading
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class InvalidationRule:
    """Rule for cache invalidation."""
    rule_type: str  # 'model_version', 'corpus_change', 'semantic_drift', 'time_based'
    threshold: float
    action: str  # 'invalidate', 'refresh', 'warn'
    affected_levels: List[str]
    description: str


class AdaptiveInvalidation:
    """
    Adaptive invalidation system that monitors and invalidates cache entries
    based on various triggers to maintain ≤5% stale entries.
    """
    
    def __init__(self, config, cache_storage, ttl_manager, eviction_policy):
        """
        Initialize adaptive invalidation system.
        
        Args:
            config: Cache configuration
            cache_storage: Cache storage engine
            ttl_manager: TTL manager
            eviction_policy: Eviction policy
        """
        self.config = config
        self.cache_storage = cache_storage
        self.ttl_manager = ttl_manager
        self.eviction_policy = eviction_policy
        
        # Version tracking
        self.model_versions: Dict[str, str] = {}
        self.corpus_versions: Dict[str, str] = {}
        self.cache_entry_versions: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Semantic drift tracking
        self.embedding_history: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.drift_threshold = 0.1  # 10% drift threshold
        
        # Invalidation rules
        self.invalidation_rules = self._setup_default_rules()
        
        # Statistics
        self.stats = {
            'invalidations': 0,
            'stale_entries_detected': 0,
            'model_version_changes': 0,
            'corpus_changes': 0,
            'semantic_drift_detected': 0,
            'total_entries_checked': 0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.monitoring_thread = None
        self.running = False
        
        print("Adaptive Invalidation System initialized")
    
    def _setup_default_rules(self) -> List[InvalidationRule]:
        """Setup default invalidation rules."""
        return [
            InvalidationRule(
                rule_type='model_version',
                threshold=0.0,  # Any change
                action='invalidate',
                affected_levels=['result', 'generation'],
                description='Invalidate when model version changes'
            ),
            InvalidationRule(
                rule_type='corpus_change',
                threshold=0.0,  # Any change
                action='invalidate',
                affected_levels=['context', 'retrieval'],
                description='Invalidate when corpus changes'
            ),
            InvalidationRule(
                rule_type='semantic_drift',
                threshold=0.1,  # 10% drift
                action='refresh',
                affected_levels=['query', 'embedding'],
                description='Refresh when semantic drift detected'
            ),
            InvalidationRule(
                rule_type='time_based',
                threshold=86400,  # 24 hours
                action='warn',
                affected_levels=['all'],
                description='Warn about old entries'
            )
        ]
    
    def start_monitoring(self):
        """Start background monitoring for invalidation triggers."""
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("Adaptive invalidation monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("Adaptive invalidation monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                self._check_invalidation_triggers()
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(f"Error in invalidation monitoring: {e}")
                time.sleep(60)
    
    def _check_invalidation_triggers(self):
        """Check all invalidation triggers."""
        with self.lock:
            # Check model version changes
            self._check_model_version_changes()
            
            # Check corpus changes
            self._check_corpus_changes()
            
            # Check semantic drift
            self._check_semantic_drift()
            
            # Check stale entries
            self._check_stale_entries()
    
    def _check_model_version_changes(self):
        """Check for model version changes."""
        # This would integrate with your model versioning system
        # For now, we'll simulate checking
        current_model_version = self._get_current_model_version()
        
        for level, stored_version in self.model_versions.items():
            if stored_version != current_model_version:
                print(f"Model version changed for {level}: {stored_version} -> {current_model_version}")
                self._invalidate_by_rule('model_version', level)
                self.model_versions[level] = current_model_version
                self.stats['model_version_changes'] += 1
    
    def _check_corpus_changes(self):
        """Check for corpus/document changes."""
        # This would integrate with your document management system
        current_corpus_version = self._get_current_corpus_version()
        
        for level, stored_version in self.corpus_versions.items():
            if stored_version != current_corpus_version:
                print(f"Corpus version changed for {level}: {stored_version} -> {current_corpus_version}")
                self._invalidate_by_rule('corpus_change', level)
                self.corpus_versions[level] = current_corpus_version
                self.stats['corpus_changes'] += 1
    
    def _check_semantic_drift(self):
        """Check for semantic drift in embeddings."""
        for query_hash, embedding_history in self.embedding_history.items():
            if len(embedding_history) < 2:
                continue
            
            # Calculate drift between first and last embedding
            first_embedding = embedding_history[0]
            last_embedding = embedding_history[-1]
            
            # Calculate cosine similarity
            similarity = np.dot(first_embedding, last_embedding) / (
                np.linalg.norm(first_embedding) * np.linalg.norm(last_embedding)
            )
            
            drift = 1 - similarity
            if drift > self.drift_threshold:
                print(f"Semantic drift detected for {query_hash}: {drift:.3f}")
                self._invalidate_by_rule('semantic_drift', 'query')
                self.stats['semantic_drift_detected'] += 1
    
    def _check_stale_entries(self):
        """Check for stale entries based on age and usage."""
        current_time = time.time()
        stale_count = 0
        total_count = 0
        
        # Check all cache entries across all levels
        for level_name, level_storage in self.cache_storage.storage.items():
            for key, entry_data in level_storage.items():
                total_count += 1
                entry_time = entry_data.get('timestamp', 0)
                age = current_time - entry_time
                
                # Consider stale if older than 24 hours and not accessed recently
                if age > 86400:  # 24 hours
                    last_access = self.eviction_policy.access_order.get(key, 0)
                    if current_time - last_access > 3600:  # Not accessed in 1 hour
                        stale_count += 1
        
        # Calculate stale percentage
        if total_count > 0:
            stale_percentage = (stale_count / total_count) * 100
            if stale_percentage > 5:  # More than 5% stale
                print(f"High stale percentage detected: {stale_percentage:.1f}%")
                self._cleanup_stale_entries()
                self.stats['stale_entries_detected'] += stale_count
        
        self.stats['total_entries_checked'] = total_count
    
    def _invalidate_by_rule(self, rule_type: str, level: str):
        """Invalidate cache entries based on a rule."""
        rule = next((r for r in self.invalidation_rules if r.rule_type == rule_type), None)
        if not rule:
            return
        
        if level in rule.affected_levels or 'all' in rule.affected_levels:
            if rule.action == 'invalidate':
                self._invalidate_level(level)
            elif rule.action == 'refresh':
                self._refresh_level(level)
            elif rule.action == 'warn':
                print(f"Warning: {rule.description} for level {level}")
            
            self.stats['invalidations'] += 1
    
    def _invalidate_level(self, level: str):
        """Invalidate all entries in a specific level."""
        keys_to_remove = []
        if level in self.cache_storage.storage:
            for key, entry_data in self.cache_storage.storage[level].items():
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.cache_storage.delete(key, level)
            self.ttl_manager.remove_ttl(key)
            self.eviction_policy.remove_key(key)
        
        print(f"Invalidated {len(keys_to_remove)} entries from level {level}")
    
    def _refresh_level(self, level: str):
        """Refresh entries in a specific level (extend TTL)."""
        refreshed_count = 0
        for key, entry_data in self.cache_storage.storage.items():
            if entry_data.get('level') == level:
                # Extend TTL by default amount
                self.ttl_manager.extend_ttl(key)
                refreshed_count += 1
        
        print(f"Refreshed {refreshed_count} entries in level {level}")
    
    def _cleanup_stale_entries(self):
        """Clean up stale entries to maintain ≤5% stale rate."""
        current_time = time.time()
        stale_keys = []
        
        for level_name, level_storage in self.cache_storage.storage.items():
            for key, entry_data in level_storage.items():
                entry_time = entry_data.get('timestamp', 0)
                age = current_time - entry_time
                
                if age > 86400:  # Older than 24 hours
                    last_access = self.eviction_policy.access_order.get(key, 0)
                    if current_time - last_access > 3600:  # Not accessed in 1 hour
                        stale_keys.append(key)
        
        # Remove stale entries
        for key in stale_keys:
            # Find which level this key belongs to
            key_level = None
            for level_name, level_storage in self.cache_storage.storage.items():
                if key in level_storage:
                    key_level = level_name
                    break
            
            if key_level:
                self.cache_storage.delete(key, key_level)
                self.ttl_manager.remove_ttl(key)
                self.eviction_policy.remove_key(key)
        
        print(f"Cleaned up {len(stale_keys)} stale entries")
    
    def _get_current_model_version(self) -> str:
        """Get current model version (placeholder implementation)."""
        # This would integrate with your model versioning system
        return f"model_v{int(time.time() / 3600)}"  # Change every hour for testing
    
    def _get_current_corpus_version(self) -> str:
        """Get current corpus version (placeholder implementation)."""
        # This would integrate with your document management system
        return f"corpus_v{int(time.time() / 1800)}"  # Change every 30 minutes for testing
    
    def track_embedding_change(self, query_hash: str, embedding: np.ndarray):
        """Track embedding changes for drift detection."""
        with self.lock:
            self.embedding_history[query_hash].append(embedding.copy())
            
            # Keep only last 10 embeddings to prevent memory growth
            if len(self.embedding_history[query_hash]) > 10:
                self.embedding_history[query_hash] = self.embedding_history[query_hash][-10:]
    
    def get_invalidation_stats(self) -> Dict[str, Any]:
        """Get invalidation statistics."""
        with self.lock:
            total_entries = self.stats['total_entries_checked']
            stale_entries = self.stats['stale_entries_detected']
            stale_percentage = (stale_entries / total_entries * 100) if total_entries > 0 else 0
            
            return {
                'total_invalidations': self.stats['invalidations'],
                'stale_entries_detected': stale_entries,
                'stale_percentage': stale_percentage,
                'model_version_changes': self.stats['model_version_changes'],
                'corpus_changes': self.stats['corpus_changes'],
                'semantic_drift_detected': self.stats['semantic_drift_detected'],
                'total_entries_checked': total_entries,
                'goal_achieved': stale_percentage <= 5.0
            }

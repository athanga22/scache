"""
Cache Warming System
Implements intelligent cache warming strategies for optimal performance.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class WarmingStrategy:
    """Represents a cache warming strategy."""
    name: str
    priority: int  # Higher number = higher priority
    enabled: bool
    frequency: int  # How often to run (seconds)
    last_run: float = 0
    success_count: int = 0
    failure_count: int = 0


class CacheWarming:
    """
    Intelligent cache warming system.
    
    Features:
    - Access pattern analysis
    - Popular content identification
    - Background warming processes
    - Configurable warming strategies
    - Performance monitoring
    """
    
    def __init__(self, cache, config):
        """
        Initialize cache warming system.
        
        Args:
            cache: Cache instance
            config: Cache configuration
        """
        self.cache = cache
        self.config = config
        
        # Access pattern tracking
        self.access_patterns = defaultdict(int)  # key -> access_count
        self.access_timestamps = defaultdict(list)  # key -> [timestamps]
        self.query_patterns = defaultdict(int)  # query -> access_count
        
        # Warming strategies
        self.strategies = {
            'popular_content': WarmingStrategy(
                name='popular_content',
                priority=1,
                enabled=True,
                frequency=300  # 5 minutes
            ),
            'frequent_queries': WarmingStrategy(
                name='frequent_queries',
                priority=2,
                enabled=True,
                frequency=600  # 10 minutes
            ),
            'semantic_clusters': WarmingStrategy(
                name='semantic_clusters',
                priority=3,
                enabled=True,
                frequency=900  # 15 minutes
            )
        }
        
        # Warming thread
        self.warming_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'warming_operations': 0,
            'successful_warming': 0,
            'failed_warming': 0,
            'popular_items_identified': 0,
            'last_warming_run': None
        }
        
        print("Cache Warming system initialized")
    
    def start_warming(self):
        """Start the cache warming thread."""
        if not self.running:
            self.running = True
            self.warming_thread = threading.Thread(
                target=self._warming_loop,
                daemon=True,
                name="CacheWarmingThread"
            )
            self.warming_thread.start()
            print("Cache warming started")
    
    def stop_warming(self):
        """Stop the cache warming thread."""
        self.running = False
        if self.warming_thread and self.warming_thread.is_alive():
            self.warming_thread.join(timeout=5)
            print("Cache warming stopped")
    
    def record_access(self, key: str, query: Optional[str] = None):
        """
        Record cache access for pattern analysis.
        
        Args:
            key: Cache key that was accessed
            query: Original query (if available)
        """
        try:
            with self.lock:
                current_time = time.time()
                
                # Record access pattern
                self.access_patterns[key] += 1
                self.access_timestamps[key].append(current_time)
                
                # Keep only recent timestamps (last 24 hours)
                cutoff_time = current_time - (24 * 3600)
                self.access_timestamps[key] = [
                    ts for ts in self.access_timestamps[key] if ts > cutoff_time
                ]
                
                # Record query pattern if available
                if query:
                    self.query_patterns[query] += 1
                
        except Exception as e:
            print(f"Error recording access: {e}")
    
    def _warming_loop(self):
        """Main warming loop."""
        while self.running:
            try:
                current_time = time.time()
                
                # Run enabled strategies
                for strategy in self.strategies.values():
                    if strategy.enabled:
                        if current_time - strategy.last_run >= strategy.frequency:
                            self._run_strategy(strategy)
                            strategy.last_run = current_time
                
                # Sleep for a short time
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Warming loop error: {e}")
                time.sleep(60)
    
    def _run_strategy(self, strategy: WarmingStrategy):
        """Run a specific warming strategy."""
        try:
            print(f"Running warming strategy: {strategy.name}")
            
            if strategy.name == 'popular_content':
                self._warm_popular_content()
            elif strategy.name == 'frequent_queries':
                self._warm_frequent_queries()
            elif strategy.name == 'semantic_clusters':
                self._warm_semantic_clusters()
            
            strategy.success_count += 1
            self.stats['successful_warming'] += 1
            
        except Exception as e:
            print(f"Strategy {strategy.name} failed: {e}")
            strategy.failure_count += 1
            self.stats['failed_warming'] += 1
        
        finally:
            self.stats['warming_operations'] += 1
            self.stats['last_warming_run'] = time.time()
    
    def _warm_popular_content(self):
        """Warm popular content based on access patterns."""
        try:
            # Get most frequently accessed items
            popular_items = sorted(
                self.access_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10 popular items
            
            for key, access_count in popular_items:
                # Check if item is still in cache
                if not self.cache.exists(key):
                    # Try to reload from persistence or external source
                    # This is a placeholder - in real implementation, you'd reload from source
                    print(f"   Warming popular item: {key} (accessed {access_count} times)")
            
            self.stats['popular_items_identified'] = len(popular_items)
            
        except Exception as e:
            print(f"Error warming popular content: {e}")
    
    def _warm_frequent_queries(self):
        """Warm cache based on frequent query patterns."""
        try:
            # Get most frequent queries
            frequent_queries = sorted(
                self.query_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5 frequent queries
            
            for query, frequency in frequent_queries:
                # Check if we have a cached result for this query
                if not self.cache.exists(query, "result"):
                    # In a real implementation, you'd pre-compute the result
                    print(f"   Warming frequent query: {query[:50]}... (used {frequency} times)")
            
        except Exception as e:
            print(f"Error warming frequent queries: {e}")
    
    def _warm_semantic_clusters(self):
        """Warm cache based on semantic query clusters."""
        try:
            # Get similarity engine stats
            similarity_stats = self.cache.get_similarity_stats()
            
            # Find queries with high similarity scores
            if similarity_stats.get('embeddings_cached', 0) > 0:
                print(f"   Warming semantic clusters: {similarity_stats['embeddings_cached']} embeddings cached")
                
                # In a real implementation, you'd identify semantic clusters
                # and pre-warm related content
            
        except Exception as e:
            print(f"Error warming semantic clusters: {e}")
    
    def get_popular_items(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of popular cache items."""
        try:
            popular_items = sorted(
                self.access_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]
            
            return [
                {
                    'key': key,
                    'access_count': count,
                    'recent_accesses': len(self.access_timestamps.get(key, [])),
                    'last_access': max(self.access_timestamps.get(key, [0]))
                }
                for key, count in popular_items
            ]
            
        except Exception as e:
            print(f"Error getting popular items: {e}")
            return []
    
    def get_frequent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of frequent queries."""
        try:
            frequent_queries = sorted(
                self.query_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]
            
            return [
                {
                    'query': query,
                    'frequency': count,
                    'cached': self.cache.exists(query, "result")
                }
                for query, count in frequent_queries
            ]
            
        except Exception as e:
            print(f"Error getting frequent queries: {e}")
            return []
    
    def get_warming_recommendations(self) -> List[Dict[str, Any]]:
        """Get cache warming recommendations."""
        recommendations = []
        
        try:
            # Analyze access patterns
            total_accesses = sum(self.access_patterns.values())
            if total_accesses > 0:
                # Find items with high access frequency
                for key, count in self.access_patterns.items():
                    if count > total_accesses * 0.1:  # More than 10% of total accesses
                        if not self.cache.exists(key):
                            recommendations.append({
                                'type': 'popular_item',
                                'priority': 'high',
                                'message': f'Item "{key}" is frequently accessed but not cached',
                                'action': f'Consider pre-loading item "{key}"'
                            })
            
            # Analyze query patterns
            total_queries = sum(self.query_patterns.values())
            if total_queries > 0:
                for query, count in self.query_patterns.items():
                    if count > total_queries * 0.05:  # More than 5% of total queries
                        if not self.cache.exists(query, "result"):
                            recommendations.append({
                                'type': 'frequent_query',
                                'priority': 'medium',
                                'message': f'Query "{query[:50]}..." is frequent but not cached',
                                'action': f'Consider pre-computing result for "{query[:50]}..."'
                            })
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def enable_strategy(self, strategy_name: str):
        """Enable a warming strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = True
            print(f"Enabled warming strategy: {strategy_name}")
    
    def disable_strategy(self, strategy_name: str):
        """Disable a warming strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = False
            print(f"Disabled warming strategy: {strategy_name}")
    
    def set_strategy_frequency(self, strategy_name: str, frequency: int):
        """Set the frequency for a warming strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].frequency = frequency
            print(f"Set {strategy_name} frequency to {frequency}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache warming statistics."""
        return {
            'warming_active': self.running,
            'strategies': {
                name: {
                    'enabled': strategy.enabled,
                    'frequency': strategy.frequency,
                    'success_count': strategy.success_count,
                    'failure_count': strategy.failure_count,
                    'last_run': strategy.last_run
                }
                for name, strategy in self.strategies.items()
            },
            'access_patterns_tracked': len(self.access_patterns),
            'query_patterns_tracked': len(self.query_patterns),
            'total_warming_operations': self.stats['warming_operations'],
            'successful_warming': self.stats['successful_warming'],
            'failed_warming': self.stats['failed_warming'],
            'popular_items_identified': self.stats['popular_items_identified'],
            'last_warming_run': self.stats['last_warming_run']
        }

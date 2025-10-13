"""
Cache Metrics and Monitoring
Provides comprehensive monitoring and metrics for the caching system.
"""

from typing import Dict, Any


class CacheMetrics:
    """Monitors and reports cache performance metrics."""
    
    def __init__(self, cache):
        """Initialize metrics collection."""
        self.cache = cache
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics."""
        stats = self.cache.get_stats()
        
        return {
            'performance': {
                'hit_rate': stats.get('hit_rate', 0),
                'total_requests': stats.get('total_requests', 0),
                'hits': stats.get('hits', 0),
                'misses': stats.get('misses', 0)
            },
            'memory': {
                'usage_bytes': stats.get('memory_usage', {}).get('total_bytes', 0),
                'usage_mb': stats.get('memory_usage', {}).get('total_mb', 0),
                'limit_bytes': stats.get('memory_usage', {}).get('limit_bytes', 0),
                'usage_percentage': stats.get('memory_usage', {}).get('usage_percentage', 0)
            },
            'operations': {
                'sets': stats.get('sets', 0),
                'deletes': stats.get('deletes', 0),
                'evictions': stats.get('evictions', 0)
            },
            'storage': {
                'total_entries': stats.get('total_entries', 0),
                'entries_by_level': stats.get('entries_by_level', {})
            }
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary of cache performance."""
        stats = self.cache.get_stats()
        
        return f"""
Cache Performance Summary:
- Hit Rate: {stats.get('hit_rate', 0)}%
- Total Requests: {stats.get('total_requests', 0)}
- Memory Usage: {stats.get('memory_usage', {}).get('total_mb', 0)} MB / {stats.get('memory_usage', {}).get('limit_mb', 0)} MB
- Total Entries: {stats.get('total_entries', 0)}
- Memory Pressure: {stats.get('memory_pressure', 'low')}
        """.strip()
    
    def is_performing_well(self) -> bool:
        """Check if cache is performing well."""
        stats = self.cache.get_stats()
        hit_rate = stats.get('hit_rate', 0)
        memory_usage = stats.get('memory_usage', {}).get('usage_percentage', 0)
        
        return hit_rate > 50 and memory_usage < 90

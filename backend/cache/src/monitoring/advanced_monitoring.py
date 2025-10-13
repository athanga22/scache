"""
Advanced Monitoring and Alerting System
Provides comprehensive monitoring, alerting, and optimization for the cache system.
"""

import time
import threading
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict


@dataclass
class Alert:
    """Represents a monitoring alert."""
    timestamp: float
    level: str  # 'info', 'warning', 'critical'
    category: str  # 'memory', 'performance', 'error'
    message: str
    details: Dict[str, Any]


class AdvancedMonitoring:
    """
    Advanced monitoring system with real-time alerts and optimization.
    
    Features:
    - Real-time memory monitoring
    - Performance degradation detection
    - Automatic optimization recommendations
    - Alert system with configurable thresholds
    - Historical performance tracking
    """
    
    def __init__(self, cache, config):
        """
        Initialize advanced monitoring.
        
        Args:
            cache: Cache instance to monitor
            config: Cache configuration
        """
        self.cache = cache
        self.config = config
        
        # Monitoring settings
        self.monitoring_interval = 30  # seconds
        self.history_size = 1000
        self.alert_cooldown = 300  # 5 minutes
        
        # Data storage
        self.performance_history = deque(maxlen=self.history_size)
        self.memory_history = deque(maxlen=self.history_size)
        self.alerts = deque(maxlen=100)
        
        # Alert thresholds
        self.thresholds = {
            'memory_warning': 0.8,      # 80%
            'memory_critical': 0.95,    # 95%
            'hit_rate_low': 0.3,        # 30%
            'response_time_slow': 0.1,  # 100ms
            'error_rate_high': 0.05     # 5%
        }
        
        # Alert cooldowns (prevent spam)
        self.last_alerts = {}
        
        # Monitoring thread
        self.monitoring_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Performance tracking
        self.operation_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        print("Advanced Monitoring initialized")
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if not self.running:
            self.running = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="AdvancedMonitoringThread"
            )
            self.monitoring_thread.start()
            print(f"Advanced monitoring started (interval: {self.monitoring_interval}s)")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
            print("Advanced monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_metrics()
                self._check_thresholds()
                self._generate_recommendations()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self._create_alert('critical', 'error', f"Monitoring error: {e}")
                time.sleep(60)  # Wait a minute on error
    
    def _collect_metrics(self):
        """Collect current system metrics."""
        try:
            # Get cache statistics
            cache_stats = self.cache.get_stats()
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            # Create metrics record
            metrics = {
                'timestamp': time.time(),
                'cache_hit_rate': cache_stats.get('hit_rate', 0),
                'cache_memory_usage': cache_stats.get('memory_usage', {}).get('usage_percentage', 0),
                'cache_total_entries': cache_stats.get('total_entries', 0),
                'system_memory_usage': system_memory.percent,
                'system_memory_available': system_memory.available,
                'cache_operations': {
                    'hits': cache_stats.get('hits', 0),
                    'misses': cache_stats.get('misses', 0),
                    'sets': cache_stats.get('sets', 0),
                    'deletes': cache_stats.get('deletes', 0)
                }
            }
            
            # Store in history
            with self.lock:
                self.performance_history.append(metrics)
                self.memory_history.append({
                    'timestamp': time.time(),
                    'cache_memory': cache_stats.get('memory_usage', {}).get('total_bytes', 0),
                    'system_memory': system_memory.used
                })
            
        except Exception as e:
            self._create_alert('critical', 'error', f"Metrics collection error: {e}")
    
    def _check_thresholds(self):
        """Check if any thresholds have been exceeded."""
        if not self.performance_history:
            return
        
        latest_metrics = self.performance_history[-1]
        
        # Check memory thresholds
        cache_memory_usage = latest_metrics['cache_memory_usage'] / 100  # Convert to decimal
        if cache_memory_usage >= self.thresholds['memory_critical']:
            self._create_alert('critical', 'memory', 
                             f"Cache memory usage critical: {cache_memory_usage:.1%}")
        elif cache_memory_usage >= self.thresholds['memory_warning']:
            self._create_alert('warning', 'memory', 
                             f"Cache memory usage high: {cache_memory_usage:.1%}")
        
        # Check hit rate
        hit_rate = latest_metrics['cache_hit_rate'] / 100  # Convert to decimal
        if hit_rate < self.thresholds['hit_rate_low']:
            self._create_alert('warning', 'performance', 
                             f"Cache hit rate low: {hit_rate:.1%}")
        
        # Check system memory
        system_memory_usage = latest_metrics['system_memory_usage'] / 100
        if system_memory_usage >= 0.9:  # 90% system memory
            self._create_alert('critical', 'memory', 
                             f"System memory usage critical: {system_memory_usage:.1%}")
    
    def _create_alert(self, level: str, category: str, message: str, details: Dict = None):
        """Create a new alert."""
        try:
            alert_key = f"{level}_{category}_{message}"
            current_time = time.time()
            
            # Check cooldown
            if alert_key in self.last_alerts:
                if current_time - self.last_alerts[alert_key] < self.alert_cooldown:
                    return  # Skip due to cooldown
            
            # Create alert
            alert = Alert(
                timestamp=current_time,
                level=level,
                category=category,
                message=message,
                details=details or {}
            )
            
            with self.lock:
                self.alerts.append(alert)
                self.last_alerts[alert_key] = current_time
            
            # Print alert
            emoji = "[CRITICAL]" if level == "critical" else "[WARNING]" if level == "warning" else "[INFO]"
            print(f"{emoji} ALERT [{level.upper()}] {category}: {message}")
            
        except Exception as e:
            print(f"Error creating alert: {e}")
    
    def _generate_recommendations(self):
        """Generate optimization recommendations."""
        if len(self.performance_history) < 10:
            return  # Need some history
        
        try:
            # Analyze recent performance
            recent_metrics = list(self.performance_history)[-10:]
            
            # Calculate averages
            avg_hit_rate = sum(m['cache_hit_rate'] for m in recent_metrics) / len(recent_metrics)
            avg_memory_usage = sum(m['cache_memory_usage'] for m in recent_metrics) / len(recent_metrics)
            
            # Generate recommendations
            recommendations = []
            
            if avg_hit_rate < 50:
                recommendations.append({
                    'type': 'hit_rate',
                    'priority': 'high',
                    'message': 'Consider increasing cache size or adjusting TTL values',
                    'details': {'current_hit_rate': avg_hit_rate}
                })
            
            if avg_memory_usage > 80:
                recommendations.append({
                    'type': 'memory',
                    'priority': 'high',
                    'message': 'Consider reducing cache size or implementing more aggressive eviction',
                    'details': {'current_memory_usage': avg_memory_usage}
                })
            
            # Store recommendations (could be used for auto-optimization)
            if recommendations:
                self._create_alert('info', 'optimization', 
                                 f"Generated {len(recommendations)} optimization recommendations")
            
        except Exception as e:
            self._create_alert('critical', 'error', f"Recommendation generation error: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        if not self.performance_history:
            return {}
        
        try:
            latest_metrics = self.performance_history[-1]
            
            # Calculate trends
            if len(self.performance_history) >= 10:
                recent_avg = sum(m['cache_hit_rate'] for m in list(self.performance_history)[-10:]) / 10
                older_avg = sum(m['cache_hit_rate'] for m in list(self.performance_history)[-20:-10]) / 10
                hit_rate_trend = recent_avg - older_avg
            else:
                hit_rate_trend = 0
            
            return {
                'current_metrics': latest_metrics,
                'hit_rate_trend': hit_rate_trend,
                'total_alerts': len(self.alerts),
                'recent_alerts': len([a for a in self.alerts if time.time() - a.timestamp < 3600]),
                'monitoring_duration': time.time() - (self.performance_history[0]['timestamp'] if self.performance_history else time.time()),
                'recommendations': self._get_current_recommendations()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_current_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations."""
        recommendations = []
        
        if not self.performance_history:
            return recommendations
        
        latest = self.performance_history[-1]
        
        # Memory recommendations
        if latest['cache_memory_usage'] > 80:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'high',
                'message': 'High memory usage detected',
                'action': 'Consider reducing cache size or implementing more aggressive eviction'
            })
        
        # Hit rate recommendations
        if latest['cache_hit_rate'] < 50:
            recommendations.append({
                'type': 'hit_rate_optimization',
                'priority': 'medium',
                'message': 'Low hit rate detected',
                'action': 'Consider increasing cache size or adjusting TTL values'
            })
        
        return recommendations
    
    def get_alerts(self, level: Optional[str] = None, 
                   category: Optional[str] = None, 
                   hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            filtered_alerts = []
            for alert in self.alerts:
                if alert.timestamp < cutoff_time:
                    continue
                
                if level and alert.level != level:
                    continue
                
                if category and alert.category != category:
                    continue
                
                filtered_alerts.append({
                    'timestamp': alert.timestamp,
                    'datetime': datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    'level': alert.level,
                    'category': alert.category,
                    'message': alert.message,
                    'details': alert.details
                })
            
            return filtered_alerts
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def get_memory_analysis(self) -> Dict[str, Any]:
        """Get detailed memory usage analysis."""
        if not self.memory_history:
            return {}
        
        try:
            latest = self.memory_history[-1]
            
            # Calculate memory growth rate
            if len(self.memory_history) >= 10:
                recent_avg = sum(m['cache_memory'] for m in list(self.memory_history)[-10:]) / 10
                older_avg = sum(m['cache_memory'] for m in list(self.memory_history)[-20:-10]) / 10
                growth_rate = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            else:
                growth_rate = 0
            
            return {
                'current_cache_memory': latest['cache_memory'],
                'current_system_memory': latest['system_memory'],
                'memory_growth_rate': growth_rate,
                'memory_pressure_level': self._get_memory_pressure_level(latest),
                'recommendations': self._get_memory_recommendations(latest)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_memory_pressure_level(self, memory_data: Dict) -> str:
        """Get current memory pressure level."""
        cache_memory = memory_data['cache_memory']
        system_memory = memory_data['system_memory']
        
        # This would need to be compared against cache limits
        # For now, use simple heuristics
        if cache_memory > 100 * 1024 * 1024:  # 100MB
            return 'high'
        elif cache_memory > 50 * 1024 * 1024:  # 50MB
            return 'medium'
        else:
            return 'low'
    
    def _get_memory_recommendations(self, memory_data: Dict) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        
        cache_memory = memory_data['cache_memory']
        
        if cache_memory > 100 * 1024 * 1024:  # 100MB
            recommendations.append("Consider reducing cache size")
            recommendations.append("Implement more aggressive eviction policy")
        
        if cache_memory > 50 * 1024 * 1024:  # 50MB
            recommendations.append("Monitor memory usage closely")
        
        return recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            'monitoring_active': self.running,
            'monitoring_interval': self.monitoring_interval,
            'performance_history_size': len(self.performance_history),
            'memory_history_size': len(self.memory_history),
            'total_alerts': len(self.alerts),
            'thresholds': self.thresholds,
            'last_monitoring': self.performance_history[-1]['timestamp'] if self.performance_history else None
        }

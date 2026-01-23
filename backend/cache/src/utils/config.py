"""
Cache Configuration Management
Handles all configuration settings for the caching system.
"""

import os
import psutil
from typing import Dict, Any, Union
from dataclasses import dataclass, field


@dataclass
class CacheConfig:
    """
    Configuration class for the caching system.
    
    Attributes:
        memory_limit: Memory limit for the cache (e.g., "25%", "2GB")
        ttl_enabled: Whether TTL expiration is enabled
        eviction_policy: Eviction policy to use (lru, lfu, hybrid)
        ttl_default: Default TTL in seconds
        ttl_extension_on_hit: Whether to extend TTL on cache hits
        cleanup_interval: Background cleanup interval in seconds
        memory_thresholds: Memory usage thresholds for eviction
        similarity_threshold: Threshold for semantic similarity matching
        persistence_enabled: Whether persistence is enabled
        snapshot_interval: Snapshot saving interval in seconds
    """
    
    # Memory settings
    memory_limit: str = "25%"
    memory_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "warning": 0.8,    # 80% - warning level
        "eviction": 0.9,   # 90% - start eviction
        "critical": 0.95   # 95% - aggressive eviction
    })
    
    # TTL settings
    ttl_enabled: bool = True
    ttl_default: int = 3600  # 1 hour
    ttl_extension_on_hit: bool = True
    ttl_by_level: Dict[str, int] = field(default_factory=lambda: {
        "query": 300,      # 5 minutes
        "embedding": 1800, # 30 minutes
        "context": 900,    # 15 minutes
        "result": 3600     # 1 hour
    })
    
    # Eviction settings
    eviction_policy: str = "lru"  # lru, lfu, hybrid
    eviction_batch_size: int = 10
    max_entries: int = 50  # Maximum number of cache entries
    eviction_threshold: float = 0.8  # Evict when 80% of max_entries reached
    
    # Performance settings
    cleanup_interval: int = 60  # 1 minute
    similarity_threshold: float = 0.75  # Higher threshold for better quality matches
    
    # Embedding provider settings
    embedding_provider: str = "sentence-transformers"  # Options: "google", "openai", "sentence-transformers", "huggingface"
    embedding_model: str = "all-mpnet-base-v2"  # Model name for sentence-transformers/huggingface
    
    # Persistence settings
    persistence_enabled: bool = True
    snapshot_interval: int = 300  # 5 minutes
    log_retention: int = 86400    # 24 hours
    
    # Threading settings
    max_threads: int = 4
    lock_timeout: int = 30
    
    def __post_init__(self):
        """Post-initialization processing."""
        self._parse_memory_limit()
        self._validate_config()
    
    def _parse_memory_limit(self):
        """Parse memory limit string into bytes."""
        if isinstance(self.memory_limit, str):
            if self.memory_limit.endswith('%'):
                # Percentage of system RAM
                percentage = float(self.memory_limit.rstrip('%')) / 100
                self.memory_limit_bytes = int(psutil.virtual_memory().total * percentage)
            else:
                # Parse size strings like "2GB", "512MB"
                self.memory_limit_bytes = self._parse_size_string(self.memory_limit)
        else:
            self.memory_limit_bytes = self.memory_limit
    
    def _parse_size_string(self, size_str: str) -> int:
        """Parse size string into bytes."""
        size_str = size_str.upper().strip()
        
        # Define size multipliers - ORDER MATTERS! Longer units first
        multipliers = [
            ('TB', 1024 ** 4),
            ('GB', 1024 ** 3),
            ('MB', 1024 ** 2),
            ('KB', 1024),
            ('B', 1),
            # Also support single letter versions
            ('T', 1024 ** 4),
            ('G', 1024 ** 3),
            ('M', 1024 ** 2),
            ('K', 1024)
        ]
        
        # Find the multiplier - check longer units first
        multiplier = 1
        for unit, mult in multipliers:
            if size_str.endswith(unit):
                multiplier = mult
                size_str = size_str[:-len(unit)]
                break
        
        try:
            # Handle case where size_str might be empty (e.g., "100MB" -> "")
            if not size_str:
                raise ValueError("No numeric value found in memory size string")
            return int(float(size_str) * multiplier)
        except ValueError:
            raise ValueError(f"Invalid memory size format: {size_str}")
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        
        if self.ttl_default < 0:
            raise ValueError("TTL default must be positive")
        
        if self.cleanup_interval < 1:
            raise ValueError("Cleanup interval must be at least 1 second")
        
        if self.eviction_policy not in ["lru", "lfu", "hybrid"]:
            raise ValueError("Eviction policy must be lru, lfu, or hybrid")
    
    def get_memory_limit_bytes(self) -> int:
        """Get memory limit in bytes."""
        return self.memory_limit_bytes
    
    def get_memory_limit_mb(self) -> float:
        """Get memory limit in MB."""
        return self.memory_limit_bytes / (1024 ** 2)
    
    def get_memory_limit_gb(self) -> float:
        """Get memory limit in GB."""
        return self.memory_limit_bytes / (1024 ** 3)
    
    def get_ttl_for_level(self, level: str) -> int:
        """Get TTL value for a specific cache level."""
        return self.ttl_by_level.get(level, self.ttl_default)
    
    def get_memory_threshold(self, threshold_name: str) -> float:
        """Get memory threshold value."""
        return self.memory_thresholds.get(threshold_name, 0.9)
    
    def is_memory_warning(self, current_usage: float) -> bool:
        """Check if memory usage is at warning level."""
        return current_usage >= self.get_memory_threshold("warning")
    
    def is_memory_eviction(self, current_usage: float) -> bool:
        """Check if memory usage requires eviction."""
        return current_usage >= self.get_memory_threshold("eviction")
    
    def is_memory_critical(self, current_usage: float) -> bool:
        """Check if memory usage is critical."""
        return current_usage >= self.get_memory_threshold("critical")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'memory_limit': self.memory_limit,
            'memory_limit_bytes': self.memory_limit_bytes,
            'ttl_enabled': self.ttl_enabled,
            'ttl_default': self.ttl_default,
            'ttl_by_level': self.ttl_by_level,
            'eviction_policy': self.eviction_policy,
            'cleanup_interval': self.cleanup_interval,
            'similarity_threshold': self.similarity_threshold,
            'persistence_enabled': self.persistence_enabled,
            'snapshot_interval': self.snapshot_interval,
            'memory_thresholds': self.memory_thresholds
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CacheConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Create configuration from environment variables."""
        config = cls(
            memory_limit=os.getenv('CACHE_MEMORY_LIMIT', '25%'),
            ttl_enabled=os.getenv('CACHE_TTL_ENABLED', 'true').lower() == 'true',
            eviction_policy=os.getenv('CACHE_EVICTION_POLICY', 'lru'),
            similarity_threshold=float(os.getenv('CACHE_SIMILARITY_THRESHOLD', '0.85')),
            persistence_enabled=os.getenv('CACHE_PERSISTENCE_ENABLED', 'true').lower() == 'true'
        )
        # Set embedding provider from environment
        config.embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'sentence-transformers')
        config.embedding_model = os.getenv('EMBEDDING_MODEL', 'all-mpnet-base-v2')
        return config
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"CacheConfig(memory_limit={self.memory_limit}, "
                f"ttl_enabled={self.ttl_enabled}, "
                f"eviction_policy={self.eviction_policy})")


# Default configuration
DEFAULT_CONFIG = CacheConfig()

# Production configuration
PRODUCTION_CONFIG = CacheConfig(
    memory_limit="10%",
    ttl_enabled=True,
    eviction_policy="hybrid",
    similarity_threshold=0.3,
    persistence_enabled=True,
    cleanup_interval=30,
    snapshot_interval=60
)

# Development configuration
DEVELOPMENT_CONFIG = CacheConfig(
    memory_limit="50%",
    ttl_enabled=True,
    eviction_policy="lru",
    similarity_threshold=0.3,
    persistence_enabled=False,
    cleanup_interval=120,
    snapshot_interval=300
)

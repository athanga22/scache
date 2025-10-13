"""
Enhanced Cache Key Manager

Implements sophisticated cache key design combining:
- Vector hash (embedding-based)
- Pipeline stage (query, retrieval, generation)
- Parameters (top-k, model version, etc.)
- Version awareness
"""

import hashlib
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

@dataclass
class CacheKeyComponents:
    """Components of a cache key."""
    vector_hash: str
    pipeline_stage: str
    parameters: Dict[str, Any]
    model_version: str
    corpus_version: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class CacheKeyManager:
    """
    Manages cache keys with version awareness and parameter tracking.
    """
    
    def __init__(self, model_version: str = "default", corpus_version: str = "default"):
        """
        Initialize cache key manager.
        
        Args:
            model_version: Version of the model being used
            corpus_version: Version of the corpus/documents
        """
        self.model_version = model_version
        self.corpus_version = corpus_version
        self.version_history = []
    
    def create_vector_hash(self, embedding: np.ndarray, precision: int = 3) -> str:
        """
        Create hash from embedding vector.
        
        Args:
            embedding: The embedding vector
            precision: Decimal precision for rounding
            
        Returns:
            Hash string
        """
        # Round to specified precision to handle floating point variations
        rounded_embedding = np.round(embedding, precision)
        
        # Convert to bytes
        embedding_bytes = rounded_embedding.tobytes()
        
        # Create hash
        return hashlib.sha256(embedding_bytes).hexdigest()[:16]  # Use first 16 chars
    
    def create_parameter_hash(self, parameters: Dict[str, Any]) -> str:
        """
        Create hash from parameters.
        
        Args:
            parameters: Dictionary of parameters
            
        Returns:
            Hash string
        """
        # Sort parameters for consistent hashing
        sorted_params = dict(sorted(parameters.items()))
        
        # Convert to JSON string
        params_json = json.dumps(sorted_params, sort_keys=True)
        
        # Create hash
        return hashlib.md5(params_json.encode()).hexdigest()[:8]  # Use first 8 chars
    
    def create_cache_key(self, 
                        embedding: np.ndarray,
                        pipeline_stage: str,
                        parameters: Optional[Dict[str, Any]] = None,
                        custom_suffix: Optional[str] = None) -> str:
        """
        Create a comprehensive cache key.
        
        Args:
            embedding: The embedding vector
            pipeline_stage: Stage in pipeline (query, retrieval, generation)
            parameters: Additional parameters
            custom_suffix: Custom suffix for the key
            
        Returns:
            Complete cache key
        """
        if parameters is None:
            parameters = {}
        
        # Create components
        vector_hash = self.create_vector_hash(embedding)
        param_hash = self.create_parameter_hash(parameters)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Build key components
        key_components = CacheKeyComponents(
            vector_hash=vector_hash,
            pipeline_stage=pipeline_stage,
            parameters=parameters,
            model_version=self.model_version,
            corpus_version=self.corpus_version,
            timestamp=timestamp
        )
        
        # Create the key
        key_parts = [
            pipeline_stage,
            vector_hash,
            param_hash,
            self.model_version,
            self.corpus_version
        ]
        
        if custom_suffix:
            key_parts.append(custom_suffix)
        
        cache_key = ":".join(key_parts)
        
        # Store version history
        self.version_history.append({
            'key': cache_key,
            'components': key_components.to_dict(),
            'created_at': timestamp
        })
        
        return cache_key
    
    def parse_cache_key(self, cache_key: str) -> Optional[CacheKeyComponents]:
        """
        Parse a cache key back to components.
        
        Args:
            cache_key: The cache key to parse
            
        Returns:
            Cache key components or None if invalid
        """
        try:
            parts = cache_key.split(":")
            if len(parts) < 5:
                return None
            
            pipeline_stage = parts[0]
            vector_hash = parts[1]
            param_hash = parts[2]
            model_version = parts[3]
            corpus_version = parts[4]
            
            return CacheKeyComponents(
                vector_hash=vector_hash,
                pipeline_stage=pipeline_stage,
                parameters={},  # Would need to be reconstructed from param_hash
                model_version=model_version,
                corpus_version=corpus_version,
                timestamp=""
            )
        except Exception:
            return None
    
    def is_key_valid(self, cache_key: str) -> bool:
        """
        Check if a cache key is valid for current version.
        
        Args:
            cache_key: The cache key to validate
            
        Returns:
            True if key is valid, False otherwise
        """
        components = self.parse_cache_key(cache_key)
        if not components:
            return False
        
        # Check if model version matches
        if components.model_version != self.model_version:
            return False
        
        # Check if corpus version matches
        if components.corpus_version != self.corpus_version:
            return False
        
        return True
    
    def invalidate_keys_by_version(self, old_model_version: str = None, 
                                 old_corpus_version: str = None) -> List[str]:
        """
        Get list of keys that should be invalidated due to version changes.
        
        Args:
            old_model_version: Previous model version
            old_corpus_version: Previous corpus version
            
        Returns:
            List of keys to invalidate
        """
        keys_to_invalidate = []
        
        for entry in self.version_history:
            components = entry['components']
            
            should_invalidate = False
            
            if old_model_version and components['model_version'] == old_model_version:
                should_invalidate = True
            
            if old_corpus_version and components['corpus_version'] == old_corpus_version:
                should_invalidate = True
            
            if should_invalidate:
                keys_to_invalidate.append(entry['key'])
        
        return keys_to_invalidate
    
    def update_model_version(self, new_version: str):
        """Update model version."""
        self.model_version = new_version
    
    def update_corpus_version(self, new_version: str):
        """Update corpus version."""
        self.corpus_version = new_version
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get current version information."""
        return {
            'model_version': self.model_version,
            'corpus_version': self.corpus_version,
            'total_keys_created': len(self.version_history),
            'latest_timestamp': self.version_history[-1]['created_at'] if self.version_history else None
        }
    
    def create_semantic_key(self, query: str, pipeline_stage: str, 
                          parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a semantic cache key for text queries.
        
        Args:
            query: The text query
            pipeline_stage: Stage in pipeline
            parameters: Additional parameters
            
        Returns:
            Semantic cache key
        """
        # Create a simple embedding-like hash from the query
        query_normalized = query.lower().strip()
        query_hash = hashlib.sha256(query_normalized.encode()).hexdigest()[:16]
        
        if parameters is None:
            parameters = {}
        
        param_hash = self.create_parameter_hash(parameters)
        
        # Build key
        key_parts = [
            pipeline_stage,
            query_hash,
            param_hash,
            self.model_version,
            self.corpus_version
        ]
        
        return ":".join(key_parts)
    
    def create_retrieval_key(self, query: str, top_k: int = 4, 
                           similarity_threshold: float = 0.85) -> str:
        """Create cache key for retrieval stage."""
        parameters = {
            'top_k': top_k,
            'similarity_threshold': similarity_threshold
        }
        return self.create_semantic_key(query, 'retrieval', parameters)
    
    def create_generation_key(self, query: str, context: List[str], 
                            model_params: Optional[Dict[str, Any]] = None) -> str:
        """Create cache key for generation stage."""
        if model_params is None:
            model_params = {}
        
        # Include context in parameters
        parameters = {
            'context_hash': hashlib.md5("|".join(context).encode()).hexdigest()[:8],
            **model_params
        }
        
        return self.create_semantic_key(query, 'generation', parameters)
    
    def create_query_key(self, query: str, preprocessing_params: Optional[Dict[str, Any]] = None) -> str:
        """Create cache key for query processing stage."""
        if preprocessing_params is None:
            preprocessing_params = {}
        
        return self.create_semantic_key(query, 'query', preprocessing_params)


class VersionAwareCache:
    """
    Version-aware cache that automatically invalidates stale entries.
    """
    
    def __init__(self, cache_key_manager: CacheKeyManager, base_cache):
        """
        Initialize version-aware cache.
        
        Args:
            cache_key_manager: The cache key manager
            base_cache: The underlying cache implementation
        """
        self.key_manager = cache_key_manager
        self.base_cache = base_cache
        self.invalidated_keys = set()
    
    def get(self, key: str, level: str = "default") -> Optional[Any]:
        """Get value from cache with version checking."""
        if not self.key_manager.is_key_valid(key):
            self.invalidated_keys.add(key)
            return None
        
        return self.base_cache.get(key, level)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, level: str = "default") -> bool:
        """Set value in cache."""
        return self.base_cache.set(key, value, ttl, level)
    
    def invalidate_by_version_change(self, old_model_version: str = None, 
                                   old_corpus_version: str = None) -> int:
        """
        Invalidate cache entries due to version changes.
        
        Args:
            old_model_version: Previous model version
            old_corpus_version: Previous corpus version
            
        Returns:
            Number of keys invalidated
        """
        keys_to_invalidate = self.key_manager.invalidate_keys_by_version(
            old_model_version, old_corpus_version
        )
        
        invalidated_count = 0
        for key in keys_to_invalidate:
            if self.base_cache.delete(key):
                invalidated_count += 1
                self.invalidated_keys.add(key)
        
        return invalidated_count
    
    def get_invalidation_stats(self) -> Dict[str, Any]:
        """Get statistics about invalidated keys."""
        return {
            'total_invalidated': len(self.invalidated_keys),
            'invalidated_keys': list(self.invalidated_keys),
            'version_info': self.key_manager.get_version_info()
        }

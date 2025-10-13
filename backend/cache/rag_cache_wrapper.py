#!/usr/bin/env python3
"""
RAG Cache Wrapper - Integrates with your existing notebook RAG system
Just wraps your existing graph with caching functionality
"""

import sys
import os
from pathlib import Path

# Add the cache src directory to Python path
cache_dir = Path(__file__).parent
src_dir = cache_dir / "src"
sys.path.insert(0, str(src_dir))

# Import with absolute paths
from storage.storage_engine import StorageEngine
from ttl.ttl_manager import TTLManager
from eviction.eviction_policy import EvictionPolicy
from utils.config import CacheConfig
from persistence.persistence import PersistenceManager
from similarity.similarity_engine import SimilarityEngine
from monitoring.advanced_monitoring import AdvancedMonitoring
from warming.cache_warming import CacheWarming

# Import thread safety with different name to avoid conflict
import sys
sys.path.insert(0, str(src_dir / "threading"))
from thread_safety import ThreadSafety

def create_cache(memory_limit="100MB", ttl_enabled=True, eviction_policy="lru", similarity_threshold=0.85):
    """Create a cache instance with the given configuration."""
    config = CacheConfig(
        memory_limit=memory_limit,
        ttl_enabled=ttl_enabled,
        eviction_policy=eviction_policy,
        similarity_threshold=similarity_threshold
    )
    
    from core.cache import Cache
    return Cache(config)

class RAGCacheWrapper:
    """
    Simple wrapper that adds caching to your existing RAG graph.
    """
    
    def __init__(self, rag_graph, memory_limit="100MB"):
        """
        Initialize cache wrapper with your existing RAG graph.
        
        Args:
            rag_graph: Your existing LangGraph RAG pipeline
            memory_limit: Memory limit for cache
        """
        self.rag_graph = rag_graph
        self.cache = create_cache(memory_limit=memory_limit, ttl_enabled=True)
        print("RAG Cache Wrapper initialized")
    
    def invoke(self, state):
        """
        Invoke your RAG graph with caching.
        
        Args:
            state: State dict with 'question' key
            
        Returns:
            RAG response with caching
        """
        question = state.get("question", "")
        
        if not question:
            return self.rag_graph.invoke(state)
        
        # Check for exact cache hit first
        cached_result = self.cache.get(question, level="result")
        if cached_result:
            print(f"EXACT CACHE HIT: Found cached result for '{question[:30]}...'")
            return {"answer": cached_result}
        
        # Check for semantic similarity (use research-backed threshold for duplicate detection)
        semantic_result = self.cache.get_rag_result(question, threshold=0.85)
        if semantic_result:
            print(f"SEMANTIC CACHE HIT: Found similar result for '{question[:30]}...'")
            return {"answer": semantic_result}
        
        # Cache miss - run your existing RAG graph
        print(f"CACHE MISS: Running RAG pipeline for '{question[:30]}...'")
        response = self.rag_graph.invoke(state)
        
        # Cache the result
        if "answer" in response:
            self.cache.cache_rag_result(question, response["answer"], ttl=3600)
        
        return response
    
    def get_cache_stats(self):
        """Get cache statistics."""
        return {
            "cache_stats": self.cache.get_stats(),
            "similarity_stats": self.cache.get_similarity_stats()
        }
    
    def clear_cache(self):
        """Clear cache."""
        self.cache.clear()
        print("Cache cleared")
    
    def shutdown(self):
        """Shutdown cache."""
        self.cache.shutdown()

# Example usage for your notebook:
"""
# In your notebook, after creating your graph:
from rag_cache_wrapper import RAGCacheWrapper

# Wrap your existing graph with caching
cached_graph = RAGCacheWrapper(graph)

# Use it exactly like your original graph
response = cached_graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])

# Check cache stats
stats = cached_graph.get_cache_stats()
print(f"Hit rate: {stats['cache_stats']['hit_rate']:.1f}%")
"""

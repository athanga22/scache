"""
Multi-Stage RAG Cache Wrapper

Implements caching at different stages of the RAG pipeline:
1. Query stage - Cache processed queries
2. Retrieval stage - Cache retrieved documents/context
3. Generation stage - Cache generated answers
"""

import time
import hashlib
from typing import Dict, Any, List, Optional, TypedDict
from pathlib import Path
import sys

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from core.cache import Cache, CacheConfig
from similarity.similarity_engine import SimilarityEngine
from langchain_core.documents import Document

# Import thread safety with different name to avoid conflict
sys.path.insert(0, str(src_dir / "threading"))
from thread_safety import ThreadSafety

class RAGState(TypedDict):
    """State for RAG pipeline with caching support."""
    question: str
    context: List[Document]
    answer: str
    query_hash: Optional[str]
    retrieval_hash: Optional[str]
    generation_hash: Optional[str]

class MultiStageCacheWrapper:
    """
    Multi-stage cache wrapper for RAG pipeline.
    
    Caches at three stages:
    1. Query processing (query normalization, embedding)
    2. Retrieval (similarity search results)
    3. Generation (final answer)
    """
    
    def __init__(self, rag_graph, cache_config: Optional[CacheConfig] = None):
        """
        Initialize multi-stage cache wrapper.
        
        Args:
            rag_graph: The original RAG graph
            cache_config: Cache configuration
        """
        if rag_graph is None:
            raise ValueError("RAG graph cannot be None")
            
        self.rag_graph = rag_graph
        self.cache_config = cache_config or CacheConfig()
        
        # Validate RAG graph structure
        self._validate_rag_graph()
        
        # Create separate caches for each stage
        self.query_cache = Cache(self.cache_config)
        self.retrieval_cache = Cache(self.cache_config)
        self.generation_cache = Cache(self.cache_config)
        
        # Thread safety
        self.thread_safety = ThreadSafety()
        
        # Performance metrics
        self.metrics = {
            'query_hits': 0,
            'query_misses': 0,
            'retrieval_hits': 0,
            'retrieval_misses': 0,
            'generation_hits': 0,
            'generation_misses': 0,
            'total_queries': 0,
            'total_time': 0.0,
            'errors': 0
        }
        
        print("Multi-stage cache wrapper initialized successfully")
    
    def _validate_rag_graph(self):
        """Validate that the RAG graph has the required structure."""
        try:
            # Check if graph has the expected methods
            if not hasattr(self.rag_graph, 'get_node'):
                print("Warning: RAG graph doesn't have get_node method")
                return
                
            # Check for retrieve node
            retrieve_node = self.rag_graph.get_node("retrieve")
            if retrieve_node is None:
                print("Warning: RAG graph doesn't have 'retrieve' node")
                
            # Check for generate node  
            generate_node = self.rag_graph.get_node("generate")
            if generate_node is None:
                print("Warning: RAG graph doesn't have 'generate' node")
                
        except Exception as e:
            print(f"Warning: Could not validate RAG graph structure: {e}")
    
    def _create_query_hash(self, question: str) -> str:
        """Create hash for query normalization."""
        normalized = question.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _create_retrieval_hash(self, question: str, top_k: int = 4) -> str:
        """Create hash for retrieval parameters."""
        content = f"{question.lower().strip()}:top_k={top_k}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _create_generation_hash(self, question: str, context: List[Any]) -> str:
        """Create hash for generation parameters."""
        context_texts = []
        for doc in context:
            if hasattr(doc, 'page_content'):
                context_texts.append(doc.page_content[:100])
            elif isinstance(doc, dict):
                context_texts.append(doc.get('page_content', str(doc))[:100])
            else:
                context_texts.append(str(doc)[:100])
        
        context_text = "|".join(context_texts)
        content = f"{question.lower().strip()}:{context_text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _serialize_documents(self, docs: List[Any]) -> List[Dict]:
        """Serialize documents for caching."""
        serialized = []
        for doc in docs:
            if hasattr(doc, 'page_content'):
                # Document object
                serialized.append({
                    'page_content': doc.page_content,
                    'metadata': doc.metadata
                })
            elif isinstance(doc, dict):
                # Already a dictionary
                serialized.append(doc)
            else:
                # Convert to dict
                serialized.append({
                    'page_content': str(doc),
                    'metadata': {}
                })
        return serialized
    
    def _deserialize_documents(self, docs_data: List[Dict]) -> List[Any]:
        """Deserialize documents from cache."""
        deserialized = []
        for doc in docs_data:
            if isinstance(doc, dict):
                # Create Document object if possible
                try:
                    from langchain_core.documents import Document
                    deserialized.append(Document(
                        page_content=doc['page_content'], 
                        metadata=doc.get('metadata', {})
                    ))
                except ImportError:
                    # Fallback to dict
                    deserialized.append(doc)
            else:
                deserialized.append(doc)
        return deserialized
    
    def cached_retrieve(self, state: RAGState) -> RAGState:
        """
        Cached retrieval stage.
        
        Args:
            state: RAG state with question
            
        Returns:
            Updated state with context
        """
        # Ensure required keys exist
        question = state.get("question", "")
        
        if not question:
            print("Error: No question in state for retrieval")
            return state
            
        retrieval_hash = self._create_retrieval_hash(question)
        
        # Check cache first
        cached_context = self.retrieval_cache.get(retrieval_hash, level="context")
        if cached_context:
            print(f"RETRIEVAL CACHE HIT: Found cached context for '{question[:30]}...'")
            self.metrics['retrieval_hits'] += 1
            return {
                **state,
                "context": self._deserialize_documents(cached_context),
                "retrieval_hash": retrieval_hash
            }
        
        # Cache miss - run original retrieval
        print(f"RETRIEVAL CACHE MISS: Running retrieval for '{question[:30]}...'")
        self.metrics['retrieval_misses'] += 1
        
        # Get the original retrieve function from the graph
        try:
            retrieve_func = self.rag_graph.get_node("retrieve")
            if retrieve_func:
                result = retrieve_func(state)
                
                # Validate result has context
                if "context" not in result:
                    print("Error: Retrieve function didn't return context")
                    self.metrics['errors'] += 1
                    return state
                
                # Cache the result
                cache_success = self.retrieval_cache.set(
                    retrieval_hash,
                    self._serialize_documents(result["context"]),
                    ttl=self.cache_config.get_ttl_for_level("context"),
                    level="context"
                )
                
                if not cache_success:
                    print("Warning: Failed to cache retrieval result")
                
                return {
                    **result,
                    "retrieval_hash": retrieval_hash
                }
            else:
                print("Error: No retrieve function found in RAG graph")
                self.metrics['errors'] += 1
                return state
                
        except Exception as e:
            print(f"Error in cached_retrieve: {e}")
            self.metrics['errors'] += 1
            return state
    
    def cached_generate(self, state: RAGState) -> RAGState:
        """
        Cached generation stage.
        
        Args:
            state: RAG state with question and context
            
        Returns:
            Updated state with answer
        """
        # Ensure required keys exist
        question = state.get("question", "")
        context = state.get("context", [])
        
        if not question:
            print("Error: No question in state for generation")
            return state
            
        generation_hash = self._create_generation_hash(question, context)
        
        # Check cache first
        cached_answer = self.generation_cache.get(generation_hash, level="result")
        if cached_answer:
            print(f"GENERATION CACHE HIT: Found cached answer for '{question[:30]}...'")
            self.metrics['generation_hits'] += 1
            return {
                **state,
                "answer": cached_answer,
                "generation_hash": generation_hash
            }
        
        # Cache miss - run original generation
        print(f"GENERATION CACHE MISS: Running generation for '{question[:30]}...'")
        self.metrics['generation_misses'] += 1
        
        # Get the original generate function from the graph
        try:
            generate_func = self.rag_graph.get_node("generate")
            if generate_func:
                result = generate_func(state)
                
                # Validate result has answer
                if "answer" not in result:
                    print("Error: Generate function didn't return answer")
                    self.metrics['errors'] += 1
                    return state
                
                # Cache the result
                cache_success = self.generation_cache.set(
                    generation_hash,
                    result["answer"],
                    ttl=self.cache_config.get_ttl_for_level("result"),
                    level="result"
                )
                
                if not cache_success:
                    print("Warning: Failed to cache generation result")
                
                return {
                    **result,
                    "generation_hash": generation_hash
                }
            else:
                print("Error: No generate function found in RAG graph")
                self.metrics['errors'] += 1
                return state
                
        except Exception as e:
            print(f"Error in cached_generate: {e}")
            self.metrics['errors'] += 1
            return state
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the RAG pipeline with multi-stage caching.
        
        Args:
            state: Input state with 'question' key
            
        Returns:
            RAG response with caching
        """
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        try:
            question = state.get("question", "")
            if not question:
                print("Warning: No question provided, falling back to original RAG")
                return self.rag_graph.invoke(state)
            
            # Create query hash for tracking
            query_hash = self._create_query_hash(question)
            
            # Convert to RAGState
            rag_state: RAGState = {
                "question": question,
                "context": [],
                "answer": "",
                "query_hash": query_hash,
                "retrieval_hash": None,
                "generation_hash": None
            }
            
            # Stage 1: Cached Retrieval
            rag_state = self.cached_retrieve(rag_state)
            
            # Stage 2: Cached Generation (only if retrieval succeeded)
            if "context" in rag_state and rag_state["context"]:
                rag_state = self.cached_generate(rag_state)
            else:
                print("Warning: No context from retrieval, falling back to original RAG")
                try:
                    fallback_result = self.rag_graph.invoke(rag_state)
                    rag_state.update(fallback_result)
                except Exception as e:
                    print(f"Error in fallback RAG: {e}")
                    self.metrics['errors'] += 1
                    return {
                        "answer": "Error: Could not process query",
                        "context": [],
                        "cache_metrics": self.get_cache_metrics()
                    }
            
            # Update metrics
            end_time = time.time()
            self.metrics['total_time'] += (end_time - start_time)
            
            return {
                "answer": rag_state.get("answer", ""),
                "context": rag_state.get("context", []),
                "cache_metrics": self.get_cache_metrics()
            }
            
        except Exception as e:
            print(f"Error in invoke: {e}")
            self.metrics['errors'] += 1
            end_time = time.time()
            self.metrics['total_time'] += (end_time - start_time)
            
            # Fallback to original RAG
            try:
                return self.rag_graph.invoke(state)
            except Exception as fallback_error:
                print(f"Error in fallback RAG: {fallback_error}")
                return {
                    "answer": "Error: Could not process query",
                    "context": [],
                    "cache_metrics": self.get_cache_metrics()
                }
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics."""
        total_retrieval = self.metrics['retrieval_hits'] + self.metrics['retrieval_misses']
        total_generation = self.metrics['generation_hits'] + self.metrics['generation_misses']
        
        return {
            'retrieval_hit_rate': self.metrics['retrieval_hits'] / total_retrieval if total_retrieval > 0 else 0,
            'generation_hit_rate': self.metrics['generation_hits'] / total_generation if total_generation > 0 else 0,
            'overall_hit_rate': (self.metrics['retrieval_hits'] + self.metrics['generation_hits']) / 
                               (total_retrieval + total_generation) if (total_retrieval + total_generation) > 0 else 0,
            'average_response_time': self.metrics['total_time'] / self.metrics['total_queries'] if self.metrics['total_queries'] > 0 else 0,
            'error_rate': self.metrics['errors'] / self.metrics['total_queries'] if self.metrics['total_queries'] > 0 else 0,
            'total_queries': self.metrics['total_queries'],
            'retrieval_hits': self.metrics['retrieval_hits'],
            'retrieval_misses': self.metrics['retrieval_misses'],
            'generation_hits': self.metrics['generation_hits'],
            'generation_misses': self.metrics['generation_misses'],
            'errors': self.metrics['errors']
        }
    
    def clear_cache(self, stage: Optional[str] = None):
        """Clear cache for specific stage or all stages."""
        if stage is None or stage == "retrieval":
            self.retrieval_cache.clear()
        if stage is None or stage == "generation":
            self.generation_cache.clear()
        if stage is None or stage == "query":
            self.query_cache.clear()
    
    def shutdown(self):
        """Shutdown all caches."""
        self.query_cache.shutdown()
        self.retrieval_cache.shutdown()
        self.generation_cache.shutdown()


def create_multi_stage_cache(rag_graph, memory_limit="100MB", ttl_enabled=True, 
                           eviction_policy="lru", similarity_threshold=0.85):
    """Create a multi-stage cache wrapper."""
    config = CacheConfig(
        memory_limit=memory_limit,
        ttl_enabled=ttl_enabled,
        eviction_policy=eviction_policy,
        similarity_threshold=similarity_threshold
    )
    return MultiStageCacheWrapper(rag_graph, config)

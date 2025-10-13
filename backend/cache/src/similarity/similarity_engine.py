"""
Semantic Similarity Engine
Handles similarity matching between queries and cached responses using embeddings.
"""

import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import hashlib
import re
import faiss


class SimilarityEngine:
    """
    Semantic similarity engine for RAG query matching.
    
    Features:
    - Cosine similarity calculation
    - Query normalization and preprocessing
    - Embedding caching
    - Fast similarity search
    - Configurable similarity thresholds
    """
    
    def __init__(self, config):
        """
        Initialize similarity engine.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        
        # Query embeddings cache: query_hash -> embedding
        self.query_embeddings: Dict[str, np.ndarray] = {}
        
        # FAISS indices for each level
        self.faiss_indices: Dict[str, faiss.IndexFlatIP] = {}
        self.faiss_metadata: Dict[str, List[Dict]] = {}  # level -> [metadata]
        
        # Initialize FAISS indices for each level
        for level in ['query', 'embedding', 'context', 'result']:
            # Use IndexFlatIP for cosine similarity (inner product on normalized vectors)
            self.faiss_indices[level] = faiss.IndexFlatIP(768)  # Google embeddings are 768-dim
            self.faiss_metadata[level] = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'similarity_searches': 0,
            'similarity_hits': 0,
            'embeddings_cached': 0,
            'queries_normalized': 0,
            'average_similarity': 0.0
        }
        
        print(f"Similarity Engine initialized with FAISS (threshold: {config.similarity_threshold})")
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query for better matching.
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query string
        """
        try:
            # Convert to lowercase
            normalized = query.lower()
            
            # Remove extra whitespace
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            # Remove punctuation (keep alphanumeric and spaces)
            normalized = re.sub(r'[^\w\s]', '', normalized)
            
            # Update statistics
            self.stats['queries_normalized'] += 1
            
            return normalized
            
        except Exception as e:
            print(f"Error normalizing query: {e}")
            return query
    
    def generate_query_hash(self, query: str) -> str:
        """
        Generate a hash for a query.
        
        Args:
            query: Query string
            
        Returns:
            Hash string
        """
        normalized = self.normalize_query(query)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def create_simple_embedding(self, query: str) -> np.ndarray:
        """
        Create an embedding for a query using Google Generative AI.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector
        """
        try:
            # Check cache first
            query_hash = self.generate_query_hash(query)
            if query_hash in self.query_embeddings:
                print(f"Using cached embedding for: {query[:30]}...")
                return self.query_embeddings[query_hash]
            
            # Use Google Generative AI embeddings
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            import os
            
            # Check for API key
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("GOOGLE_API_KEY not set, cannot create real embeddings")
                raise ValueError("Google API key required for semantic similarity")
            
            # Create embeddings with the API key
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            embedding = embeddings.embed_query(query)
            embedding_array = np.array(embedding, dtype=np.float32)
            
            # Cache the embedding
            self.query_embeddings[query_hash] = embedding_array
            self.stats['embeddings_cached'] += 1
            
            print(f"Created Google AI embedding for query: {query[:30]}... (dim: {len(embedding_array)})")
            return embedding_array
            
        except Exception as e:
            print(f"Error creating Google AI embedding: {e}")
            raise e  # Don't use fallback, fail fast
    
    def _create_fallback_embedding(self, query: str) -> np.ndarray:
        """
        Create a simple fallback embedding based on query characteristics.
        This should create UNIQUE embeddings for different queries.
        
        Args:
            query: Query string
            
        Returns:
            Fallback embedding vector
        """
        try:
            # Create a unique embedding for each query
            import hashlib
            
            # Normalize query
            normalized = self.normalize_query(query)
            
            # Create multiple hashes for more uniqueness
            query_hash1 = hashlib.md5(normalized.encode('utf-8')).hexdigest()
            query_hash2 = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
            
            # Convert hashes to embedding-like vector
            embedding = np.zeros(768, dtype=np.float32)
            
            # Use hash bytes to fill embedding with more randomness
            hash1_bytes = bytes.fromhex(query_hash1)
            hash2_bytes = bytes.fromhex(query_hash2)
            
            # Fill embedding with hash data
            for i in range(min(32, len(embedding))):
                if i < len(hash1_bytes):
                    embedding[i] = (hash1_bytes[i] - 128) / 128.0
                if i + 32 < len(embedding) and i < len(hash2_bytes):
                    embedding[i + 32] = (hash2_bytes[i] - 128) / 128.0
            
            # Add query-specific features that make each query unique
            embedding[64] = len(query) / 1000.0  # Query length
            embedding[65] = len(query.split()) / 100.0  # Word count
            embedding[66] = hash(normalized) % 1000 / 1000.0  # Hash-based uniqueness
            embedding[67] = sum(ord(c) for c in normalized[:10]) / 10000.0  # Character sum
            
            # Fill remaining with more hash-based randomness
            for i in range(68, len(embedding)):
                hash_val = hash(normalized + str(i)) % 256
                embedding[i] = (hash_val - 128) / 128.0
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            print(f"Using fallback embedding for: {query[:30]}...")
            return embedding
            
        except Exception as e:
            print(f"Error creating fallback embedding: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def cache_embedding(self, query: str, embedding: np.ndarray, level: str = "query") -> str:
        """
        Cache an embedding for a query using FAISS.
        
        Args:
            query: Query string
            embedding: Embedding vector
            level: Cache level
            
        Returns:
            Query hash
        """
        try:
            with self.lock:
                query_hash = self.generate_query_hash(query)
                
                # Cache the embedding
                self.query_embeddings[query_hash] = embedding.copy()
                
                # Normalize embedding for cosine similarity
                normalized_embedding = embedding.copy()
                norm = np.linalg.norm(normalized_embedding)
                if norm > 0:
                    normalized_embedding = normalized_embedding / norm
                
                # Add to FAISS index
                faiss_index = self.faiss_indices[level]
                faiss_index.add(normalized_embedding.reshape(1, -1).astype('float32'))
                
                # Store metadata
                metadata = {
                    'query': query,
                    'normalized_query': self.normalize_query(query),
                    'timestamp': time.time(),
                    'level': level,
                    'query_hash': query_hash,
                    'faiss_index': faiss_index.ntotal - 1  # Store the index position
                }
                self.faiss_metadata[level].append(metadata)
                
                # Update statistics
                self.stats['embeddings_cached'] += 1
                
                return query_hash
                
        except Exception as e:
            print(f"Error caching embedding: {e}")
            return ""
    
    def find_similar_queries(self, query: str, level: str = "query", 
                           threshold: Optional[float] = None) -> List[Tuple[str, float, Dict]]:
        """
        Find similar queries in the cache using FAISS.
        
        Args:
            query: Query to find similarities for
            level: Cache level to search in
            threshold: Similarity threshold (uses config default if None)
            
        Returns:
            List of (query_hash, similarity_score, metadata) tuples
        """
        try:
            with self.lock:
                if threshold is None:
                    threshold = self.config.similarity_threshold
                
                # Generate embedding for the query
                query_embedding = self.create_simple_embedding(query)
                
                # Normalize query embedding for cosine similarity
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm
                
                # Get FAISS index for this level
                faiss_index = self.faiss_indices[level]
                metadata_list = self.faiss_metadata[level]
                
                # If no embeddings in this level, return empty
                if faiss_index.ntotal == 0:
                    return []
                
                # Search for similar embeddings using FAISS
                # Search for top k results (we'll filter by threshold later)
                k = min(faiss_index.ntotal, 10)  # Search top 10 or all if less
                similarities, indices = faiss_index.search(
                    query_embedding.reshape(1, -1).astype('float32'), k
                )
                
                # Filter by threshold and build results
                similar_queries = []
                for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                    if similarity >= threshold and idx < len(metadata_list):
                        metadata = metadata_list[idx]
                        query_hash = metadata.get('query_hash', '')
                        similar_queries.append((query_hash, float(similarity), metadata))
                
                # Sort by similarity score (highest first)
                similar_queries.sort(key=lambda x: x[1], reverse=True)
                
                # Update statistics
                self.stats['similarity_searches'] += 1
                if similar_queries:
                    self.stats['similarity_hits'] += 1
                    # Update average similarity
                    avg_sim = sum(sim for _, sim, _ in similar_queries) / len(similar_queries)
                    self.stats['average_similarity'] = avg_sim
                
                return similar_queries
                
        except Exception as e:
            print(f"Error finding similar queries: {e}")
            return []
    
    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Ensure embeddings are numpy arrays
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)
            
            # Calculate dot product
            dot_product = np.dot(embedding1, embedding2)
            
            # Calculate norms
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def get_best_match(self, query: str, level: str = "query", 
                      threshold: Optional[float] = None) -> Optional[Tuple[str, float, Dict]]:
        """
        Get the best matching query from cache.
        
        Args:
            query: Query to match
            level: Cache level to search in
            threshold: Similarity threshold
            
        Returns:
            Best match tuple (query_hash, similarity, metadata) or None
        """
        similar_queries = self.find_similar_queries(query, level, threshold)
        return similar_queries[0] if similar_queries else None
    
    def remove_query(self, query_hash: str, level: str = "query") -> bool:
        """
        Remove a query from the similarity index.
        
        Note: FAISS doesn't support efficient removal, so we'll mark for removal
        and rebuild the index when needed.
        
        Args:
            query_hash: Hash of query to remove
            level: Cache level
            
        Returns:
            True if removed, False otherwise
        """
        try:
            with self.lock:
                # Remove from embeddings cache
                if query_hash in self.query_embeddings:
                    del self.query_embeddings[query_hash]
                
                # Find and remove from metadata
                metadata_list = self.faiss_metadata[level]
                for i, metadata in enumerate(metadata_list):
                    if metadata.get('query_hash') == query_hash:
                        del metadata_list[i]
                        # Note: FAISS doesn't support efficient removal
                        # In production, you'd want to rebuild the index periodically
                        return True
                
                return False
                
        except Exception as e:
            print(f"Error removing query: {e}")
            return False
    
    def clear_level(self, level: str):
        """
        Clear all queries from a specific level.
        
        Args:
            level: Cache level to clear
        """
        try:
            with self.lock:
                # Clear FAISS index for this level
                self.faiss_indices[level] = faiss.IndexFlatIP(768)
                self.faiss_metadata[level] = []
                
        except Exception as e:
            print(f"Error clearing level: {e}")
    
    def clear_all(self):
        """Clear all similarity data."""
        try:
            with self.lock:
                self.query_embeddings.clear()
                for level in ['query', 'embedding', 'context', 'result']:
                    self.faiss_indices[level] = faiss.IndexFlatIP(768)
                    self.faiss_metadata[level] = []
                
        except Exception as e:
            print(f"Error clearing all: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get similarity engine statistics."""
        with self.lock:
            total_queries = sum(len(metadata) for metadata in self.faiss_metadata.values())
            
            return {
                'similarity_searches': self.stats['similarity_searches'],
                'similarity_hits': self.stats['similarity_hits'],
                'embeddings_cached': self.stats['embeddings_cached'],
                'queries_normalized': self.stats['queries_normalized'],
                'average_similarity': round(self.stats['average_similarity'], 3),
                'total_cached_queries': total_queries,
                'queries_by_level': {
                    level: len(metadata) for level, metadata in self.faiss_metadata.items()
                },
                'similarity_threshold': self.config.similarity_threshold,
                'hit_rate': (
                    self.stats['similarity_hits'] / self.stats['similarity_searches'] * 100
                    if self.stats['similarity_searches'] > 0 else 0
                )
            }
    
    def get_similarity_info(self, query: str, level: str = "query") -> Dict[str, Any]:
        """
        Get detailed similarity information for a query.
        
        Args:
            query: Query to analyze
            level: Cache level
            
        Returns:
            Dictionary with similarity information
        """
        try:
            similar_queries = self.find_similar_queries(query, level)
            
            return {
                'query': query,
                'normalized_query': self.normalize_query(query),
                'query_hash': self.generate_query_hash(query),
                'level': level,
                'similar_queries_found': len(similar_queries),
                'best_match': similar_queries[0] if similar_queries else None,
                'all_matches': similar_queries[:5],  # Top 5 matches
                'threshold_used': self.config.similarity_threshold
            }
            
        except Exception as e:
            print(f"Error getting similarity info: {e}")
            return {}

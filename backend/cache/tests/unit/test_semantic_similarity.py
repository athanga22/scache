#!/usr/bin/env python3
"""
Semantic Similarity Unit Tests

Tests the core semantic similarity functionality:
- Google AI embeddings generation
- FAISS vector search
- Similarity threshold matching
- Cache hit/miss accuracy
"""

import os
import sys
import time
import json
import pandas as pd
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from core.cache import Cache
from utils.config import CacheConfig


def test_semantic_similarity_accuracy():
    """Test semantic similarity accuracy with real Quora dataset."""
    print("SEMANTIC SIMILARITY ACCURACY TEST")
    print("=" * 50)
    
    # Setup API keys for Google AI embeddings
    try:
        import sys
        cache_dir = os.path.join(os.path.dirname(__file__), '../..')
        sys.path.append(cache_dir)
        from api_config import setup_api_keys
        setup_api_keys()
        print("API keys configured for semantic similarity test")
    except Exception as e:
        print(f"Could not setup API keys: {e}")
        print("Continuing with existing environment...")
    
    # Load Quora dataset
    csv_path = "/Users/ashishthanga/Documents/GH repos/scache/questions.csv"
    print(f"Loading Quora dataset...")
    
    df_sample = pd.read_csv(csv_path, nrows=5000)
    duplicate_pairs = df_sample[df_sample['is_duplicate'] == 1].sample(n=250, random_state=42)
    non_duplicate_pairs = df_sample[df_sample['is_duplicate'] == 0].sample(n=250, random_state=42)
    all_pairs = pd.concat([duplicate_pairs, non_duplicate_pairs]).sample(frac=1, random_state=42)
    
    print(f"Loaded {len(all_pairs)} question pairs (250 duplicates + 250 non-duplicates)")
    
    # Test with threshold 0.85 only
    threshold = 0.85
    print(f"\nTesting with threshold: {threshold}")
    
    # Initialize cache with threshold 0.85
    config = CacheConfig()
    config.similarity_threshold = threshold
    cache = Cache(config)
    cache.clear()
    
    # Cache first question of each pair
    for i, row in all_pairs.iterrows():
        question1 = str(row['question1'])
        result = {"answer": f"Answer for {question1}", "context": f"Context for {question1}"}
        cache.cache_rag_result(question1, result, ttl=3600)
    
    # Test semantic matches
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    total_tests = 0
    
    for i, row in all_pairs.iterrows():
        question2 = str(row['question2'])
        expected_duplicate = int(row['is_duplicate'])
        total_tests += 1
        
        result = cache.get_rag_result(question2, threshold=threshold)
        got_cache_hit = 1 if result is not None else 0
        
        if expected_duplicate == 1 and got_cache_hit == 1:
            true_positives += 1
        elif expected_duplicate == 0 and got_cache_hit == 0:
            true_negatives += 1
        elif expected_duplicate == 0 and got_cache_hit == 1:
            false_positives += 1
        elif expected_duplicate == 1 and got_cache_hit == 0:
            false_negatives += 1
    
    # Calculate metrics
    accuracy = (true_positives + true_negatives) / total_tests if total_tests > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    duplicate_hit_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'duplicate_hit_rate': duplicate_hit_rate,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
    
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Precision: {precision:.1%}")
    print(f"   Recall: {recall:.1%}")
    print(f"   F1-Score: {f1_score:.3f}")
    print(f"   Duplicate Hit Rate: {duplicate_hit_rate:.1%}")
    
    cache.shutdown()
    
    print(f"\nTHRESHOLD: {threshold}")
    print(f"   F1-Score: {f1_score:.3f}")
    print(f"   Duplicate Hit Rate: {duplicate_hit_rate:.1%}")
    print(f"   Accuracy: {accuracy:.1%}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/semantic_similarity_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: results/semantic_similarity_results.json")
    
    return results['duplicate_hit_rate'] >= 0.7


if __name__ == "__main__":
    success = test_semantic_similarity_accuracy()
    if success:
        print(f"\nSEMANTIC SIMILARITY TEST PASSED!")
    else:
        print(f"\nSEMANTIC SIMILARITY TEST FAILED!")

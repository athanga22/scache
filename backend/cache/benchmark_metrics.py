#!/usr/bin/env python3
"""
Benchmark Script - Measures All 5 Key Metrics
Run this anytime to measure cache performance
"""

import os
# CRITICAL: Disable multiprocessing BEFORE any imports to prevent resource leaks
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'  # Disable OpenMP threading

import sys
import json
import time
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

cache_dir = os.path.dirname(__file__)
sys.path.append(cache_dir)
from api_config import setup_api_keys
setup_api_keys()

from core.cache import Cache
from utils.config import CacheConfig

# Import real RAG pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests/integration'))
from test_rag_cache_integration import RealRAGPipeline

def benchmark_metrics(num_pairs=200, threshold=0.75, use_fast_model=False, cache_size=1000):
    """
    Measure all 5 key metrics.
    
    Args:
        num_pairs: Number of question pairs to test (default: 200)
        threshold: Similarity threshold (default: 0.75)
        use_fast_model: Use Claude Haiku instead of Sonnet (default: False)
        cache_size: Maximum cache entries (default: 1000)
    """
    print("=" * 60)
    print("CACHE PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    csv_path = "/Users/ashishthanga/Documents/GH repos/scache/questions.csv"
    print(f"\nLoading {num_pairs} question pairs from Quora dataset...")
    
    df = pd.read_csv(csv_path, nrows=5000)
    duplicate_pairs = df[df['is_duplicate'] == 1].sample(n=num_pairs//2, random_state=42)
    non_duplicate_pairs = df[df['is_duplicate'] == 0].sample(n=num_pairs//2, random_state=42)
    all_pairs = pd.concat([duplicate_pairs, non_duplicate_pairs]).sample(frac=1, random_state=42)
    
    print(f"   - {len(duplicate_pairs)} duplicate pairs")
    print(f"   - {len(non_duplicate_pairs)} non-duplicate pairs")
    print(f"   - Total: {len(all_pairs)} pairs")
    print(f"   - Similarity threshold: {threshold}")
    
    # Initialize cache and RAG pipeline
    config = CacheConfig()
    config.similarity_threshold = threshold
    config.embedding_provider = "sentence-transformers"
    # Cache size: configurable for testing different sizes
    config.max_entries = cache_size
    # Disable persistence for benchmarks to avoid accumulating files
    config.persistence_enabled = False
    cache = Cache(config)
    cache.clear()
    
    print(f"   Cache size: {cache_size} entries (max)")
    
    # Initialize real RAG pipeline with Claude
    print(f"\nInitializing RAG pipeline with Claude...")
    if use_fast_model:
        print("   Using Claude Haiku (faster/cheaper) for benchmarking")
    rag_pipeline = RealRAGPipeline(use_fast_model=use_fast_model)
    if rag_pipeline.use_mock:
        print("WARNING: RAG pipeline is using MOCK - metrics will not reflect real API calls!")
        print("   Make sure ANTHROPIC_API_KEY is set correctly")
    else:
        model_type = "Claude Haiku" if use_fast_model else "Claude Sonnet"
        print(f"Real RAG pipeline initialized with {model_type}")
    
    # Phase 1: Cache all question1 with REAL RAG responses
    print(f"\nPhase 1: Caching {len(all_pairs)} questions with real RAG...")
    cache_start = time.time()
    llm_calls_made = 0
    for i, row in enumerate(all_pairs.itertuples()):
        q1 = str(row.question1)
        try:
            # Get real answer from Claude
            rag_result = rag_pipeline.answer(q1)
            llm_calls_made += 1
            # Cache the real result
            cache.cache_rag_result(q1, rag_result, ttl=3600)
            
            # Periodic cleanup and progress reporting
            if (i + 1) % 10 == 0:
                elapsed = time.time() - cache_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(all_pairs) - (i + 1)) / rate if rate > 0 else 0
                print(f"   [{i+1}/{len(all_pairs)}] LLM calls: {llm_calls_made} | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.0f}s")
                sys.stdout.flush()  # Force output to show immediately
                # Force eviction if we're getting close to max entries
                stats = cache.get_stats()
                if stats.get('total_entries', 0) > config.max_entries * 0.8:
                    cache.eviction_policy.evict_entries(cache.storage)
        except Exception as e:
            print(f"   Error caching question {i+1}: {e}")
            continue
    
    cache_time = time.time() - cache_start
    print(f"   Cached {len(all_pairs)} questions in {cache_time:.1f}s")
    print(f"   Total LLM calls made: {llm_calls_made}")
    
    # Final stats check after Phase 1
    stats_phase1 = cache.get_stats()
    evictions_phase1 = stats_phase1.get('evictions', 0)
    print(f"   Cache stats: {stats_phase1.get('total_entries', 0)} entries, {evictions_phase1} evictions")
    
    # Explicit cleanup between Phase 1 and Phase 2
    print(f"\nCleaning up Phase 1 resources...")
    import gc
    gc.collect()  # Force garbage collection
    
    # If using torch/transformers, clear CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   Cleared CUDA cache")
    except ImportError:
        pass  # torch not available, skip
    
    print(f"   Cleanup complete")
    
    # Phase 2: Test all question2 and measure REAL performance
    print(f"\nPhase 2: Testing queries and measuring REAL metrics...")
    
    cache_hits = 0
    cache_misses = 0
    cache_hit_times = []
    cache_miss_times = []
    # Confusion matrix for complete evaluation
    true_positives = 0   # Duplicates that hit cache (correct)
    false_negatives = 0  # Duplicates that miss cache (should have hit)
    false_positives = 0  # Non-duplicates that hit cache (shouldn't have hit)
    true_negatives = 0   # Non-duplicates that miss cache (correct)
    additional_llm_calls = 0
    
    test_start = time.time()
    for i, row in enumerate(all_pairs.itertuples()):
        q2 = str(row.question2)
        expected_duplicate = int(row.is_duplicate)
        
        try:
            start = time.time()
            result = cache.get_rag_result(q2, threshold=threshold)
            elapsed = (time.time() - start) * 1000  # Convert to ms
            
            if result:
                cache_hits += 1
                cache_hit_times.append(elapsed)
                if expected_duplicate == 1:
                    true_positives += 1  # Duplicate hit cache (correct)
                else:
                    false_positives += 1  # Non-duplicate hit cache (incorrect - too similar)
            else:
                cache_misses += 1
                # Make REAL LLM API call for cache miss
                llm_start = time.time()
                rag_result = rag_pipeline.answer(q2)
                llm_elapsed = (time.time() - llm_start) * 1000  # Convert to ms
                cache_miss_times.append(elapsed + llm_elapsed)  # Total time = cache lookup + LLM call
                additional_llm_calls += 1
                if expected_duplicate == 1:
                    false_negatives += 1  # Duplicate missed cache (incorrect - should have hit)
                else:
                    true_negatives += 1  # Non-duplicate missed cache (correct)
            
            # Periodic progress and cleanup
            if (i + 1) % 10 == 0:
                elapsed = time.time() - test_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(all_pairs) - (i + 1)) / rate if rate > 0 else 0
                print(f"   [{i+1}/{len(all_pairs)}] Hits: {cache_hits} | Misses: {cache_misses} | "
                      f"LLM calls: {additional_llm_calls} | ETA: {remaining:.0f}s")
                sys.stdout.flush()  # Force output to show immediately
                # Force eviction if needed
                stats = cache.get_stats()
                if stats.get('total_entries', 0) > config.max_entries * 0.8:
                    cache.eviction_policy.evict_entries(cache.storage)
        except Exception as e:
            print(f"   Error testing question {i+1}: {e}")
            cache_misses += 1
            additional_llm_calls += 1
            continue
    
    test_time = time.time() - test_start
    total_llm_calls = llm_calls_made + additional_llm_calls
    
    # Get final cache stats including evictions
    final_stats = cache.get_stats()
    total_evictions = final_stats.get('evictions', 0)
    final_entries = final_stats.get('total_entries', 0)
    
    # Calculate metrics
    total_queries = len(all_pairs)
    
    # METRIC 1: Cache Hit Rate
    cache_hit_rate = (cache_hits / total_queries * 100) if total_queries > 0 else 0
    
    # METRIC 2: API Call Reduction (REAL - based on actual LLM calls avoided)
    api_calls_avoided = cache_hits  # Each cache hit avoided one LLM call
    api_reduction = (api_calls_avoided / total_queries * 100) if total_queries > 0 else 0
    
    # METRIC 3: Latency Improvement
    avg_cache_hit_time = sum(cache_hit_times) / len(cache_hit_times) if cache_hit_times else 0
    avg_cache_miss_time = sum(cache_miss_times) / len(cache_miss_times) if cache_miss_times else 0
    speedup = avg_cache_miss_time / avg_cache_hit_time if avg_cache_hit_time > 0 else 0
    
    # Confusion Matrix Metrics
    total_duplicates = true_positives + false_negatives
    total_non_duplicates = false_positives + true_negatives
    
    # Precision: Of all cache hits, how many were correct (duplicates)?
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    # Recall: Of all duplicates, how many did we catch? (Same as duplicate hit rate)
    recall = true_positives / total_duplicates if total_duplicates > 0 else 0
    duplicate_hit_rate = recall  # Alias for clarity
    
    # Accuracy: Overall correctness
    accuracy = (true_positives + true_negatives) / total_queries if total_queries > 0 else 0
    
    # F1 Score: Harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # False positive rate: Non-duplicates that incorrectly hit cache
    false_positive_rate = false_positives / total_non_duplicates if total_non_duplicates > 0 else 0
    
    metrics = {
        'cache_hit_rate': round(cache_hit_rate, 1),
        'api_call_reduction': round(api_reduction, 1),
        'api_calls_avoided': api_calls_avoided,
        'latency_improvement': {
            'avg_cache_hit_time_ms': round(avg_cache_hit_time, 1),
            'avg_cache_miss_time_ms': round(avg_cache_miss_time, 1),
            'speedup': round(speedup, 1)
        },
        'similarity_threshold': threshold,
        'dataset_size': total_queries,
        'duplicate_hit_rate': round(duplicate_hit_rate * 100, 1),
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
        'total_llm_calls': total_llm_calls,
        'llm_calls_avoided': api_calls_avoided,
        'test_duration_seconds': round(test_time, 1),
        # Complete Confusion Matrix
        'confusion_matrix': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        },
        # Classification Metrics
        'precision': round(precision * 100, 1),  # Of hits, how many were correct?
        'recall': round(recall * 100, 1),  # Of duplicates, how many caught?
        'accuracy': round(accuracy * 100, 1),  # Overall correctness
        'f1_score': round(f1_score, 3),
        'false_positive_rate': round(false_positive_rate * 100, 1),  # % of non-duplicates incorrectly cached
        # Cache configuration and performance
        'cache_size': cache_size,
        'final_cache_entries': final_entries,
        'total_evictions': total_evictions,
        'eviction_rate': round((total_evictions / (num_pairs + additional_llm_calls) * 100), 1) if (num_pairs + additional_llm_calls) > 0 else 0
    }
    
    # Display results
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"1. Cache Hit Rate: {cache_hit_rate:.1f}% (Target: 60-75%)")
    if cache_hit_rate >= 60:
        print(f"   ACHIEVED")
    else:
        print(f"   Below target")
    
    print(f"\n2. API Call Reduction: {api_reduction:.1f}% (Target: ≥40%)")
    print(f"   - Total LLM calls made: {total_llm_calls}")
    print(f"   - LLM calls avoided (cache hits): {api_calls_avoided}")
    print(f"   - Avoided {api_calls_avoided} API calls out of {total_queries} queries")
    if api_reduction >= 40:
        print(f"   ACHIEVED")
    else:
        print(f"   Below target")
    
    print(f"\n3. Latency Improvement:")
    print(f"   - Cache Hit: {avg_cache_hit_time:.1f}ms")
    print(f"   - Cache Miss: {avg_cache_miss_time:.1f}ms")
    print(f"   - Speedup: {speedup:.1f}× (Target: 10-30×)")
    if speedup >= 10:
        print(f"   ACHIEVED")
    else:
        print(f"   Below target")
    
    print(f"\n4. Similarity Threshold: {threshold}")
    print(f"   (Duplicate Hit Rate: {duplicate_hit_rate:.1%})")
    
    print(f"\n5. Dataset Size: {total_queries} queries")
    print(f"   Test Duration: {test_time:.1f}s")
    
    print(f"\n6. Cache Performance:")
    print(f"   Cache Size: {cache_size} entries (max)")
    print(f"   Final Entries: {final_entries} entries")
    print(f"   Total Evictions: {total_evictions}")
    print(f"   Eviction Rate: {(total_evictions / (num_pairs + additional_llm_calls) * 100) if (num_pairs + additional_llm_calls) > 0 else 0:.1f}%")
    
    # Complete Confusion Matrix
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX (Complete Picture)")
    print(f"{'='*60}")
    print(f"True Positives (TP):  {true_positives:3d} - Duplicates that hit cache")
    print(f"False Negatives (FN): {false_negatives:3d} - Duplicates that missed")
    print(f"False Positives (FP):  {false_positives:3d} - Non-duplicates that hit")
    print(f"True Negatives (TN):   {true_negatives:3d} - Non-duplicates that missed")
    print(f"\nTotal Duplicates:     {total_duplicates:3d}")
    print(f"Total Non-Duplicates:  {total_non_duplicates:3d}")
    
    # Classification Metrics
    print(f"\n{'='*60}")
    print("CLASSIFICATION METRICS")
    print(f"{'='*60}")
    print(f"Precision: {precision:.1%} (Of {cache_hits} cache hits, {true_positives} were correct duplicates)")
    print(f"Recall:    {recall:.1%} (Of {total_duplicates} duplicates, caught {true_positives})")
    print(f"Accuracy:  {accuracy:.1%} (Overall correctness)")
    print(f"F1 Score:  {f1_score:.3f} (Harmonic mean of precision & recall)")
    print(f"False Positive Rate: {false_positive_rate:.1%} (Non-duplicates incorrectly cached)")
    
    print(f"{'='*60}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/benchmark_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nResults saved to: results/benchmark_metrics.json")
    
    # Overall success
    all_achieved = (
        cache_hit_rate >= 60 and
        api_reduction >= 40 and
        speedup >= 10
    )
    
    if all_achieved:
        print(f"\nALL METRICS ACHIEVED!")
    else:
        print(f"\nSome metrics need improvement")
    
    try:
        cache.shutdown()
    except:
        pass
    
    return metrics

def test_multiple_cache_sizes(num_pairs=200, cache_sizes=None, threshold=0.75, use_fast_model=True):
    """
    Test multiple cache sizes and generate comparison report.
    
    Args:
        num_pairs: Number of question pairs to test
        cache_sizes: List of cache sizes to test (default: 25%, 50%, 100% of num_pairs)
        threshold: Similarity threshold
        use_fast_model: Use Claude Haiku for faster/cheaper testing
    """
    # Default: 25%, 50%, 100% of num_pairs
    if cache_sizes is None:
        cache_sizes = [
            int(num_pairs * 0.25),  # 25% of queries
            int(num_pairs * 0.50),  # 50% of queries
            int(num_pairs * 1.00)   # 100% of queries
        ]
    
    print("=" * 80)
    print("CACHE SIZE OPTIMIZATION TEST")
    print("=" * 80)
    print(f"Testing {len(cache_sizes)} cache sizes with {num_pairs} query pairs")
    print(f"Cache sizes: {cache_sizes} ({[f'{int(s/num_pairs*100)}%' for s in cache_sizes]})")
    print("=" * 80)
    
    all_results = []
    
    for i, cache_size in enumerate(cache_sizes, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(cache_sizes)}: Cache Size = {cache_size} entries")
        print(f"{'='*80}")
        
        try:
            metrics = benchmark_metrics(
                num_pairs=num_pairs,
                threshold=threshold,
                use_fast_model=use_fast_model,
                cache_size=cache_size
            )
            
            # Extract key metrics for comparison
            result = {
                'cache_size': cache_size,
                'num_pairs': num_pairs,
                'total_queries': metrics['dataset_size'],
                'cache_hit_rate': metrics['cache_hit_rate'],
                'api_call_reduction': metrics['api_call_reduction'],
                'total_llm_calls': metrics['total_llm_calls'],
                'llm_calls_avoided': metrics['llm_calls_avoided'],
                'speedup': metrics['latency_improvement']['speedup'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'final_cache_entries': metrics['final_cache_entries'],
                'total_evictions': metrics['total_evictions'],
                'eviction_rate': metrics['eviction_rate'],
                'test_duration_seconds': metrics['test_duration_seconds']
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"ERROR testing cache_size={cache_size}: {e}")
            continue
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print("CACHE SIZE COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # Header
    print(f"\n{'Cache Size':<12} {'Hit Rate':<12} {'API Saved':<12} {'Evictions':<12} {'Eviction %':<12} {'Speedup':<10} {'Precision':<12} {'Duration':<12}")
    print("-" * 100)
    
    # Results
    for r in all_results:
        print(f"{r['cache_size']:<12} {r['cache_hit_rate']:<11.1f}% {r['llm_calls_avoided']:<12} "
              f"{r['total_evictions']:<12} {r['eviction_rate']:<11.1f}% {r['speedup']:<10.1f}× "
              f"{r['precision']:<11.1f}% {r['test_duration_seconds']:<11.1f}s")
    
    # Summary insights
    print(f"\n{'='*80}")
    print("INSIGHTS")
    print(f"{'='*80}")
    
    if len(all_results) > 0:
        best_hit_rate = max(all_results, key=lambda x: x['cache_hit_rate'])
        lowest_evictions = min(all_results, key=lambda x: x['total_evictions'])
        best_speedup = max(all_results, key=lambda x: x['speedup'])
        
        print(f"\nBest Hit Rate: {best_hit_rate['cache_hit_rate']:.1f}% at {best_hit_rate['cache_size']} entries")
        print(f"Lowest Evictions: {lowest_evictions['total_evictions']} at {lowest_evictions['cache_size']} entries")
        print(f"Best Speedup: {best_speedup['speedup']:.1f}× at {best_speedup['cache_size']} entries")
        
        # Find optimal (balance of hit rate and evictions)
        optimal = max(all_results, key=lambda x: x['cache_hit_rate'] - (x['eviction_rate'] * 0.1))
        print(f"\nOptimal Cache Size: {optimal['cache_size']} entries")
        print(f"  - Hit Rate: {optimal['cache_hit_rate']:.1f}%")
        print(f"  - Evictions: {optimal['total_evictions']} ({optimal['eviction_rate']:.1f}%)")
        print(f"  - Memory: ~{optimal['cache_size'] * 3 / 1024:.1f}MB")
    
    # Save comprehensive results
    comparison_results = {
        'test_config': {
            'num_pairs': num_pairs,
            'threshold': threshold,
            'cache_sizes_tested': cache_sizes,
            'use_fast_model': use_fast_model
        },
        'results': all_results,
        'summary': {
            'best_hit_rate': best_hit_rate if len(all_results) > 0 else None,
            'lowest_evictions': lowest_evictions if len(all_results) > 0 else None,
            'optimal_size': optimal if len(all_results) > 0 else None
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/cache_size_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    print(f"\nFull comparison results saved to: results/cache_size_comparison.json")
    
    return all_results


if __name__ == "__main__":
    try:
        # Default: Test multiple cache sizes (25%, 50%, 100% of num_pairs)
        test_multiple_cache_sizes(num_pairs=200, use_fast_model=True)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

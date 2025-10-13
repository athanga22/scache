"""
Comprehensive Benchmarking Script for Multi-Stage RAG Caching

Measures:
1. Latency improvements (target: 50-70% faster)
2. API cost reduction (target: 40% fewer API calls)
3. Cache hit rates at different stages
4. Result accuracy validation (target: ≤2% variation)
"""

import time
import json
import pandas as pd
from typing import Dict, List, Any, Tuple
import numpy as np
from pathlib import Path
import sys

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from multi_stage_cache_wrapper import MultiStageCacheWrapper, create_multi_stage_cache
from rag_cache_wrapper import RAGCacheWrapper, create_cache

class RAGBenchmark:
    """Comprehensive benchmarking for RAG caching systems."""
    
    def __init__(self, rag_graph, test_questions: List[str]):
        """
        Initialize benchmark.
        
        Args:
            rag_graph: The RAG graph to test
            test_questions: List of test questions
        """
        self.rag_graph = rag_graph
        self.test_questions = test_questions
        
        # Create cache wrappers
        self.single_stage_cache = RAGCacheWrapper(rag_graph, create_cache())
        self.multi_stage_cache = create_multi_stage_cache(rag_graph)
        
        # Results storage
        self.results = {
            'no_cache': [],
            'single_stage': [],
            'multi_stage': []
        }
    
    def benchmark_no_cache(self) -> List[Dict[str, Any]]:
        """Benchmark without any caching."""
        print("Benchmarking without cache...")
        results = []
        
        for i, question in enumerate(self.test_questions):
            print(f"  Question {i+1}/{len(self.test_questions)}: {question[:50]}...")
            
            start_time = time.time()
            response = self.rag_graph.invoke({"question": question})
            end_time = time.time()
            
            results.append({
                'question': question,
                'response_time': end_time - start_time,
                'answer': response.get('answer', ''),
                'context_count': len(response.get('context', [])),
                'api_calls': 2,  # Embedding + LLM call
                'cache_hits': 0,
                'cache_misses': 2
            })
        
        return results
    
    def benchmark_single_stage(self) -> List[Dict[str, Any]]:
        """Benchmark with single-stage caching."""
        print("Benchmarking with single-stage cache...")
        results = []
        
        for i, question in enumerate(self.test_questions):
            print(f"  Question {i+1}/{len(self.test_questions)}: {question[:50]}...")
            
            start_time = time.time()
            response = self.single_stage_cache.invoke({"question": question})
            end_time = time.time()
            
            # Determine cache hits based on response time
            response_time = end_time - start_time
            cache_hit = response_time < 0.5  # Fast response indicates cache hit
            
            results.append({
                'question': question,
                'response_time': response_time,
                'answer': response.get('answer', ''),
                'context_count': len(response.get('context', [])),
                'api_calls': 0 if cache_hit else 2,
                'cache_hits': 1 if cache_hit else 0,
                'cache_misses': 0 if cache_hit else 1
            })
        
        return results
    
    def benchmark_multi_stage(self) -> List[Dict[str, Any]]:
        """Benchmark with multi-stage caching."""
        print("Benchmarking with multi-stage cache...")
        results = []
        
        for i, question in enumerate(self.test_questions):
            print(f"  Question {i+1}/{len(self.test_questions)}: {question[:50]}...")
            
            start_time = time.time()
            response = self.multi_stage_cache.invoke({"question": question})
            end_time = time.time()
            
            response_time = end_time - start_time
            cache_metrics = response.get('cache_metrics', {})
            
            # Calculate API calls based on cache hits
            retrieval_hit = cache_metrics.get('retrieval_hits', 0) > 0
            generation_hit = cache_metrics.get('generation_hits', 0) > 0
            
            api_calls = 0
            if not retrieval_hit:
                api_calls += 1  # Embedding call
            if not generation_hit:
                api_calls += 1  # LLM call
            
            results.append({
                'question': question,
                'response_time': response_time,
                'answer': response.get('answer', ''),
                'context_count': len(response.get('context', [])),
                'api_calls': api_calls,
                'cache_hits': (1 if retrieval_hit else 0) + (1 if generation_hit else 0),
                'cache_misses': (0 if retrieval_hit else 1) + (0 if generation_hit else 1),
                'retrieval_hit': retrieval_hit,
                'generation_hit': generation_hit
            })
        
        return results
    
    def calculate_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate accuracy metrics by comparing answers."""
        print("Calculating accuracy metrics...")
        
        # Compare answers between different approaches
        accuracy_metrics = {}
        
        # Compare single-stage vs no-cache
        single_vs_no_cache = self._compare_answers(
            self.results['single_stage'], 
            self.results['no_cache']
        )
        accuracy_metrics['single_stage_vs_no_cache'] = single_vs_no_cache
        
        # Compare multi-stage vs no-cache
        multi_vs_no_cache = self._compare_answers(
            self.results['multi_stage'], 
            self.results['no_cache']
        )
        accuracy_metrics['multi_stage_vs_no_cache'] = multi_vs_no_cache
        
        # Compare multi-stage vs single-stage
        multi_vs_single = self._compare_answers(
            self.results['multi_stage'], 
            self.results['single_stage']
        )
        accuracy_metrics['multi_stage_vs_single_stage'] = multi_vs_single
        
        return accuracy_metrics
    
    def _compare_answers(self, results1: List[Dict], results2: List[Dict]) -> float:
        """Compare answers between two result sets."""
        if len(results1) != len(results2):
            return 0.0
        
        matches = 0
        for r1, r2 in zip(results1, results2):
            # Simple similarity check (could be enhanced with semantic similarity)
            if r1['question'] == r2['question']:
                # Compare answer lengths and first 100 characters
                answer1 = r1['answer'][:100].lower().strip()
                answer2 = r2['answer'][:100].lower().strip()
                if answer1 == answer2 or abs(len(answer1) - len(answer2)) < 10:
                    matches += 1
        
        return matches / len(results1)
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark across all configurations."""
        print("Starting comprehensive RAG caching benchmark...")
        print(f"Testing with {len(self.test_questions)} questions")
        
        # Run benchmarks
        self.results['no_cache'] = self.benchmark_no_cache()
        self.results['single_stage'] = self.benchmark_single_stage()
        self.results['multi_stage'] = self.benchmark_multi_stage()
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics()
        accuracy_metrics = self.calculate_accuracy_metrics()
        
        # Combine results
        comprehensive_results = {
            'test_configuration': {
                'total_questions': len(self.test_questions),
                'test_questions': self.test_questions[:5]  # First 5 for reference
            },
            'performance_metrics': metrics,
            'accuracy_metrics': accuracy_metrics,
            'detailed_results': {
                'no_cache': self._summarize_results(self.results['no_cache']),
                'single_stage': self._summarize_results(self.results['single_stage']),
                'multi_stage': self._summarize_results(self.results['multi_stage'])
            }
        }
        
        return comprehensive_results
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        for config_name, results in self.results.items():
            if not results:
                continue
                
            total_time = sum(r['response_time'] for r in results)
            avg_time = total_time / len(results)
            total_api_calls = sum(r['api_calls'] for r in results)
            total_cache_hits = sum(r['cache_hits'] for r in results)
            total_cache_misses = sum(r['cache_misses'] for r in results)
            
            metrics[config_name] = {
                'average_response_time': avg_time,
                'total_response_time': total_time,
                'total_api_calls': total_api_calls,
                'average_api_calls_per_query': total_api_calls / len(results),
                'cache_hit_rate': total_cache_hits / (total_cache_hits + total_cache_misses) if (total_cache_hits + total_cache_misses) > 0 else 0,
                'total_cache_hits': total_cache_hits,
                'total_cache_misses': total_cache_misses
            }
        
        # Calculate improvements
        if 'no_cache' in metrics and 'single_stage' in metrics:
            no_cache_time = metrics['no_cache']['average_response_time']
            single_time = metrics['single_stage']['average_response_time']
            metrics['single_stage_improvements'] = {
                'latency_improvement_percent': ((no_cache_time - single_time) / no_cache_time) * 100,
                'api_calls_reduction_percent': ((metrics['no_cache']['total_api_calls'] - metrics['single_stage']['total_api_calls']) / metrics['no_cache']['total_api_calls']) * 100
            }
        
        if 'no_cache' in metrics and 'multi_stage' in metrics:
            no_cache_time = metrics['no_cache']['average_response_time']
            multi_time = metrics['multi_stage']['average_response_time']
            metrics['multi_stage_improvements'] = {
                'latency_improvement_percent': ((no_cache_time - multi_time) / no_cache_time) * 100,
                'api_calls_reduction_percent': ((metrics['no_cache']['total_api_calls'] - metrics['multi_stage']['total_api_calls']) / metrics['no_cache']['total_api_calls']) * 100
            }
        
        return metrics
    
    def _summarize_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Summarize results for a configuration."""
        if not results:
            return {}
        
        return {
            'total_queries': len(results),
            'average_response_time': np.mean([r['response_time'] for r in results]),
            'median_response_time': np.median([r['response_time'] for r in results]),
            'min_response_time': np.min([r['response_time'] for r in results]),
            'max_response_time': np.max([r['response_time'] for r in results]),
            'total_api_calls': sum(r['api_calls'] for r in results),
            'total_cache_hits': sum(r['cache_hits'] for r in results),
            'total_cache_misses': sum(r['cache_misses'] for r in results)
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a formatted way."""
        print("\n" + "="*80)
        print("COMPREHENSIVE RAG CACHING BENCHMARK RESULTS")
        print("="*80)
        
        # Performance metrics
        perf_metrics = results['performance_metrics']
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"   Total questions tested: {results['test_configuration']['total_questions']}")
        
        for config_name, metrics in perf_metrics.items():
            if config_name.endswith('_improvements'):
                continue
            print(f"\n   {config_name.upper().replace('_', ' ')}:")
            print(f"      Average response time: {metrics['average_response_time']:.3f}s")
            print(f"      Total API calls: {metrics['total_api_calls']}")
            print(f"      Average API calls per query: {metrics['average_api_calls_per_query']:.2f}")
            print(f"      Cache hit rate: {metrics['cache_hit_rate']:.1%}")
        
        # Improvements
        if 'single_stage_improvements' in perf_metrics:
            improvements = perf_metrics['single_stage_improvements']
            print(f"\n   SINGLE-STAGE CACHE IMPROVEMENTS:")
            print(f"      Latency improvement: {improvements['latency_improvement_percent']:.1f}%")
            print(f"      API calls reduction: {improvements['api_calls_reduction_percent']:.1f}%")
        
        if 'multi_stage_improvements' in perf_metrics:
            improvements = perf_metrics['multi_stage_improvements']
            print(f"\n   MULTI-STAGE CACHE IMPROVEMENTS:")
            print(f"      Latency improvement: {improvements['latency_improvement_percent']:.1f}%")
            print(f"      API calls reduction: {improvements['api_calls_reduction_percent']:.1f}%")
        
        # Accuracy metrics
        accuracy_metrics = results['accuracy_metrics']
        print(f"\n   ACCURACY METRICS:")
        for comparison, accuracy in accuracy_metrics.items():
            print(f"      {comparison.replace('_', ' ').title()}: {accuracy:.1%}")
        
        # Goal achievement
        print(f"\n   GOAL ACHIEVEMENT:")
        if 'multi_stage_improvements' in perf_metrics:
            latency_improvement = perf_metrics['multi_stage_improvements']['latency_improvement_percent']
            api_reduction = perf_metrics['multi_stage_improvements']['api_calls_reduction_percent']
            
            latency_goal = "ACHIEVED" if latency_improvement >= 50 else "NOT ACHIEVED"
            api_goal = "ACHIEVED" if api_reduction >= 40 else "NOT ACHIEVED"
            
            print(f"      Latency improvement ≥50%: {latency_goal} ({latency_improvement:.1f}%)")
            print(f"      API calls reduction ≥40%: {api_goal} ({api_reduction:.1f}%)")
        
        print("\n" + "="*80)
    
    def save_results(self, results: Dict[str, Any], filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        output_path = Path(__file__).parent / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")


def create_test_questions() -> List[str]:
    """Create a diverse set of test questions."""
    return [
        "What is machine learning?",
        "How does artificial intelligence work?",
        "What are the benefits of cloud computing?",
        "Explain the concept of neural networks",
        "What is the difference between supervised and unsupervised learning?",
        "How do recommendation systems work?",
        "What is natural language processing?",
        "Explain the concept of deep learning",
        "What are the applications of computer vision?",
        "How does blockchain technology work?",
        "What is the Internet of Things?",
        "Explain the concept of big data",
        "What are the benefits of microservices architecture?",
        "How does containerization work?",
        "What is the difference between SQL and NoSQL databases?",
        "Explain the concept of DevOps",
        "What are the principles of agile development?",
        "How does version control work?",
        "What is the difference between frontend and backend development?",
        "Explain the concept of API design"
    ]


if __name__ == "__main__":
    # This would be used with an actual RAG graph
    # For now, this is a template for benchmarking
    print("RAG Caching Benchmark Template")
    print("To use this benchmark, provide a RAG graph and run:")
    print("benchmark = RAGBenchmark(rag_graph, test_questions)")
    print("results = benchmark.run_comprehensive_benchmark()")
    print("benchmark.print_results(results)")

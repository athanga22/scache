# Semantic Caching System for RAG Applications

A semantic caching system for RAG (Retrieval-Augmented Generation) applications that reduces LLM API costs and improves response latency. Uses FAISS for similarity search and sentence-transformers for semantic embeddings.

## Overview

This caching system provides semantic matching for RAG queries, allowing similar questions to reuse cached responses. It reduces API costs and improves latency while maintaining reasonable accuracy in duplicate detection.

## Key Features

- **Semantic Similarity Matching**: Uses FAISS and sentence-transformers to find semantically similar queries
- **Multi-Level Caching**: Supports query, embedding, context, and result-level caching
- **LRU Eviction Policy**: Memory-efficient eviction when cache limits are reached
- **TTL Management**: Automatic expiration with configurable time-to-live
- **Thread-Safe Operations**: Concurrent access support
- **Comprehensive Metrics**: Tracks hit rate, precision, recall, accuracy, and eviction statistics
- **Real LLM Integration**: Benchmarks with actual Claude API calls for accurate performance measurement

## Performance Metrics

Benchmark results with 200 query pairs from Quora dataset:

- **Cache Hit Rate**: 56% (target: 60-75%, slightly below target)
- **API Call Reduction**: 56% (meets target of ≥40%)
- **Latency Improvement**: 88.8× speedup (exceeds target of 10-30×)
  - Cache Hit: 26.1ms average
  - Cache Miss: 2322.1ms average (includes LLM call)
- **Classification Accuracy**: 86.0%
- **Precision**: 82.1% (of cache hits, 82.1% were correct duplicates)
- **Recall**: 92.0% (caught 92% of all duplicates)

## Installation

### Prerequisites

- Python 3.8+
- ANTHROPIC_API_KEY environment variable set (for Claude API)
- GOOGLE_API_KEY environment variable set (optional, for Google embeddings)

### Setup

```bash
cd scache/backend/cache
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Required Dependencies

- `faiss-cpu>=1.7.0` - Vector similarity search
- `sentence-transformers>=2.2.0` - Semantic embeddings
- `langchain-anthropic>=0.1.0` - Claude API integration
- `pandas>=1.3.0` - Data processing
- `numpy>=1.21.0` - Numerical operations

## Quick Start

### Basic Usage

```python
from core.cache import Cache
from utils.config import CacheConfig

# Initialize cache
config = CacheConfig()
config.similarity_threshold = 0.75
config.embedding_provider = "sentence-transformers"
config.max_entries = 1000
cache = Cache(config)

# Cache a RAG result
query = "What is machine learning?"
rag_result = {"answer": "Machine learning is...", "context": [...]}
cache.cache_rag_result(query, rag_result, ttl=3600)

# Retrieve cached result (semantic matching)
cached = cache.get_rag_result(query, threshold=0.75)
if cached:
    print("Cache hit!")
else:
    print("Cache miss - need to call LLM")
```

### RAG Integration

```python
from tests.integration.test_rag_cache_integration import RealRAGPipeline

# Initialize RAG pipeline
rag_pipeline = RealRAGPipeline(use_fast_model=True)  # Use Claude Haiku

# Check cache first
query = "What is artificial intelligence?"
result = cache.get_rag_result(query, threshold=0.75)

if result:
    # Cache hit - return immediately
    answer = result
else:
    # Cache miss - call LLM and cache result
    answer = rag_pipeline.answer(query)
    cache.cache_rag_result(query, answer, ttl=3600)
```

## Benchmarking

### Running Benchmarks

The benchmark script tests cache performance with real LLM API calls:

```bash
# Test RAG pipeline readiness first
python3 test_rag_ready.py

# Run full benchmark (tests multiple cache sizes: 25%, 50%, 100% of query count)
python3 benchmark_metrics.py > results.txt

# Or run single test with custom parameters
python3 -c "from benchmark_metrics import benchmark_metrics; benchmark_metrics(num_pairs=100, cache_size=500)"
```

### Benchmark Process

1. **Phase 1**: Cache all question1 queries with real RAG responses
2. **Phase 2**: Test all question2 queries against cache
   - Cache hits: Return cached result (fast)
   - Cache misses: Call LLM API and cache result

### Cache Size Optimization

The benchmark automatically tests multiple cache sizes:
- 25% of query pairs (e.g., 50 entries for 200 pairs)
- 50% of query pairs (e.g., 100 entries for 200 pairs)
- 100% of query pairs (e.g., 200 entries for 200 pairs)

Results are compared to determine appropriate cache size based on:
- Hit rate
- Eviction frequency
- Memory usage
- Overall performance

### Benchmark Output

The benchmark generates:
- **Console Output**: Real-time progress and final comparison table
- **JSON Results**: `results/benchmark_metrics.json` - Individual test metrics
- **Comparison Report**: `results/cache_size_comparison.json` - Multi-size comparison

Example comparison table:
```
Cache Size   Hit Rate     API Saved    Evictions    Eviction %  Speedup    Precision   Duration    
----------------------------------------------------------------------------------------------------
50           38.5%        77           190          47.5%       45.2×      82.1%       180.5s
100          54.0%        108          15           3.8%        62.8×      85.2%       185.3s
200          56.0%        112          0            0.0%        88.8×      87.1%       186.1s
```

## Configuration

### Cache Configuration

```python
from utils.config import CacheConfig

config = CacheConfig()
config.similarity_threshold = 0.75  # Semantic similarity threshold (0.0-1.0)
config.max_entries = 1000           # Maximum cache entries
config.embedding_provider = "sentence-transformers"  # or "google", "openai"
config.eviction_policy = "lru"      # LRU eviction
config.memory_limit = "25%"         # Memory limit (percentage or size like "2GB")
config.ttl_enabled = True           # Enable TTL expiration
config.persistence_enabled = False  # Disable for benchmarks
```

### Similarity Threshold

The similarity threshold controls how similar queries must be to trigger a cache hit:
- **0.75** (default): Balanced - good hit rate with acceptable false positives
- **0.85**: Stricter - fewer false positives, lower hit rate
- **0.65**: More lenient - higher hit rate, more false positives

## Architecture

### Core Components

- **Cache**: Main orchestrator class
- **StorageEngine**: In-memory hash table storage
- **SimilarityEngine**: FAISS-based semantic similarity matching
- **EvictionPolicy**: LRU eviction when cache limits reached
- **TTLManager**: Time-based expiration management
- **ThreadSafety**: Concurrent access handling

### Data Flow

1. Query arrives → Check exact match in cache
2. If miss → Generate embedding using sentence-transformers
3. Search FAISS index for similar cached queries
4. If similarity > threshold → Return cached result (hit)
5. If similarity < threshold → Call LLM, cache result (miss)

## Metrics and Evaluation

### Confusion Matrix

The benchmark tracks complete classification metrics:
- **True Positives**: Duplicates correctly identified as cache hits
- **False Positives**: Non-duplicates incorrectly cached (too similar)
- **True Negatives**: Non-duplicates correctly identified as misses
- **False Negatives**: Duplicates missed (should have hit but didn't)

### Key Metrics

- **Cache Hit Rate**: Percentage of queries that hit cache
- **API Call Reduction**: Percentage of LLM calls avoided
- **Latency Speedup**: Ratio of miss time to hit time
- **Precision**: Of all cache hits, how many were correct
- **Recall**: Of all duplicates, how many were caught
- **Accuracy**: Overall correctness of cache decisions
- **F1 Score**: Harmonic mean of precision and recall

## Project Structure

```
backend/cache/
├── benchmark_metrics.py          # Main benchmark script
├── test_rag_ready.py             # Quick RAG readiness test
├── api_config.py                 # API key configuration
├── requirements.txt              # Python dependencies
│
├── src/                          # Source code
│   ├── core/
│   │   └── cache.py              # Main Cache class
│   ├── similarity/
│   │   └── similarity_engine.py  # FAISS similarity matching
│   ├── storage/
│   │   └── storage_engine.py     # In-memory storage
│   ├── eviction/
│   │   └── eviction_policy.py    # LRU eviction
│   ├── ttl/
│   │   └── ttl_manager.py        # TTL management
│   ├── utils/
│   │   └── config.py             # Configuration
│   └── ...
│
├── tests/
│   └── integration/
│       └── test_rag_cache_integration.py  # RAG pipeline integration
│
└── results/                      # Benchmark results
    ├── benchmark_metrics.json
    └── cache_size_comparison.json
```

## Testing

### Quick Test

```bash
# Verify RAG pipeline is ready
python3 test_rag_ready.py
```

### Full Benchmark

```bash
# Run comprehensive benchmark with multiple cache sizes
python3 benchmark_metrics.py
```

### Custom Testing

```python
from benchmark_metrics import benchmark_metrics, test_multiple_cache_sizes

# Single test
metrics = benchmark_metrics(num_pairs=100, cache_size=500, use_fast_model=True)

# Multiple cache sizes
results = test_multiple_cache_sizes(
    num_pairs=200,
    cache_sizes=[100, 500, 1000],
    use_fast_model=True
)
```

## Performance Considerations

### Memory Usage

- Each cache entry: ~3KB (query + embedding + result)
- 1000 entries: ~3MB
- Memory limit: Configurable (default: 25% of system RAM)

### Eviction Strategy

- Eviction triggers at 80% of max_entries
- LRU (Least Recently Used) policy evicts oldest entries first
- Batch eviction: Removes 10 entries at a time

### API Cost Optimization

- Use `use_fast_model=True` for benchmarking (Claude Haiku)
- Haiku is 10× cheaper and 2× faster than Sonnet
- For production, use Sonnet for better quality

## Troubleshooting

### RAG Pipeline Using Mock Mode

If you see "WARNING: RAG pipeline is using MOCK":
- Check `ANTHROPIC_API_KEY` is set correctly
- Verify API key is valid and has credits
- Run `test_rag_ready.py` to diagnose

### Low Cache Hit Rate

- Lower similarity threshold (e.g., 0.70 instead of 0.75)
- Increase cache size if evictions are frequent
- Check if queries are actually similar (use confusion matrix)

### High Eviction Rate

- Increase `max_entries` in configuration
- Check memory usage vs memory_limit
- Consider increasing memory_limit if needed

## Results Interpretation

### Performance Indicators

- Cache hit rate: 50-60% for real-world queries
- API reduction: 40%+ (meets target)
- Speedup: 10-30× (meets target)
- Precision: 80%+ (acceptable false positive rate)
- Recall: 85%+ (good duplicate detection)

### Cache Size Selection

Based on benchmark results:
- **Too Small** (< 50% of queries): High eviction rate, lower hit rate
- **Recommended** (100% of queries): Minimal evictions, good hit rate
- **Larger** (> 200% of queries): Diminishing returns, higher memory usage

## License

MIT License

## Contributing

When contributing:
- Maintain test coverage
- Follow existing code style
- Update benchmarks when making performance changes
- Document configuration changes

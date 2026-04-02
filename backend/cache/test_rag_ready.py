#!/usr/bin/env python3
"""
Quick RAG Pipeline Test
Checks if RAG is ready before running full benchmark
"""

import os
import sys

# Disable multiprocessing
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
cache_dir = os.path.dirname(__file__)
sys.path.append(cache_dir)

from api_config import setup_api_keys
setup_api_keys()

# Import RAG pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests/integration'))
from test_rag_cache_integration import RealRAGPipeline

def test_rag_ready():
    """Quick test to verify RAG pipeline is working."""
    print("=" * 60)
    print("RAG PIPELINE READINESS TEST")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return False
    print(f"ANTHROPIC_API_KEY configured ({len(api_key)} chars)")
    
    # Initialize RAG pipeline
    print("\nInitializing RAG pipeline...")
    try:
        rag_pipeline = RealRAGPipeline(use_fast_model=True)
    except Exception as e:
        print(f"ERROR: Failed to initialize RAG pipeline: {e}")
        return False
    
    # Check if using mock
    if rag_pipeline.use_mock:
        print("ERROR: RAG pipeline is using MOCK mode")
        print("   Real API calls will not be made")
        return False
    
    print("RAG pipeline initialized")
    
    # Test with a simple question
    print("\nTesting with sample question...")
    test_question = "What is machine learning?"
    
    try:
        import time
        start = time.time()
        response = rag_pipeline.answer(test_question)
        elapsed = time.time() - start
        
        if not response or "answer" not in response:
            print("ERROR: Invalid response format")
            return False
        
        answer = response.get("answer", "")
        if "Mock" in answer or len(answer) < 10:
            print("ERROR: Received mock or invalid response")
            return False
        
        print(f"RAG response received in {elapsed:.2f}s")
        print(f"Answer preview: {answer[:100]}...")
        print(f"\nRAG PIPELINE IS READY")
        print(f"   Model: Claude Haiku")
        print(f"   Response time: {elapsed:.2f}s")
        return True
        
    except Exception as e:
        print(f"ERROR: RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_ready()
    sys.exit(0 if success else 1)

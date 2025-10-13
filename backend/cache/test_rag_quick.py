#!/usr/bin/env python3
"""
Quick RAG Integration Test

Tests if the RAG pipeline is working correctly before running the full test suite.
"""

import os
import sys
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup API keys
try:
    from api_config import setup_api_keys
    setup_api_keys()
    print("API keys configured")
except Exception as e:
    print(f"API key setup failed: {e}")
    sys.exit(1)

def test_rag_pipeline():
    """Quick test of the RAG pipeline."""
    print("QUICK RAG PIPELINE TEST")
    print("=" * 40)
    
    try:
        # Import the RAG pipeline
        from tests.integration.test_rag_cache_integration import RealRAGPipeline
        
        print("Initializing RAG pipeline...")
        rag_pipeline = RealRAGPipeline()
        
        if rag_pipeline.use_mock:
            print("RAG pipeline is using mock - not real RAG!")
            return False
        
        print("Real RAG pipeline initialized")
        
        # Test with a simple question
        print("Testing with sample question...")
        test_question = "What is machine learning?"
        
        start_time = time.time()
        response = rag_pipeline.answer(test_question)
        end_time = time.time()
        
        if response and "answer" in response and len(response["answer"]) > 10:
            print(f"RAG response received in {end_time - start_time:.2f}s")
            print(f"Answer: {response['answer'][:100]}...")
            print(f"Context sources: {len(response.get('context', []))}")
            print(f"Confidence: {response.get('confidence', 'N/A')}")
            
            # Check if it's a real response or mock
            if "Mock answer" in response["answer"]:
                print("WARNING: Received mock response - RAG pipeline not working properly")
                return False
            else:
                print("Real RAG response confirmed")
                return True
        else:
            print("No valid response received")
            return False
            
    except Exception as e:
        print(f"RAG pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_pipeline()
    if success:
        print("\nRAG PIPELINE IS READY!")
        print("You can now run the full integration test")
    else:
        print("\nRAG PIPELINE HAS ISSUES!")
        print("Fix the issues before running the full test")
    
    sys.exit(0 if success else 1)

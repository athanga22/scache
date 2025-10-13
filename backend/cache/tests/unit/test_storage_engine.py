"""
Unit tests for StorageEngine
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from storage.storage_engine import StorageEngine
from utils.config import CacheConfig


def test_storage_engine_basic():
    """Test basic storage engine functionality."""
    print("Testing Storage Engine Basic Functionality...")
    
    # Create config and storage engine
    config = CacheConfig(memory_limit="100MB")
    storage = StorageEngine(config)
    
    # Test 1: Basic set and get
    print("  Testing set/get operations...")
    success = storage.set("test_key", "test_value", level="query")
    assert success, "Set operation should succeed"
    
    value = storage.get("test_key", level="query")
    assert value == "test_value", "Get operation should return correct value"
    
    # Test 2: Memory tracking
    print("  Testing memory tracking...")
    memory_usage = storage.get_memory_usage()
    assert memory_usage['total_bytes'] > 0, "Memory usage should be tracked"
    assert memory_usage['by_level']['query']['bytes'] > 0, "Level memory should be tracked"
    
    # Test 3: Exists check
    print("  Testing exists operation...")
    assert storage.exists("test_key", level="query"), "Key should exist"
    assert not storage.exists("nonexistent_key", level="query"), "Non-existent key should not exist"
    
    # Test 4: Delete operation
    print("  Testing delete operation...")
    success = storage.delete("test_key", level="query")
    assert success, "Delete operation should succeed"
    assert not storage.exists("test_key", level="query"), "Key should not exist after deletion"
    
    # Test 5: Clear operation
    print("  Testing clear operation...")
    storage.set("key1", "value1", level="query")
    storage.set("key2", "value2", level="query")
    success = storage.clear(level="query")
    assert success, "Clear operation should succeed"
    assert storage.get_memory_usage()['by_level']['query']['bytes'] == 0, "Level should be empty"
    
    print("Storage Engine Basic Tests Passed!")


def test_storage_engine_memory_limits():
    """Test memory limit enforcement."""
    print("Testing Memory Limit Enforcement...")
    
    # Create config with small memory limit
    config = CacheConfig(memory_limit="1KB")
    storage = StorageEngine(config)
    
    # Try to store a large string
    large_string = "x" * 2000  # 2KB string
    success = storage.set("large_key", large_string, level="query")
    
    # Should fail due to memory limit
    assert not success, "Should not be able to store data exceeding memory limit"
    
    print("Memory Limit Tests Passed!")


def test_storage_engine_multiple_levels():
    """Test multiple cache levels."""
    print("Testing Multiple Cache Levels...")
    
    config = CacheConfig(memory_limit="10MB")
    storage = StorageEngine(config)
    
    # Store data in different levels
    storage.set("query_key", "query_value", level="query")
    storage.set("embedding_key", "embedding_value", level="embedding")
    storage.set("context_key", "context_value", level="context")
    storage.set("result_key", "result_value", level="result")
    
    # Verify data is stored in correct levels
    assert storage.get("query_key", level="query") == "query_value"
    assert storage.get("embedding_key", level="embedding") == "embedding_value"
    assert storage.get("context_key", level="context") == "context_value"
    assert storage.get("result_key", level="result") == "result_value"
    
    # Verify level-specific operations
    stats = storage.get_stats()
    assert stats['entries_by_level']['query'] == 1
    assert stats['entries_by_level']['embedding'] == 1
    assert stats['entries_by_level']['context'] == 1
    assert stats['entries_by_level']['result'] == 1
    
    print("Multiple Levels Tests Passed!")


def test_storage_engine_error_handling():
    """Test error handling."""
    print("Testing Error Handling...")
    
    config = CacheConfig(memory_limit="10MB")
    storage = StorageEngine(config)
    
    # Test invalid level
    result = storage.set("key", "value", level="invalid_level")
    assert not result, "Should handle invalid level gracefully"
    
    # Test invalid key types (should still work for now)
    result = storage.set("", "value", level="query")  # Empty string key
    assert result, "Should handle empty string keys"
    
    print("Error Handling Tests Passed!")


if __name__ == "__main__":
    print("Running Storage Engine Tests...")
    
    try:
        test_storage_engine_basic()
        test_storage_engine_memory_limits()
        test_storage_engine_multiple_levels()
        test_storage_engine_error_handling()
        
        print("\nALL TESTS PASSED! Storage Engine is working correctly!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

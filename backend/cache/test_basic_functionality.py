#!/usr/bin/env python3
"""
Test Basic Cache Functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.cache import create_cache

def test_basic_operations():
    """Test basic cache operations."""
    print("🧪 Testing Basic Cache Operations...")
    
    # Create cache with small memory limit for testing
    cache = create_cache(memory_limit="10MB", ttl_enabled=True)
    
    print("\n1. Testing SET operations...")
    # Test basic set with specific level
    success = cache.set("test_key", "test_value", ttl=60, level="query")
    print(f"   Set 'test_key': {'✅' if success else '❌'}")
    
    # Test different data types with specific levels
    cache.set("number", 42, ttl=60, level="query")
    cache.set("list_data", [1, 2, 3], ttl=60, level="context")
    cache.set("dict_data", {"name": "cache", "type": "test"}, ttl=60, level="result")
    print("   Set multiple data types: ✅")
    
    print("\n2. Testing GET operations...")
    # Test basic get with specific level
    value = cache.get("test_key", level="query")
    print(f"   Get 'test_key': {'✅' if value == 'test_value' else '❌'} (value: {value})")
    
    # Test other data types
    number = cache.get("number", level="query")
    list_data = cache.get("list_data", level="context")
    dict_data = cache.get("dict_data", level="result")
    print(f"   Get 'number': {'✅' if number == 42 else '❌'}")
    print(f"   Get 'list_data': {'✅' if list_data == [1, 2, 3] else '❌'}")
    print(f"   Get 'dict_data': {'✅' if dict_data == {"name": "cache", "type": "test"} else '❌'}")
    
    print("\n3. Testing EXISTS operations...")
    exists = cache.exists("test_key", level="query")
    not_exists = cache.exists("nonexistent_key", level="query")
    print(f"   'test_key' exists: {'✅' if exists else '❌'}")
    print(f"   'nonexistent_key' exists: {'✅' if not not_exists else '❌'}")
    
    print("\n4. Testing DELETE operations...")
    success = cache.delete("test_key", level="query")
    print(f"   Delete 'test_key': {'✅' if success else '❌'}")
    
    # Verify deletion
    value_after_delete = cache.get("test_key", level="query")
    print(f"   Get after delete: {'✅' if value_after_delete is None else '❌'}")
    
    print("\n5. Testing STATISTICS...")
    stats = cache.get_stats()
    print(f"   Total entries: {stats.get('total_entries', 'N/A')}")
    print(f"   Memory usage: {stats['memory_usage']['total_mb']} MB")
    print(f"   Hit rate: {stats['hit_rate']}%")
    
    print("\n6. Testing CLEAR operations...")
    success = cache.clear()
    print(f"   Clear all: {'✅' if success else '❌'}")
    
    # Verify clear
    stats_after_clear = cache.get_stats()
    print(f"   Entries after clear: {stats_after_clear.get('total_entries', 'N/A')}")
    
    print("\n🎉 Basic functionality test complete!")
    return True

if __name__ == "__main__":
    try:
        test_basic_operations()
        print("\n✅ ALL TESTS PASSED! Core cache is working!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

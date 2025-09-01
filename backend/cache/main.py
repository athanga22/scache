#!/usr/bin/env python3
"""
Custom Caching System for RAG Applications
Main Entry Point

This module provides the main interface for the custom caching system.
It orchestrates all components and provides a simple API for integration.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.cache import Cache
from src.api.cache_api import CacheAPI
from src.monitoring.metrics import CacheMetrics
from src.utils.config import CacheConfig

def main():
    """Main entry point for the caching system."""
    print("🚀 Starting Custom Caching System for RAG...")
    
    # Load configuration
    config = CacheConfig()
    
    # Initialize cache
    cache = Cache(config)
    
    # Initialize API
    api = CacheAPI(cache)
    
    # Initialize monitoring
    metrics = CacheMetrics(cache)
    
    print(f"✅ Cache initialized with {config.memory_limit} memory limit")
    print(f"✅ TTL enabled: {config.ttl_enabled}")
    print(f"✅ Eviction policy: {config.eviction_policy}")
    
    return cache, api, metrics

if __name__ == "__main__":
    cache, api, metrics = main()
    print("🎯 Caching system ready for integration!")

#!/usr/bin/env python3
"""
Test Runner

Runs all tests in proper order:
1. Unit tests (semantic similarity, adaptive invalidation)
2. Integration tests (RAG + Cache)
3. Performance tests
4. Generates comprehensive report
"""

import os
import sys
import time
import json
import subprocess
from typing import Dict, Any, List

def run_test(test_path: str, test_name: str) -> Dict[str, Any]:
    """Run a single test and return results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test with real-time output
        print(f"   Running: {test_path}")
        result = subprocess.run([
            sys.executable, test_path
        ], cwd=os.path.dirname(__file__))
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse output for key metrics (we'll get this from the test results file)
        output = ""  # We'll read from results files instead
        success = result.returncode == 0
        
        # Extract key metrics from results files
        metrics = {}
        try:
            # Try to read results from JSON files
            results_file = f"results/{test_name.lower().replace(' ', '_')}_results.json"
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                    if 'duplicate_hit_rate' in results_data:
                        hit_rate = results_data['duplicate_hit_rate']
                        metrics['status'] = 'ACHIEVED' if hit_rate >= 0.7 else 'NOT ACHIEVED'
                        metrics['hit_rate'] = hit_rate
                    elif 'stale_percentage' in results_data:
                        stale_pct = results_data['stale_percentage']
                        metrics['status'] = 'ACHIEVED' if stale_pct <= 5.0 else 'NOT ACHIEVED'
                        metrics['stale_percentage'] = stale_pct
                    else:
                        metrics['status'] = 'UNKNOWN'
            else:
                metrics['status'] = 'UNKNOWN'
        except Exception as e:
            metrics['status'] = 'UNKNOWN'
        
        return {
            'test_name': test_name,
            'test_path': test_path,
            'success': success,
            'duration': duration,
            'return_code': result.returncode,
            'stdout': output,
            'stderr': result.stderr if hasattr(result, 'stderr') else '',
            'metrics': metrics
        }
        
    except Exception as e:
        return {
            'test_name': test_name,
            'test_path': test_path,
            'success': False,
            'duration': 0,
            'return_code': -1,
            'stdout': '',
            'stderr': str(e),
            'metrics': {'status': 'ERROR'}
        }

def main():
    """Run all tests and generate comprehensive report."""
    print("SEMANTIC CACHE TEST SUITE")
    print("=" * 60)
    
    # Define test suite
    tests = [
        {
            'path': 'tests/unit/test_semantic_similarity.py',
            'name': 'Semantic Similarity Unit Test',
            'category': 'unit'
        },
        {
            'path': 'tests/unit/test_adaptive_invalidation.py', 
            'name': 'Adaptive Invalidation Unit Test',
            'category': 'unit'
        },
        {
            'path': 'tests/integration/test_rag_cache_integration.py',
            'name': 'RAG + Cache Integration Test',
            'category': 'integration'
        }
    ]
    
    # Run tests
    results = []
    total_start_time = time.time()
    
    for test in tests:
        if os.path.exists(test['path']):
            result = run_test(test['path'], test['name'])
            result['category'] = test['category']
            results.append(result)
        else:
            print(f"Test file not found: {test['path']}")
    
    total_duration = time.time() - total_start_time
    
    # Generate report
    print(f"\n{'='*60}")
    print(f"TEST SUITE REPORT")
    print(f"{'='*60}")
    
    # Summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    print(f"Total duration: {total_duration:.2f}s")
    
    # Detailed results
    print(f"\nDETAILED RESULTS:")
    for result in results:
        status = "PASS" if result['success'] else "FAIL"
        print(f"   {status} {result['test_name']} ({result['duration']:.2f}s)")
        if result['metrics'].get('status'):
            print(f"      Status: {result['metrics']['status']}")
            if 'hit_rate' in result['metrics']:
                print(f"      Hit Rate: {result['metrics']['hit_rate']:.1%}")
            if 'stale_percentage' in result['metrics']:
                print(f"      Stale %: {result['metrics']['stale_percentage']:.1f}%")
    
    # Overall assessment
    print(f"\nOVERALL ASSESSMENT:")
    
    # Check if key goals are achieved
    semantic_achieved = any(
        r['success'] and 'ACHIEVED' in r['stdout'] 
        for r in results 
        if 'semantic' in r['test_name'].lower()
    )
    
    invalidation_achieved = any(
        r['success'] and 'ACHIEVED' in r['stdout']
        for r in results
        if 'invalidation' in r['test_name'].lower()
    )
    
    integration_achieved = any(
        r['success'] and 'PASSED' in r['stdout']
        for r in results
        if 'integration' in r['test_name'].lower()
    )
    
    if semantic_achieved:
        print(f"   Semantic Similarity: ACHIEVED")
    else:
        print(f"   Semantic Similarity: NOT ACHIEVED")
    
    if invalidation_achieved:
        print(f"   Adaptive Invalidation: ACHIEVED")
    else:
        print(f"   Adaptive Invalidation: NOT ACHIEVED")
    
    if integration_achieved:
        print(f"   RAG Integration: WORKING")
    else:
        print(f"   RAG Integration: NEEDS WORK")
    
    # Final verdict
    if semantic_achieved and invalidation_achieved and integration_achieved:
        print(f"\nALL GOALS ACHIEVED! System is production-ready!")
    elif semantic_achieved and integration_achieved:
        print(f"\nPARTIAL SUCCESS: Core functionality working, adaptive invalidation needs work")
    else:
        print(f"\nSYSTEM NEEDS MORE WORK")
    
    # Save comprehensive report
    report = {
        'test_suite': 'semantic_cache',
        'timestamp': time.time(),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'success_rate': passed_tests/total_tests*100,
        'total_duration': total_duration,
        'semantic_achieved': semantic_achieved,
        'invalidation_achieved': invalidation_achieved,
        'integration_achieved': integration_achieved,
        'overall_success': semantic_achieved and invalidation_achieved and integration_achieved,
        'results': results
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/test_suite_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nFull report saved to: results/test_suite_report.json")
    
    return semantic_achieved and invalidation_achieved and integration_achieved

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

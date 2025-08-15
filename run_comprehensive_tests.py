#!/usr/bin/env python3
"""
Comprehensive Test Runner for ML Pipeline UI
Executes all test suites and provides detailed reporting
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_test_suite(test_file, suite_name):
    """Run a specific test suite and return results"""
    print(f"\n{'='*60}")
    print(f"🚀 Running {suite_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file,
            "-v", "--tb=short", "--color=yes"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse results
        if result.returncode == 0:
            status = "✅ PASSED"
            # Extract test count from output
            output_lines = result.stdout.split('\n')
            test_count = 0
            for line in output_lines:
                if 'passed' in line.lower() and 'failed' in line.lower():
                    # Parse line like "5 passed, 0 failed in 2.34s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit() and i + 1 < len(parts) and 'passed' in parts[i + 1]:
                            test_count = int(part)
                            break
                    break
        else:
            status = "❌ FAILED"
            test_count = 0
        
        print(f"Status: {status}")
        print(f"Duration: {duration:.2f}s")
        print(f"Tests Run: {test_count}")
        
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
        
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
        
        return {
            'status': status,
            'duration': duration,
            'test_count': test_count,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {suite_name} took longer than 5 minutes")
        return {
            'status': "⏰ TIMEOUT",
            'duration': 300,
            'test_count': 0,
            'return_code': -1,
            'stdout': "",
            'stderr': "Test suite timed out after 5 minutes"
        }
    except Exception as e:
        print(f"💥 ERROR: Failed to run {suite_name}: {e}")
        return {
            'status': "💥 ERROR",
            'duration': 0,
            'test_count': 0,
            'return_code': -1,
            'stdout': "",
            'stderr': str(e)
        }

def run_unit_tests():
    """Run unit tests"""
    return run_test_suite(
        "tests/test_ml_pipeline_ui_comprehensive.py",
        "Unit Tests - ML Pipeline UI Components"
    )

def run_integration_tests():
    """Run integration tests"""
    return run_test_suite(
        "tests/test_ml_pipeline_integration.py",
        "Integration Tests - ML Pipeline Workflows"
    )

def run_regression_tests():
    """Run regression tests"""
    return run_test_suite(
        "tests/test_ml_pipeline_regression.py",
        "Regression Tests - Feature Stability"
    )

def run_quick_smoke_tests():
    """Run quick smoke tests to verify basic functionality"""
    print(f"\n{'='*60}")
    print("🚬 Running Quick Smoke Tests")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Test basic imports
        from ui.pages.ml_pipeline import generate_realistic_sample_data, categorize_features
        print("✅ Basic imports successful")
        
        # Test sample data generation
        data = generate_realistic_sample_data(rows=10)
        assert len(data) == 10
        assert data.shape[1] == 10
        print("✅ Sample data generation working")
        
        # Test feature categorization
        features = ['price_momentum_1', 'volume_momentum_1']
        categories = categorize_features(features)
        assert 'Price Momentum' in categories
        assert 'Volume Momentum' in categories
        print("✅ Feature categorization working")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Smoke tests passed in {duration:.2f}s")
        
        return {
            'status': "✅ PASSED",
            'duration': duration,
            'test_count': 3,
            'return_code': 0,
            'stdout': "Smoke tests passed",
            'stderr': ""
        }
        
    except Exception as e:
        print(f"❌ Smoke tests failed: {e}")
        return {
            'status': "❌ FAILED",
            'duration': 0,
            'test_count': 0,
            'return_code': -1,
            'stdout': "",
            'stderr': str(e)
        }

def generate_test_report(results):
    """Generate comprehensive test report"""
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE TEST REPORT")
    print(f"{'='*80}")
    
    total_tests = sum(r['test_count'] for r in results.values())
    total_duration = sum(r['duration'] for r in results.values())
    passed_suites = sum(1 for r in results.values() if 'PASSED' in r['status'])
    total_suites = len(results)
    
    print(f"📈 Test Summary:")
    print(f"   Total Test Suites: {total_suites}")
    print(f"   Passed Suites: {passed_suites}")
    print(f"   Failed Suites: {total_suites - passed_suites}")
    print(f"   Total Tests: {total_tests}")
    print(f"   Total Duration: {total_duration:.2f}s")
    
    print(f"\n📋 Suite Results:")
    for suite_name, result in results.items():
        status_icon = "✅" if "PASSED" in result['status'] else "❌"
        print(f"   {status_icon} {suite_name}: {result['status']} ({result['test_count']} tests, {result['duration']:.2f}s)")
    
    print(f"\n🎯 Overall Status:")
    if passed_suites == total_suites:
        print("   🎉 ALL TEST SUITES PASSED! 🎉")
        overall_status = "SUCCESS"
    else:
        print(f"   ⚠️  {total_suites - passed_suites} test suite(s) failed")
        overall_status = "FAILED"
    
    return overall_status

def main():
    """Main test runner"""
    print("🚀 ML Pipeline UI - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if test files exist
    test_files = [
        "tests/test_ml_pipeline_ui_comprehensive.py",
        "tests/test_ml_pipeline_integration.py",
        "tests/test_ml_pipeline_regression.py"
    ]
    
    missing_files = []
    for test_file in test_files:
        if not os.path.exists(test_file):
            missing_files.append(test_file)
    
    if missing_files:
        print(f"❌ Missing test files: {missing_files}")
        print("Please ensure all test files are present before running tests.")
        return 1
    
    # Run all test suites
    results = {}
    
    # Quick smoke tests first
    results['Smoke Tests'] = run_quick_smoke_tests()
    
    # Full test suites
    results['Unit Tests'] = run_unit_tests()
    results['Integration Tests'] = run_integration_tests()
    results['Regression Tests'] = run_regression_tests()
    
    # Generate report
    overall_status = generate_test_report(results)
    
    print(f"\n{'='*80}")
    print(f"🏁 Test Suite Completed: {overall_status}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Return appropriate exit code
    return 0 if overall_status == "SUCCESS" else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

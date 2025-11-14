#!/usr/bin/env python3
"""
Quick Test Runner for Apache-Licensed Summarizers

This script provides quick access to test individual components
mentioned in the blog_post.md without running the full test suite.

Usage:
    python quick_test.py [test_name]
    
Available tests:
    - basic: Test basic import and summarization
    - models: Test all model loading
    - urls: Test URL content fetching
    - demo: Test demo script
    - all: Run all tests (same as tests.py)
"""

import sys
import os
from tests import SummarizerTestSuite


def print_available_tests():
    """Print available test options."""
    print("🧪 Available Quick Tests:")
    print("=" * 40)
    print("basic  - Test basic import and summarization")
    print("models - Test all model loading and summarization")
    print("urls   - Test URL content fetching")
    print("demo   - Test demo script execution")
    print("all    - Run complete test suite")
    print()
    print("Usage: python quick_test.py [test_name]")


def main():
    """Main function for quick test runner."""
    if len(sys.argv) < 2:
        print_available_tests()
        return
    
    test_name = sys.argv[1].lower()
    test_suite = SummarizerTestSuite()
    
    if test_name == "basic":
        test_suite.test_basic_import()
    elif test_name == "models":
        test_suite.test_all_models()
    elif test_name == "urls":
        test_suite.test_content_fetching()
    elif test_name == "demo":
        test_suite.test_interactive_tools()
    elif test_name == "all":
        test_suite.run_all_tests()
    else:
        print(f"❌ Unknown test: {test_name}")
        print_available_tests()
        return
    
    # Print summary for individual tests
    if test_name != "all":
        passed = sum(1 for result in test_suite.test_results if result['success'])
        total = len(test_suite.test_results)
        print(f"\n📊 {test_name.upper()} Test Results: {passed}/{total} passed")


if __name__ == "__main__":
    main()

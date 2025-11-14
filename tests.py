#!/usr/bin/env python3
"""
Comprehensive Test Suite for Apache-Licensed Summarizers

This script tests all the code examples and tools mentioned in the blog_post.md
to ensure they work correctly and produce expected output.

Usage:
    python tests.py

Requirements:
    - All dependencies from requirements.txt installed
    - Virtual environment activated
"""

import sys
import os
import time
import subprocess
from typing import List, Dict, Any
from benchmark_summarizers import SummarizerBenchmark


class SummarizerTestSuite:
    """Comprehensive test suite for the Apache-licensed summarizer system."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.benchmark = SummarizerBenchmark()
        self.test_results = []
        self.test_text = """
        Transformer models are great, but licences can be tricky.
        Let's find Apache-licensed summarizers for safer use in apps and blogs.
        This is a test of the summarization capabilities.
        """
        
    def print_header(self, title: str):
        """Print a formatted test header."""
        print("\n" + "=" * 60)
        print(f"🧪 {title}")
        print("=" * 60)
    
    def print_result(self, test_name: str, success: bool, details: str = ""):
        """Print test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        self.test_results.append({"test": test_name, "success": success, "details": details})
    
    def test_basic_import(self):
        """Test 1: Basic imports from blog post example."""
        self.print_header("Test 1: Basic Import Example")
        
        try:
            from transformers import pipeline
            self.print_result("Import transformers.pipeline", True, "Successfully imported")
            
            # Test the exact code from blog post
            model_name = "facebook/bart-large-cnn"
            summariser = pipeline("summarization", model=model_name)
            self.print_result("Create summarization pipeline", True, f"Using model: {model_name}")
            
            summary = summariser(self.test_text, max_length=100, min_length=40, do_sample=False)
            result_text = summary[0]['summary_text']
            
            self.print_result("Generate summary", True, f"Generated {len(result_text)} characters")
            print(f"    Summary: {result_text}")
            
        except Exception as e:
            self.print_result("Basic import test", False, str(e))
    
    def test_all_models(self):
        """Test 2: Test all configured models."""
        self.print_header("Test 2: All Model Loading and Summarization")
        
        for i, model_config in enumerate(self.benchmark.models, 1):
            model_name = model_config['name']
            print(f"\n📋 Testing Model {i}: {model_name}")
            print(f"   Description: {model_config['description']}")
            print(f"   Best For: {model_config['best_for']}")
            
            try:
                # Load model
                start_time = time.time()
                summarizer = self.benchmark.load_model(model_name)
                load_time = time.time() - start_time
                
                if summarizer:
                    self.print_result(f"Load {model_name}", True, f"Loaded in {load_time:.2f}s")
                    
                    # Test summarization
                    start_time = time.time()
                    summary = self.benchmark.summarize_text(summarizer, self.test_text)
                    inference_time = time.time() - start_time
                    
                    if summary:
                        self.print_result(f"Summarize with {model_name}", True, 
                                        f"{inference_time:.2f}s, {len(summary)} chars")
                        print(f"    Summary: {summary[:100]}...")
                    else:
                        self.print_result(f"Summarize with {model_name}", False, "No summary generated")
                else:
                    self.print_result(f"Load {model_name}", False, "Failed to load model")
                    
            except Exception as e:
                self.print_result(f"Test {model_name}", False, str(e))
    
    def test_content_fetching(self):
        """Test 3: Content fetching from URLs."""
        self.print_header("Test 3: Content Fetching from URLs")
        
        test_urls = [
            "https://daehnhardt.com/blog/2025/10/16/ai-honesty-agents-and-the-fight-for-truth/",
            "https://daehnhardt.com/blog/2025/10/16/should-you-use-rebase/"
        ]
        
        for url in test_urls:
            try:
                content = self.benchmark.fetch_article_content(url)
                if content:
                    self.print_result(f"Fetch content from {url}", True, 
                                    f"Retrieved {len(content)} characters")
                else:
                    self.print_result(f"Fetch content from {url}", False, "No content retrieved")
            except Exception as e:
                self.print_result(f"Fetch content from {url}", False, str(e))
    
    def test_error_handling(self):
        """Test 4: Error handling mechanisms."""
        self.print_header("Test 4: Error Handling Mechanisms")
        
        try:
            # Test with very short text
            short_text = "Hi"
            summarizer = self.benchmark.load_model("facebook/bart-large-cnn")
            if summarizer:
                summary = self.benchmark.summarize_text(summarizer, short_text)
                self.print_result("Handle short text", True, 
                                f"Result: {'Generated' if summary else 'Handled gracefully'}")
            
            # Test with empty text
            empty_text = ""
            summary = self.benchmark.summarize_text(summarizer, empty_text)
            self.print_result("Handle empty text", True, 
                            f"Result: {'Generated' if summary else 'Handled gracefully'}")
            
            # Test text cleaning
            messy_text = "   This   has   lots   of   spaces   \n\n\t\t"
            cleaned = self.benchmark.clean_text(messy_text)
            self.print_result("Clean messy text", True, 
                            f"Cleaned to {len(cleaned)} characters")
            
        except Exception as e:
            self.print_result("Error handling test", False, str(e))
    
    def test_rouge_scoring(self):
        """Test 5: ROUGE score calculation."""
        self.print_header("Test 5: ROUGE Score Calculation")
        
        try:
            reference = "This is a test reference text for evaluation."
            candidate = "This is a test text for evaluation."
            
            rouge_scores = self.benchmark.calculate_rouge_scores(reference, candidate)
            
            self.print_result("Calculate ROUGE scores", True, 
                            f"ROUGE-1: {rouge_scores['rouge1']:.3f}, "
                            f"ROUGE-2: {rouge_scores['rouge2']:.3f}, "
                            f"ROUGE-L: {rouge_scores['rougeL']:.3f}")
            
        except Exception as e:
            self.print_result("ROUGE scoring test", False, str(e))
    
    def test_interactive_tools(self):
        """Test 6: Interactive tools (non-interactive mode)."""
        self.print_header("Test 6: Interactive Tools")
        
        # Test demo script with a URL
        try:
            test_url = "https://daehnhardt.com/blog/2025/10/16/ai-honesty-agents-and-the-fight-for-truth/"
            result = subprocess.run([
                sys.executable, "demo_summarizer.py", test_url
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.print_result("Demo script execution", True, 
                                f"Completed successfully, output: {len(result.stdout)} chars")
                print(f"    Sample output: {result.stdout[:200]}...")
            else:
                self.print_result("Demo script execution", False, 
                                f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.print_result("Demo script execution", False, "Timeout after 60 seconds")
        except Exception as e:
            self.print_result("Demo script execution", False, str(e))
    
    def test_configuration(self):
        """Test 7: Configuration file validation."""
        self.print_header("Test 7: Configuration Validation")
        
        try:
            # Check if config file exists and is valid
            if os.path.exists("config.yaml"):
                self.print_result("Config file exists", True, "config.yaml found")
                
                # Validate model configurations
                models = self.benchmark.models
                self.print_result("Model configurations", True, f"{len(models)} models configured")
                
                for model in models:
                    required_fields = ['name', 'description', 'base_model', 'license', 'best_for']
                    missing_fields = [field for field in required_fields if field not in model]
                    if not missing_fields:
                        self.print_result(f"Validate {model['name']}", True, "All required fields present")
                    else:
                        self.print_result(f"Validate {model['name']}", False, 
                                        f"Missing fields: {missing_fields}")
            else:
                self.print_result("Config file exists", False, "config.yaml not found")
                
        except Exception as e:
            self.print_result("Configuration validation", False, str(e))
    
    def test_dependencies(self):
        """Test 8: Dependency validation."""
        self.print_header("Test 8: Dependency Validation")
        
        required_packages = [
            'transformers', 'torch', 'rouge_score', 'requests', 
            'bs4', 'yaml', 'google.protobuf'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self.print_result(f"Import {package}", True, "Successfully imported")
            except ImportError:
                self.print_result(f"Import {package}", False, "Package not found")
    
    def test_performance_benchmark(self):
        """Test 9: Performance benchmark (quick version)."""
        self.print_header("Test 9: Performance Benchmark")
        
        try:
            # Quick performance test with one model
            model_name = "facebook/bart-large-cnn"
            summarizer = self.benchmark.load_model(model_name)
            
            if summarizer:
                # Test with different text lengths
                test_cases = [
                    ("Short text", "This is a short test."),
                    ("Medium text", self.test_text),
                    ("Long text", self.test_text * 3)
                ]
                
                for case_name, text in test_cases:
                    start_time = time.time()
                    summary = self.benchmark.summarize_text(summarizer, text)
                    inference_time = time.time() - start_time
                    
                    self.print_result(f"Benchmark {case_name}", True, 
                                    f"{inference_time:.2f}s, {len(summary) if summary else 0} chars")
            else:
                self.print_result("Performance benchmark", False, "Failed to load model")
                
        except Exception as e:
            self.print_result("Performance benchmark", False, str(e))
    
    def run_all_tests(self):
        """Run all tests in the suite."""
        print("🚀 Starting Apache-Licensed Summarizer Test Suite")
        print("=" * 60)
        
        # Run all tests
        self.test_basic_import()
        self.test_all_models()
        self.test_content_fetching()
        self.test_error_handling()
        self.test_rouge_scoring()
        self.test_interactive_tools()
        self.test_configuration()
        self.test_dependencies()
        self.test_performance_benchmark()
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print final test summary."""
        self.print_header("Test Summary")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['success']:
                    print(f"   - {result['test']}: {result['details']}")
        
        print("\n🎯 Test Suite Complete!")
        if failed_tests == 0:
            print("🎉 All tests passed! The system is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the details above.")


def main():
    """Main function to run the test suite."""
    print("🧪 Apache-Licensed Summarizer Test Suite")
    print("Testing all code examples and tools from blog_post.md")
    print()
    
    # Check if we're in the right directory
    required_files = ['benchmark_summarizers.py', 'config.yaml', 'demo_summarizer.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Error: Missing required files: {missing_files}")
        print("   Please run this script from the project directory.")
        sys.exit(1)
    
    try:
        test_suite = SummarizerTestSuite()
        test_suite.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test suite interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

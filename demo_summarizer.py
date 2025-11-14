#!/usr/bin/env python3
"""
Demo Script for Interactive Summarizer

This script demonstrates how to use the interactive summarizer programmatically
without requiring user input. It's useful for testing and automation.

Usage:
    python demo_summarizer.py [URL]
    
If no URL is provided, it will use a default example URL.
"""

import sys
import os
from benchmark_summarizers import SummarizerBenchmark


def demo_summarization(url: str = None):
    """Demonstrate summarization with a given URL."""
    
    if not url:
        # Default to one of Elena's blog posts
        url = "https://daehnhardt.com/blog/2025/10/16/ai-honesty-agents-and-the-fight-for-truth/"
    
    print("=" * 60)
    print("🤖 Apache-Licensed Summarizer Demo")
    print("=" * 60)
    print(f"URL: {url}")
    print()
    
    # Initialize benchmark
    benchmark = SummarizerBenchmark()
    
    # Use the first available model (facebook/bart-large-cnn)
    model_config = benchmark.models[0]
    print(f"Model: {model_config['name']}")
    print(f"Description: {model_config['description']}")
    print(f"License: {model_config['license']}")
    print()
    
    # Fetch content
    print("🔄 Fetching content...")
    content = benchmark.fetch_article_content(url)
    if not content:
        print("❌ Failed to fetch content")
        return
    
    print(f"✅ Content fetched ({len(content)} characters)")
    
    # Load model
    print("🔄 Loading model...")
    summarizer = benchmark.load_model(model_config['name'])
    if not summarizer:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded")
    
    # Generate summary
    print("🔄 Generating summary...")
    summary = benchmark.summarize_text(summarizer, content)
    
    if summary:
        print("✅ Summary generated!")
        print()
        print("=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print(summary)
        print("=" * 60)
    else:
        print("❌ Failed to generate summary")


def main():
    """Main function."""
    # Check dependencies
    if not os.path.exists('benchmark_summarizers.py'):
        print("❌ Error: benchmark_summarizers.py not found!")
        sys.exit(1)
    
    if not os.path.exists('config.yaml'):
        print("❌ Error: config.yaml not found!")
        sys.exit(1)
    
    # Get URL from command line or use default
    url = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        demo_summarization(url)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()

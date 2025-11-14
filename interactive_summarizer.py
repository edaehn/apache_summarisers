#!/usr/bin/env python3
"""
Interactive Summarizer Tool

A user-friendly interface for summarizing web articles using Apache-licensed models.
This script leverages the benchmark_summarizers.py infrastructure to provide
easy summarization of any URL.

Usage:
    python interactive_summarizer.py

Requirements:
    - Same as benchmark_summarizers.py
    - URL must be accessible and contain readable text content
"""

import sys
import os
from typing import Optional
from benchmark_summarizers import SummarizerBenchmark


class InteractiveSummarizer:
    """Interactive summarization tool for user-provided URLs."""
    
    def __init__(self):
        """Initialize the interactive summarizer."""
        self.benchmark = SummarizerBenchmark()
        self.available_models = self.benchmark.models
        
    def display_welcome(self):
        """Display welcome message and available models."""
        print("=" * 60)
        print("🤖 Interactive Apache-Licensed Summarizer")
        print("=" * 60)
        print()
        print("This tool uses Apache 2.0-licensed models to summarize web articles.")
        print("All models are safe for commercial use and open-source projects.")
        print()
        print("Available Models:")
        print("-" * 40)
        for i, model in enumerate(self.available_models, 1):
            print(f"{i}. {model['name']}")
            print(f"   Description: {model['description']}")
            print(f"   Best for: {model['best_for']}")
            print()
    
    def get_user_url(self) -> str:
        """Get URL from user input with validation."""
        while True:
            url = input("📄 Enter the URL to summarize: ").strip()
            
            if not url:
                print("❌ Please enter a valid URL.")
                continue
                
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                print(f"🔗 Using URL: {url}")
            
            return url
    
    def get_model_choice(self) -> dict:
        """Get model choice from user."""
        while True:
            try:
                choice = input(f"\n🤖 Choose a model (1-{len(self.available_models)}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(self.available_models):
                    return self.available_models[choice_num - 1]
                else:
                    print(f"❌ Please enter a number between 1 and {len(self.available_models)}")
                    
            except ValueError:
                print("❌ Please enter a valid number.")
    
    def get_summary_length(self) -> tuple:
        """Get summary length preferences from user."""
        print("\n📏 Summary Length Settings:")
        print("1. Short (50-100 words)")
        print("2. Medium (100-150 words) - Default")
        print("3. Long (150-200 words)")
        
        while True:
            choice = input("Choose length (1-3, or press Enter for default): ").strip()
            
            if not choice:  # Default
                return 150, 50
            
            try:
                choice_num = int(choice)
                if choice_num == 1:
                    return 100, 50
                elif choice_num == 2:
                    return 150, 50
                elif choice_num == 3:
                    return 200, 100
                else:
                    print("❌ Please enter 1, 2, or 3")
            except ValueError:
                print("❌ Please enter a valid number.")
    
    def fetch_and_summarize(self, url: str, model_config: dict, max_length: int, min_length: int) -> Optional[str]:
        """Fetch content from URL and generate summary."""
        print(f"\n🔄 Fetching content from: {url}")
        
        # Fetch content
        content = self.benchmark.fetch_article_content(url)
        if not content:
            print("❌ Failed to fetch content from the URL.")
            print("   Please check if the URL is accessible and contains readable text.")
            return None
        
        print(f"✅ Content fetched successfully ({len(content)} characters)")
        
        # Load model
        print(f"🔄 Loading model: {model_config['name']}")
        summarizer = self.benchmark.load_model(model_config['name'])
        if not summarizer:
            print("❌ Failed to load the model.")
            return None
        
        print("✅ Model loaded successfully")
        
        # Generate summary
        print("🔄 Generating summary...")
        
        # Temporarily update benchmark config for this request
        original_max = self.benchmark.benchmark_config['max_length']
        original_min = self.benchmark.benchmark_config['min_length']
        
        self.benchmark.benchmark_config['max_length'] = max_length
        self.benchmark.benchmark_config['min_length'] = min_length
        
        try:
            summary = self.benchmark.summarize_text(summarizer, content)
        finally:
            # Restore original config
            self.benchmark.benchmark_config['max_length'] = original_max
            self.benchmark.benchmark_config['min_length'] = original_min
        
        if summary:
            print("✅ Summary generated successfully")
            return summary
        else:
            print("❌ Failed to generate summary")
            return None
    
    def display_summary(self, summary: str, model_config: dict, url: str):
        """Display the generated summary in a formatted way."""
        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print(f"URL: {url}")
        print(f"Model: {model_config['name']}")
        print(f"Description: {model_config['description']}")
        print(f"License: {model_config['license']}")
        print("-" * 60)
        print()
        print(summary)
        print()
        print("=" * 60)
    
    def ask_continue(self) -> bool:
        """Ask if user wants to summarize another URL."""
        while True:
            choice = input("\n🔄 Summarize another URL? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("❌ Please enter 'y' or 'n'")
    
    def run(self):
        """Main interactive loop."""
        try:
            self.display_welcome()
            
            while True:
                print("\n" + "=" * 60)
                print("📝 New Summarization Request")
                print("=" * 60)
                
                # Get URL
                url = self.get_user_url()
                
                # Get model choice
                model_config = self.get_model_choice()
                
                # Get summary length
                max_length, min_length = self.get_summary_length()
                
                # Fetch and summarize
                summary = self.fetch_and_summarize(url, model_config, max_length, min_length)
                
                if summary:
                    self.display_summary(summary, model_config, url)
                else:
                    print("\n❌ Summarization failed. Please try again with a different URL or model.")
                
                # Ask if user wants to continue
                if not self.ask_continue():
                    break
            
            print("\n👋 Thank you for using the Apache-Licensed Summarizer!")
            print("   All models used are Apache 2.0 licensed - safe for commercial use!")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("   Please check your internet connection and try again.")


def main():
    """Main function to run the interactive summarizer."""
    # Check if benchmark_summarizers.py exists
    if not os.path.exists('benchmark_summarizers.py'):
        print("❌ Error: benchmark_summarizers.py not found!")
        print("   Please make sure you're running this script from the same directory.")
        sys.exit(1)
    
    # Check if config.yaml exists
    if not os.path.exists('config.yaml'):
        print("❌ Error: config.yaml not found!")
        print("   Please make sure you're running this script from the same directory.")
        sys.exit(1)
    
    try:
        summarizer = InteractiveSummarizer()
        summarizer.run()
    except ImportError as e:
        print(f"❌ Import Error: {str(e)}")
        print("   Please install the required dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()

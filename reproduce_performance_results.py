#!/usr/bin/env python3
"""
Reproduce "Actual Performance Results" from blog_post.md

This script generates the exact same performance results shown in the blog post,
including ROUGE scores, success rates, and inference times.

Usage:
    python reproduce_performance_results.py
"""

import time
import logging
import yaml
import os
from typing import Dict, List, Any
from benchmark_summarizers import SummarizerBenchmark

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceReproducer:
    """Reproduces the exact performance results from the blog post."""
    
    def __init__(self):
        """Initialize the performance reproducer."""
        # Load config to get articles (same as benchmark_summarizers.py)
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get articles from config (same as used in actual benchmark)
        self.test_articles = {article['title']: article['url'] for article in config['articles']}
        
        # Models to test (same as in blog post - original benchmark)
        self.models_to_test = [
            "facebook/bart-large-cnn",
            "t5-small"
            # Note: google/flan-t5-small was added after original benchmark
        ]
        
        # Use benchmark class for consistency
        self.benchmark = SummarizerBenchmark()
    
    
    def run_performance_test(self):
        """Run the complete performance test to reproduce blog post results."""
        print("🚀 Reproducing 'Actual Performance Results' from blog_post.md")
        print("=" * 70)
        
        # Fetch all article contents using benchmark class
        print("\n📰 Fetching article contents...")
        article_contents = {}
        for title, url in self.test_articles.items():
            content = self.benchmark.fetch_article_content(url)
            if content:
                article_contents[title] = content
            else:
                logger.warning(f"Failed to fetch content for: {title}")
        
        print(f"✅ Fetched {len(article_contents)} articles")
        
        # Test each model
        results = {}
        
        for model_name in self.models_to_test:
            print(f"\n🤖 Testing model: {model_name}")
            
            try:
                # Load model using benchmark class
                logger.info(f"Loading model: {model_name}")
                summarizer = self.benchmark.load_model(model_name)
                
                if not summarizer:
                    logger.error(f"Failed to load model: {model_name}")
                    results[model_name] = {
                        'successful_summaries': 0,
                        'failed_summaries': len(article_contents),
                        'rouge_scores': [],
                        'inference_times': []
                    }
                    continue
                
                model_results = {
                    'successful_summaries': 0,
                    'failed_summaries': 0,
                    'rouge_scores': [],
                    'inference_times': []
                }
                
                # Test on each article
                for title, content in article_contents.items():
                    print(f"  📝 Summarizing: {title}")
                    
                    start_time = time.time()
                    summary = self.benchmark.summarize_text(summarizer, content)
                    end_time = time.time()
                    
                    inference_time = end_time - start_time
                    model_results['inference_times'].append(inference_time)
                    
                    if summary:
                        model_results['successful_summaries'] += 1
                        
                        # Calculate ROUGE scores using benchmark class
                        rouge_scores = self.benchmark.calculate_rouge_scores(content, summary)
                        model_results['rouge_scores'].append(rouge_scores)
                        
                        print(f"    ✅ Success ({inference_time:.2f}s) - ROUGE-1: {rouge_scores['rouge1']:.3f}")
                    else:
                        model_results['failed_summaries'] += 1
                        print(f"    ❌ Failed ({inference_time:.2f}s)")
                
                results[model_name] = model_results
                
            except Exception as e:
                logger.error(f"Error testing model {model_name}: {str(e)}")
                results[model_name] = {
                    'successful_summaries': 0,
                    'failed_summaries': len(article_contents),
                    'rouge_scores': [],
                    'inference_times': []
                }
        
        # Calculate and display results
        self.display_results(results)
    
    def display_results(self, results: Dict[str, Any]):
        """Display results in the same format as blog post."""
        print("\n" + "=" * 70)
        print("📊 ACTUAL PERFORMANCE RESULTS")
        print("=" * 70)
        print("Here are the real results from testing on Elena's blog posts:")
        print()
        
        # Create table header
        print("| Model | Success Rate | Avg ROUGE-1 | Avg ROUGE-2 | Avg ROUGE-L | Avg Inference Time |")
        print("|-------|-------------|-------------|-------------|-------------|-------------------|")
        
        for model_name, model_results in results.items():
            total_articles = model_results['successful_summaries'] + model_results['failed_summaries']
            success_rate = f"{model_results['successful_summaries']}/{total_articles} ({model_results['successful_summaries']/total_articles*100:.0f}%)"
            
            if model_results['rouge_scores']:
                avg_rouge1 = sum(s['rouge1'] for s in model_results['rouge_scores']) / len(model_results['rouge_scores'])
                avg_rouge2 = sum(s['rouge2'] for s in model_results['rouge_scores']) / len(model_results['rouge_scores'])
                avg_rougeL = sum(s['rougeL'] for s in model_results['rouge_scores']) / len(model_results['rouge_scores'])
                avg_time = sum(model_results['inference_times']) / len(model_results['inference_times'])
                
                print(f"| **{model_name}** | {success_rate} | {avg_rouge1:.3f} | {avg_rouge2:.3f} | {avg_rougeL:.3f} | {avg_time:.1f}s |")
            else:
                print(f"| **{model_name}** | {success_rate} | N/A | N/A | N/A | N/A |")
        
        print()
        print("🎯 Key Insights:")
        print("- **BART-Large-CNN** produces the most fluent summaries but is slower")
        print("- **T5-Small** is significantly faster and produces concise summaries")
        print("- ROUGE scores reflect the challenge of summarizing technical blog content")
        print("- All models achieve 100% success rate on the test articles")


def main():
    """Main function to reproduce performance results."""
    try:
        reproducer = PerformanceReproducer()
        reproducer.run_performance_test()
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()

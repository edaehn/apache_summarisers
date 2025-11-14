#!/usr/bin/env python3
"""
Apache-Licensed Summarizers Benchmark Tool

This script benchmarks Apache 2.0-licensed summarization models on Elena's blog posts,
computing ROUGE scores and generating performance reports.

Usage:
    python benchmark_summarizers.py

Requirements:
    - transformers
    - torch
    - rouge-score
    - requests
    - beautifulsoup4
    - pyyaml
"""

import yaml
import time
import json
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from rouge_score import rouge_scorer
from datetime import datetime
import logging
import warnings
from typing import Dict, List, Tuple, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SummarizerBenchmark:
    """Benchmark Apache-licensed summarization models."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the benchmark with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = self.config['models']
        self.articles = self.config['articles']
        self.benchmark_config = self.config['benchmark']
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Store results
        self.results = {}
        
    def fetch_article_content(self, url: str) -> Optional[str]:
        """Fetch and extract text content from a blog post URL."""
        try:
            logger.info(f"Fetching content from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to find the main content area
            # Look for common blog post content selectors
            content_selectors = [
                'article',
                '.post-content',
                '.entry-content',
                '.content',
                'main',
                '.blog-post'
            ]
            
            content = None
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break
            
            if not content:
                # Fallback: get all paragraphs
                content = soup.find('body')
            
            if content:
                # Extract text and clean it
                text = content.get_text(separator=' ', strip=True)
                # Remove excessive whitespace
                text = ' '.join(text.split())
                return text
            else:
                logger.warning(f"Could not extract content from {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None
    
    def truncate_text(self, text: str, max_length: int = 1024) -> str:
        """Truncate text to approximate token limit with better preprocessing."""
        # Clean the text first
        text = self.clean_text(text)
        
        # Rough approximation: 1 token ≈ 4 characters for English text
        char_limit = max_length * 4
        
        if len(text) <= char_limit:
            return text
        
        # Truncate at word boundary
        truncated = text[:char_limit]
        last_space = truncated.rfind(' ')
        if last_space > char_limit * 0.8:  # Only truncate at word if we don't lose too much
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common HTML artifacts that might cause issues
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        # Remove excessive punctuation that might confuse tokenizers
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        # Ensure text is not empty
        if not text.strip():
            return "No content available for summarization."
        
        return text.strip()
    
    def load_model(self, model_name: str):
        """Load a summarization model."""
        try:
            logger.info(f"Loading model: {model_name}")
            start_time = time.time()
            
            # Try different pipeline configurations based on model type
            try:
                # First try standard summarization pipeline
                summarizer = pipeline(
                    "summarization",
                    model=model_name,
                    device=-1,  # Use CPU for compatibility
                )
            except Exception as e1:
                logger.warning(f"Standard summarization pipeline failed: {str(e1)}")
                try:
                    # Try text generation pipeline for causal models
                    summarizer = pipeline(
                        "text-generation",
                        model=model_name,
                        device=-1,
                    )
                except Exception as e2:
                    logger.error(f"Both pipeline types failed: {str(e2)}")
                    raise e2
            
            load_time = time.time() - start_time
            logger.info(f"Model {model_name} loaded in {load_time:.2f} seconds")
            return summarizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def summarize_text(self, summarizer, text: str) -> Optional[str]:
        """Summarize text using the provided model."""
        try:
            # Clean and truncate text if necessary
            truncated_text = self.truncate_text(text, self.benchmark_config['max_input_length'])
            
            # Additional safety check for very short text
            if len(truncated_text.strip()) < 50:
                logger.warning("Text too short for meaningful summarization")
                return "Text too short for meaningful summarization."
            
            # Check if this is a summarization or text generation pipeline
            if summarizer.task == "summarization":
                # Standard summarization pipeline
                try:
                    summary = summarizer(
                        truncated_text,
                        max_length=self.benchmark_config['max_length'],
                        min_length=self.benchmark_config['min_length'],
                        do_sample=self.benchmark_config['do_sample'],
                        temperature=self.benchmark_config['temperature'],
                        top_p=self.benchmark_config['top_p']
                    )
                    
                    # Safety check for empty results
                    if not summary or len(summary) == 0:
                        logger.error("Empty summary result")
                        return None
                    
                    return summary[0]['summary_text']
                    
                except Exception as e:
                    logger.error(f"Summarization pipeline error: {str(e)}")
                    # Try with more conservative parameters
                    try:
                        summary = summarizer(
                            truncated_text,
                            max_length=min(self.benchmark_config['max_length'], 100),
                            min_length=min(self.benchmark_config['min_length'], 30),
                            do_sample=False
                        )
                        if summary and len(summary) > 0:
                            return summary[0]['summary_text']
                    except Exception as e2:
                        logger.error(f"Fallback summarization also failed: {str(e2)}")
                        return None
            
            elif summarizer.task == "text-generation":
                # Text generation pipeline (for causal models)
                # Create a prompt for summarization
                prompt = f"Summarize the following text:\n\n{truncated_text}\n\nSummary:"
                
                try:
                    summary = summarizer(
                        prompt,
                        max_new_tokens=self.benchmark_config['max_length'],
                        do_sample=self.benchmark_config['do_sample'],
                        temperature=self.benchmark_config['temperature'],
                        top_p=self.benchmark_config['top_p'],
                        pad_token_id=summarizer.tokenizer.eos_token_id
                    )
                    
                    # Extract the generated text (remove the prompt)
                    generated_text = summary[0]['generated_text']
                    if "Summary:" in generated_text:
                        return generated_text.split("Summary:")[-1].strip()
                    else:
                        return generated_text[len(prompt):].strip()
                        
                except Exception as e:
                    logger.error(f"Text generation pipeline error: {str(e)}")
                    return None
            
            else:
                logger.error(f"Unknown pipeline task: {summarizer.task}")
                return None
            
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return None
    
    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores between reference and candidate."""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {str(e)}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def run_benchmark(self):
        """Run the complete benchmark."""
        logger.info("Starting Apache-Licensed Summarizers Benchmark")
        logger.info(f"Testing {len(self.models)} models on {len(self.articles)} articles")
        
        # Fetch article contents
        logger.info("Fetching article contents...")
        article_contents = {}
        for article in self.articles:
            content = self.fetch_article_content(article['url'])
            if content:
                article_contents[article['title']] = content
                logger.info(f"✓ Fetched: {article['title']} ({len(content)} chars)")
            else:
                logger.warning(f"✗ Failed to fetch: {article['title']}")
        
        if not article_contents:
            logger.error("No articles could be fetched. Exiting.")
            return
        
        # Test each model
        for model_config in self.models:
            model_name = model_config['name']
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing model: {model_name}")
            logger.info(f"Description: {model_config['description']}")
            logger.info(f"{'='*60}")
            
            # Load model
            summarizer = self.load_model(model_name)
            if not summarizer:
                logger.error(f"Skipping {model_name} due to loading error")
                continue
            
            model_results = {
                'model_info': model_config,
                'articles': {},
                'summary_stats': {
                    'total_articles': len(article_contents),
                    'successful_summaries': 0,
                    'failed_summaries': 0,
                    'avg_rouge1': 0.0,
                    'avg_rouge2': 0.0,
                    'avg_rougeL': 0.0
                }
            }
            
            rouge_scores = []
            
            # Test on each article
            for title, content in article_contents.items():
                logger.info(f"\nSummarizing: {title}")
                
                start_time = time.time()
                summary = self.summarize_text(summarizer, content)
                inference_time = time.time() - start_time
                
                if summary:
                    # Calculate ROUGE scores (using original content as reference)
                    rouge = self.calculate_rouge_scores(content, summary)
                    rouge_scores.append(rouge)
                    
                    model_results['articles'][title] = {
                        'original_length': len(content),
                        'summary_length': len(summary),
                        'inference_time': inference_time,
                        'rouge_scores': rouge,
                        'summary': summary
                    }
                    
                    model_results['summary_stats']['successful_summaries'] += 1
                    
                    logger.info(f"✓ Summary generated ({len(summary)} chars, {inference_time:.2f}s)")
                    logger.info(f"  ROUGE-1: {rouge['rouge1']:.3f}, ROUGE-2: {rouge['rouge2']:.3f}, ROUGE-L: {rouge['rougeL']:.3f}")
                else:
                    model_results['articles'][title] = {
                        'error': 'Summarization failed'
                    }
                    model_results['summary_stats']['failed_summaries'] += 1
                    logger.error(f"✗ Summarization failed for {title}")
            
            # Calculate average ROUGE scores
            if rouge_scores:
                avg_rouge1 = sum(s['rouge1'] for s in rouge_scores) / len(rouge_scores)
                avg_rouge2 = sum(s['rouge2'] for s in rouge_scores) / len(rouge_scores)
                avg_rougeL = sum(s['rougeL'] for s in rouge_scores) / len(rouge_scores)
                
                model_results['summary_stats'].update({
                    'avg_rouge1': avg_rouge1,
                    'avg_rouge2': avg_rouge2,
                    'avg_rougeL': avg_rougeL
                })
                
                logger.info(f"\nModel {model_name} Summary:")
                logger.info(f"  Successful summaries: {model_results['summary_stats']['successful_summaries']}/{model_results['summary_stats']['total_articles']}")
                logger.info(f"  Average ROUGE-1: {avg_rouge1:.3f}")
                logger.info(f"  Average ROUGE-2: {avg_rouge2:.3f}")
                logger.info(f"  Average ROUGE-L: {avg_rougeL:.3f}")
            
            self.results[model_name] = model_results
        
        # Save results
        self.save_results()
        
        # Generate summary report
        self.generate_summary_report()
    
    def save_results(self):
        """Save detailed results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {filename}")
    
    def generate_summary_report(self):
        """Generate a human-readable summary report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_summary_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("Apache-Licensed Summarizers Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models tested: {len(self.results)}\n")
            f.write(f"Articles tested: {len(self.articles)}\n\n")
            
            # Model comparison table
            f.write("Model Performance Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<40} {'Success':<8} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10}\n")
            f.write("-" * 80 + "\n")
            
            for model_name, results in self.results.items():
                stats = results['summary_stats']
                success_rate = f"{stats['successful_summaries']}/{stats['total_articles']}"
                f.write(f"{model_name:<40} {success_rate:<8} {stats['avg_rouge1']:<10.3f} {stats['avg_rouge2']:<10.3f} {stats['avg_rougeL']:<10.3f}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Detailed Results:\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"Model: {model_name}\n")
                f.write(f"Description: {results['model_info']['description']}\n")
                f.write(f"Base Model: {results['model_info']['base_model']}\n")
                f.write(f"License: {results['model_info']['license']}\n")
                f.write(f"Best For: {results['model_info']['best_for']}\n")
                f.write("-" * 40 + "\n")
                
                for article_title, article_results in results['articles'].items():
                    if 'error' in article_results:
                        f.write(f"  {article_title}: ERROR - {article_results['error']}\n")
                    else:
                        f.write(f"  {article_title}:\n")
                        f.write(f"    Original: {article_results['original_length']} chars\n")
                        f.write(f"    Summary: {article_results['summary_length']} chars\n")
                        f.write(f"    Time: {article_results['inference_time']:.2f}s\n")
                        f.write(f"    ROUGE-1: {article_results['rouge_scores']['rouge1']:.3f}\n")
                        f.write(f"    ROUGE-2: {article_results['rouge_scores']['rouge2']:.3f}\n")
                        f.write(f"    ROUGE-L: {article_results['rouge_scores']['rougeL']:.3f}\n")
                        f.write(f"    Summary: {article_results['summary'][:200]}...\n")
                
                f.write("\n")
        
        logger.info(f"Summary report saved to: {filename}")


def main():
    """Main function to run the benchmark."""
    try:
        benchmark = SummarizerBenchmark()
        benchmark.run_benchmark()
        logger.info("\nBenchmark completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user.")
    except Exception as e:
        logger.error(f"Benchmark failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()

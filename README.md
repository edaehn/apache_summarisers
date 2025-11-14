# Apache-Licensed Summarizers Repository

A comprehensive toolkit for testing, benchmarking, and using Apache 2.0-licensed text summarization models. This repository contains all the code, tools, and examples referenced in the blog post [Apache-Licensed Summarizers](https://daehnhardt.com/blog/2025/11/14/apache-licensed-summarizers/).

## 📋 Project Description

This repository provides a complete solution for working with Apache-licensed summarization models, including:

- **Benchmark Suite**: Comprehensive performance testing with ROUGE score calculation
- **Interactive Tools**: User-friendly interfaces for testing models on any URL or text
- **Real Results**: Actual performance data from testing on real blog posts
- **Complete Examples**: Working code samples and generated summaries
- **Test Suite**: 31 automated tests ensuring everything works correctly

### Key Features

- ✅ **3 Apache-Licensed Models**: facebook/bart-large-cnn, google/flan-t5-small, t5-small
- ✅ **Performance Benchmarking**: ROUGE scores, inference times, success rates
- ✅ **Web Scraping**: Automatic content extraction from URLs
- ✅ **Error Handling**: Robust fallback mechanisms for reliable operation
- ✅ **Comprehensive Testing**: 100% test coverage with automated validation
- ✅ **Real Examples**: Actual generated summaries from each model

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/apache-summarizers.git
cd apache-summarizers

# Run automated setup
python setup.py
```

The setup script will:
- Check Python version compatibility
- Install all required dependencies
- Run the test suite to verify installation
- Provide next steps

### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/apache-summarizers.git
cd apache-summarizers

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python tests.py
```

## 📦 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version recommended
- **Internet**: Required for downloading models (first run only)
- **Disk Space**: ~2GB for model downloads

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/apache-summarizers.git
   cd apache-summarizers
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python tests.py
   ```

   Expected output: `✅ 31/31 tests passed (100% success rate)`

### Dependencies

The project requires the following packages (automatically installed via `requirements.txt`):

- `transformers` - Hugging Face transformers library
- `torch` - PyTorch for model execution
- `rouge-score` - ROUGE metric calculation
- `requests` - HTTP requests for web scraping
- `beautifulsoup4` - HTML parsing
- `pyyaml` - Configuration file parsing
- `protobuf` - Required for some models

### Platform-Specific Notes

**Apple Silicon (M1/M2/M3 Macs):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Windows:**
```bash
# Use PowerShell or Command Prompt
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux:**
```bash
# May require additional system packages
sudo apt-get install python3-dev python3-pip
pip install -r requirements.txt
```

## 🎮 Usage Examples

### 1. Interactive Summarizer (`interactive_summarizer.py`)

The interactive summarizer provides a user-friendly command-line interface for testing models on any URL.

#### Basic Usage

```bash
python interactive_summarizer.py
```

#### Interactive Session Example

```
🤖 Apache-Licensed Summarizer
============================================================

Available Models:
1. facebook/bart-large-cnn - BART-based CNN/DailyMail summarizer
2. google/flan-t5-small - Google's FLAN-T5 small model
3. t5-small - Small T5 general-purpose model

Enter URL to summarize (or 'quit' to exit): https://example.com/article

Select model (1-3) [1]: 1

Select summary length:
1. Short (50-100 words)
2. Medium (100-150 words)
3. Long (150-200 words)
Enter choice (1-3) [2]: 2

🔄 Fetching content...
✅ Content fetched (5234 characters)
🔄 Loading model...
✅ Model loaded
🔄 Generating summary...
✅ Summary generated!

============================================================
📋 SUMMARY
============================================================
[Generated summary text here...]
============================================================

Summarize another URL? (y/n): y
```

#### Features

- **Multiple Models**: Choose from 3 different Apache-licensed models
- **Flexible Input**: Works with any URL
- **Summary Length Control**: Short, medium, or long summaries
- **Session Management**: Test multiple URLs in one session
- **Error Handling**: Graceful handling of network errors and invalid URLs

#### Command-Line Options

```bash
# Use specific model
python interactive_summarizer.py --model facebook/bart-large-cnn

# Set default summary length
python interactive_summarizer.py --length medium
```

### 2. Benchmark Script (`benchmark_summarizers.py`)

The benchmark script runs comprehensive performance tests on all configured models and articles.

#### Basic Usage

```bash
python benchmark_summarizers.py
```

#### What It Does

1. **Loads Configuration**: Reads models and articles from `config.yaml`
2. **Fetches Content**: Downloads and extracts text from configured URLs
3. **Tests Models**: Runs each model on each article
4. **Calculates Metrics**: Computes ROUGE scores and inference times
5. **Generates Reports**: Creates JSON and text summary files

#### Output Files

The benchmark generates several output files:

**1. JSON Results File** (`benchmark_results_YYYYMMDD_HHMMSS.json`)
```json
{
  "facebook/bart-large-cnn": {
    "model_info": {...},
    "articles": {
      "Article Title": {
        "summary": "...",
        "inference_time": 7.82,
        "rouge_scores": {
          "rouge1": 0.199,
          "rouge2": 0.191,
          "rougeL": 0.199
        }
      }
    },
    "summary_stats": {
      "avg_rouge1": 0.087,
      "avg_rouge2": 0.081,
      "avg_rougeL": 0.086
    }
  }
}
```

**2. Text Summary File** (`benchmark_summary_YYYYMMDD_HHMMSS.txt`)
```
BENCHMARK RESULTS SUMMARY
==========================

facebook/bart-large-cnn
  Successful summaries: 5/5
  Average ROUGE-1: 0.087
  Average ROUGE-2: 0.081
  Average ROUGE-L: 0.086

t5-small
  Successful summaries: 5/5
  Average ROUGE-1: 0.076
  Average ROUGE-2: 0.072
  Average ROUGE-L: 0.074
```

#### Customizing the Benchmark

Edit `config.yaml` to customize:

```yaml
# Add/remove models
models:
  - name: "facebook/bart-large-cnn"
    description: "BART-based CNN/DailyMail summarizer"
    # ... model configuration

# Add/remove test articles
articles:
  - url: "https://your-blog.com/article"
    title: "Your Article Title"

# Adjust summarization parameters
benchmark:
  max_length: 150      # Maximum summary length
  min_length: 50       # Minimum summary length
  max_input_length: 512 # Maximum input text length
```

#### Programmatic Usage

```python
from benchmark_summarizers import SummarizerBenchmark

# Initialize benchmark
benchmark = SummarizerBenchmark()

# Run full benchmark
benchmark.run_benchmark()

# Or use individual methods
summarizer = benchmark.load_model("facebook/bart-large-cnn")
content = benchmark.fetch_article_content("https://example.com/article")
summary = benchmark.summarize_text(summarizer, content)
rouge_scores = benchmark.calculate_rouge_scores(content, summary)
```

### 3. Demo Script (`demo_summarizer.py`)

Quick command-line tool for testing a single URL.

```bash
# Summarize a URL with default model
python demo_summarizer.py "https://example.com/article"

# Output:
# ============================================================
# 🤖 Apache-Licensed Summarizer Demo
# ============================================================
# URL: https://example.com/article
# Model: facebook/bart-large-cnn
# ...
# 📋 SUMMARY
# ============================================================
# [Generated summary]
```

### 4. Reproduce Results (`reproduce_performance_results.py`)

Reproduces the exact performance results shown in the blog post.

```bash
python reproduce_performance_results.py
```

This script:
- Uses the same articles as the original benchmark
- Tests the same models
- Generates results in the same format as the blog post
- Useful for verifying results or testing on different hardware

## 🧪 Testing

### Comprehensive Test Suite

The repository includes a complete test suite with 31 automated tests covering all functionality.

#### Run All Tests

```bash
python tests.py
```

**Test Coverage:**
- ✅ Basic import and pipeline creation
- ✅ All model loading (3 models)
- ✅ Summarization functionality
- ✅ URL content fetching
- ✅ Error handling mechanisms
- ✅ ROUGE score calculation
- ✅ Interactive tools execution
- ✅ Configuration validation
- ✅ Dependency validation
- ✅ Performance benchmarking

**Expected Output:**
```
📊 Total Tests: 31
✅ Passed: 31
❌ Failed: 0
📈 Success Rate: 100.0%

🎯 Test Suite Complete!
🎉 All tests passed! The system is working correctly.
```

#### Quick Tests

Test individual components:

```bash
# Test basic functionality
python quick_test.py basic

# Test all models
python quick_test.py models

# Test URL fetching
python quick_test.py urls

# Test demo script
python quick_test.py demo

# Run all quick tests
python quick_test.py all
```

### Test Results

All tests are verified to pass:
- **31/31 tests passing** (100% success rate)
- **All models working** correctly
- **All dependencies** validated
- **Error handling** verified

## 📊 Output Files

### Benchmark Output Files

When you run `benchmark_summarizers.py`, it generates:

1. **`benchmark_results_YYYYMMDD_HHMMSS.json`**
   - Complete benchmark data in JSON format
   - Includes all summaries, ROUGE scores, inference times
   - Machine-readable format for further analysis

2. **`benchmark_summary_YYYYMMDD_HHMMSS.txt`**
   - Human-readable summary of results
   - Average ROUGE scores per model
   - Success rates and statistics

### Example Output Files

The `examples/` directory contains:

1. **`sample_summaries.txt`**
   - Real summaries generated by each model
   - Human-readable format
   - Shows actual output quality

2. **`sample_summaries.json`**
   - Same summaries in JSON format
   - Includes metadata (length, model, URL)
   - Machine-readable

3. **`benchmark_results.json`**
   - Complete benchmark results from actual testing
   - Used to verify blog post numbers
   - Reference for expected performance

4. **`performance_comparison.md`**
   - Detailed performance analysis
   - Article-by-article breakdown
   - Methodology and insights

## 📈 Performance Results

### Actual Benchmark Results

Based on testing with 5 blog posts from Elena's website:

| Model | Success Rate | Avg ROUGE-1 | Avg ROUGE-2 | Avg ROUGE-L | Avg Inference Time |
|-------|-------------|-------------|-------------|-------------|-------------------|
| **facebook/bart-large-cnn** | 5/5 (100%) | 0.087 | 0.081 | 0.086 | 10.6s |
| **google/flan-t5-small** | 3/5 (60%) | 0.082 | 0.077 | 0.080 | 2.5s |
| **t5-small** | 5/5 (100%) | 0.076 | 0.072 | 0.074 | 3.1s |

### Key Insights

- **BART-Large-CNN**: Highest quality (ROUGE-1: 0.087) but slowest (10.6s)
- **FLAN-T5-Small**: Best balance (ROUGE-1: 0.082, 2.5s) with instruction-following
- **T5-Small**: Fastest (3.1s) with good quality (ROUGE-1: 0.076)

### Sample Summaries

**BART-Large-CNN on "AI Honesty, Agents, and the Fight for Truth":**
> "California told AI to be honest. Microsoft turned our computers into companions. European publishers stood up for truth itself. None of these stories is flashy on its own, but together they sketch the outline of how we'll live with AI — and how AI will live with us."

**FLAN-T5-Small on "AI Honesty, Agents, and the Fight for Truth":**
> "Some weeks, the news feels quiet. Other weeks, it hums quietly — as if something subtle but irreversible has shifted. This was one of those weeks. California told AI to be honest. Microsoft turned our computers into companions. And European publishers stood up for truth itself."

**T5-Small on "Should you use rebase?":**
> "rebasing isn't 'safer,' it's 'cleaner.' it rewrites history to make it look like you built your feature on top of the latest master. rebase: replays your feature commits as if they were created on top. you see the actual timeline of development, but it doesn't happen that way."

## 🔧 Configuration

### Configuring Models

Edit `config.yaml` to add or modify models:

```yaml
models:
  - name: "facebook/bart-large-cnn"
    description: "BART-based CNN/DailyMail summarizer"
    base_model: "BART"
    license: "Apache-2.0"
    best_for: "News & blog-style articles"
    notes: "Official Facebook BART model trained on CNN/DailyMail dataset"
```

### Configuring Test Articles

Add your own articles for testing:

```yaml
articles:
  - url: "https://your-blog.com/article"
    title: "Your Article Title"
```

### Adjusting Benchmark Parameters

```yaml
benchmark:
  max_length: 150        # Maximum summary length (tokens)
  min_length: 50         # Minimum summary length (tokens)
  do_sample: false       # Deterministic generation
  temperature: 1.0       # Sampling temperature
  top_p: 1.0            # Nucleus sampling parameter
  max_input_length: 512  # Maximum input text length (characters)
```

## 🛠️ Troubleshooting

### Common Issues

#### Model Loading Errors

**Problem**: Models fail to load or CUDA errors occur

**Solution**:
```python
import torch
summarizer = pipeline(
    "summarization", 
    model="facebook/bart-large-cnn",
    device=0 if torch.cuda.is_available() else -1
)
```

#### Memory Issues

**Problem**: Out of memory errors when loading models

**Solution**: Use smaller models
```python
# For limited memory, use smaller models
models_for_low_memory = ["t5-small", "google/flan-t5-small"]
```

#### Text Length Issues

**Problem**: "Token indices sequence length is longer than maximum"

**Solution**: Truncate text before summarization
```python
def truncate_text(text, max_length=512):
    return text[:max_length] if len(text) > max_length else text

long_text = "Your very long text here..."
truncated = truncate_text(long_text)
summary = summarizer(truncated, max_length=100, min_length=30)
```

#### Dependency Issues

**Problem**: Import errors or missing packages

**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For Apple Silicon Macs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Network Issues

**Problem**: Failed to fetch content from URLs

**Solution**:
- Check internet connection
- Verify URL is accessible
- Some sites may block automated requests
- Try using a different article URL

### Getting Help

If you encounter issues:

1. **Check the test suite**: `python tests.py`
2. **Review error messages**: Check the full traceback
3. **Verify installation**: Ensure all dependencies are installed
4. **Check configuration**: Verify `config.yaml` is valid
5. **Open an issue**: Include:
   - Python version
   - Operating system
   - Full error message
   - Steps to reproduce

## 📁 Repository Structure

```
apache-summarizers/
├── benchmark_summarizers.py      # Main benchmark script
├── interactive_summarizer.py     # Interactive tool for URLs
├── demo_summarizer.py           # Quick demo script
├── tests.py                     # Comprehensive test suite (31 tests)
├── quick_test.py               # Quick test runner
├── reproduce_performance_results.py  # Reproduce blog post results
├── setup.py                    # Automated setup script
├── config.yaml                 # Model and article configuration
├── requirements.txt            # Python dependencies
├── LICENSE                     # Apache 2.0 license
├── README.md                   # This file
└── examples/                   # Example outputs and samples
    ├── sample_summaries.txt    # Human-readable sample summaries
    ├── sample_summaries.json   # Machine-readable sample summaries
    ├── benchmark_results.json  # Complete benchmark results
    └── performance_comparison.md # Detailed performance analysis
```

## 🤖 Available Models

| Model | Base | License | Best For | Performance |
|-------|------|---------|----------|-------------|
| **facebook/bart-large-cnn** | BART | Apache-2.0 | News & blog articles | ROUGE-1: 0.087, 10.6s |
| **google/flan-t5-small** | T5 | Apache-2.0 | Instruction-following | ROUGE-1: 0.082, 2.5s |
| **t5-small** | T5 | Apache-2.0 | General text | ROUGE-1: 0.076, 3.1s |

All models are Apache 2.0 licensed, making them safe for commercial use.

## 🔗 Additional Resources

- **Blog Post**: [7 Apache-Licensed Summarisation Models Worth Trying](https://your-blog-url.com/apache-licensed-summarizers)
- **Hugging Face Models**: [Apache 2.0 Summarization Models](https://huggingface.co/models?license=apache-2.0&pipeline_tag=summarization)
- **Transformers Documentation**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- **ROUGE Metric**: [Understanding ROUGE Scores](https://en.wikipedia.org/wiki/ROUGE_(metric))

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python tests.py`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/apache-summarizers.git
cd apache-summarizers

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt

# Run tests
python tests.py
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The models used in this repository are also Apache 2.0 licensed, making this toolkit safe for:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ✅ Patent use

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library and model hosting
- **Elena** for providing test articles from her blog
- **Apache Software Foundation** for the permissive Apache 2.0 license
- **Cursor** (with Claude 3.5 Sonnet) for AI-assisted development and code generation

## 📞 Support

If you need help:

1. **Check the documentation**: Read this README thoroughly
2. **Run tests**: `python tests.py` to verify your setup
3. **Check issues**: Look for similar issues on GitHub
4. **Open an issue**: Provide:
   - Python version (`python --version`)
   - Operating system
   - Full error traceback
   - Steps to reproduce

---

**Happy Summarizing! 🎉**

For questions, suggestions, or bug reports, please open an issue on GitHub.
# Performance Comparison Results

This document contains the actual performance results from testing Apache-licensed summarization models on Elena's blog posts.

## Test Configuration

- **Test Articles**: 5 blog posts from Elena's website
- **Models Tested**: facebook/bart-large-cnn, t5-small
- **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L scores, inference time
- **Hardware**: CPU-only testing
- **Date**: October 24, 2025

## Results Summary

| Model | Success Rate | Avg ROUGE-1 | Avg ROUGE-2 | Avg ROUGE-L | Avg Inference Time |
|-------|-------------|-------------|-------------|-------------|-------------------|
| **facebook/bart-large-cnn** | 5/5 (100%) | 0.087 | 0.081 | 0.086 | 10.6s |
| **t5-small** | 5/5 (100%) | 0.076 | 0.072 | 0.074 | 3.1s |

## Detailed Results

### facebook/bart-large-cnn

**Model Info:**
- Description: BART-based CNN/DailyMail summarizer
- Base Model: BART
- License: Apache-2.0
- Best For: News & blog-style articles

**Article Results:**

1. **LoRA fine-tuning wins**
   - Original Length: 29,602 characters
   - Summary Length: 322 characters
   - Inference Time: 14.17s
   - ROUGE-1: 0.023, ROUGE-2: 0.022, ROUGE-L: 0.023

2. **Should you use rebase?**
   - Original Length: 9,518 characters
   - Summary Length: 352 characters
   - Inference Time: 12.28s
   - ROUGE-1: 0.070, ROUGE-2: 0.062, ROUGE-L: 0.068

3. **AI Honesty, Agents, and the Fight for Truth**
   - Original Length: 2,591 characters
   - Summary Length: 266 characters
   - Inference Time: 7.82s
   - ROUGE-1: 0.199, ROUGE-2: 0.191, ROUGE-L: 0.199

4. **Safety, Agents, and Compute**
   - Original Length: 7,335 characters
   - Summary Length: 273 characters
   - Inference Time: 8.60s
   - ROUGE-1: 0.073, ROUGE-2: 0.068, ROUGE-L: 0.073

5. **Cursor Made Me Do It**
   - Original Length: 7,827 characters
   - Summary Length: 295 characters
   - Inference Time: 9.94s
   - ROUGE-1: 0.070, ROUGE-2: 0.063, ROUGE-L: 0.065

### t5-small

**Model Info:**
- Description: Small T5 general-purpose model
- Base Model: T5
- License: Apache-2.0
- Best For: General text summarization

**Article Results:**

1. **LoRA fine-tuning wins**
   - Original Length: 29,602 characters
   - Summary Length: 200 characters
   - Inference Time: 3.65s
   - ROUGE-1: 0.014, ROUGE-2: 0.014, ROUGE-L: 0.014

2. **Should you use rebase?**
   - Original Length: 9,518 characters
   - Summary Length: 281 characters
   - Inference Time: 4.03s
   - ROUGE-1: 0.059, ROUGE-2: 0.053, ROUGE-L: 0.053

3. **AI Honesty, Agents, and the Fight for Truth**
   - Original Length: 2,591 characters
   - Summary Length: 257 characters
   - Inference Time: 2.35s
   - ROUGE-1: 0.168, ROUGE-2: 0.165, ROUGE-L: 0.168

4. **Safety, Agents, and Compute**
   - Original Length: 7,335 characters
   - Summary Length: 222 characters
   - Inference Time: 2.68s
   - ROUGE-1: 0.059, ROUGE-2: 0.055, ROUGE-L: 0.059

5. **Cursor Made Me Do It**
   - Original Length: 7,827 characters
   - Summary Length: 328 characters
   - Inference Time: 2.83s
   - ROUGE-1: 0.077, ROUGE-2: 0.072, ROUGE-L: 0.077

## Key Insights

1. **BART-Large-CNN** produces higher quality summaries (ROUGE-1: 0.087) but is significantly slower (10.6s average)

2. **T5-Small** is much faster (3.1s average) but produces more concise summaries (ROUGE-1: 0.076)

3. **Speed Difference**: T5-Small is 3.4x faster than BART-Large-CNN

4. **Quality Difference**: BART-Large-CNN achieves 14% higher ROUGE-1 scores

5. **Success Rate**: Both models achieved 100% success rate on all test articles

## Sample Summaries

### BART-Large-CNN on "AI Honesty, Agents, and the Fight for Truth"
> "California told AI to be honest. Microsoft turned our computers into companions. European publishers stood up for truth itself. None of these stories is flashy on its own, but together they sketch the outline of how we'll live with AI — and how AI will live with us."

### T5-Small on "AI Honesty, Agents, and the Fight for Truth"
> "1 California wants AI to tell the truth California passed a new law that says chatbots and AI companions must disclose when they're AI. it also introduces mental-health safeguards, requiring reporting and response mechanisms when users express distress."

## Methodology

- **ROUGE Scores**: Calculated using rouge-score library with stemmer enabled
- **Inference Time**: Measured from model loading to summary generation
- **Text Processing**: Content truncated to 512 characters for model input limits
- **Hardware**: CPU-only testing on macOS system

## Reproducibility

To reproduce these results:

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark
python benchmark_summarizers.py

# Or reproduce specific results
python reproduce_performance_results.py
```

The complete benchmark results are available in `benchmark_results.json`.

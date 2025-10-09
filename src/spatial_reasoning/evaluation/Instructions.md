# Evaluation Instructions

This guide explains how to run the spatial reasoning benchmark evaluation and visualize the results.

## Prerequisites

### API Keys Setup

Before running the evaluation, you need to set up your API keys. Create a `.env` file in the project root with the following:

```bash
# .env
OPENAI_API_KEY=your-openai-api-key-here
GEMINI_API_KEY=your-google-gemini-api-key-here
XAI_API_KEY=your-xai-api-key-here
```

Get your API keys from:
- **OpenAI**: https://platform.openai.com/api-keys
- **Gemini**: https://makersuite.google.com/app/apikey
- **XAI**: https://console.x.ai/

## Running the Evaluation

The evaluation process consists of two main steps:

### Step 1: Run the Benchmark

Execute the benchmark script to evaluate different agents on your dataset:

```bash
python -m spatial_reasoning.evaluation.benchmark \
  --agents gemini openai_vanilla_reasoning xai_vanilla_reasoning openai_advanced_reasoning xai_advanced_reasoning \
  --data data/benchmark_data.json \
  --save-location /tmp/benchmark \
  --num-workers 200
```

### Impact of normalization:

```bash
python -m spatial_reasoning.evaluation.benchmark_normalized \
  --agents openai_vanilla_reasoning xai_vanilla_reasoning \
  --data data/benchmark_data.json \
  --save-location ./output \
  --num-workers 64
```


**Parameters:**
- `--agents`: List of agents to evaluate (space-separated). Add as many as you'd like to evaluate.
  - `gemini`: Google's Gemini model
  - `openai_vanilla_reasoning`: OpenAI o4-mini baseline
  - `xai_vanilla_reasoning`: Grok4 fast-reasoning baseline
  - `openai_advanced_reasoning`: OpenAI o4-mini with tool-use
  - `xai_advanced_reasoning`: xAI fast-reasoning with tool-use
- `--data`: Path to your parsed dataset JSON file
- `--save-location`: Directory where benchmark results will be saved
- `--num-workers`: Number of parallel workers for processing (adjust based on your system)

### Step 2: Visualize Results

After the benchmark completes, generate visualization plots:

```bash
python evaluation/analysis.py /tmp/benchmark --output-dir /tmp/benchmark/plots
```

**Parameters:**
- First argument: Path to the benchmark results directory (same as `--save-location` from Step 1)
- `--output-dir`: Directory where visualization plots will be saved

## Output

After running both steps, you'll find:
- **Benchmark results**: Raw evaluation data in `/tmp/benchmark`
- **Visualization plots**: Performance charts and comparisons in `/tmp/benchmark/plots`

## Tips

- Adjust `--num-workers` based on your system capabilities and API rate limits (general rule of thumb is 5-10x your number of CPU cores. On an H100, I tried with 200 workers and completed running the benchmark in 5 images, that's 252-1008 requests in 300 seconds). Note: xAI has strong rate limiters of 400 RPM, so if the system appears to glitch, it's most likely a rate limited. Can be alleviated by reducing the number of workers.
- Ensure you have sufficient API credits for all the agents you're testing
- The benchmark may take some time depending on dataset size and number of agents

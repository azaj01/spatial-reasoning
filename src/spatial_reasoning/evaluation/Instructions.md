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
  --data /home/qasim/data/reasoning/Advex/SR/parsed_dataset.json \
  --save-location /tmp/sr-benchmark \
  --num-workers 300
```

**Parameters:**
- `--agents`: List of agents to evaluate (space-separated)
  - `gemini`: Google's Gemini model
  - `openai_vanilla_reasoning`: OpenAI vanilla reasoning model
  - `xai_vanilla_reasoning`: XAI vanilla reasoning model
  - `openai_advanced_reasoning`: OpenAI advanced reasoning model
  - `xai_advanced_reasoning`: XAI advanced reasoning model
- `--data`: Path to your parsed dataset JSON file
- `--save-location`: Directory where benchmark results will be saved
- `--num-workers`: Number of parallel workers for processing (adjust based on your system)

### Step 2: Visualize Results

After the benchmark completes, generate visualization plots:

```bash
python evaluation/analysis.py /tmp/sr-benchmark --output-dir /tmp/sr-benchmark/plots
```

**Parameters:**
- First argument: Path to the benchmark results directory (same as `--save-location` from Step 1)
- `--output-dir`: Directory where visualization plots will be saved

## Output

After running both steps, you'll find:
- **Benchmark results**: Raw evaluation data in `/tmp/sr-benchmark`
- **Visualization plots**: Performance charts and comparisons in `/tmp/sr-benchmark/plots`

## Tips

- Adjust `--num-workers` based on your system capabilities and API rate limits (general rule of thumb is 5-10x your number of CPU cores. On an H100, I tried with 200 workers and completed running the benchmark in 5 minutes, that's 252-1008 requests in 300 seconds).
- Ensure you have sufficient API credits for all the agents you're testing
- The benchmark may take some time depending on dataset size and number of agents

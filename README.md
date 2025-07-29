# Vision Evals

A powerful Python package for object detection using advanced vision and reasoning models, including OpenAI's models and Google's Gemini.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Running Locally](#running-locally)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Requirements](#requirements)
- [Advanced Usage](#advanced-usage)
- [License](#license)

## Features

- **Multiple Detection Models**: Support for various state-of-the-art models
  - Advanced Reasoning Model (OpenAI)
  - Vanilla Reasoning Model  
  - Vision Model with SAM2 integration
  - Gemini Model (Google)
  - Multi Advanced Reasoning Model

- **Easy-to-use API**: Simple `detect()` function for quick integration
- **Flexible Output**: Returns bounding boxes, visualized images, and overlay images
- **CLI Support**: Command-line interface for quick testing
- **Customizable**: Support for custom parameters and thresholds

## Installation

### From PyPI (once published)
```bash
pip install vision-evals
```

### From Source (for development)
```bash
git clone https://github.com/yourusername/vision-evals.git
cd vision_evals
pip install -e .
```

## Setup

### Environment Variables

This package requires API keys for the AI models. You have two options:

#### Option 1: Using a .env file (Recommended)

Create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=your-openai-api-key-here
GEMINI_API_KEY=your-google-gemini-api-key-here
```

#### Option 2: Export Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export GEMINI_API_KEY="your-google-gemini-api-key-here"
```

**Note**: 
- Get your OpenAI API key from: https://platform.openai.com/api-keys
- Get your Gemini API key from: https://makersuite.google.com/app/apikey

## Quick Start

### Python API

```python
from vision_evals import detect

# Basic detection
result = detect(
    image_path="https://ix-cdn.b2e5.com/images/27094/27094_3063d356a3a54cc3859537fd23c5ba9d_1539205710.jpeg",  # or image path
    object_of_interest="farthest scooter in the image",
    task_type="advanced_reasoning_model"  # options: ["gemini", "vanilla_reasoning_model", "vision_model"]
)

# Access results
bounding_boxes = result['bboxs']
visualized_image = result['visualized_image']
print(f"Found {len(bounding_boxes)} objects")
```

### Command Line Interface

```bash
vision-evals --image-path "image.jpg" --object-of-interest "person" --task-type "advanced_reasoning_model"
```

## Running Locally

### Using the CLI Tool

If you're running from the source code (not installed via pip):

```bash
# Navigate to the src directory
cd vision_evals/src

# Run the CLI tool
python run_cli.py --image-path "path/to/image.jpg" --object-of-interest "car"
```

#### CLI Examples:

1. **Basic detection with local image:**
```bash
python run_cli.py \
  --image-path "/path/to/local/image.jpg" \
  --object-of-interest "person wearing hat" \
  --task-type "advanced_reasoning_model"
```

2. **Detection with URL image:**
```bash
python run_cli.py \
  --image-path "https://example.com/image.jpg" \
  --object-of-interest "red car" \
  --task-type "gemini"
```

3. **With custom parameters:**
```bash
python run_cli.py \
  --image-path "image.jpg" \
  --object-of-interest "text in image" \
  --task-type "advanced_reasoning_model" \
  --task-kwargs '{"nms_threshold": 0.7, "multiple_predictions": true}' \
  --output-folder-path "./my_results"
```

### Using the Python API

If running from source:

```python
import sys
sys.path.append('/path/to/vision_evals/src')

from api import detect

# Now use the detect function
result = detect(
    image_path="path/to/image.jpg",
    object_of_interest="bicycle",
    task_type="advanced_reasoning_model"
)

print(f"Found {len(result['bboxs'])} objects")
print(f"Bounding boxes: {result['bboxs']}")

# Save the visualized image
result['visualized_image'].save("output_with_boxes.jpg")
```

## API Reference

### `detect()` Function

Main detection function that processes an image and returns bounding boxes for objects of interest.

#### Parameters:

- `image_path` (str): Path to the image file or URL
- `object_of_interest` (str): Description of what to detect in the image
- `task_type` (str): Type of detection task. Options:
  - `"advanced_reasoning_model"` (default)
  - `"vanilla_reasoning_model"`
  - `"vision_model"`
  - `"gemini"`
  - `"multi_advanced_reasoning_model"`
- `task_kwargs` (dict, optional): Additional parameters for the task
  - Example: `{"nms_threshold": 0.7, "multiple_predictions": True}`
- `save_outputs` (bool): Whether to save output files to disk (default: False)
- `output_folder_path` (str, optional): Where to save outputs if save_outputs=True
- `return_overlay_images` (bool): Whether to return overlay images in the result (default: True)

#### Returns:

Dictionary containing:
- `bboxs`: List of bounding boxes `[[x1, y1, x2, y2], ...]`
- `visualized_image`: PIL Image with bounding boxes drawn
- `original_image`: Original PIL Image
- `overlay_images`: List of overlay images (if any)
- `total_time`: Processing time in seconds
- `object_of_interest`: The object that was searched for
- `task_type`: The task type that was used
- `task_kwargs`: The task parameters that were used

## Examples

### Advanced Detection with Custom Parameters

```python
from vision_evals import detect

result = detect(
    image_path="https://example.com/image.jpg",
    object_of_interest="time to first token chart",
    task_type="advanced_reasoning_model",
    task_kwargs={
        "nms_threshold": 0.7,
        "multiple_predictions": True
    },
    save_outputs=True,
    output_folder_path="./my_outputs"
)

# Display the visualized result
result['visualized_image'].show()
```

### Using Different Models

```python
# Using Gemini model
result = detect(
    image_path="image.jpg",
    object_of_interest="cat",
    task_type="gemini"
)

# Using Vision model with SAM2
result = detect(
    image_path="image.jpg", 
    object_of_interest="building",
    task_type="vision_model"
)
```

### CLI with Advanced Options

```bash
vision-evals \
  --image-path "https://example.com/image.jpg" \
  --object-of-interest "holes in shoes" \
  --task-type "advanced_reasoning_model" \
  --task-kwargs '{"nms_threshold": 0.7, "multiple_predictions": true}' \
  --output-folder-path "./results"
```

## Requirements

### System Requirements
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (optional but recommended for vision_model task)

### API Keys Required
- **OpenAI API key**: Required for `advanced_reasoning_model`, `vanilla_reasoning_model`, and `vision_model` tasks
- **Gemini API key**: Required for `gemini` task

### Python Dependencies
All dependencies are automatically installed with the package. Key dependencies include:
- `numpy>=1.21.0`
- `torch>=2.0.0`
- `Pillow>=9.0.0`
- `opencv-python>=4.5.0`
- `openai>=1.0.0`
- `google-generativeai>=0.8.0`
- `transformers>=4.30.0`
- `python-dotenv>=0.19.0`

For a complete list, see `requirements.txt`.

## Advanced Usage

### Custom Agent Creation

```python
from vision_evals import AgentFactory

# Create a custom agent
agent = AgentFactory.create_agent(
    model="o4-mini",
    platform_name="openai"
)
```

### Working with Cell Data

```python
from vision_evals import Cell

# Cell objects represent detected regions
cell = Cell(row=0, col=0, bbox=(10, 20, 100, 200))
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'vision_evals'**
   - Make sure you've installed the package: `pip install -e .` (from the root directory)
   - If running from source, add the src directory to your Python path

2. **API Key Errors**
   ```
   ValueError: OpenAI API key is required
   ```
   - Ensure your `.env` file is in the correct location
   - Check that the environment variables are set correctly
   - Verify your API keys are valid

3. **CUDA/GPU Issues**
   - The package will automatically fall back to CPU if CUDA is not available
   - For best performance with `vision_model`, ensure you have CUDA installed

4. **Output Directory Issues**
   - The package will create output directories automatically
   - Ensure you have write permissions in the specified output path

### Debug Mode

To see more detailed error messages, set the environment variable:
```bash
export VISION_EVALS_DEBUG=1
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and feature requests, please use the [GitHub issue tracker](https://github.com/yourusername/vision-evals/issues).
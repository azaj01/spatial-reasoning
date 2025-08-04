"""
Spatial Reasoning: A PyPI package for object detection using advanced vision models.

This package provides a unified API for detecting objects in images using various
state-of-the-art vision and reasoning models including OpenAI's models and Google's Gemini.
"""

import os
import sys
import warnings

# Try to detect if flash_attn is available and compatible
_flash_attn_available = False
_flash_attn_error = None

try:
    import flash_attn
    import flash_attn_2_cuda

    _flash_attn_available = True
except ImportError as e:
    _flash_attn_error = str(e)
    # Only mock if flash_attn is installed but incompatible
    if "flash_attn" in sys.modules or "undefined symbol" in str(e):
        warnings.warn(
            f"Flash Attention is installed but incompatible with current environment: {e}. "
            "Disabling flash attention support.",
            UserWarning,
        )

        # Set environment variables to disable flash attention in transformers
        os.environ["FLASH_ATTENTION_SKIP_IMPORT"] = "1"
        os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"

        # Mock the modules to prevent import errors
        class MockFlashAttn:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        sys.modules["flash_attn"] = MockFlashAttn()
        sys.modules["flash_attn.flash_attn_interface"] = MockFlashAttn()
        sys.modules["flash_attn_2_cuda"] = MockFlashAttn()


__version__ = "0.1.8"
__author__ = "Qasim Wani"
__email__ = "qasim31wani@gmail.com"

# Import key classes for advanced usage
from .agents import AgentFactory, BaseAgent, GeminiAgent, OpenAIAgent
# Import main API function for easy access
from .api import detect, detect_stream
from .data import BaseDataset, Cell
from .tasks import (AdvancedReasoningModelTask, BaseTask, GeminiTask,
                    MultiAdvancedReasoningModelTask,
                    StreamAdvancedReasoningModelTask,
                    VanillaReasoningModelTask, VisionModelTask)

__all__ = [
    "detect",
    "detect_stream",
    "AgentFactory",
    "BaseAgent",
    "GeminiAgent",
    "OpenAIAgent",
    "BaseDataset",
    "Cell",
    "AdvancedReasoningModelTask",
    "BaseTask",
    "GeminiTask",
    "MultiAdvancedReasoningModelTask",
    "VanillaReasoningModelTask",
    "StreamAdvancedReasoningModelTask",
    "VisionModelTask",
]

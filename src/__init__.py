"""
Vision Evals: A PyPI package for object detection using advanced vision models.

This package provides a unified API for detecting objects in images using various
state-of-the-art vision and reasoning models including OpenAI's models and Google's Gemini.
"""

__version__ = "0.1.0"
__author__ = "Qasim Wani"
__email__ = "qasim31wani@gmail.com"

# Import key classes for advanced usage
from agents import AgentFactory, BaseAgent, GeminiAgent, OpenAIAgent
# Import main API function for easy access
from api import detect
from data import BaseDataset, Cell
from tasks import (AdvancedReasoningModelTask, BaseTask, GeminiTask,
                    MultiAdvancedReasoningModelTask, VanillaReasoningModelTask,
                    VisionModelTask)

__all__ = [
    "detect",
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
    "VisionModelTask",
]
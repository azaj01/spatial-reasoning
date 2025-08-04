import os

from .base_agent import BaseAgent
from .gemini_agent import GeminiAgent
from .openai_agent import OpenAIAgent


class AgentFactory:
    """Factory for creating agent instances based on model name."""

    @staticmethod
    def get_api_key(platform_name: str):
        if "openai" in platform_name.lower():
            return os.getenv("OPENAI_API_KEY")
        if "gemini" in platform_name.lower():
            return os.getenv("GEMINI_API_KEY")

    @staticmethod
    def create_agent(model: str, platform_name: str) -> BaseAgent:
        """Create an appropriate agent based on the model name."""
        api_key: str = AgentFactory.get_api_key(platform_name)
        if OpenAIAgent.is_supported_model(model):
            return OpenAIAgent(model, api_key)
        elif GeminiAgent.is_supported_model(model):
            return GeminiAgent(model, api_key)
        else:
            raise ValueError(f"Model '{model}' is not supported by any available agent")

    @staticmethod
    def get_supported_models() -> dict:
        """Get a dictionary of supported models by agent type."""
        return {
            "openai": ["gpt-4o", "o4-mini-high", "o3"],
            "gemini": ["gemini-2.5-flash"],
        }

from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePrompt(ABC):
    """Abstract base class for all prompt templates."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def get_system_prompt(self, **kwargs) -> str:
        """Get the system prompt template."""
        pass

    @abstractmethod
    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt template."""
        pass

    def format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with provided parameters."""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for prompt formatting: {e}")

    @abstractmethod
    def get_required_parameters(self) -> Dict[str, str]:
        """Get a dictionary of required parameters and their descriptions."""
        pass

    def get_prompt_info(self) -> Dict[str, Any]:
        """Get information about this prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "required_parameters": self.get_required_parameters(),
        }

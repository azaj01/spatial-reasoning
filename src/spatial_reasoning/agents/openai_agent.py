import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base_agent import BaseAgent


class OpenAIAgent(BaseAgent):
    """OpenAI GPT agent implementation."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model, api_key)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

    @property
    def client(self):
        """Get or initialize the OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send messages to OpenAI and get response."""
        params = {"model": self.model, "input": self._format_messages(messages)}

        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        reasoning = kwargs.pop("reasoning", None)
        if reasoning:
            # Sample: {"effort": "low", "summary": "auto"}
            params["reasoning"] = reasoning

        params.update(kwargs)
        response = self.client.responses.create(**params)

        try:
            o_text = response.output[1].content[0].text
        except:
            o_text = response.output[0].content[0].text

        if reasoning:
            r_text = response.output[0].summary
            return {"reasoning": r_text, "output": o_text}
        else:
            return {"output": o_text}

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API (no transformation needed)."""
        return messages

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        """Check if the model is supported by OpenAI."""
        openai_models = ["gpt-4o", "o3", "o4-mini"]
        return any(model_name in model.lower() for model_name in openai_models)

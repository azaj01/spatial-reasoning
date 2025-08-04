import os
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from ..utils.image_utils import base64_to_image

from .base_agent import BaseAgent


class GeminiAgent(BaseAgent):
    """Google Gemini agent implementation."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model, api_key)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required")

    @property
    def client(self):
        """Get or initialize the Gemini client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send messages to Gemini and get response."""
        reasoning = kwargs.pop("reasoning", None)

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )

        response = self.client.models.generate_content(
            model=self.model, contents=self._format_messages(messages), config=config
        )

        if reasoning:
            return {"reasoning": "", "output": response.text}
        return {"output": response.text}

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Any]:
        """Format messages for Gemini API."""
        contents = []

        for msg in messages:
            content = msg.get("content")

            if isinstance(content, str):
                contents.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part["type"] == "input_text":
                        contents.append(part["text"])
                    elif part["type"] == "input_image":
                        # Handle base64 image
                        if part["image_url"].startswith("data:image"):
                            base64_data = part["image_url"].split(",")[1]
                            contents.append(base64_to_image(base64_data))
                        else:
                            raise NotImplementedError("URL images need preprocessing")

        return contents

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        """Check if the model is supported by Gemini."""
        gemini_models = ["gemini-2.5-flash"]
        return any(m in model.lower() for m in gemini_models)

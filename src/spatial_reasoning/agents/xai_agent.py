import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base_agent import BaseAgent


class XAIAgent(BaseAgent):
    """XAI Grok agent implementation."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model, api_key)
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI API key is required")

    @property
    def client(self):
        """Get or initialize the XAI client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1"
            )
        return self._client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send messages to XAI and get response."""
        params = {
            "model": self.model,
            "messages": self._format_messages(messages)
        }
        
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        reasoning_requested = kwargs.pop("reasoning", None)
        params.update(kwargs)
        
        # Make API call
        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content
        
        # Handle reasoning if requested
        if reasoning_requested:
            reasoning_info = self._extract_reasoning(response)
            return {"reasoning": reasoning_info, "output": content}
        
        return {"output": content}
    
    def _extract_reasoning(self, response) -> List:
        """Extract reasoning information from response."""
        class ReasoningChunk:
            def __init__(self, text):
                self.text = text
        
        # Check if XAI used reasoning tokens internally
        # NOTE: Grok doesn't expose reasoning tokens to prevent companies from training on them
        # so we can't really know what's happening behind the hood.
        # As such, we modified the detection_prompt a bit to output <thinking> tags for a more insight
        # into what the model is doing when it outputs the final response.
        if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens_details'):
            details = response.usage.completion_tokens_details
            if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens > 0:
                note = f"[Grok-4 used {details.reasoning_tokens} reasoning tokens internally]"
                return [ReasoningChunk(note)]
        
        return []

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for XAI API (OpenAI-compatible format)."""
        formatted = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if isinstance(content, str):
                formatted.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Handle multimodal content
                parts = []
                for part in content:
                    if part["type"] == "input_text":
                        parts.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "input_image":
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": part["image_url"]}
                        })
                formatted.append({"role": role, "content": parts})
        
        return formatted

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        """Check if the model is supported by XAI."""
        xai_models = ["grok-4-0709", "grok-4-fast-reasoning"]
        return any(model_name in model.lower() for model_name in xai_models)

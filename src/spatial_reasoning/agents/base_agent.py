import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from ..utils.image_utils import image_to_base64


class BaseAgent(ABC):
    """Abstract base class for all vision model agents."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._client = None

    @property
    @abstractmethod
    def client(self):
        """Get or initialize the client for the specific service."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Send messages to the model and get response."""
        pass

    @abstractmethod
    def _format_messages(self, messages: List[Dict[str, Any]]) -> Any:
        """Format messages for the specific API format."""
        pass

    def safe_chat(
        self, messages: List[Dict[str, Any]], max_attempts: int = 2, **kwargs
    ) -> Dict[str, Any]:
        """Call chat with exponential-backoff retries."""

        for attempt in range(1, max_attempts + 1):
            try:
                return self.chat(messages, **kwargs)
            except Exception as e:
                if attempt == max_attempts:
                    raise
                wait = (2**attempt) + random.random()
                print(f"[retry {attempt}/{max_attempts}] {e}. retrying in {wait:.1f}s")
                if "Too many tokens" in str(e):
                    print("Rate limit exceeded. Sleeping for 1 minute.")
                    time.sleep(60)
                    continue
                time.sleep(wait)

    @staticmethod
    def create_text_message(role: str, content: str) -> Dict[str, Any]:
        """Create a text-only message."""
        return {"role": role, "content": content}

    @staticmethod
    def create_multimodal_message(
        role: str,
        text: str,
        images: List[Union[Image.Image, str]],
    ) -> Dict[str, Any]:
        """Create a multimodal message with text and image."""
        content = [{"type": "input_text", "text": text}]

        for image in images:
            if isinstance(image, Image.Image):
                image_url = f"data:image/png;base64,{image_to_base64(image)}"
            elif not image.startswith("http"):
                raise ValueError("Image must be a valid URL")
            else:
                image_url = image
            content.append({"type": "input_image", "image_url": image_url})

        return {"role": role, "content": content}

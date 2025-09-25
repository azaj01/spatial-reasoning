import json
import os
import time
from threading import Lock
from typing import Dict, Generator, List, Optional, Union

from dotenv import load_dotenv
from PIL import Image

from .agents.agent_factory import AgentFactory
from .data import BaseDataset, Cell
from .tasks import (AdvancedReasoningModelTask, GeminiTask,
                    StreamAdvancedReasoningModelTask,
                    VanillaReasoningModelTask, VisionModelTask)
from .utils.io_utils import (convert_list_of_cells_to_list_of_bboxes,
                             download_image, get_timestamp)

load_dotenv()


class DetectionAPI:
    """API that initializes agents lazily and avoids GPU memory churn."""

    def __init__(self):
        self._agents, self._tasks = {}, {}
        self._initialized = False
        self._init_lock = Lock()

    def _initialize_if_needed(self):
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            print("Initializing agents and tasks...")

            self._agents["openai"] = self._safe_agent("o4-mini", "openai")
            if self._agents["openai"]:
                self._tasks.update({
                    "openai_advanced_reasoning": AdvancedReasoningModelTask(self._agents["openai"]),
                    "openai_stream_reasoning": StreamAdvancedReasoningModelTask(self._agents["openai"]),
                    "openai_vanilla_reasoning": VanillaReasoningModelTask(self._agents["openai"], prompt_type="vanilla"),
                    "vision_model": VisionModelTask(self._agents["openai"]),
                })

            self._agents["gemini"] = self._safe_agent("gemini-2.5-flash", "gemini")
            if self._agents["gemini"]:
                self._tasks["gemini"] = GeminiTask(self._agents["gemini"])

            self._agents["xai"] = self._safe_agent("grok-4-fast-reasoning", "xai")
            if self._agents["xai"]:
                self._tasks.update({
                    "xai_advanced_reasoning": AdvancedReasoningModelTask(self._agents["xai"]),
                    "xai_vanilla_reasoning": VanillaReasoningModelTask(self._agents["xai"], prompt_type="vanilla"),
                })

            self._initialized = True
            print("Initialization complete!")

    def _safe_agent(self, model: str, platform: str):
        try:
            return AgentFactory.create_agent(model=model, platform_name=platform)
        except Exception as e:
            print(f"Error initializing {platform}: {e}")
            return None

    def _load_image(self, path: str) -> Image.Image:
        return download_image(path) if path.startswith("http") else Image.open(path).convert("RGB")

    def detect(
        self,
        image_path: str,
        object_of_interest: str,
        task_type: str,
        task_kwargs: Optional[Dict] = None,
        save_outputs: bool = False,
        output_folder_path: Optional[str] = None,
        return_overlay_images: bool = True,
    ) -> Dict[str, Union[List, float, Image.Image]]:
        """Non-streaming detection entrypoint (keeps same args as your CLI)."""
        self._initialize_if_needed()
        if task_type not in self._tasks:
            raise ValueError(f"Unsupported task type: {task_type}")

        task_kwargs = task_kwargs or {}
        image = self._load_image(image_path)
        start = time.perf_counter()

        output = self._tasks[task_type].execute(image=image, prompt=object_of_interest, **task_kwargs)
        if output.get("bboxs") and isinstance(output["bboxs"][0], Cell):
            output["bboxs"] = convert_list_of_cells_to_list_of_bboxes(output["bboxs"])

        total_time = time.perf_counter() - start
        vis = BaseDataset.visualize_image(image, output["bboxs"], return_image=True)

        result = {
            "bboxs": output["bboxs"],
            "visualized_image": vis,
            "original_image": image,
            "overlay_images": output.get("overlay_images", []),
            "crop_metadata": output.get("crop_metadata", {}),
            "total_time": total_time,
            "object_of_interest": object_of_interest,
            "task_type": task_type,
            "task_kwargs": task_kwargs,
        }

        if save_outputs:
            if output_folder_path is None:
                output_folder_path = f"./output/{get_timestamp()}"
            save_outputs_to_disk(output_folder_path, result)

        if not return_overlay_images:
            result.pop("overlay_images", None)

        return result

    def detect_stream(
        self,
        image_path: str,
        object_of_interest: str,
        task_type: str = "openai_stream_reasoning",
        task_kwargs: Optional[Dict] = None,
    ) -> Generator[Dict, None, None]:
        """Streaming detection (keeps same args)."""
        self._initialize_if_needed()
        if task_type not in self._tasks:
            yield {"type": "error", "error": f"Streaming not supported: {task_type}"}
            return

        task_kwargs = task_kwargs or {}
        image = self._load_image(image_path)
        start = time.perf_counter()

        for result in self._tasks[task_type].execute_streaming(image=image, prompt=object_of_interest, **task_kwargs):
            result["elapsed_time"] = time.perf_counter() - start
            if result["type"] == "final" and "bboxs" in result:
                if isinstance(result["bboxs"][0], Cell):
                    result["bboxs"] = convert_list_of_cells_to_list_of_bboxes(result["bboxs"])
                result.update({
                    "object_of_interest": object_of_interest,
                    "task_type": task_type,
                    "task_kwargs": task_kwargs,
                    "total_time": time.perf_counter() - start,
                })
            yield result


_api = DetectionAPI()


def detect(*args, **kwargs):
    return _api.detect(*args, **kwargs)


def detect_stream(*args, **kwargs):
    return _api.detect_stream(*args, **kwargs)


def save_outputs_to_disk(folder: str, result: Dict) -> None:
    os.makedirs(folder, exist_ok=True)

    def _save(name: str, img: Image.Image):
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        img.save(os.path.join(folder, name))

    _save("original.jpg", result["original_image"])
    _save("visualized.jpg", result["visualized_image"])

    with open(os.path.join(folder, "output.json"), "w") as f:
        json.dump({
            "object_of_interest": result["object_of_interest"],
            "task_type": result["task_type"],
            "task_kwargs": result["task_kwargs"],
            "bboxs": result["bboxs"],
            "crop_metadata": result.get("crop_metadata", {}),
            "total_time": result["total_time"],
        }, f, indent=2)

    for i, overlay in enumerate(result.get("overlay_images", [])):
        if overlay:
            _save(f"overlay_{i}.jpg", overlay)

    print(f"Outputs saved to: {folder}")



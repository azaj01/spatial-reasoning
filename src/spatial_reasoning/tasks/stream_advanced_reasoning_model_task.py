import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from itertools import product
from typing import Dict, Generator, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..agents import BaseAgent
from ..data import BaseDataset, Cell
from ..prompts import GridCellDetectionPrompt
from ..utils.io_utils import get_original_bounding_box, parse_detection_output
from .base_task import BaseTask
from .vanilla_reasoning_model_task import VanillaReasoningModelTask
from .vision_model_task import VisionModelTask


class StreamAdvancedReasoningModelTask(BaseTask):
    """
    Agent that utilizes CV tools and FMs with streaming support
    """

    def __init__(self, agent: BaseAgent, **kwargs):
        super().__init__(agent, **kwargs)
        self.prompt: GridCellDetectionPrompt = GridCellDetectionPrompt()
        self.vanilla_agent: VanillaReasoningModelTask = VanillaReasoningModelTask(
            agent, prompt_type="advanced", **kwargs
        )
        self.vision_agent: VisionModelTask = VisionModelTask(agent, **kwargs)

    def run_agents_parallel(self, **kwargs) -> Tuple[dict, dict]:
        """
        Run both vision and vanilla agents in parallel and return both outputs.

        Returns:
            tuple: (vision_output, vanilla_output)
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            future_to_agent = {
                executor.submit(self.vision_agent.execute, **kwargs): "vision",
                executor.submit(self.vanilla_agent.execute, **kwargs): "vanilla",
            }

            results = {}
            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_type = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent_type] = result
                except Exception as e:
                    print(f"Agent {agent_type} generated an exception: {e}")
                    results[agent_type] = {"error": str(e)}

        return results.get("vision", {}), results.get("vanilla", {})

    def execute(self, **kwargs) -> dict:
        """
        Run reasoning model (non-streaming version for backward compatibility)
        """
        # Collect all results from the generator
        results = list(self.execute_streaming(**kwargs))
        # Return the final result
        return results[-1] if results else {}

    def execute_streaming(self, **kwargs) -> Generator[dict, None, None]:
        """
        Run reasoning model with streaming support
        Yields intermediate results as they become available
        """
        original_image: Image.Image = kwargs["image"]
        object_of_interest: str = kwargs["prompt"]
        grid_size = kwargs.get("grid_size", (4, 3))
        max_crops = kwargs.get("max_crops", 2)
        top_k = kwargs.get("top_k", -1)
        confidence_threshold = kwargs.get("confidence_threshold", 0.5)
        convergence_threshold = kwargs.get("convergence_threshold", 0.5)

        origin_coordinates = (0, 0)
        overlay_samples = []
        is_terminal_state = False
        crop_iteration = 0
        # Yield the very first image
        yield {
            "type": "intermediate",
            "iteration": 0,
            "overlay_image": self._image_to_base64(
                BaseTask.overlay_grid_on_image(
                    original_image, grid_size[0], grid_size[1]
                )[0]
            ),
            "is_terminal": is_terminal_state,
            "total_crops": 0,
            "max_crops": max_crops,
            "message": "Initializing grid. May take up to 2 minutes...",
        }

        image = original_image.copy()
        while not is_terminal_state and len(overlay_samples) < max_crops:
            crop_iteration += 1

            if image.width < 512 and image.height < 512 and len(overlay_samples) > 0:
                _grid_size = (3, 2)
            else:
                _grid_size = grid_size

            # Run single crop process
            overlay_image, image, origin_coordinates, is_terminal_state = (
                self.run_single_crop_process(
                    image.copy(),
                    object_of_interest,
                    origin_coordinates,
                    _grid_size,
                    top_k,
                    confidence_threshold,
                    convergence_threshold,
                )
            )

            overlay_samples.append(overlay_image)

            # Yield intermediate result
            message = f"Processing crop {crop_iteration}/{max_crops}..."
            if crop_iteration == max_crops or is_terminal_state:
                message = "Starting final model detection (ETA ~20 seconds)"
            yield {
                "type": "intermediate",
                "iteration": crop_iteration,
                "overlay_image": self._image_to_base64(
                    BaseTask.overlay_grid_on_image(image, _grid_size[0], _grid_size[1])[
                        0
                    ]
                ),
                "is_terminal": is_terminal_state,
                "total_crops": len(overlay_samples),
                "max_crops": max_crops,
                "message": message,
            }

        # Final detection on the cropped image
        kwargs["image"] = image
        kwargs["grid_size"] = (
            (2, 2) if (image.width < 512 and image.height < 512) else _grid_size
        )
        kwargs["confidence_threshold"] = 0.8
        vision_out, vanilla_out = self.run_agents_parallel(**kwargs)

        out = vanilla_out if vanilla_out["bboxs"] else vision_out

        # Now, if both vision and vanilla agents return no detections, we need to yeild no detections
        if "bboxs" not in out or len(out["bboxs"]) == 0:
            yield {
                "type": "final",
                "bboxs": [],
                "overlay_images": [None] * len(overlay_samples),
                "final_image": self._image_to_base64(image),
                "total_iterations": crop_iteration + 1,
                "message": "No detections found",
            }
            return

        # Display final cropped model prediction
        cropped_visualized_image = BaseDataset.visualize_image(
            image, [cell.to_tuple() for cell in out["bboxs"]], return_image=True
        )

        # Yield the cropped visualization
        yield {
            "type": "intermediate",
            "iteration": crop_iteration + 1,
            "overlay_image": self._image_to_base64(cropped_visualized_image),
            "is_terminal": True,
            "total_crops": len(overlay_samples) + 1,
            "max_crops": max_crops,
            "message": "Final model detection complete!",
        }

        # Restore to original coordinates
        restored_bboxs = get_original_bounding_box(
            cropped_bounding_boxs=out["bboxs"],
            crop_origin=origin_coordinates,
        )

        # Create final visualization
        visualized_image = BaseDataset.visualize_image(
            original_image,
            [cell.to_tuple() for cell in restored_bboxs],
            return_image=True,
        )

        # Yield final result
        final_result = {
            "type": "final",
            "bboxs": restored_bboxs,
            "overlay_images": [self._image_to_base64(img) for img in overlay_samples],
            "final_image": self._image_to_base64(visualized_image),
            "total_iterations": crop_iteration,
            "message": "Detection complete!",
        }

        yield final_result

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    @staticmethod
    def is_terminal_state(
        source_image: Image.Image,
        target_image: Image.Image,
        convergence_threshold: float,
    ) -> bool:
        """
        If the target image is similar in size to the source image, return True
        """
        src_width, src_height = source_image.size
        target_width, target_height = target_image.size
        area_ratio = (target_width * target_height) / (src_width * src_height)
        print(
            f"Area ratio: {area_ratio}, Target image size: {target_image.size}, Source image size: {source_image.size}"
        )
        return area_ratio >= convergence_threshold

    def run_single_crop_process(
        self,
        image: Image.Image,
        object_of_interest: str,
        origin_coordinates: tuple,
        grid_size: tuple,
        top_k: int,
        confidence_threshold: float,
        convergence_threshold: float,
    ) -> dict:
        """
        Run crop process
        """
        overlay_image, cell_lookup = BaseTask.overlay_grid_on_image(
            image, grid_size[0], grid_size[1]
        )

        messages = [
            self.agent.create_text_message(
                "system",
                self.prompt.get_system_prompt(
                    resolution=image.size,
                    object_of_interest=object_of_interest,
                    grid_size=grid_size,
                    cell_lookup=cell_lookup,
                ),
            ),
            self.agent.create_multimodal_message(
                "user",
                self.prompt.get_user_prompt(
                    resolution=image.size,
                    object_of_interest=object_of_interest,
                    grid_size=grid_size,
                    cell_lookup=cell_lookup,
                ),
                [overlay_image],
            ),
        ]

        raw_response = self.agent.safe_chat(
            messages, reasoning={"effort": "medium", "summary": "detailed"}
        )
        structured_response = parse_detection_output(raw_response["output"])

        # DEBUGGING PURPOSES ONLY TO SEE WHAT THE REASONING MODEL IS SAYING
        print("----------------LOGGING REASONING ----------------")
        if "reasoning" in raw_response:
            for reasoning in raw_response["reasoning"]:
                print(reasoning.text)
        print("-------------Final Output ----------------")
        print(raw_response["output"])
        print("----------------LOGGING REASONING ----------------")

        try:
            cropped_image_data: dict = BaseTask.crop_image(
                image,
                structured_response,
                cell_lookup,
                top_k=top_k,
                confidence_threshold=confidence_threshold,
            )
            assert cropped_image_data is not None, "Cropped image data is None"
        except Exception as e:
            print(f"Error cropping image: {e}")
            return overlay_image, image, origin_coordinates, True

        crop_origin = (
            origin_coordinates[0] + cropped_image_data["crop_origin"][0],
            origin_coordinates[1] + cropped_image_data["crop_origin"][1],
        )

        return (
            overlay_image,
            cropped_image_data["cropped_image"],
            crop_origin,
            self.is_terminal_state(
                image, cropped_image_data["cropped_image"], convergence_threshold
            ),
        )

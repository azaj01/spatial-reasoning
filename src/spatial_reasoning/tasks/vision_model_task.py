import warnings

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from ..agents.base_agent import BaseAgent
from ..data import Cell
from ..utils.image_utils import nms
from .base_task import BaseTask


class VisionModelTask(BaseTask):
    def __init__(self, agent: BaseAgent, **kwargs):
        super().__init__(agent, **kwargs)
        self.cpu_mode = kwargs.get("cpu_mode", False)

        # Suppress the specific warning about meta parameters
        warnings.filterwarnings("ignore", message="copying from a non-meta parameter")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        )

        # Load model with proper device mapping
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base", torch_dtype=torch.float32
        )

        # Move model to appropriate device
        if torch.cuda.is_available() and not self.cpu_mode:
            self.model = self.model.cuda()
        else:
            print("Using CPU mode")
            self.model = self.model.cpu()

    def execute(self, **kwargs) -> dict:
        """
        Run GroundingDino + SAM
        Arguments:
            image: Image.Image
            prompt: str
            nms_threshold: float
        """

        nms_threshold = kwargs.get("nms_threshold", 0.5)
        multiple_predictions = kwargs.get("multiple_predictions", False)

        bbox_detections = self.detect_grounding_dino(
            kwargs["image"], kwargs["prompt"], nms_threshold, multiple_predictions
        )

        if bbox_detections is None:
            return {"bboxs": [], "overlay_images": []}

        overlay_images = [None] * len(bbox_detections)

        return {"bboxs": bbox_detections, "overlay_images": overlay_images}

    def detect_grounding_dino(
        self,
        image: Image.Image,
        prompt: str,
        nms_threshold: float,
        multiple_predictions: bool,
    ) -> Cell:
        # Step 1: Use Grounding DINO to get bounding box from text
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        # Move inputs to the same device as the model
        if torch.cuda.is_available() and not self.cpu_mode:
            inputs = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
        else:
            inputs = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }  # For sanity reasons, explicitly move to CPU

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process outputs
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=0.15,
            text_threshold=0.25,
            target_sizes=[(image.height, image.width)],
        )[0]

        if len(results["boxes"]) == 0:
            print(f"No objects found for prompt: {prompt}.")
            return []

        # Get the best box (highest score)
        detections = []

        # ADD NMS thresholding to minimize false positive
        filtered_results = nms(
            results["boxes"].cpu().numpy(),
            results["scores"].cpu().numpy(),
            nms_threshold,
        )
        if multiple_predictions:
            for idx, box in enumerate(filtered_results["boxes"]):
                detections.append(
                    Cell(
                        id=idx + 1,
                        left=int(box[0]),
                        top=int(box[1]),
                        right=int(box[2]),
                        bottom=int(box[3]),
                    )
                )
        else:
            best_box = filtered_results["boxes"][np.argmax(filtered_results["scores"])]
            detections.append(
                Cell(
                    id=1,
                    left=int(best_box[0]),
                    top=int(best_box[1]),
                    right=int(best_box[2]),
                    bottom=int(best_box[3]),
                )
            )

        return detections

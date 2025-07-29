import os
import warnings

# Try to import with flash attention, but disable if it fails
try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
except ImportError as e:
    if "flash_attn" in str(e):
        warnings.warn("Flash Attention not available or incompatible. Disabling for compatibility.")
        os.environ["FLASH_ATTENTION_SKIP_IMPORT"] = "1"
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    else:
        raise

import cv2
import numpy as np
import torch
from ..agents.base_agent import BaseAgent
from ..data import Cell
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ..utils.image_utils import nms

from .base_task import BaseTask


class VisionModelTask(BaseTask):
    def __init__(self, agent: BaseAgent, **kwargs):
        super().__init__(agent, **kwargs)
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
        self.sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    
    def execute(self, **kwargs) -> dict:
        """
        Run GroundingDino + SAM
        Arguments:
            image: Image.Image
            prompt: str
            nms_threshold: float
        """
        
        return_sam_masks = kwargs.get("return_sam_masks", False)
        nms_threshold = kwargs.get("nms_threshold", 0.7)
        multiple_predictions = kwargs.get("multiple_predictions", False)

        bbox_detections = self.detect_grounding_dino(
            kwargs["image"], 
            kwargs["prompt"], 
            nms_threshold,
            multiple_predictions
        )
        
        if bbox_detections is None:
            return {'bboxs': [], 'overlay_images': []}
        
        # Given that we don't support Segmentation as a task, there's no need
        # to return the overlay images. Added as optional logic for future use.

        overlay_images = []
        
        if return_sam_masks:
            for bbox in bbox_detections:
                _, overlay_image = self.detect_sam(
                    kwargs["image"], bbox
                )
                overlay_images.append(overlay_image)
        else:
            overlay_images = [None] * len(bbox_detections)

        return {
            "bboxs": bbox_detections,
            "overlay_images": overlay_images
        }

    def detect_grounding_dino(
        self,
        image: Image.Image,
        prompt: str,
        nms_threshold: float,
        multiple_predictions: bool
    ) -> Cell:
        # Step 1: Use Grounding DINO to get bounding box from text
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.15,
            text_threshold=0.25,
            target_sizes=[(image.height, image.width)]
        )[0]
        
        if len(results["boxes"]) == 0:
            print(f"No objects found for prompt: {prompt}. Trying again with lower threshold.")
            return []
            with torch.no_grad():
                outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.2,
                text_threshold=0.25,
                target_sizes=[(image.height, image.width)]
            )[0]
            if len(results["boxes"]) == 0:
                return []

        # Get the best box (highest score)
        detections = []

        # ADD NMS thresholding to minimize false positive
        filtered_results = nms(results['boxes'].cpu().numpy(), results['scores'].cpu().numpy(), nms_threshold)
        if multiple_predictions:
            for idx, box in enumerate(filtered_results['boxes']):
                detections.append(Cell(
                    id=idx + 1,
                    left=int(box[0]),
                    top=int(box[1]),
                    right=int(box[2]),
                    bottom=int(box[3])
                ))
        else:
            best_box = filtered_results['boxes'][np.argmax(filtered_results['scores'])]
            detections.append(Cell(
                id=1,
                left=int(best_box[0]),
                top=int(best_box[1]),
                right=int(best_box[2]),
                bottom=int(best_box[3])
            ))

        return detections

    def detect_sam(
        self, image: Image.Image, bbox: Cell
    ) -> tuple[Cell, Image.Image]:
        """
        Run SAM to get a mask for the object of interest
        """
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            self.sam2_predictor.set_image(image)
            _bbox = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
            masks, _, _ = self.sam2_predictor.predict(box=_bbox, multimask_output=False)
            mask: np.ndarray = masks[0]

            # Create a semi-transparent mask
            overlay_image = image.copy()
            mask_color = Image.new('RGBA', image.size, (255, 0, 0, 128))
            mask_alpha = Image.fromarray((mask * 128).astype(np.uint8))
            mask_color.putalpha(mask_alpha)

            # Composite the mask onto the image
            overlay_image = Image.alpha_composite(
                overlay_image.convert('RGBA'),
                mask_color
            ).convert('RGB')

            # I don't think there's a point in returning the bounding box
            # since GroundingDINO already returns the bounding box and
            # is equivalent to the bounding box returned by SAM. However,
            # for the sake of consistency with the other tasks, we're
            # returning the bounding box.
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            return Cell(
                id=bbox.id, left=x, top=y, right=x+w, bottom=y+h
            ), overlay_image

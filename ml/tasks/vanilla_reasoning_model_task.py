import random

from agents import BaseAgent
from data import Cell
from PIL import Image
from prompts import SimpleDetectionPrompt
from utils.image_utils import nms
from utils.io_utils import parse_detection_output

from .base_task import BaseTask


class VanillaReasoningModelTask(BaseTask):
    def __init__(self, agent: BaseAgent, **kwargs):
        super().__init__(agent, **kwargs)
        self.prompt: SimpleDetectionPrompt = SimpleDetectionPrompt()
    
    def execute(self, **kwargs) -> dict:
        """
        Run reasoning model
        Arguments:
            image: Image.Image
            prompt: str
        """
        image: Image.Image = kwargs['image']
        object_of_interest: str = kwargs['prompt']
        confidence_threshold: float = kwargs.get("confidence_threshold", 0.65)  # Treated as the NMS threshold
        multiple_predictions: bool = kwargs.get('multiple_predictions', False)

        messages = [
            self.agent.create_text_message("system", self.prompt.get_system_prompt()),
            self.agent.create_multimodal_message("user", self.prompt.get_user_prompt(resolution=image.size, object_of_interest=object_of_interest), [image])
        ]
        
        raw_response = self.agent.safe_chat(messages)
        structured_response = parse_detection_output(raw_response['output'])
        if not structured_response or "bbox" not in structured_response:
            return {
                "bboxs": [],
                "overlay_images": []
            }
        
        bboxs: list[Cell] = []
        confidence_scores: list[float] = []
        for i, bbox in enumerate(structured_response['bbox']):
            x, y, w, h = bbox
            confidence = structured_response['confidence'][i]
            cell = Cell(id=i, left=x, top=y, right=x+w, bottom=y+h)
            confidence_scores.append(confidence)
            bboxs.append(cell)
        
        # Filter out all bboxs that have confidence less than the threshold
        filtered_results = nms(bboxs, confidence_scores, confidence_threshold)
        bboxs = filtered_results['boxes']
        
        if multiple_predictions:
            return {
                "bboxs": bboxs,
                "overlay_images": [None] * len(bboxs)
            }
        else:
            random_index = random.randint(0, len(bboxs) - 1)
            return {
                "bboxs": [bboxs[random_index]],
                "overlay_images": [None]
            }

import random

from PIL import Image

from ..agents import BaseAgent
from ..data import Cell
from ..prompts import SimpleDetectionPrompt
from ..utils.io_utils import parse_detection_output
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
        
        print(f"Vanilla reasoning model task. Confidence threshold: {confidence_threshold}")

        messages = [
            self.agent.create_text_message("system", self.prompt.get_system_prompt()),
            self.agent.create_multimodal_message("user", self.prompt.get_user_prompt(resolution=image.size, object_of_interest=object_of_interest), [image])
        ]
        
        raw_response = self.agent.safe_chat(messages, reasoning={"effort": "medium", "summary": "auto"})
        structured_response = parse_detection_output(raw_response['output'])

        # DEBUGGING PURPOSES ONLY TO SEE WHAT THE REASONING MODEL IS SAYING
        if "reasoning" in raw_response:
            for reasoning in raw_response["reasoning"]:
                print(reasoning.text)
        print("--------------------------------")
        print(raw_response["output"])
        
        if not structured_response or "bbox" not in structured_response or len(structured_response['bbox']) == 0:
            return {
                "bboxs": [],
                "overlay_images": [None]
            }
        
        bboxs: list[Cell] = []
        confidence_scores: list[float] = []
        for i, bbox in enumerate(structured_response['bbox']):
            x, y, w, h = bbox
            confidence = structured_response['confidence'][i]
            cell = Cell(id=i, left=x, top=y, right=x+w, bottom=y+h)
            confidence_scores.append(confidence)
            bboxs.append(cell)
        
        # Filter out all bboxs that have confidence less than the threshold. utilize confidence from bboxs and confidence_scores
        filtered_bboxs = [(box, confidence) for box, confidence in zip(bboxs, confidence_scores) if confidence >= confidence_threshold]
        filtered_bboxs.sort(key=lambda x: x[1], reverse=True)
        filtered_bboxs = [box for box, _ in filtered_bboxs]

        if multiple_predictions:
            return {
                "bboxs": filtered_bboxs,
                "overlay_images": [None] * len(filtered_bboxs)
            }
        else:
            return {
                "bboxs": [filtered_bboxs[0]],
                "overlay_images": [None]
            }

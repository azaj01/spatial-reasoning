from typing import List

from agents import BaseAgent
from PIL import Image
from prompts import GeminiPrompt
from utils.io_utils import parse_detection_output

from .base_task import BaseTask


class GeminiTask(BaseTask):
    def __init__(self, agent: BaseAgent, **kwargs):
        super().__init__(agent, **kwargs)
        self.prompt: GeminiPrompt = GeminiPrompt()
    
    def execute(self, **kwargs) -> dict:
        """
        Run reasoning model
        Arguments:
            image: Image.Image
            prompt: str
        """
        image: Image.Image = kwargs['image']
        object_of_interest: str = kwargs['prompt']
        normalization_factor: float = kwargs.get('normalization_factor', 1000)

        messages = [
            self.agent.create_text_message("user", self.prompt.get_system_prompt(normalization_factor=normalization_factor)),
            self.agent.create_multimodal_message("system", self.prompt.get_user_prompt(object_of_interest=object_of_interest, normalization_factor=normalization_factor), [image])
        ]
        
        raw_response = self.agent.safe_chat(messages)
        try:
            structured_response = parse_detection_output(raw_response['output'])
            bounding_boxes = GeminiTask.extract_bounding_boxes(structured_response, image, normalization_factor)
        except Exception as e:
            print(f"Error parsing structured response: {e}")
            return {
                "bboxs": [],
                "overlay_images": []
            }
        
        return {
            "bboxs": bounding_boxes,
            "overlay_images": [None] * len(bounding_boxes)
        }
       
    @staticmethod
    def extract_bounding_boxes(
        responses: list,
        image: Image.Image,
        normalization_factor: float
    ) -> List[List[int]]:
        """Convert normalized bounding boxes to absolute coordinates."""
        width, height = image.size
        converted_bounding_boxes = []
        for response in responses:
            box = response["box_2d"]
            abs_y1 = int(box[0] / normalization_factor * height)
            abs_x1 = int(box[1] / normalization_factor * width)
            abs_y2 = int(box[2] / normalization_factor * height)
            abs_x2 = int(box[3] / normalization_factor * width)
            converted_bounding_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])
        
        return converted_bounding_boxes

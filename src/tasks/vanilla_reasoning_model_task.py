from agents import BaseAgent
from data import Cell
from PIL import Image
from prompts import SimpleDetectionPrompt
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

        messages = [
            self.agent.create_text_message("system", self.prompt.get_system_prompt()),
            self.agent.create_multimodal_message("user", self.prompt.get_user_prompt(resolution=image.size, object_of_interest=object_of_interest), [image])
        ]
        
        raw_response = self.agent.safe_chat(messages)
        structured_response = parse_detection_output(raw_response['output'])
        
        bboxs: list[Cell] = []
        for i, bbox in enumerate(structured_response['bbox']):
            x, y, w, h = bbox
            cell = Cell(id=i, left=x, top=y, right=x+w, bottom=y+h)
            bboxs.append(cell)
        
        return {
            "bboxs": bboxs,
            "overlay_images": [None] * len(bboxs)
        }

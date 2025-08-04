from typing import List

from PIL import Image

from ..agents import BaseAgent
from ..data import Cell
from ..prompts import GeminiPrompt
from ..utils.io_utils import parse_detection_output
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
            multiple_predictions: bool
        """
        image: Image.Image = kwargs["image"]
        object_of_interest: str = kwargs["prompt"]
        normalization_factor: float = kwargs.get("normalization_factor", 1000)
        multiple_predictions: bool = kwargs.get("multiple_predictions", False)

        messages = [
            self.agent.create_text_message(
                "user",
                self.prompt.get_system_prompt(
                    normalization_factor=normalization_factor
                ),
            ),
            self.agent.create_multimodal_message(
                "system",
                self.prompt.get_user_prompt(
                    object_of_interest=object_of_interest,
                    normalization_factor=normalization_factor,
                ),
                [image],
            ),
        ]

        raw_response = self.agent.safe_chat(messages)

        # DEBUGGING PURPOSES ONLY TO SEE WHAT THE REASONING MODEL IS SAYING
        print("----------------Gemini Task LOGGING REASONING ----------------")
        print(raw_response["output"])
        print("----------------Gemini Task LOGGING REASONING ----------------")

        try:
            structured_response = parse_detection_output(raw_response["output"])
            bounding_boxes = GeminiTask.extract_bounding_boxes(
                structured_response, image, normalization_factor
            )
        except Exception as e:
            print(
                f"Error parsing structured response: {e}. Returning empty bounding boxes. Raw response: {raw_response['output']}"
            )
            return {"bboxs": [], "overlay_images": []}
        if multiple_predictions:
            return {
                "bboxs": bounding_boxes,
                "overlay_images": [None] * len(bounding_boxes),
            }
        else:
            return {"bboxs": [bounding_boxes[0]], "overlay_images": [None]}

    @staticmethod
    def extract_bounding_boxes(
        responses: list, image: Image.Image, normalization_factor: float
    ) -> List[Cell]:
        """Convert normalized bounding boxes to absolute coordinates."""
        width, height = image.size
        converted_bounding_boxes = []
        for i, response in enumerate(responses):
            box = response["box_2d"]
            abs_y1 = int(box[0] / normalization_factor * height)
            abs_x1 = int(box[1] / normalization_factor * width)
            abs_y2 = int(box[2] / normalization_factor * height)
            abs_x2 = int(box[3] / normalization_factor * width)
            cell = Cell(id=i, left=abs_x1, top=abs_y1, right=abs_x2, bottom=abs_y2)
            converted_bounding_boxes.append(cell)

        return converted_bounding_boxes

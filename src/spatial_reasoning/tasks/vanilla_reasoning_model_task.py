import random

from PIL import Image

from ..agents import BaseAgent
from ..data import Cell
from ..prompts import BboxDetectionWithGridCellPrompt, SimpleDetectionPrompt
from ..utils.io_utils import parse_detection_output
from .base_task import BaseTask


class VanillaReasoningModelTask(BaseTask):
    def __init__(self, agent: BaseAgent, **kwargs):
        super().__init__(agent, **kwargs)
        vanilla_prompt: SimpleDetectionPrompt = SimpleDetectionPrompt()
        advanced_prompt: BboxDetectionWithGridCellPrompt = (
            BboxDetectionWithGridCellPrompt()
        )

        prompt_type = self.kwargs.get("prompt_type", "vanilla")
        self.prompt = None
        if prompt_type == "vanilla":
            self.prompt: SimpleDetectionPrompt = vanilla_prompt
        elif prompt_type == "advanced":
            self.prompt: BboxDetectionWithGridCellPrompt = advanced_prompt
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")

    def execute(self, **kwargs) -> dict:
        """
        Run reasoning model
        Arguments:
            image: Image.Image
            prompt: str
        """
        image: Image.Image = kwargs["image"]  # Passed to both prompts
        object_of_interest: str = kwargs["prompt"]  # Passed to both prompts
        confidence_threshold: int = int(kwargs.get("confidence_threshold", 0.65) * 100)
        multiple_predictions: bool = kwargs.get("multiple_predictions", False)
        resolution = kwargs.get("resolution", image.size)  # BBOX Prompt
        grid_size = kwargs.get("grid_size", (4, 3))  # BBOX Prompt only

        if isinstance(self.prompt, BboxDetectionWithGridCellPrompt):
            image, cell_lookup = BaseTask.overlay_grid_on_image(
                image, grid_size[0], grid_size[1]
            )
            prompt_kwargs = {
                "resolution": resolution,
                "object_of_interest": object_of_interest,
                "grid_size": grid_size,
                "cell_lookup": cell_lookup,
            }
        else:
            prompt_kwargs = {
                "object_of_interest": object_of_interest,
                "resolution": resolution,
            }

        messages = [
            self.agent.create_text_message(
                "system", self.prompt.get_system_prompt(**prompt_kwargs)
            ),
            self.agent.create_multimodal_message(
                "user", self.prompt.get_user_prompt(**prompt_kwargs), [image]
            ),
        ]

        raw_response = self.agent.safe_chat(
            messages, reasoning={"effort": "medium", "summary": "auto"}
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

        if (
            not structured_response
            or "bbox" not in structured_response
            or len(structured_response["bbox"]) == 0
        ):
            return {"bboxs": [], "overlay_images": [None]}

        def typecast_confidence(confidence: float) -> int:
            if 0.0 <= confidence <= 1.0:
                return int(confidence * 100)
            elif 1.0 < confidence <= 100.0:
                return int(confidence)
            return confidence

        bboxs: list[Cell] = []
        confidence_scores: list[float] = []
        for i, bbox in enumerate(structured_response["bbox"]):
            x, y, w, h = bbox
            confidence = structured_response["confidence"][i]

            cell = Cell(id=i, left=x, top=y, right=x + w, bottom=y + h)
            if (
                isinstance(self.prompt, BboxDetectionWithGridCellPrompt)
                and confidence == 100
            ):  # Observation: whenever confidence is a 100, model is highly unreliable. Applied only to BBOX Prompt (used by the advanced reasoning model)
                continue

            if typecast_confidence(confidence) >= confidence_threshold:
                confidence_scores.append(confidence)
                bboxs.append(cell)

        if multiple_predictions:
            return {"bboxs": bboxs, "overlay_images": [None] * len(bboxs)}
        else:
            return {"bboxs": [bboxs[0]] if bboxs else [], "overlay_images": [None]}

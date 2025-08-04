from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

from PIL import Image

from ..agents import BaseAgent
from ..data import BaseDataset
from ..prompts import GridCellDetectionPrompt
from ..utils.io_utils import get_original_bounding_box, parse_detection_output
from .base_task import BaseTask
from .vanilla_reasoning_model_task import VanillaReasoningModelTask


class AdvancedReasoningModelTask(BaseTask):
    """
    Agent that utilizes CV tools and FMs
    """

    def __init__(self, agent: BaseAgent, **kwargs):
        super().__init__(agent, **kwargs)
        self.prompt: GridCellDetectionPrompt = GridCellDetectionPrompt()
        self.vanilla_agent: VanillaReasoningModelTask = VanillaReasoningModelTask(
            agent, prompt_type="advanced", **kwargs
        )

    def execute(self, **kwargs) -> dict:
        """
        Run reasoning model
        Arguments:
            image: Image.Image
            prompt: str
        """
        image: Image.Image = kwargs["image"]
        object_of_interest: str = kwargs["prompt"]
        grid_size = kwargs.get("grid_size", (4, 3))  # num_rows x num_cols
        max_crops = kwargs.get("max_crops", 4)
        top_k = kwargs.get("top_k", -1)
        confidence_threshold = kwargs.get("confidence_threshold", 0.5)
        convergence_threshold = kwargs.get("convergence_threshold", 0.5)

        origin_coordinates = (0, 0)

        overlay_samples = []
        is_terminal_state = False
        while not is_terminal_state and len(overlay_samples) < max_crops:
            if (
                image.width < 512 and image.height < 512 and len(overlay_samples) > 0
            ):  # initial state, pick the default grid size
                _grid_size = (3, 2)
            else:
                _grid_size = grid_size
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

        kwargs["image"] = image
        kwargs["grid_size"] = (
            (2, 2) if (image.width < 512 and image.height < 512) else _grid_size
        )
        out = self.vanilla_agent.execute(**kwargs)

        # also upload the final image to the output
        cropped_visualized_image = BaseDataset.visualize_image(
            image, [cell.to_tuple() for cell in out["bboxs"]], return_image=True
        )
        overlay_samples.append(cropped_visualized_image)
        # Restore to original coordinates
        restored_bboxs = get_original_bounding_box(
            cropped_bounding_boxs=out["bboxs"],
            crop_origin=origin_coordinates,
        )
        out["bboxs"] = restored_bboxs
        out["overlay_images"] = overlay_samples
        return out

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
                    confidence_threshold=confidence_threshold,
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

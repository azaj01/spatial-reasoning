from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Dict, Tuple, Union

import numpy as np
import torch
from agents import BaseAgent
from data import Cell
from PIL import Image, ImageDraw, ImageFont
from prompts import GridCellTwoImagesDetectionPrompt
from utils.io_utils import parse_detection_output

from .base_task import BaseTask
from .vanilla_reasoning_model_task import VanillaReasoningModelTask
from .vision_model_task import VisionModelTask


class AdvancedReasoningModelTask(BaseTask):
    """
    Agent that utilizes CV tools and FMs
    """
    def __init__(self, agent: BaseAgent, **kwargs):
        super().__init__(agent, **kwargs)
        self.prompt: GridCellTwoImagesDetectionPrompt = GridCellTwoImagesDetectionPrompt()
        # Tool use -and- foundation model agents
        self.vanilla_agent: VanillaReasoningModelTask = VanillaReasoningModelTask(agent, **kwargs)
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
                executor.submit(self.vision_agent.execute, **kwargs): 'vision',
                executor.submit(self.vanilla_agent.execute, **kwargs): 'vanilla'
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
                    results[agent_type] = {'error': str(e)}
            
        return results.get('vision', {}), results.get('vanilla', {})
    
    def execute(self, **kwargs) -> dict:
        """
        Run reasoning model
        Arguments:
            image: Image.Image
            prompt: str
        """
        image: Image.Image = kwargs['image']
        object_of_interest: str = kwargs['prompt']
        
        grid_size = self.kwargs.get("grid_size", (3, 4))  # num_rows x num_cols
        max_crops = self.kwargs.get('max_crops', 2)  # TODO: make max crops an LMM decision.
        top_k = self.kwargs.get("top_k", -1)  # TODO: give user the flexibility if they want to detect one object or multiple
        use_expert: bool = self.kwargs.get("use_expert", True)
        
        origin_coordinates = (0, 0)
        
        overlay_samples = []
        for _ in range(max_crops):
            # TODO: convert this into a streaming application
            overlay_image, image, origin_coordinates = self.run_single_crop_process(image.copy(), object_of_interest, origin_coordinates, grid_size, top_k)
            overlay_samples.append(overlay_image)

        # Run both agents in parallel
        vision_out, vanilla_out = self.run_agents_parallel(**kwargs)
        
        # Compare outputs - you can customize this comparison logic
        print(f"Vision agent output: {vision_out}")
        print(f"Vanilla agent output: {vanilla_out}")
        
        # You can modify this to implement your own comparison/selection logic
        out = vision_out if len(vision_out['bboxs']) > 0 else vanilla_out

        out['overlay_images'] = overlay_samples
        return out

    def run_single_crop_process(self, image: Image.Image, object_of_interest: str, origin_coordinates: tuple, grid_size: tuple, top_k: int) -> dict:
        """
        Run crop process
        """
        overlay_image, cell_lookup = self.overlay_grid_on_image(
            image, grid_size[0], grid_size[1], font_size=40, width=2
        )  # TODO: make font size and width function of image size

        messages = [
            self.agent.create_text_message("system", self.prompt.get_system_prompt()),
            self.agent.create_multimodal_message(
                "user",
                self.prompt.get_user_prompt(
                    resolution=image.size,
                    object_of_interest=object_of_interest,
                    grid_size=grid_size
                ),
                [image, overlay_image]
            )
        ]
        raw_response = self.agent.safe_chat(messages)
        structured_response = parse_detection_output(raw_response['output'])

        cropped_image_data: dict = AdvancedReasoningModelTask.crop_image(
            image, structured_response, cell_lookup, top_k=top_k
        )
        if not cropped_image_data:
            print(
                "Unable to get object in the grid, most likely due to it not being found in the image."
            )
            return [
                None,
                None,
                None
            ] 

        # Update cumulative image coordinates
        # BUG: this is not correct. we need to be able to do proper scaling for the bounding box to work properly.
        # How to reproduce: The cropped image has a bounding box that spans the entire crop. This will yield in the original 
        # space to the whole image which doesn't make sense.
        crop_origin = (
            origin_coordinates[0] + cropped_image_data["crop_origin"][0],
            origin_coordinates[1] + cropped_image_data["crop_origin"][1]
        )

        # Update current image for next iteration
        cropped_image: Image.Image = cropped_image_data["cropped_image"]

        return overlay_image, cropped_image, crop_origin


    @staticmethod
    def overlay_grid_on_image(
        image: Union[Image.Image, torch.Tensor],
        num_rows: int,
        num_cols: int,
        color: str = "red",
        font_size: int = 20,
        width: int = 2,
    ) -> Tuple[Union[Image.Image, torch.Tensor], Dict[int, Cell]]:
        """
        Draw a rows x cols grid over `image`, label cells 1..rows*cols, and return:
        (image_with_grid, {cell_number: {left, top, right, bottom, cell_dims}})
        """
        if num_rows + num_cols <= 2:
            raise ValueError(f"Too few rows ({num_rows}) and columns ({num_cols}).")

        # --- to PIL ---
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            arr = (image.detach().cpu().permute(1, 2, 0)
                   .mul(255).clamp(0, 255).to(torch.uint8).numpy())
            pil = Image.fromarray(arr)
        else:
            pil = image.copy()

        original_image_width, original_image_height = pil.size
        cell_width, cell_height = original_image_width // num_cols, original_image_height // num_rows

        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            print("Unable to load font. Loading default")
            font = ImageFont.load_default(size=font_size)

        # --- generate grid lines ---
        for x in range(cell_width, original_image_width, cell_width):
            draw.line([(x, 0), (x, original_image_height)], fill=color, width=width)
        for y in range(cell_height, original_image_height, cell_height):
            draw.line([(0, y), (original_image_width, y)], fill=color, width=width)

        # --- create cell + label map ---
        table: Dict[int, Cell] = {}
        for n, (r, c) in enumerate(product(range(num_rows), range(num_cols)), 1):
            left, top = c * cell_width, r * cell_height
            right, bottom = min(left + cell_width, original_image_width), min(top + cell_height, original_image_height)
            cell = Cell(n, left, top, right, bottom)
            table[n] = cell
            draw.text(((cell.left + cell.right) // 2,
                       (cell.top + cell.bottom) // 2),
                      str(n), fill=color, font=font, anchor="mm")

        # --- back to original type ---
        if is_tensor:
            out = torch.from_numpy(np.array(pil)).permute(2, 0, 1)
            out = out.float().div(255) if image.dtype.is_floating_point else out
        else:
            out = pil

        return out, table
    
    @staticmethod
    def crop_image(
        pil_image: Image.Image,
        scores_grid: dict,
        cell_lookup: dict,
        pad: int = 0,
        top_k: int = 1
    ):
        """
        Crop image using top-k most confident cell groups from `scores_grid`.
        scores_grid = {
            "confidence": [(65, 65), (75, 62)],
            "cells": [(3, 6), (5, 6)]
        }
        Note: this function returns just one cropped image, not a list of cropped images. TODO: add support for multiple crops.
        """
        # Basic error checking
        if not scores_grid or not scores_grid.get("cells") or not scores_grid.get("confidence"):
            return None
        
        grouped = sorted(
            zip(scores_grid["cells"], scores_grid["confidence"]),
            key=lambda g: np.mean(g[1]),
            reverse=True
        )

        if top_k != -1:
            grouped = grouped[:top_k]

        bounds = []
        for cell_ids, _ in grouped:
            for cid in cell_ids:
                c = cell_lookup[cid]
                l, r = sorted([c.left, c.right])
                t, b = sorted([c.top, c.bottom])
                bounds.append((l, t, r, b))

        if not bounds:
            raise ValueError("No cells to crop from.")

        ls, ts, rs, bs = zip(*bounds)
        crop_box = (
            max(0, min(ls) - pad),
            max(0, min(ts) - pad),
            min(pil_image.width,  max(rs) + pad),
            min(pil_image.height, max(bs) + pad)
        )

        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise ValueError(f"Bad crop box: {crop_box}")

        cropped = pil_image.crop(crop_box)

        return {
            "original_dims": pil_image.size,
            "new_dims":      (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
            "crop_box":      crop_box,
            "crop_origin":   (crop_box[0], crop_box[1]),
            "cropped_image": cropped
        }

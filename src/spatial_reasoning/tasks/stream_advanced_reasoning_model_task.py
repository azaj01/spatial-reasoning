import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from itertools import product
from typing import Dict, Generator, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..agents import BaseAgent
from ..data import BaseDataset, Cell
from ..prompts import (GridCellDetectionPrompt,
                       SingleObjectGridCellTwoImagesDetectionPrompt)
from ..utils.io_utils import get_original_bounding_box, parse_detection_output
from .base_task import BaseTask
from .vanilla_reasoning_model_task import VanillaReasoningModelTask
from .vision_model_task import VisionModelTask


class StreamAdvancedReasoningModelTask(BaseTask):
    """
    Agent that utilizes CV tools and FMs with streaming support
    """
    def __init__(self, agent: BaseAgent, **kwargs):
        super().__init__(agent, **kwargs)
        self.prompt: GridCellDetectionPrompt = GridCellDetectionPrompt()
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
        Run reasoning model (non-streaming version for backward compatibility)
        """
        # Collect all results from the generator
        results = list(self.execute_streaming(**kwargs))
        # Return the final result
        return results[-1] if results else {}
    
    def execute_streaming(self, **kwargs) -> Generator[dict, None, None]:
        """
        Run reasoning model with streaming support
        Yields intermediate results as they become available
        """
        original_image: Image.Image = kwargs['image']
        object_of_interest: str = kwargs['prompt']
        grid_size = kwargs.get("grid_size", (4, 3))
        max_crops = kwargs.get('max_crops', 4)
        top_k = kwargs.get("top_k", -1)
        confidence_threshold = kwargs.get("confidence_threshold", 0.5)
        convergence_threshold = kwargs.get("convergence_threshold", 0.6)

        origin_coordinates = (0, 0)
        overlay_samples = []
        is_terminal_state = False
        crop_iteration = 0
        # Yield the very first image
        yield {
            'type': 'intermediate',
            'iteration': 0,
            'overlay_image': self._image_to_base64(self.overlay_grid_on_image(original_image, grid_size[0], grid_size[1])[0]),
            'is_terminal': is_terminal_state,
            'total_crops': 0,
            'max_crops': max_crops,
            'message': f'Initializing grid. May take up to 2 minutes...'
        }
        
        image = original_image.copy()
        while not is_terminal_state and len(overlay_samples) < max_crops:
            crop_iteration += 1
            
            if image.width < 512 and image.height < 512 and len(overlay_samples) > 0:
                _grid_size = (3, 2)
            else:
                _grid_size = grid_size
                
            # Run single crop process
            overlay_image, image, origin_coordinates, is_terminal_state = self.run_single_crop_process(
                image.copy(), 
                object_of_interest, 
                origin_coordinates, 
                _grid_size, 
                top_k, 
                confidence_threshold, 
                convergence_threshold
            )
            
            overlay_samples.append(overlay_image)
            
            # Yield intermediate result
            yield {
                'type': 'intermediate',
                'iteration': crop_iteration,
                'overlay_image': self._image_to_base64(self.overlay_grid_on_image(image, _grid_size[0], _grid_size[1])[0]),
                'is_terminal': is_terminal_state,
                'total_crops': len(overlay_samples),
                'max_crops': max_crops,
                'message': f'Processing crop {crop_iteration}/{max_crops}...'
            }
        
        # Final detection on the cropped image
        kwargs['image'] = image
        kwargs['confidence_threshold'] = 0.9
        vision_out, vanilla_out = self.run_agents_parallel(**kwargs)
        out = vanilla_out if vanilla_out['bboxs'] else vision_out
            
        # Display final cropped model prediction
        cropped_visualized_image = BaseDataset.visualize_image(
            image,
            [cell.to_tuple() for cell in out['bboxs']],
            return_image=True
        )
        
        # Yield the cropped visualization
        yield {
            'type': 'intermediate',
            'iteration': crop_iteration + 1,
            'overlay_image': self._image_to_base64(cropped_visualized_image),
            'is_terminal': True,
            'total_crops': len(overlay_samples) + 1,
            'max_crops': max_crops,
            'message': 'Model detection complete!'
        }

        # overlay_samples.append(cropped_visualized_image)

        # Restore to original coordinates
        restored_bboxs = get_original_bounding_box(
            cropped_bounding_boxs=out['bboxs'],
            crop_origin=origin_coordinates,
        )

        # Create final visualization
        visualized_image = BaseDataset.visualize_image(
            original_image,
            [cell.to_tuple() for cell in restored_bboxs],
            return_image=True
        )
        

        # Yield final result
        final_result = {
            'type': 'final',
            'bboxs': restored_bboxs,
            'overlay_images': [self._image_to_base64(img) for img in overlay_samples],
            'final_image': self._image_to_base64(visualized_image),
            'total_iterations': crop_iteration,
            'message': 'Detection complete!'
        }

        # Include any other data from the vanilla agent output
        # for key, value in out.items():
        #     if key not in ['bboxs', 'overlay_images']:
        #         final_result[key] = value
                
        yield final_result
    
    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    # ... rest of the methods remain the same ...
    
    @staticmethod
    def is_terminal_state(source_image: Image.Image,target_image: Image.Image, convergence_threshold: float) -> bool:
        """
        If the target image is similar in size to the source image, return True
        """
        src_width, src_height = source_image.size
        target_width, target_height = target_image.size
        area_ratio = (target_width * target_height) / (src_width * src_height)
        print(f"Area ratio: {area_ratio}, Target image size: {target_image.size}, Source image size: {source_image.size}")
        return area_ratio >= convergence_threshold
        
    def run_single_crop_process(self, image: Image.Image, object_of_interest: str, origin_coordinates: tuple, grid_size: tuple, top_k: int, confidence_threshold: float, convergence_threshold: float) -> dict:
        """
        Run crop process
        """
        overlay_image, cell_lookup = self.overlay_grid_on_image(
            image, grid_size[0], grid_size[1]
        )

        messages = [
            self.agent.create_text_message("system", self.prompt.get_system_prompt(
                resolution=image.size,
                object_of_interest=object_of_interest,
                grid_size=grid_size,
                confidence_threshold=confidence_threshold,
                cell_lookup=cell_lookup
            )),
            self.agent.create_multimodal_message(
                "user",
                self.prompt.get_user_prompt(
                    resolution=image.size,
                    object_of_interest=object_of_interest,
                    grid_size=grid_size,
                    cell_lookup=cell_lookup
                ),
                [overlay_image]
            )
        ]

        raw_response = self.agent.safe_chat(messages, reasoning={'effort' : 'medium', 'summary' : 'detailed'})
        structured_response = parse_detection_output(raw_response['output'])

        # DEBUGGING PURPOSES ONLY TO SEE WHAT THE REASONING MODEL IS SAYING
        if "reasoning" in raw_response:
            for reasoning in raw_response["reasoning"]:
                print(reasoning.text)
        print("--------------------------------")
        print(raw_response["output"])

        try:
            cropped_image_data: dict = StreamAdvancedReasoningModelTask.crop_image(
                image, structured_response, cell_lookup, top_k=top_k, confidence_threshold=confidence_threshold
            )
            assert cropped_image_data is not None, "Cropped image data is None"
        except Exception as e:
            print(f"Error cropping image: {e}")
            return overlay_image, image, origin_coordinates, True

        crop_origin = (
            origin_coordinates[0] + cropped_image_data["crop_origin"][0],
            origin_coordinates[1] + cropped_image_data["crop_origin"][1]
        )

        return overlay_image, cropped_image_data["cropped_image"], crop_origin, self.is_terminal_state(image, cropped_image_data["cropped_image"], convergence_threshold)

    @staticmethod
    def overlay_grid_on_image(
        image: Union[Image.Image, torch.Tensor],
        num_rows: int,
        num_cols: int,
        color: str = "red",
        font_size: int = None,
        width: int = None,
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
        
        # Auto-calculate font size and width if not provided
        if font_size is None:
            min_cell_dim = min(cell_width, cell_height)
            font_size = int(min_cell_dim * 0.3)
            font_size = max(10, min(font_size, 80))  # Clamp between 10 and 80
        
        if width is None:
            width = max(1, font_size // 20)  # Scale line width with font size

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
        pad: int = 50,
        top_k: int = -1,
        confidence_threshold: float = 0.65
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
        # filter out all groups that have confidence less than the threshold
        grouped = [g for g in grouped if np.mean(g[1]) >= confidence_threshold]

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
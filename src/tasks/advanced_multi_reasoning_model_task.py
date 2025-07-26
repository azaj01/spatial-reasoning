from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from agents import BaseAgent
from PIL import Image
from prompts import GridCellTwoImagesDetectionPrompt
from utils.io_utils import get_original_bounding_box, parse_detection_output

from .advanced_reasoning_model_task import AdvancedReasoningModelTask
from .base_task import BaseTask
from .vanilla_reasoning_model_task import VanillaReasoningModelTask
from .vision_model_task import VisionModelTask


@dataclass
class Node:
    image: Image.Image
    coordinates: Tuple[int, int]
    depth: int
    parent: Optional['Node'] = None

    def __str__(self):
        return f"Node(image={self.image}, coordinates={self.coordinates}, depth={self.depth}, parent={self.parent})"


class MultiAdvancedReasoningModelTask(BaseTask):
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
        max_crops = self.kwargs.get('max_crops', 3)  # TODO: make max crops an LMM decision.
        top_k = self.kwargs.get("top_k", -1)  # TODO: give user the flexibility if they want to detect one object or multiple
        confidence_threshold = self.kwargs.get("confidence_threshold", 0.65)
        convergence_threshold = self.kwargs.get("convergence_threshold", 0.5)
        
        results = self.bfs(
            image,
            object_of_interest,
            grid_size,
            top_k,
            confidence_threshold,
            convergence_threshold,
            max_crops,
        )
        
        node_to_result = {id(r['node']): r for r in results}
        
        final_bboxs = []
        final_overlay_images = []
        
        final_nodes_data = []
        # TODO: add the final bboxs and overlay images
        return {
            "bboxs": final_bboxs,
            "overlay_images": final_overlay_images
        }

    def bfs(
        self,
        initial_image: Image.Image,
        object_of_interest: str,
        grid_size: Tuple[int, int],
        top_k: int,
        confidence_threshold: float,
        convergence_threshold: float,
        max_crops: int
    ):

        # Start with root node
        root = Node(image=initial_image, coordinates=(0, 0), depth=0)
        queue = deque([root])
        
        # Store all processed nodes with their results
        results = []
        
        while queue:
            node = queue.popleft()
            
            # Stop at max depth
            if node.depth >= max_crops:
                results.append({'node': node, 'children': [], 'overlay': None})
                continue
            
            # Process this image
            print(f"Processing node: {node}")
            out = self.run_single_crop_process(
                node.image,
                object_of_interest,
                node.coordinates,
                grid_size,
                top_k,
                confidence_threshold,
                convergence_threshold
            )
            
            # Create child nodes
            children = []
            list_of_is_terminal = out.get('list_of_is_terminal', [])
            for i, (img, coords) in enumerate(zip(out['list_of_crop_image_data'], out['list_of_crop_origin_coordinates'])):
                child = Node(
                    image=img,
                    coordinates=coords,
                    depth=node.depth + 1,
                    parent=node
                )
                children.append(child)
                
                # Only add to queue if this specific crop hasn't converged
                is_terminal = list_of_is_terminal[i] if i < len(list_of_is_terminal) else False
                if not is_terminal:
                    queue.append(child)
            
            results.append({
                'node': node,
                'children': children,
                'overlay': out['overlay_image'],
                'is_terminal': out.get('is_terminal', False),
                'reason': out.get('reason', '')
            })
        
        return results

    @staticmethod
    def is_terminal_state(cell_group: list[int], grid_size: tuple, convergence_threshold: float) -> bool:
        """
        Convergence is defined as the ratio of the number of cells in the cell group
        to the total number of cells in the grid. If the ratio is greater than or
        equal to the convergence threshold, then the cell group is a terminal state.
        """
        total_cells = grid_size[0] * grid_size[1]
        return len(cell_group) >= total_cells * convergence_threshold

    def run_single_crop_process(self, image: Image.Image, object_of_interest: str, origin_coordinates: tuple, grid_size: tuple, top_k: int, confidence_threshold: float, convergence_threshold: float):
        """
        Run crop process
        """
        overlay_image, cell_lookup = AdvancedReasoningModelTask.overlay_grid_on_image(
            image, grid_size[0], grid_size[1]
        )
        
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
        
        
        # Add logging
        print("*"*100)
        print("Grouped cells:")
        print(structured_response['cells'])
        print("-" * 100)
        print("Grouped confidence:")
        print(structured_response['confidence'])
        print("*"*100)
        # Check convergence for each group
        total_cells = grid_size[0] * grid_size[1]
        threshold_cells = total_cells * convergence_threshold
        
        # Get crops for ALL groups
        list_of_cropped_image_data: list[dict] = MultiAdvancedReasoningModelTask.crop_image(
            image, 
            structured_response,
            cell_lookup, 
            top_k=top_k, 
            confidence_threshold=confidence_threshold
        )
        
        if not list_of_cropped_image_data:
            print("Unable to get object in the grid, most likely due to it not being found in the image.")
            return {
                "overlay_image": overlay_image,
                "list_of_crop_image_data": [],
                "list_of_crop_origin_coordinates": [],
                "is_terminal": True,
                "reason": "no objects found"
            }

        # Process all crops but mark which ones are terminal
        list_of_crop_origin_coordinates = []
        list_of_crop_image_data = []
        list_of_is_terminal = []  # Track which crops are terminal
        
        for i, (group, crop_image_data) in enumerate(zip(structured_response, list_of_cropped_image_data)):
            crop_origin = (
                origin_coordinates[0] + crop_image_data["crop_origin"][0],
                origin_coordinates[1] + crop_image_data["crop_origin"][1]
            )
            list_of_crop_image_data.append(crop_image_data["cropped_image"])
            list_of_crop_origin_coordinates.append(crop_origin)
            
            # Check if this specific group has converged
            is_converged = len(group) >= threshold_cells
            list_of_is_terminal.append(is_converged)
            
            if is_converged:
                print(f"Group {i} has converged ({len(group)}/{total_cells} cells = {len(group)/total_cells:.1%})")

        return {
            "overlay_image": overlay_image,
            "list_of_crop_image_data": list_of_crop_image_data,
            "list_of_crop_origin_coordinates": list_of_crop_origin_coordinates,
            "list_of_is_terminal": list_of_is_terminal,  # NEW: terminal status per crop
            "is_terminal": False,  # The overall process continues
            "reason": f"found {len(list_of_crop_image_data)} crops"
        }

    @staticmethod
    def crop_image(
        pil_image: Image.Image,
        scores_grid: dict,
        cell_lookup: dict,
        pad: int = 0,
        top_k: int = 1,
        confidence_threshold: float = 0.65
    ) -> list[dict]:
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

        # Each cell group will be transformed into a crop.
        # This enables us to detect multiple objects in the image
        # without the model having to infer everything from a zoomed out view.
        crops = []
        for group_idx, (cell_ids, confidences) in enumerate(grouped):
            bounds = []
            for cid in cell_ids:
                c = cell_lookup[cid]
                l, r = sorted([c.left, c.right])
                t, b = sorted([c.top, c.bottom])
                bounds.append((l, t, r, b))

            if not bounds:
                continue

            ls, ts, rs, bs = zip(*bounds)
            crop_box = (
                max(0, min(ls) - pad),
                max(0, min(ts) - pad),
                min(pil_image.width,  max(rs) + pad),
                min(pil_image.height, max(bs) + pad)
            )

            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                print(f"Warning: Bad crop box for group {group_idx}: {crop_box}")
                continue

            cropped = pil_image.crop(crop_box)

            crops.append({
                "original_dims": pil_image.size,
                "new_dims":      (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
                "crop_box":      crop_box,
                "crop_origin":   (crop_box[0], crop_box[1]),
                "cropped_image": cropped,
                "cells": cell_ids,  # Track which cells this crop came from
                "confidence": np.mean(confidences)
            })
        
        return crops

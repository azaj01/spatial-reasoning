from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from agents import BaseAgent
from PIL import Image
from prompts import MultiObjectGridCellTwoImagesDetectionPrompt
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
        self.prompt: MultiObjectGridCellTwoImagesDetectionPrompt = MultiObjectGridCellTwoImagesDetectionPrompt()
        # Tool use -and- foundation model agents
        self.vanilla_agent: VanillaReasoningModelTask = VanillaReasoningModelTask(agent, **kwargs)
        vision_agent = deepcopy(agent)
        vision_agent.model = "o4-mini"
        self.vision_agent: VisionModelTask = VisionModelTask(vision_agent, **kwargs)
    
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
        max_crops = self.kwargs.get('max_crops', 3)
        print(max_crops)
        top_k = self.kwargs.get("top_k", -1)  # TODO: give user the flexibility if they want to detect one object or multiple
        confidence_threshold = self.kwargs.get("confidence_threshold", 0.7)
        
        results = self.bfs(
            image,
            object_of_interest,
            grid_size,
            top_k,
            confidence_threshold,
            max_crops,
        )
        
        # Create mappings
        all_nodes_in_results = {id(r['node']) for r in results}
        node_to_result = {id(r['node']): r for r in results}

        # Collect leaf nodes with their parent result info
        final_crops_data = []

        for r in results:
            # Case 1: Children that are NOT in results (terminal FOUND)
            for child in r['children']:
                if id(child) not in all_nodes_in_results:
                    final_crops_data.append({
                        'crop_image': child.image,
                        'crop_coordinates': child.coordinates,
                        'depth': child.depth,
                        'overlay_image': r['overlay'],  # Parent's overlay showing where this crop came from
                    })

            # Case 2: Node with no children (max_depth or no detections)
            if not r['children']:
                # For leaf nodes, we need their parent's overlay
                parent_overlay = None
                if r['node'].parent:
                    parent_result = node_to_result.get(id(r['node'].parent))
                    if parent_result:
                        parent_overlay = parent_result['overlay']

                final_crops_data.append({
                    'crop_image': r['node'].image,
                    'crop_coordinates': r['node'].coordinates,
                    'depth': r['node'].depth,
                    'overlay_image': parent_overlay,  # Parent's overlay (None for root)
                })

        # Run through the vision encoder
        final_data = {"bboxs": [], "overlay_images": []}
        for item in final_crops_data:
            vision_out, vanilla_out = self.run_agents_parallel(image=item['crop_image'], prompt=object_of_interest)

            out = vision_out if len(vision_out['bboxs']) > 0 else vanilla_out
            
            # Restore to original coordinates
            restored_bboxs = get_original_bounding_box(
                cropped_bounding_boxs=out['bboxs'],
                crop_origin=item['crop_coordinates'],
            )
            final_data["bboxs"].append(restored_bboxs)
            final_data["overlay_images"].append(item['overlay_image'])

        return final_data

    def bfs(
        self,
        initial_image: Image.Image,
        object_of_interest: str,
        grid_size: Tuple[int, int],
        top_k: int,
        confidence_threshold: float,
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
                confidence_threshold
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
                'reason': out.get('reason', '')
            })
        
        return results


    def run_single_crop_process(self, image: Image.Image, object_of_interest: str, origin_coordinates: tuple, grid_size: tuple, top_k: int, confidence_threshold: float):
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
        raw_response = self.agent.safe_chat(messages, reasoning={'effort' : 'medium', 'summary' : 'detailed'})
        # raw_response = self.agent.safe_chat(messages)
        print(raw_response)
        
        structured_response = parse_detection_output(raw_response['output'])
        
        if not structured_response['detections']:
            return {
                "overlay_image": overlay_image,
                "list_of_crop_image_data": [],
                "list_of_crop_origin_coordinates": [],
                "list_of_is_terminal": [],
                "reason": "no objects found"
            }
            
        # Get crops for ALL detections (both FOUND and ZOOM)
        all_crops = MultiAdvancedReasoningModelTask.crop_image(
            image,
            structured_response['detections'],
            cell_lookup,
            action_filter=None,  # Process both FOUND and ZOOM
            top_k=top_k,
            confidence_threshold=confidence_threshold
        )
        
        print("**" * 100)
        print(all_crops)
        print("**" * 100)
        
        # Process crops
        list_of_crop_origin_coordinates = []
        list_of_crop_image_data = []
        list_of_is_terminal = []
        
        for crop_data in all_crops:
            crop_origin = (
                origin_coordinates[0] + crop_data["crop_origin"][0],
                origin_coordinates[1] + crop_data["crop_origin"][1]
            )
            list_of_crop_image_data.append(crop_data["cropped_image"])
            list_of_crop_origin_coordinates.append(crop_origin)
            
            # FOUND = terminal, ZOOM = not terminal
            is_terminal = (crop_data["action"] == "FOUND")
            list_of_is_terminal.append(is_terminal)
        
        return {
            "overlay_image": overlay_image,
            "list_of_crop_image_data": list_of_crop_image_data,
            "list_of_crop_origin_coordinates": list_of_crop_origin_coordinates,
            "list_of_is_terminal": list_of_is_terminal,
            "reason": f"found {len(structured_response['detections'])} detections",
            "all_detections": structured_response['detections']
        }

    @staticmethod
    def crop_image(
        pil_image: Image.Image,
        detections: list[dict],
        cell_lookup: dict,
        pad: int = 50,
        top_k: int = -1,
        confidence_threshold: float = 0.65,
        action_filter: str = None  # 'FOUND', 'ZOOM', or None for both
    ) -> list[dict]:
        """
        Crop image using detections from the new format.
        detections = [
            {
                "action": "FOUND",
                "cells": [14, 15, 24, 25],
                "confidence": [85, 90, 88, 92]
            },
            {
                "action": "ZOOM",
                "cells": [31, 32, 41, 42],
                "confidence": [75, 70, 72, 68],
                "reason": "Multiple small objects clustered together"
            }
        ]
        """
        # Basic error checking
        if not detections:
            return []
        
        # Filter by action if specified
        if action_filter:
            detections = [d for d in detections if d.get("action") == action_filter]
        
        # Convert to (cells, confidence, action) tuples and sort by mean confidence
        processed_detections = []
        for det in detections:
            cells = det.get("cells", [])
            confidence = det.get("confidence", [])
            action = det.get("action", "UNKNOWN")
            
            if cells and confidence and len(cells) == len(confidence):
                mean_conf = np.mean(confidence)
                if mean_conf >= confidence_threshold:
                    processed_detections.append((cells, confidence, action, det))
        
        # Sort by mean confidence
        processed_detections.sort(key=lambda x: np.mean(x[1]), reverse=True)
        
        # Apply top_k if specified
        if top_k != -1:
            processed_detections = processed_detections[:top_k]
        
        # Create crops
        crops = []
        for group_idx, (cell_ids, confidences, action, original_det) in enumerate(processed_detections):
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
                min(pil_image.width, max(rs) + pad),
                min(pil_image.height, max(bs) + pad)
            )
            
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                print(f"Warning: Bad crop box for group {group_idx}: {crop_box}")
                continue
            
            cropped = pil_image.crop(crop_box)
            
            crops.append({
                "original_dims": pil_image.size,
                "new_dims": (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
                "crop_box": crop_box,
                "crop_origin": (crop_box[0], crop_box[1]),
                "cropped_image": cropped,
                "cells": cell_ids,
                "confidence": np.mean(confidences),
                "action": action,  # Track whether this was FOUND or ZOOM
                "reason": original_det.get("reason", None)  # For ZOOM actions
            })
        
        return crops

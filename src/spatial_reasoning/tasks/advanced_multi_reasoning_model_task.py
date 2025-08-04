""" 
This task is an advanced version of the AdvancedReasoningModelTask.
Currently, crop-based reasoning looks at the set of all valid cells in the grid and takes the cumulative overlap before cropping
to the next stage.

Design to motivate the problem: https://imgur.com/a/HI3yACH

The challenge with the approach is as follows:
In the case multiple objects exist in vastly different parts of the image
the new crop will be very similar to the original image.
This defeats the purpose of crop-based reasoning.

To solve this, we can use a BFS approach to crop the image.
Where all distinct objects are cropped individually and we re-run the cropping + reasoning mechanism on each cropped group of cells
in parallel (a la BFS).

That said, this approach has a few challenges:
1. Computationally expensive.
2. Assumes the existing grid lines separate the objects of interest as perfectly as possible which is mostly false.
This is critical because in the case this is violated, we will now be running parallel `run_single_crop_process`
calls on subset of cells that contain the same object that may be found in another cell. This will yield to an 
incorrect number of objects and corresponding bounding boxes reported by the model.

I pulled an all-nighter to implement this and my initial results were lackluster.
I leave this as an exercise for the curious reader to take a plunge and see if they can improve upon this.

Motivation: https://x.com/mrsiipa/status/1951929207392514185
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import label

from ..agents import BaseAgent
from ..prompts import SimplifiedGridCellDetectionPrompt
from ..utils.io_utils import get_original_bounding_box, parse_detection_output
from .advanced_reasoning_model_task import AdvancedReasoningModelTask
from .base_task import BaseTask
from .vanilla_reasoning_model_task import VanillaReasoningModelTask
from .vision_model_task import VisionModelTask


@dataclass
class Node:
    image: Image.Image
    coordinates: Tuple[int, int]
    depth: int
    parent: Optional["Node"] = None

    def __str__(self):
        return f"Node(image={self.image}, coordinates={self.coordinates}, depth={self.depth}, parent={self.parent})"


class MultiAdvancedReasoningModelTask(BaseTask):
    """
    Agent that utilizes CV tools and FMs
    """

    def __init__(self, agent: BaseAgent, **kwargs):
        super().__init__(agent, **kwargs)
        self.prompt: SimplifiedGridCellDetectionPrompt = (
            SimplifiedGridCellDetectionPrompt()
        )
        # Tool use -and- foundation model agents
        self.vanilla_agent: VanillaReasoningModelTask = VanillaReasoningModelTask(
            agent, prompt_type="advanced", **kwargs
        )
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
                executor.submit(self.vision_agent.execute, **kwargs): "vision",
                executor.submit(self.vanilla_agent.execute, **kwargs): "vanilla",
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
                    results[agent_type] = {"error": str(e)}

        return results.get("vision", {}), results.get("vanilla", {})

    def execute(self, **kwargs) -> dict:
        """
        Run reasoning model
        Arguments:
            image: Image.Image
            prompt: str
        """
        image: Image.Image = kwargs["image"]
        object_of_interest: str = kwargs["prompt"]

        grid_size = self.kwargs.get("grid_size", (3, 4))  # num_rows x num_cols
        max_crops = self.kwargs.get("max_crops", 3)
        print(max_crops)
        top_k = self.kwargs.get(
            "top_k", -1
        )  # TODO: give user the flexibility if they want to detect one object or multiple
        confidence_threshold = self.kwargs.get("confidence_threshold", 0.7)
        convergence_threshold = self.kwargs.get("convergence_threshold", 0.5)

        results = self.bfs(
            image,
            object_of_interest,
            grid_size,
            top_k,
            confidence_threshold,
            max_crops,
            convergence_threshold,
        )

        # Create mappings
        all_nodes_in_results = {id(r["node"]) for r in results}
        node_to_result = {id(r["node"]): r for r in results}

        # Collect leaf nodes with their parent result info
        final_crops_data = []

        for r in results:
            # Case 1: Children that are NOT in results (terminal FOUND)
            for child in r["children"]:
                if id(child) not in all_nodes_in_results:
                    final_crops_data.append(
                        {
                            "crop_image": child.image,
                            "crop_coordinates": child.coordinates,
                            "depth": child.depth,
                            "overlay_image": r[
                                "overlay"
                            ],  # Parent's overlay showing where this crop came from
                        }
                    )

            # Case 2: Node with no children (max_depth or no detections)
            if not r["children"]:
                # For leaf nodes, we need their parent's overlay
                parent_overlay = None
                if r["node"].parent:
                    parent_result = node_to_result.get(id(r["node"].parent))
                    if parent_result:
                        parent_overlay = parent_result["overlay"]

                final_crops_data.append(
                    {
                        "crop_image": r["node"].image,
                        "crop_coordinates": r["node"].coordinates,
                        "depth": r["node"].depth,
                        "overlay_image": parent_overlay,  # Parent's overlay (None for root)
                    }
                )

        # Run through the vision encoder
        final_data = {"bboxs": [], "overlay_images": []}
        for item in final_crops_data:
            vision_out, vanilla_out = self.run_agents_parallel(
                image=item["crop_image"], prompt=object_of_interest
            )

            out = vision_out if len(vision_out["bboxs"]) > 0 else vanilla_out

            # Restore to original coordinates
            restored_bboxs = get_original_bounding_box(
                cropped_bounding_boxs=out["bboxs"],
                crop_origin=item["crop_coordinates"],
            )
            final_data["bboxs"].append(restored_bboxs)
            final_data["overlay_images"].append(item["overlay_image"])

        return final_data

    def bfs(
        self,
        initial_image: Image.Image,
        object_of_interest: str,
        grid_size: Tuple[int, int],
        top_k: int,
        confidence_threshold: float,
        max_crops: int,
        convergence_threshold: float,
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
                results.append({"node": node, "children": [], "overlay": None})
                continue

            # Process this image
            out = self.run_single_crop_process(
                node.image,
                object_of_interest,
                node.coordinates,
                grid_size,
                top_k,
                confidence_threshold,
                convergence_threshold,
            )

            # Create child nodes
            children = []
            list_of_is_terminal = out.get("list_of_is_terminal", [])
            for i, (img, coords) in enumerate(
                zip(
                    out["list_of_crop_image_data"],
                    out["list_of_crop_origin_coordinates"],
                )
            ):
                child = Node(
                    image=img, coordinates=coords, depth=node.depth + 1, parent=node
                )
                children.append(child)

                # Only add to queue if this specific crop hasn't converged
                is_terminal = (
                    list_of_is_terminal[i] if i < len(list_of_is_terminal) else False
                )
                if not is_terminal:
                    queue.append(child)

            results.append(
                {
                    "node": node,
                    "children": children,
                    "overlay": out["overlay_image"],
                }
            )

        return results

    def connected_components(
        self, detections: dict, grid_size: tuple[int, int]
    ) -> tuple[list[list[int]], list[list[float]]]:
        """
        Find connected components in a grid of cells using 4-neighbor connectivity,
        and group their corresponding confidence values.

        Args:
            detections (dict): {cells: list of cell indices}, confidence: list of confidence values corresponding to each cell}
            grid_size: Tuple of (rows, cols).
        """
        cells, confidences = detections["cells"], detections["confidence"]
        rows, cols = grid_size
        binary_grid = np.zeros((rows, cols), dtype=np.int32)

        # Create mapping from cell number to confidence
        cell_to_confidence = dict(zip(cells, confidences))

        # Convert 1-indexed cell numbers to 2D grid coordinates
        for cell in cells:
            cell -= 1
            r, c = divmod(cell, cols)
            binary_grid[r, c] = 1

        # Use 4-connectivity structure
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)

        labeled, num_features = label(binary_grid, structure=structure)

        # Initialize components for cells and confidences
        cell_components = [[] for _ in range(num_features)]
        confidence_components = [[] for _ in range(num_features)]

        for r in range(rows):
            for c in range(cols):
                label_id = labeled[r, c]
                if label_id > 0:
                    cell_number = r * cols + c + 1  # back to 1-indexed
                    cell_components[label_id - 1].append(cell_number)
                    confidence_components[label_id - 1].append(
                        cell_to_confidence[cell_number]
                    )

        return {
            "cells": cell_components,
            "confidence": confidence_components,
            "total_cc": len(cell_components),
        }

    def is_terminating_state(self, cells: list[int], grid_size, convergence_threshold):
        total_number_of_cells: int = grid_size[0] * grid_size[1]
        return len(cells) >= total_number_of_cells * convergence_threshold

    def run_single_crop_process(
        self,
        image: Image.Image,
        object_of_interest: str,
        origin_coordinates: tuple,
        grid_size: tuple,
        top_k: int,
        confidence_threshold: float,
        convergence_threshold: float,
    ):
        """
        Run crop process
        """
        overlay_image, cell_lookup = BaseTask.overlay_grid_on_image(
            image, grid_size[0], grid_size[1]
        )

        output = {
            "list_of_is_terminal": [],
            "list_of_crop_image_data": [],
            "list_of_crop_origin_coordinates": [],
            "overlay_image": overlay_image,
        }

        messages = [
            self.agent.create_text_message(
                "system", self.prompt.get_system_prompt(grid_size=grid_size)
            ),
            self.agent.create_multimodal_message(
                "user",
                self.prompt.get_user_prompt(
                    resolution=image.size,
                    object_of_interest=object_of_interest,
                    grid_size=grid_size,
                ),
                [image, overlay_image],
            ),
        ]
        # raw_response = self.agent.safe_chat(messages, reasoning={'effort' : 'medium', 'summary' : 'detailed'})
        raw_response = self.agent.safe_chat(messages)
        print(raw_response)

        structured_response: dict = parse_detection_output(raw_response["output"])
        if not structured_response:
            return output

        components = self.connected_components(structured_response, grid_size)

        print(f"Found {components['total_cc']} objects.")

        for i in range(components["total_cc"]):
            cell_group = components["cells"][i]
            confidence_group = components["confidence"][i]
            if np.mean(confidence_group) < confidence_threshold:
                # Model not confident about the prediction of cells, should filter out
                continue

            cropped_data = MultiAdvancedReasoningModelTask.crop_image(
                image, cell_group, cell_lookup, origin_coordinates
            )

            is_terminal = self.is_terminating_state(
                cell_group, grid_size, convergence_threshold
            )

            output["list_of_is_terminal"].append(is_terminal)
            output["list_of_crop_image_data"].append(cropped_data["cropped_image"])
            output["list_of_crop_origin_coordinates"].append(
                cropped_data["crop_origin"]
            )
        return output

    @staticmethod
    def crop_image(
        pil_image: Image.Image,
        cell_group: list[int],
        cell_lookup: dict,
        origin_coordinates: tuple,
        pad: int = 0,
    ) -> dict:
        """
        Crop image based on a single group of cells.

        Args:
            pil_image: PIL Image to crop
            cell_group: List of cells, e.g. [1, 2, 5] or [7, 11]
            cell_lookup: Dict mapping cell_id to cell object with .left, .right, .top, .bottom
            pad: Padding around crop in pixels

        Returns:
            Crop dictionary with image and metadata (or None if invalid)
        """
        if not cell_group:
            return None

        # Get bounding box for all cells in group
        bounds = []
        for cid in cell_group:
            if cid not in cell_lookup:
                continue
            c = cell_lookup[cid]
            bounds.append((c.left, c.top, c.right, c.bottom))

        if not bounds:
            return None

        # Find overall bounding box
        lefts, tops, rights, bottoms = zip(*bounds)
        crop_box = (
            max(0, min(lefts) - pad),
            max(0, min(tops) - pad),
            min(pil_image.width, max(rights) + pad),
            min(pil_image.height, max(bottoms) + pad),
        )

        # Validate crop box
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return None

        # Crop image
        cropped = pil_image.crop(crop_box)

        return {
            "original_dims": pil_image.size,
            "new_dims": (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
            "crop_box": crop_box,
            "crop_origin": (
                origin_coordinates[0] + crop_box[0],
                origin_coordinates[1] + crop_box[1],
            ),
            "cropped_image": cropped,
            "cells": cell_group,
        }

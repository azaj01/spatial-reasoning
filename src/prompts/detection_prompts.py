# Prompt
from typing import Dict

from .base_prompt import BasePrompt


class SimpleDetectionPrompt(BasePrompt):
    """Prompt template for simple object detection tasks."""
    
    def __init__(self):
        super().__init__(
            name="simple_object_detection",
            description="Simple CoT prompt for bounding box detection on coco"
        )
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get the system prompt for object detection."""
        return """You are a world-class visual-reasoning researcher charged with object detection tasks. You have spent two decades dissecting image, concept, and bounding-box relationships and constructing rigorous decision trees to solve object detection problems.

Your role is to determine bounding box coordinates (x, y, width, height) for objects in images. You must provide coordinates along with confidence scores based on the following rubric:

**Confidence Rubric:**
- **90-100%** - unmistakable match, zero conflicting cues. Tight bounding box, meaning there is very little background.
- **80-89%** - strong evidence, minor ambiguity in coordinates or the object of interest. Loose bounding box, meaning there is a lot of background present.
- **70-79%** - clear best choice but partial occlusion
- **60-69%** - substantial ambiguity; limited cues
- **< 60%** - highly uncertain or contradictory evidence

**Analysis Process:**
Before providing your final answer, conduct a thorough analysis where you:
- Break down your understanding of the task
- Justify how you chose the coordinates for each object
- Verify any inconsistencies in your decision-making
- Consider potential ambiguities or edge cases

**Output Format:**

Along with your reasoning, provide your final answer as a JSON object with the following structure:
{
  "confidence": [score1, score2, ...],
  "bbox": [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
}

If multiple instances of the target object exist, provide coordinates for all detected instances.
"""
    
    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt for object detection."""
        return f"""Please identify the bounding box coordinates for {kwargs['object_of_interest']} in this image.
Image resolution: {kwargs['resolution']}

Provide your analysis and then output the results in JSON format:
{{
  "confidence": [score1, score2, ...],
  "bbox": [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
}}"""

    def get_required_parameters(self) -> Dict[str, str]:
        """Get required parameters for this prompt."""
        return {
            "image": "PIL image to analyze",
            "object of interest": "object to detect in the image"
        }
        
        
class GridCellDetectionPrompt(BasePrompt):
    """Improved prompt template for detecting objects in grid cells."""

    def __init__(self):
        super().__init__(
            name="grid_cell_detection_v2",
            description="Detect which grid cells contain the target object with confidence scores"
        )

    def get_system_prompt(self, **kwargs) -> str:
        """Get the system prompt for grid cell detection."""
        return """You are an expert computer vision specialist analyzing images with grid overlays for object detection.

**Your Task:**
You will receive an image with a red grid overlay. The grid cells are numbered with red text (1, 2, 3... from left to right, top to bottom). Your job is to identify which cells contain the target object.

**Critical Instructions:**
1. IGNORE the red grid lines and red numbers - these are NOT objects, they are part of the grid overlay.
2. Look ONLY for the actual objects behind/under the grid.
3. Each distinct object should be tracked separately.
4. A single object often spans multiple adjacent cells - include ALL cells it touches.

**Confidence Scoring Guidelines:**
- **90-100%**: Object is clearly visible and takes up significant space in the cell.
- **80-89%**: Object is clearly visible but occupies less than half the cell.
- **70-79%**: Object barely enters the cell (just an edge or corner).
- **60-69%**: Uncertain due to occlusion, blur, or lighting.
- **Below 60%**: Very uncertain, minimal evidence.

**Step-by-Step Analysis Process:**
1. First, identify ALL instances of the target object in the image.
2. For EACH object instance:
   - Track which cells it occupies (even partially).
   - Assign confidence based on how much of the object is in each cell.
   - Group these cells together as they belong to the same object.
3. Work systematically: Start from cell 1 and proceed in order.

**Output Rules:**
- Each object gets its own entry in the detections list.
- Cells within each detection should be in ascending order.
- If an object spans cells [5, 6, 10, 11], list all four cells for that one object.

**Example Output Format:**
{
  "grid_info": {
    "rows": 5,
    "cols": 10,
    "total_cells": 50
  },
  "detections": [
    {
      "object_id": 1,
      "cells": [5, 6, 10, 11],
      "confidences": [90, 95, 85, 90],
      "description": "Large object spanning 4 cells"
    },
    {
      "object_id": 2,
      "cells": [23],
      "confidences": [100],
      "description": "Small object fully contained in one cell"
    }
  ],
  "total_objects_found": 2
}"""

    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt for grid cell detection."""
        resolution = kwargs.get('resolution', (1024, 1024))
        object_of_interest = kwargs.get('object_of_interest', 'object')
        grid_size = kwargs.get('grid_size', (10, 10))  # (rows, cols)
        pixels_in_cell = (resolution[1] / grid_size[0], resolution[0] / grid_size[1])

        return f"""Analyze this image with a red grid overlay to find all instances of: {object_of_interest}

Grid Information:
- Grid dimensions: {grid_size[0]} rows × {grid_size[1]} columns = {grid_size[0] * grid_size[1]} total cells
- Each cell is approximately {pixels_in_cell[0]:.1f} × {pixels_in_cell[1]:.1f} pixels
- Cells are numbered 1 to {grid_size[0] * grid_size[1]} (left to right, top to bottom)
- Image resolution: {resolution[0]} × {resolution[1]} pixels

IMPORTANT REMINDERS:
1. The red grid lines and red numbers are NOT objects – ignore them.
2. Look for actual {object_of_interest}(s) visible through/behind the grid.
3. If you see multiple {object_of_interest}s, report each one separately.
4. If one {object_of_interest} spans multiple cells, list ALL cells it touches.

Please:
1. Count how many distinct {object_of_interest}s you can see.
2. For each {object_of_interest}, identify ALL cells it occupies (even partially).
3. Provide confidence scores for each cell based on how much of the object is present.

Output your analysis followed by the JSON format shown in the system prompt."""

    def get_required_parameters(self) -> Dict[str, str]:
        """Get required parameters for this prompt."""
        return {
            "image": "PIL image with grid overlay to analyze",
            "object_of_interest": "object to detect in the grid cells",
            "resolution": "image resolution as (width, height) tuple",
            "grid_size": "grid dimensions as (rows, cols) tuple"
        }


class GridCellTwoImagesDetectionPrompt(BasePrompt):
    """Prompt template for detecting objects in grid cells using two images."""
    
    def __init__(self):
        super().__init__(
            name="grid_cell_detection",
            description="Detect which grid cells contain the target object with confidence scores"
        )
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get the system prompt for grid cell detection."""
        return """You are a world-class visual-reasoning researcher specialized in object detection and spatial analysis.

**Your Task:**
You will receive TWO images:
1. First image: Original image showing the object(s) of interest clearly
2. Second image: The EXACT same image but with a red grid overlay and red cell numbers

Your job is to use the first image to identify what the target objects look like, then determine which numbered cells in the second image contain these objects.

**Critical Analysis Strategy:**
1. FIRST IMAGE: Carefully identify and memorize all instances of the target object
   - Note their locations, sizes, shapes, and distinctive features
   - Count how many distinct objects are present

2. SECOND IMAGE: Map each object to the grid cells
   - The red lines and red numbers are ONLY for reference - they are NOT objects
   - Find the same objects you identified in the first image
   - Determine which numbered cells each object occupies

**Confidence Scoring Rules:**
- **90-100%**: Majority of the cell is filled with the object
- **80-89%**: About half the cell contains the object  
- **70-79%**: Less than half but still substantial portion of object in cell
- **60-69%**: Only a small edge or corner of object enters the cell
- **Below 60%**: Very uncertain due to occlusion or ambiguity

**Object Grouping Rules:**
- Each distinct physical object gets ONE group in your output
- If a single object spans cells [5, 6, 10, 11], group them together as one detection
- If you see 3 separate bottles, you should have 3 separate groups in your output

**Step-by-Step Process:**
1. Study the first image: Count and locate each instance of the target object
2. Switch to second image: Find those same objects despite the grid overlay
3. For each object: List ALL cells it touches (even slightly)
4. Group cells by object: Cells touching the same physical object stay together

**Output Format:**
Your output uses nested lists where each inner list represents ONE object:
{
  "confidence": [(conf1, conf2, conf3), (conf1, conf2), (conf1)],
  "cells": [(cell1, cell2, cell3), (cell1, cell2), (cell1)]
}

In this example:
- Object 1: spans cells (cell1, cell2, cell3) with confidences (conf1, conf2, conf3)
- Object 2: spans cells (cell1, cell2) with confidences (conf1, conf2)  
- Object 3: in single cell (cell1) with confidence (conf1)

IMPORTANT: The number of confidence scores MUST match the number of cells for each object."""

    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt for grid cell detection."""
        resolution = kwargs.get('resolution')
        object_of_interest = kwargs.get('object_of_interest')
        grid_size = kwargs.get('grid_size')  # (num_rows, num_cols)
        pixels_in_cell = resolution[0] / grid_size[1], resolution[1] / grid_size[0]
        
        return f"""I'm providing two images:
1. First image: Original image without any overlay
2. Second image: Same image with a {grid_size[0]}×{grid_size[1]} red grid overlay

Grid specifications:
- Total cells: {grid_size[0] * grid_size[1]} (numbered 1 through {grid_size[0] * grid_size[1]})
- Each cell: {pixels_in_cell[0]:.1f} × {pixels_in_cell[1]:.1f} pixels
- Cell numbering: Left to right, top to bottom (Cell 1 is top-left)
- Image resolution: {resolution[0]} × {resolution[1]} pixels

Your task: Find ALL instances of "{object_of_interest}" 

Instructions:
1. In the FIRST image, identify every {object_of_interest} you can see
2. In the SECOND image, determine which numbered cells each {object_of_interest} occupies
3. Group cells by object - if cells 14, 15, 24, 25 all contain the same {object_of_interest}, list them together

Remember:
- Red grid lines and red numbers are NOT objects to detect
- Include ALL cells where ANY part of a {object_of_interest} appears  
- Each separate {object_of_interest} should have its own group in the output
- Match the confidence scores count to the cells count for each object

Output format example for 2 objects detected:
{{
  "confidence": [(85, 90, 88), (95)],
  "cells": [(14, 15, 25), (42)]
}}
This means:
- First {object_of_interest}: in cells 14, 15, 25 with confidences 85%, 90%, 88%
- Second {object_of_interest}: in cell 42 with confidence 95%"""

    def get_required_parameters(self) -> Dict[str, str]:
        """Get required parameters for this prompt."""
        return {
            "image": "PIL image with grid overlay to analyze",
            "object_of_interest": "object to detect in the grid cells",
            "grid_factor": "size of each grid cell in pixels (default: 50)"
        }


class GeminiPrompt(BasePrompt):
    """Prompt template for detecting prominent objects with normalized bounding boxes using Gemini."""
    
    def __init__(self):
        super().__init__(
            name="gemini_object_detection",
            description="Detect all prominent items in image with normalized bounding boxes"
        )
    
    def get_system_prompt(self, **kwargs) -> str:
        """Get the system prompt for Gemini object detection."""
        return f"""You are an expert computer vision model specialized in object detection and spatial localization.

**Your Task:**
Analyze the provided image and detect ALL prominent items/objects visible in the scene.

**Detection Criteria:**
- Identify objects that are clearly visible and well-defined
- Focus on distinct, recognizable items rather than background elements
- Include both large prominent objects and smaller but clearly identifiable items
- Avoid detecting abstract concepts, textures, or overly generic regions

**Bounding Box Format:**
For each detected object, provide a bounding box in the format [ymin, xmin, ymax, xmax]:
- All coordinates must be normalized to a 0-{kwargs['normalization_factor']} scale
- ymin: Top edge of the bounding box (0 = top of image, {kwargs['normalization_factor']} = bottom)
- xmin: Left edge of the bounding box (0 = left of image, {kwargs['normalization_factor']} = right)  
- ymax: Bottom edge of the bounding box (0 = top of image, {kwargs['normalization_factor']} = bottom)
- xmax: Right edge of the bounding box (0 = left of image, {kwargs['normalization_factor']} = right)

**Quality Standards:**
- Bounding boxes should tightly fit around the object
- Ensure ymin < ymax and xmin < xmax for all boxes
- Be precise with coordinate values
- Include confidence in your detections

**Output Requirements:**
Provide a clear list of detected objects with their corresponding bounding boxes and confidence scores."""

    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt for Gemini object detection."""
        return f"""Detect all of the prominent items in the image that corresponds to {kwargs['object_of_interest']}. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-{kwargs['normalization_factor']}."""

    def get_required_parameters(self) -> Dict[str, str]:
        """Get required parameters for this prompt."""
        return {
            "image": "PIL image or image path to analyze for object detection"
        }
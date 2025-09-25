# Prompt
from typing import Any, Dict, Tuple

from ..data import Cell
from .base_prompt import BasePrompt


class SimpleDetectionPrompt(BasePrompt):
    """Prompt template for simple object detection tasks."""

    def __init__(self):
        super().__init__(
            name="simple_object_detection",
            description="Simple CoT prompt for bounding box detection on coco",
        )

    def get_system_prompt(self, **kwargs) -> str:
        """Get the system prompt for object detection."""
        return """You are a world-class visual-reasoning researcher charged with object detection tasks. You have spent two decades dissecting image, concept, and bounding-box relationships and constructing rigorous decision trees to solve object detection problems.

Your role is to determine bounding box coordinates (x, y, width, height) for objects in images. You must provide coordinates along with confidence scores based on the following rubric:

**Confidence Rubric:**
- **90-100%** - unmistakable match, zero conflicting cues. Tight bounding box, meaning there is very little background.
- **80-90%** - strong evidence, minor ambiguity in coordinates or the object of interest. Loose bounding box, meaning there is a lot of background present.
- **70-80%** - clear best choice but partial occlusion
- **60-70%** - substantial ambiguity; limited cues
- **< 60%** - highly uncertain or contradictory evidence

**Analysis Process:**
Before providing your final answer, conduct a thorough analysis where you:
- Break down your understanding of the task
- Justify how you chose the coordinates for each object
- Verify any inconsistencies in your decision-making
- Consider potential ambiguities or edge cases

**Output Format:**

Provide reasoning under <thinking>...</thinking>.

Along with your reasoning, provide your final answer as a JSON object with the following structure:
{
  "confidence": [score1, score2, ...],
  "bbox": [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
}
x
If multiple instances of the target object exist, provide coordinates for all detected instances.
"""

    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt for object detection."""
        return f"""Please identify the bounding box coordinates for {kwargs["object_of_interest"]} in this image.
Image resolution: {kwargs["resolution"]}

Provide your analysis and then output the results in JSON format:
{{
  "confidence": [score1, score2, ...],
  "bbox": [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
}}"""

    def get_required_parameters(self) -> Dict[str, str]:
        """Get required parameters for this prompt."""
        return {
            "image": "PIL image to analyze",
            "object of interest": "object to detect in the image",
        }


class SimplifiedGridCellDetectionPrompt(BasePrompt):
    """Simplified prompt for detecting all cells containing the target object."""

    def __init__(self):
        super().__init__(
            name="simplified_grid_detection",
            description="Detect all grid cells containing any part of the target object",
        )

    def get_system_prompt(self, **kwargs) -> str:
        """Get the system prompt for simplified grid cell detection."""
        return f"""You are an expert visual analyst specializing in precise object detection.

**Your Task:**
You will receive TWO images:
1. First image: Original image showing the target object(s)
2. Second image: SAME image with red grid overlay and numbered cells

**Objective:** Find EVERY cell that contains ANY part of the target object.

**IMPORTANT: Grid Layout Specification**
- Grid cells are numbered **left to right, top to bottom**
- The grid is defined as **{kwargs["grid_size"][0]} rows x {kwargs["grid_size"][1]} columns**
- Numbering works like this (example for a 4x3 grid):
    - Row 1 → cells 1, 2, 3
    - Row 2 → cells 4, 5, 6
    - Row 3 → cells 7, 8, 9
    - Row 4 → cells 10, 11, 12

**Critical Instructions:**
- The red grid lines and red numbers are ONLY for reference - ignore them as objects
- A cell should be included if even the SMALLEST part of the object touches it
- Check every cell systematically from 1 to the maximum number
- Include cells even if you only see a tiny edge, shadow, or partial view

**Analysis Process:**
1. Study the first image to understand what the target object looks like
2. In the second image, scan each numbered cell methodically
3. Mark ANY cell where you see ANY part of the object

**Confidence Scoring:**
- 90-100: Object clearly visible and fills significant portion of cell
- 70-89: Object partially visible or fills moderate portion of cell
- 50-69: Small part of object visible (edge, corner, shadow)
- 30-49: Very uncertain but possible presence
- Below 30: Do not include

**Important Reminders:**
- Include EVERY cell with ANY part of the object
- When in doubt, include the cell with lower confidence
- Better to include borderline cells than miss them
- Tiny objects still count - even if just a few pixels

**How to Avoid Mistakes:**
- DO NOT assume the grid has 3 rows x 4 columns — this is a common error.
- ALWAYS use the grid layout provided above to determine which cells the object touches.
- Double-check your cell mappings by verifying the object's position against the correct row/column structure.


**Output Format:**
{{
  "cells": [list of ALL cell numbers containing any part of the object],
  "confidence": [corresponding confidence score for each cell]
}}

Example: If object appears in cells 5, 6, 10, 11, 15:
{{
  "cells": [5, 6, 10, 11, 15],
  "confidence": [85, 90, 88, 92, 70]
}}"""

    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt for simplified detection."""
        resolution = kwargs.get("resolution")
        object_of_interest = kwargs.get("object_of_interest")
        grid_size = kwargs.get("grid_size")  # (num_rows, num_cols)
        total_cells = grid_size[0] * grid_size[1]
        pixels_per_cell = (resolution[0] / grid_size[1], resolution[1] / grid_size[0])

        return f"""I need you to find ALL cells containing "{object_of_interest}".

**Image Information:**
- Image 1: Original image without overlay
- Image 2: Same image with {grid_size[0]}x{grid_size[1]} red grid (cells numbered 1-{total_cells})
- Resolution: {resolution[0]}x{resolution[1]} pixels
- Cell size: ~{pixels_per_cell[0]:.0f}×{pixels_per_cell[1]:.0f} pixels each

**Your Task:**
Find EVERY cell where ANY part of "{object_of_interest}" appears.

**Key Points:**
- Include cells with even tiny portions of {object_of_interest}
- Red grid/numbers are NOT objects - they're just reference markers
- Check all {total_cells} cells systematically
- A {object_of_interest} spanning multiple cells should have ALL those cells listed
- Grid numbering flows left to right, top to bottom:
  - Row 1 → cells 1 to {grid_size[1]}
  - Row 2 → cells {grid_size[1] + 1} to {2 * grid_size[1]}
  - ...
  - Row {grid_size[0]} → cells {total_cells - grid_size[1] + 1} to {total_cells}

**Output Format:**
{{
  "cells": [list of cell numbers],
  "confidence": [corresponding confidence scores]
}}

The "cells" and "confidence" lists must be the same length, and in the same order.

Example:
{{
  "cells": [3, 7, 8],
  "confidence": [95, 82, 87]
}}

**Remember:** If a single {object_of_interest} covers cells 14, 15, 24, 25 — include ALL four.
Only include cells where you see the {object_of_interest}, and report a confidence score for each one."""

    def get_required_parameters(self) -> Dict[str, str]:
        """Get required parameters for this prompt."""
        return {
            "image": "PIL image with grid overlay to analyze",
            "object_of_interest": "object to detect in the grid cells",
            "grid_size": "tuple of (num_rows, num_cols) for the grid",
            "resolution": "tuple of (width, height) in pixels",
        }


class GeminiPrompt(BasePrompt):
    """Prompt template for detecting prominent objects with normalized bounding boxes using Gemini."""

    def __init__(self):
        super().__init__(
            name="gemini_object_detection",
            description="Detect all prominent items in image with normalized bounding boxes",
        )

    def get_system_prompt(self, **kwargs) -> str:
        """Get the system prompt for Gemini object detection."""
        return ""

    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt for Gemini object detection."""
        return f"""Detect all of the prominent items in the image that corresponds to {kwargs["object_of_interest"]}. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-{kwargs["normalization_factor"]}."""

    def get_required_parameters(self) -> Dict[str, str]:
        """Get required parameters for this prompt."""
        return {"image": "PIL image or image path to analyze for object detection"}


class GridCellDetectionPrompt(BasePrompt):
    """
    Simplified single-image grid detector with grouped outputs.
    Returns a Python-literal dict:
    {
      "confidence": [(conf1, conf2, ...), (conf1, ...), ...],
      "cells":      [(cell1, cell2, ...), (cell1, ...), ...]
    }
    Each inner tuple corresponds to one physical object.
    """

    def __init__(self):
        super().__init__(
            name="grid_cell_detection_simple",
            description="Basic cells/confidence per object (single grid image)",
        )

    @staticmethod
    def _require(kwargs: Dict[str, Any], key: str):
        if key not in kwargs or kwargs[key] is None:
            raise KeyError(f"Missing required parameter: {key}")
        return kwargs[key]

    @classmethod
    def _make_layout(cls, rows: int, cols: int) -> str:
        lines = []
        n = 1
        for r in range(1, rows + 1):
            row_ids = ", ".join(str(i) for i in range(n, n + cols))
            lines.append(f"Row {r} - cells {row_ids}")
            n += cols
        return "\n".join(lines)

    def _extract_geo(self, kwargs):
        W, H = self._require(kwargs, "resolution")
        rows, cols = self._require(kwargs, "grid_size")
        obj = self._require(kwargs, "object_of_interest")
        return {
            "rows": rows,
            "cols": cols,
            "total": rows * cols,
            "W": W,
            "H": H,
            "layout": self._make_layout(rows, cols),
        }, obj

    def get_system_prompt(self, **kwargs) -> str:
        """Get the system prompt for grid cell detection."""
        geo, obj = self._extract_geo(kwargs)
        return f"""You are a visual reasoning model analyzing grid-overlay images.
        **Task:** Detect all instances of "{obj}" and report which grid cells each instance touches.

        **Output Format:**
        Return a Python dict literal (NOT JSON) with:
        {{
          "confidence": [(c11, c12, ...), (c21, ...), ...],
          "cells":      [(id11, id12, ...), (id21, ...), ...]
        }}
        - One tuple per physical {obj}, with matching lengths for confidence and cells.
        - If no objects found: {{"confidence": [], "cells": []}}

        **Grid Details:**
        - Image size: {geo["W"]}×{geo["H"]} px
        - Grid: {geo["rows"]} rows × {geo["cols"]} cols = {geo["total"]} cells
        - Layout:
        {geo["layout"]}

        **Instructions:**
        - Use red grid lines and numerals to determine cells.
        - Include all cells an object touches, even partially.
        - Group cells by object (e.g., if one {obj} spans 4, 5, list as (4, 5)).
        - Sort cells within each group, no duplicates.
        - Confidence Scoring (percentage of object in each cell):
        - 90-100: Most of the cell contains the object
        - 80-89: About half the cell contains the object
        - 70-79: Substantial portion but less than half
        - 60-69: Small edge or corner of object
        - Below 60: Too uncertain - omit this cell

        Provide reasoning under <thinking>...</thinking>."""

    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt for grid cell detection."""
        geo, obj = self._extract_geo(kwargs)
        return f"""Find all instances of "{obj}" in this grid image.

        **Grid Info:**
        - Dimensions: {geo["rows"]} rows × {geo["cols"]} columns = {geo["total"]} cells
        - Resolution: {geo["W"]} × {geo["H"]} pixels

        <thinking>Reasoning for your answer</thinking>

        Return a Python dict:
        {{
          "confidence": [(conf1, conf2, ...), ...],
          "cells": [(cell1, cell2, ...), ...]
        }}"""

    def get_required_parameters(self) -> Dict[str, str]:
        return {
            "image": "PIL image with grid overlay",
            "object_of_interest": "target object",
            "resolution": "(W, H) pixels of the grid image",
            "grid_size": "(rows, cols)",
        }

class BboxDetectionWithGridCellPrompt(BasePrompt):
    """
    Bounding box detector using grid overlay for visual grounding.
    Returns a Python-literal dict (not JSON):
    {
      "confidence": [conf1, conf2, ...],
      "bbox": [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
    }
    Each bbox is for one physical object instance.
    """

    def __init__(self):
        super().__init__(
            name="bbox_detection_with_grid",
            description="Bounding box detection using grid overlay for precise localization",
        )

    @staticmethod
    def _require(kwargs: Dict[str, Any], key: str):
        if key not in kwargs or kwargs[key] is None:
            raise KeyError(f"Missing required parameter: {key}")
        return kwargs[key]

    @staticmethod
    def _make_geo_from_table(
        table: Dict[int, Cell], rows: int, cols: int, W: int, H: int
    ):
        """Extract geometry information from cell lookup table."""
        # Convert all cells to (x, y, w, h) format using to_tuple()
        cell_tuples = {}
        for cell_id, cell in table.items():
            cell_tuples[cell_id] = cell.to_tuple()

        return {
            "rows": rows,
            "cols": cols,
            "total": rows * cols,
            "W": W,
            "H": H,
            "cell_tuples": cell_tuples,
        }

    def _create_ascii_cell_table(self, geo: Dict[str, Any]) -> str:
        """Create an ASCII table showing cell IDs and their pixel coordinates."""
        lines = []
        
        # Header
        lines.append("CELL REFERENCE TABLE:")
        lines.append("┌──────┬────────────┬────────────┬─────────┬─────────┬──────────────┐")
        lines.append("│ CELL │    X MIN   │    X MAX   │  Y MIN  │  Y MAX  │    CENTER    │")
        lines.append("├──────┼────────────┼────────────┼─────────┼─────────┼──────────────┤")
        
        # Data rows
        for cell_id in sorted(geo["cell_tuples"].keys()):
            x, y, w, h = geo["cell_tuples"][cell_id]
            x_min, x_max = x, x + w - 1
            y_min, y_max = y, y + h - 1
            center_x = x + w // 2
            center_y = y + h // 2
            
            lines.append(f"│ {cell_id:4d} │ {x_min:10d} │ {x_max:10d} │ {y_min:7d} │ {y_max:7d} │ ({center_x:3d}, {center_y:3d}) │")
        
        lines.append("└──────┴────────────┴────────────┴─────────┴─────────┴──────────────┘")
        return "\n".join(lines)

    def _create_ascii_grid_layout(self, geo: Dict[str, Any]) -> str:
        """Create visual ASCII grid layout showing cell positions."""
        lines = []
        lines.append(f"\nGRID LAYOUT ({geo['rows']}×{geo['cols']}):")
        lines.append("┌" + "─" * (geo['cols'] * 6 - 1) + "┐")
        
        cell_id = 1
        for r in range(geo["rows"]):
            row_parts = []
            for c in range(geo["cols"]):
                if c == 0:
                    row_parts.append(f"│ {cell_id:3d} ")
                else:
                    row_parts.append(f"│ {cell_id:3d} ")
                cell_id += 1
            row_parts.append("│")
            lines.append("".join(row_parts))
            
            if r < geo["rows"] - 1:
                lines.append("├" + "─────┼" * (geo['cols'] - 1) + "─────┤")
        
        lines.append("└" + "─" * (geo['cols'] * 6 - 1) + "┘")
        return "\n".join(lines)

    def _extract_geo(self, kwargs):
        W, H = self._require(kwargs, "resolution")
        rows, cols = self._require(kwargs, "grid_size")
        obj = self._require(kwargs, "object_of_interest")
        table = self._require(kwargs, "cell_lookup")
        geo = self._make_geo_from_table(table, int(rows), int(cols), int(W), int(H))
        return geo, str(obj)

    def get_system_prompt(self, **kwargs) -> str:
        geo, obj = self._extract_geo(kwargs)
        rows, cols, W, H = geo["rows"], geo["cols"], geo["W"], geo["H"]

        # Create ASCII table for clear cell reference
        ascii_table = self._create_ascii_cell_table(geo)
        grid_layout = self._create_ascii_grid_layout(geo)

        return f"""DETECT "{obj}" in {W}×{H}px image with {rows}×{cols} grid.

**REQUIRED OUTPUT FORMAT (EXACT):**
{{
"confidence": [conf1, conf2, ...],
"bbox": [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
}}

{ascii_table}

{grid_layout}

**DETECTION RULES:**
• Only detect clearly visible instances - NO pattern completion
• Use grid cells as spatial reference for precise edge location
• Each bbox: x=left, y=top, w=width, h=height (integers only)
• Coordinates clamped to [0,{W-1}] × [0,{H-1}]

**CONFIDENCE SCORING:**
• 90-100%: Unmistakable, tight bbox, clear edges
• 80-89%: Strong evidence, minor ambiguity, loose bbox
• 70-79%: Partially occluded but clearly identifiable
• 60-69%: Substantial uncertainty, unclear boundaries
• <60%: Discard detection

**EDGE AMBIGUITY:** When boundaries fall within cells, snap to nearest visible edge, then cell boundary if unclear.

**NO DETECTIONS FOUND:** Return {{"confidence": [], "bbox": []}}

Provide reasoning under <thinking>...</thinking>.

**OUTPUT:** Python dict only. Zero explanations.
        """

    def get_user_prompt(self, **kwargs) -> str:
        geo, obj = self._extract_geo(kwargs)
        rows, cols, W, H = geo["rows"], geo["cols"], geo["W"], geo["H"]

        # Create ASCII table for user prompt
        ascii_table = self._create_ascii_cell_table(geo)
        grid_layout = self._create_ascii_grid_layout(geo)

        return f"""DETECT: "{obj}" in {W}×{H}px image with {rows}×{cols} grid

{ascii_table}

{grid_layout}

RETURN ONLY:
{{
"confidence": [score1, score2, ...],
"bbox": [(x, y, w, h), (x, y, w, h), ...]
}}

Each bbox represents one physical object instance. Use the cell reference table above to determine precise pixel coordinates.
        """

    def get_required_parameters(self) -> Dict[str, str]:
        return {
            "image": "PIL image with grid overlay",
            "object_of_interest": "target object to detect",
            "resolution": "(W, H) pixels of the image",
            "grid_size": "(rows, cols) of the grid",
            "cell_lookup": "Dict mapping cell IDs to Cell objects with boundaries",
        }

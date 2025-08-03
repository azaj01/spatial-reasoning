# Prompt
from typing import Any, Dict, Tuple

from ..data import Cell
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


class SimplifiedGridCellDetectionPrompt(BasePrompt):
    """Simplified prompt for detecting all cells containing the target object."""
    
    def __init__(self):
        super().__init__(
            name="simplified_grid_detection",
            description="Detect all grid cells containing any part of the target object"
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
- The grid is defined as **{kwargs['grid_size'][0]} rows x {kwargs['grid_size'][1]} columns**
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
        resolution = kwargs.get('resolution')
        object_of_interest = kwargs.get('object_of_interest')
        grid_size = kwargs.get('grid_size')  # (num_rows, num_cols)
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
  - Row 2 → cells {grid_size[1]+1} to {2*grid_size[1]}
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
            "resolution": "tuple of (width, height) in pixels"
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
        return ""

    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt for Gemini object detection."""
        return f"""Detect all of the prominent items in the image that corresponds to {kwargs['object_of_interest']}. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-{kwargs['normalization_factor']}."""

    def get_required_parameters(self) -> Dict[str, str]:
        """Get required parameters for this prompt."""
        return {
            "image": "PIL image or image path to analyze for object detection"
        }


class GridCellDetectionPrompt(BasePrompt):
    """
    Single-image grid detector with grouped outputs.
    Returns a Python-literal dict (not JSON):
    {
      "confidence": [(conf1, conf2, ...), (conf1, ...), ...],
      "cells":      [(cell1, cell2, ...), (cell1, ...), ...]
    }
    Each inner tuple corresponds to one physical object. Tuple lengths must match per group.
    """

    def __init__(self):
        super().__init__(
            name="grid_cell_detection_v2",
            description="Grouped cells/confidence per object (single grid image)"
        )
    # ---------- geometry helpers ----------

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

    @classmethod
    def _make_geometry(cls, rows: int, cols: int, W: int, H: int) -> Dict[str, Any]:
        cw, ch = W / cols, H / rows
        return {
            "rows": rows,
            "cols": cols,
            "total": rows * cols,
            "W": W,
            "H": H,
            "cell_w": cw,
            "cell_h": ch,
            "layout": cls._make_layout(rows, cols),
            "formula": (
                f"row = floor(cy / {ch:.6f}) + 1\n"
                f"col = floor(cx / {cw:.6f}) + 1\n"
                f"id  = (row-1)*{cols} + col   # row-major, 1-indexed"
            ),
        }
    
    @staticmethod
    def _make_geo_from_table(table: Dict[int, Cell], rows: int, cols: int, W: int, H: int):
        # Per-cell boxes in row-major order
        def cell(r, c):
            return table[r*cols + c + 1]

        # Row ranges: min top, max bottom across all cols of that row
        row_ranges = []
        for r in range(rows):
            tops = [cell(r,c).top for c in range(cols)]
            bottoms = [cell(r,c).bottom for c in range(cols)]
            row_ranges.append((min(tops), max(bottoms)))

        # Col ranges: min left, max right across all rows of that col
        col_ranges = []
        for c in range(cols):
            lefts = [cell(r,c).left for r in range(rows)]
            rights = [cell(r,c).right for r in range(rows)]
            col_ranges.append((min(lefts), max(rights)))

        # Nominal cell size (may differ for last row/col; used only for display)
        cw = sum(col_ranges[i][1] - col_ranges[i][0] for i in range(cols)) / cols
        ch = sum(row_ranges[i][1] - row_ranges[i][0] for i in range(rows)) / rows

        # Pretty layout
        layout_lines = []
        n = 1
        for r in range(1, rows+1):
            row_ids = ", ".join(str(i) for i in range(n, n+cols))
            layout_lines.append(f"Row {r} – cells {row_ids}")
            n += cols

        # Pretty ranges
        row_text = "\n".join(
            f"Row {r+1}: y ∈ [{y0}, {y1})" if r < rows-1 else f"Row {r+1}: y ∈ [{y0}, {y1}]"
            for r, (y0, y1) in enumerate(row_ranges)
        )
        col_text = "\n".join(
            f"Col {c+1}: x ∈ [{x0}, {x1})" if c < cols-1 else f"Col {c+1}: x ∈ [{x0}, {x1}]"
            for c, (x0, x1) in enumerate(col_ranges)
        )

        return {
            "rows": rows, "cols": cols, "total": rows*cols,
            "W": W, "H": H, "cell_w": cw, "cell_h": ch,
            "row_ranges": row_ranges, "col_ranges": col_ranges,
            "row_text": row_text, "col_text": col_text,
            "layout": "\n".join(layout_lines),
        }

    def _extract_geo(self, kwargs):
        W, H = self._require(kwargs, "resolution")
        rows, cols = self._require(kwargs, "grid_size")
        obj = self._require(kwargs, "object_of_interest")
        table = self._require(kwargs, "cell_lookup")
        # Always trust the image we drew on:
        geo = self._make_geo_from_table(table, int(rows), int(cols), int(W), int(H))
        return geo, str(obj)

    def get_system_prompt(self, **kwargs) -> str:
        """Get the system prompt for grid cell detection."""
        geo, obj = self._extract_geo(kwargs)

        return f"""You are a meticulous visual reasoning model analyzing images with grid overlays.
        **Your Task:**
        Detect every instance of "{obj}" and report which grid **cell IDs** each instance touches.


        **Output Format:**
        Return a Python dict literal (NOT JSON) with this exact structure:
        {{
        "confidence": [(c11, c12, ...), (c21, ...), ...],
        "cells":      [(id11, id12, ...), (id21, ...), ...]
        }}
        Each inner tuple represents one physical {obj}. Tuple lengths must match between confidence and cells.

        **Grid facts (row-major, 1-indexed):**
        - Image size: {geo['W']}×{geo['H']} px
        - Grid: {geo['rows']} rows × {geo['cols']} cols = {geo['total']} cells
        - Layout:
        {geo['layout']}

        **Row y-ranges**
        {geo['row_text']}

        **Column x-ranges**
        {geo['col_text']}
        
        **Note:** Do NOT re-derive widths/heights; use these ranges exactly.

        Dual-evidence policy
        1) **VISUAL first.** Read red grid **lines** and **numerals** directly from the image to determine cells contain the object(s).
        2) **CROSS-CHECK** with the tabulated row/column ranges. If a visually estimated boundary differs by ≤ 10 px from a tabulated boundary, SNAP to the tabulated boundary.
        3) Numerals in the image are for **ID verification only**; never treated as objects.

        **Important Instructions:**
        - IGNORE red grid lines and numbers (pixels with R>200, G<60, B<60) - they are NOT objects
        - Find every distinct {obj} visible under/through the grid
        - Include ALL cells that an object touches, even partially
        - Group cells by object: if one {obj} spans cells 5, 6, 10, 11, list them together
        - If a red numeral overlaps an object, ignore those red pixels; use nearest non-red object pixels.

        **Confidence Scoring (percentage of object in each cell):**
        - 90-100: Most of the cell contains the object
        - 80-89: About half the cell contains the object
        - 70-79: Substantial portion but less than half
        - 60-69: Small edge or corner of object
        - Below 60: Too uncertain - omit this cell

        **Output Rules:**
        - One tuple group per physical object
        - Sort cells within each group in ascending order
        - No duplicate cells within a group
        - Drop cells outside range [1, {geo['total']}] or with confidence < 60
        - If no objects found: {{"confidence": [], "cells": []}}
        - Use tuples (), not lists []
        - Output ONLY the dict literal, no other text"""

    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt for grid cell detection."""
        geo, obj = self._extract_geo(kwargs)

        return f"""Find all instances of "{obj}" in this grid image.    

    **Grid Information:**
    - Dimensions: {geo['rows']} rows × {geo['cols']} columns = {geo['total']} cells
    - Resolution: {geo['W']} × {geo['H']} pixels  
    - Cell size: {geo['cell_w']:.1f} × {geo['cell_h']:.1f} pixels
    - Numbering: 1 to {geo['total']} (left-to-right, top-to-bottom)

    **Your Task:**
    Identify which numbered cells contain any part of "{obj}".

    Return a Python dict with grouped detections:
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
            description="Bounding box detection using grid overlay for precise localization"
        )

    # ---------- geometry helpers ----------

    @staticmethod
    def _require(kwargs: Dict[str, Any], key: str):
        if key not in kwargs or kwargs[key] is None:
            raise KeyError(f"Missing required parameter: {key}")
        return kwargs[key]

    @staticmethod
    def _make_geo_from_table(table: Dict[int, Cell], rows: int, cols: int, W: int, H: int):
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
            "cell_tuples": cell_tuples
        }

    def _format_grid_visualization(self, geo: Dict[str, Any]) -> str:
        """Create visual grid representation."""
        lines = []
        lines.append("```")
        lines.append("Cells:")
        
        cell_id = 1
        for r in range(geo['rows']):
            row_cells = []
            for c in range(geo['cols']):
                row_cells.append(f"[{cell_id:2d}]")
                cell_id += 1
            lines.append(" ".join(row_cells))
            if r < geo['rows'] - 1:  # Add blank line between rows
                lines.append("")
        
        lines.append("```")
        return "\n".join(lines)

    def _format_cell_table(self, geo: Dict[str, Any]) -> str:
        """Create detailed cell coordinate table."""
        lines = []
        lines.append("```")
        lines.append("Cell | x      | y      | width  | height | center")
        lines.append("-----|--------|--------|--------|--------|--------")
        
        for cell_id in sorted(geo['cell_tuples'].keys()):
            x, y, w, h = geo['cell_tuples'][cell_id]
            center_x = x + w // 2
            center_y = y + h // 2
            lines.append(
                f"{cell_id:4d} | {x:6d} | {y:6d} | {w:6d} | {h:6d} | ({center_x:3d},{center_y:3d})"
            )
        
        lines.append("```")
        return "\n".join(lines)

    def _extract_geo(self, kwargs):
        W, H = self._require(kwargs, "resolution")
        rows, cols = self._require(kwargs, "grid_size")
        obj = self._require(kwargs, "object_of_interest")
        table = self._require(kwargs, "cell_lookup")
        geo = self._make_geo_from_table(table, int(rows), int(cols), int(W), int(H))
        return geo, str(obj)

    def get_system_prompt(self, **kwargs) -> str:
        """Get the system prompt for bbox detection with grid."""
        geo, obj = self._extract_geo(kwargs)
        
        grid_viz = self._format_grid_visualization(geo)
        cell_table = self._format_cell_table(geo)

        return f"""You are a grid-based visual detection model. YOU MUST use the grid overlay to locate objects.

**CRITICAL METHODOLOGY - YOU MUST FOLLOW THIS PROCESS:**
1. FIRST identify and explicitly state which grid cells contain the target object
2. Look up those cell numbers in the reference table to get pixel ranges
3. ONLY THEN determine exact boundaries within those cells
4. Never skip the grid analysis - it is MANDATORY

**Grid Information:**
- Image size: {geo['W']}×{geo['H']} pixels
- Grid: {geo['rows']} rows × {geo['cols']} columns = {geo['total']} cells

**Grid Layout:**
{grid_viz}

**Cell Reference Table:**
{cell_table}

**Example Reasoning Process (YOU MUST REASON LIKE THIS):**

Example 1 - Detecting an eye:
"Looking at the grid overlay, I can see the right eye spans cells 8 and 9.
Cell 8 covers x:[200-299], y:[100-199]
Cell 9 covers x:[300-399], y:[100-199]
The eye starts at x=245 in cell 8 and extends to x=340 in cell 9.
The vertical bounds are y=130 to y=170 within the cell ranges.
Therefore: bbox = (245, 130, 95, 40), confidence = 95"

Example 2 - Detecting a complex object:
"Examining the grid, the object is located in cells 23, 24, 31, and 32.
Cell 23: x:[300-399], y:[300-399]
Cell 24: x:[400-499], y:[300-399]
Cell 31: x:[300-399], y:[400-499]
Cell 32: x:[400-499], y:[400-499]
The object starts at x=320 in cell 23 and extend to x=480 in cell 24.
Vertically, it spans from y=380 in cells 23-24 to y=420 in cells 31-32.
Therefore: bbox = (320, 380, 160, 40), confidence = 90"

**Your Task:**
Detect every instance of "{obj}" using the grid-based process above.

**Output Format:**
Return a Python dict literal (NOT JSON) with this exact structure:
{{
  "confidence": [conf1, conf2, ...],
  "bbox": [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
}}
Each bbox represents one physical {obj}. Lists must have matching lengths.

**Detection Process:**

1. **Grid Cell Identification (MANDATORY):**
   - Scan grid systematically
   - State explicitly: "I see {obj} in cells X, Y, Z"
   - List ALL cells the object touches

2. **Cell Lookup (MANDATORY):**
   - State the pixel ranges for each identified cell
   - Example: "Cell 15 covers x:[300-399], y:[200-299]"

3. **Boundary Determination:**
   - Find exact object edges WITHIN the cell ranges
   - x = leftmost object pixel
   - y = topmost object pixel
   - width = rightmost - leftmost + 1
   - height = bottommost - topmost + 1

4. **Quality Checks:**
   ✓ Did I identify grid cells first?
   ✓ Did I look up cell pixel ranges?
   ✓ Are bounds within [0,{geo['W']-1}] × [0,{geo['H']-1}]?
   ✓ Complete object captured?

**Confidence Scoring:**
- 90-100: Clear boundaries, minimal occlusion
- 80-89: Slight edge uncertainty
- 70-79: Partial occlusion
- <70: Significant uncertainty

**REMEMBER:**
- The red grid lines and numbers are reference markers, not part of objects
- You MUST state grid cells before determining coordinates
- If you don't mention grid cells, your analysis is incomplete"""


    def get_user_prompt(self, **kwargs) -> str:
        """Get the user prompt for bbox detection."""
        geo, obj = self._extract_geo(kwargs)

        return f"""Find all instances of "{obj}" using the MANDATORY grid-based process.

**Grid Reference:**
- {geo['rows']}×{geo['cols']} grid overlaid on {geo['W']}×{geo['H']} pixel image
- Red grid lines and numbers are for reference only

**REQUIRED ANALYSIS STEPS:**
1. State which grid cells contain each {obj}
2. Look up pixel ranges for those cells
3. Determine exact boundaries within the cells

**Example of REQUIRED reasoning format:**
"I'm examining the grid to find {obj}. 
I can see the first {obj} in cells 12 and 13.
Cell 12 covers x:[200-299], y:[200-299]
Cell 13 covers x:[300-399], y:[200-299]
The {obj} starts at x=250 in cell 12 and extends to x=350 in cell 13.
Vertically, it spans y=220 to y=280.
Therefore: bbox = (250, 220, 100, 60)"

**Your response MUST include this reasoning BEFORE the final output.**

Return results as:
{{
  "confidence": [score1, score2, ...],
  "bbox": [(x1, y1, w1, h1), ...]
}}

Begin by stating: "I will use the grid overlay to locate all {obj}..." """

    def get_required_parameters(self) -> Dict[str, str]:
        return {
            "image": "PIL image with grid overlay",
            "object_of_interest": "target object to detect",
            "resolution": "(W, H) pixels of the image",
            "grid_size": "(rows, cols) of the grid",
            "cell_lookup": "Dict mapping cell IDs to Cell objects with boundaries",
        }

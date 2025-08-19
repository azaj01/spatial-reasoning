from abc import ABC, abstractmethod
from itertools import product
from typing import Dict, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..agents import BaseAgent
from ..data import Cell


class BaseTask(ABC):
    """Abstract base class for all tasks."""

    def __init__(self, agent: BaseAgent, **kwargs):
        self.agent = agent
        self.kwargs = kwargs

    @abstractmethod
    def execute(self, **kwargs):
        """Run task executor"""
        pass

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
        if num_rows + num_cols < 2:
            raise ValueError(f"Too few rows ({num_rows}) and columns ({num_cols}).")

        # --- to PIL ---
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            arr = (
                image.detach()
                .cpu()
                .permute(1, 2, 0)
                .mul(255)
                .clamp(0, 255)
                .to(torch.uint8)
                .numpy()
            )
            pil = Image.fromarray(arr)
        else:
            pil = image.copy()

        original_image_width, original_image_height = pil.size
        cell_width, cell_height = (
            original_image_width // num_cols,
            original_image_height // num_rows,
        )

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
            right, bottom = (
                min(left + cell_width, original_image_width),
                min(top + cell_height, original_image_height),
            )
            cell = Cell(n, left, top, right, bottom)
            table[n] = cell
            draw.text(
                ((cell.left + cell.right) // 2, (cell.top + cell.bottom) // 2),
                str(n),
                fill=color,
                font=font,
                anchor="mm",
            )

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
        confidence_threshold: float = 0.65,
    ):
        """
        Crop image using top-k most confident cell groups from `scores_grid`.
        Returns one cropped image with padding, centered on the selected cells.
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
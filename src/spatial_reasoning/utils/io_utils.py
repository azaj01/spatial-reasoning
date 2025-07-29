import ast
import json
import re
import time
from io import BytesIO

import requests
from ..data import Cell
from PIL import Image


def download_image(url: str) -> Image.Image:
    """Download an image from a url and return a PIL Image (RGB)"""
    assert url.startswith("http"), "URL must start with http"
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def parse_detection_output(output_text):
    """forgiving parse of dicts/lists, handles outer quotes, comments & code-fences."""
    cleaned = output_text.strip()

    # 0) strip a single matching leading+trailing quote if present
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
        cleaned = cleaned[1:-1]

    # 1) strip code fences
    cleaned = re.sub(r"^```.*?\n|```$", "", cleaned, flags=re.S)
    # 2) drop any // comments
    cleaned = re.sub(r"//.*", "", cleaned)
    # 3) normalize lone (70) → [70]
    cleaned = re.sub(r"\(\s*(\d+)\s*\)", r"[\1]", cleaned)

    # 4) find first '{' or '[' and walk to its matching brace/bracket
    for i, ch in enumerate(cleaned):
        if ch in "{[":
            start, open_ch = i, ch
            break
    else:
        return None

    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    for j, ch in enumerate(cleaned[start:], start):
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                fragment = cleaned[start: j+1]
                break
    else:
        return None

    # 5) literal_eval + tuple→list via JSON round‑trip
    try:
        obj = ast.literal_eval(fragment)
    except Exception:
        return None
    return json.loads(json.dumps(obj))


def get_original_bounding_box(
    cropped_bounding_boxs: list[Cell],
    crop_origin: tuple[int, int],
) -> Cell:
    """
    Map a bounding box from a cropped image back to the original image.

    Args:
        cropped_bounding_box: Cell in the cropped image
        crop_origin: (x_offset, y_offset) top-left corner of the crop in original image

    Returns:
        Bounding box (x, y, w, h) in original image coordinates
    """
    restored_bboxs = []
    for cropped_bounding_box in cropped_bounding_boxs:
        x = cropped_bounding_box.left
        y = cropped_bounding_box.top
        w = cropped_bounding_box.right - cropped_bounding_box.left
        h = cropped_bounding_box.bottom - cropped_bounding_box.top
        crop_x, crop_y = crop_origin

        x_orig = x + crop_x
        y_orig = y + crop_y

        restored_bboxs.append(Cell(
            id=cropped_bounding_box.id,
            left=x_orig,
            top=y_orig,
            right=x_orig + w,
            bottom=y_orig + h,
        ))
    return restored_bboxs

def convert_list_of_cells_to_list_of_bboxes(list_of_cells: list[Cell]) -> list[tuple[int, int, int, int]]:
    """ Convert list of cells to list of bboxes """
    return [
        (cell.left, cell.top, cell.right - cell.left, cell.bottom - cell.top)
        for cell in list_of_cells
    ]
def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S")
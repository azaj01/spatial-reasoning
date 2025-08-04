import ast
import json
import re
import time
from io import BytesIO

import requests
from PIL import Image

from ..data import Cell


def download_image(url: str) -> Image.Image:
    """Download an image from a url and return a PIL Image (RGB)"""
    assert url.startswith("http"), "URL must start with http"
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def get_original_bounding_box(
    cropped_bounding_boxs: list[Cell],
    crop_origin: tuple[int, int],
) -> list[Cell]:
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

        restored_bboxs.append(
            Cell(
                id=cropped_bounding_box.id,
                left=x_orig,
                top=y_orig,
                right=x_orig + w,
                bottom=y_orig + h,
            )
        )
    return restored_bboxs


def convert_list_of_cells_to_list_of_bboxes(
    list_of_cells: list[Cell],
) -> list[tuple[int, int, int, int]]:
    """Convert list of cells to list of bboxes"""
    return [
        (cell.left, cell.top, cell.right - cell.left, cell.bottom - cell.top)
        for cell in list_of_cells
    ]


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


# Helper functions to parse any string to a JSON object.
# Huge kudos to Claude code for automating this for me!!!


def parse_detection_output(output_text):
    """Forgiving parse of dicts/lists, handles outer quotes, comments & code-fences."""
    if not output_text:
        return None

    cleaned = output_text.strip()

    # 0) Strip a single matching leading+trailing quote if present
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
        cleaned = cleaned[1:-1]

    # 1) Extract content from code fences if present
    code_fence_match = re.search(
        r"```(?:json|python)?\s*\n(.*?)\n```", cleaned, re.DOTALL
    )
    if code_fence_match:
        fence_content = code_fence_match.group(1).strip()
        # Try to parse the fence content first
        result = _try_parse_json_or_python(fence_content)
        if result is not None:
            return result

    # 2) Strip code fences for further processing
    cleaned = re.sub(r"^```.*?\n|```$", "", cleaned, flags=re.S)

    # 3) Drop any // comments
    cleaned = re.sub(r"//.*", "", cleaned)

    # 4) Normalize lone (70) → [70]
    cleaned = re.sub(r"\(\s*(\d+)\s*\)", r"[\1]", cleaned)

    # 5) Find and extract JSON/Python dict/list with proper string handling
    candidates = _extract_all_structures(cleaned)

    # Try each candidate (prefer dicts over lists)
    dicts = [c for c in candidates if c["text"].strip().startswith("{")]
    lists = [c for c in candidates if c["text"].strip().startswith("[")]

    # Try dicts first
    for candidate in dicts:
        result = _try_parse_json_or_python(candidate["text"])
        if result is not None:
            return result

    # Then try lists
    for candidate in lists:
        result = _try_parse_json_or_python(candidate["text"])
        if result is not None:
            return result

    return None


def _try_parse_json_or_python(text):
    """Try to parse text as JSON or Python literal."""
    if not text:
        return None

    # Try JSON first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try Python literal_eval
    try:
        obj = ast.literal_eval(text)
        # Convert to JSON-serializable format
        return json.loads(json.dumps(obj))
    except (ValueError, SyntaxError):
        pass

    # Try converting Python syntax to JSON
    converted = _convert_python_to_json(text)
    if converted != text:
        try:
            return json.loads(converted)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _convert_python_to_json(text):
    """Convert common Python syntax to JSON syntax."""
    # Replace Python constants
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    # Convert single quotes to double quotes (simple approach)
    # This is imperfect but works for most cases
    text = re.sub(r"'([^']*)'", r'"\1"', text)

    return text


def _extract_all_structures(text):
    """Extract all dict/list structures with proper string and comment handling."""
    candidates = []
    i = 0

    while i < len(text):
        # Skip whitespace
        while i < len(text) and text[i].isspace():
            i += 1

        if i >= len(text):
            break

        # Look for start of structure
        if text[i] in "{[":
            start_pos = i
            end_pos = _find_matching_bracket(text, i)
            if end_pos is not None:
                candidates.append(
                    {
                        "text": text[start_pos : end_pos + 1],
                        "start": start_pos,
                        "end": end_pos,
                    }
                )
                i = end_pos + 1
            else:
                i += 1
        else:
            i += 1

    return candidates


def _find_matching_bracket(text, start_idx):
    """Find matching bracket with proper string handling."""
    if start_idx >= len(text):
        return None

    open_ch = text[start_idx]
    if open_ch == "{":
        close_ch = "}"
    elif open_ch == "[":
        close_ch = "]"
    else:
        return None

    depth = 0
    in_string = False
    string_char = None
    escape_next = False

    for i in range(start_idx, len(text)):
        ch = text[i]

        # Handle escape sequences
        if escape_next:
            escape_next = False
            continue

        if ch == "\\" and in_string:
            escape_next = True
            continue

        # Handle strings (both single and double quotes)
        if ch in "\"'":
            if not in_string:
                in_string = True
                string_char = ch
            elif ch == string_char:
                in_string = False
                string_char = None

        # Only count brackets when not in a string
        if not in_string:
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return i

    return None

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))


def resize_image(img: Image.Image, max_size: tuple = (1024, 1024)) -> Image.Image:
    """Resize image while maintaining aspect ratio."""
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img


def crop_center(img: Image.Image, crop_size: tuple) -> Image.Image:
    """Crop image from center."""
    width, height = img.size
    crop_width, crop_height = crop_size

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    return img.crop((left, top, right, bottom))


def zoom_in_on_object_of_interest(
    image: Image.Image,
    mask: Image.Image,
    offset: int,
    draw_box: bool = True,
):
    """
    Zoom in on the object of interest in the image.
    """
    # Get the bounding box of the object of interest
    mask_array = np.array(mask).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop image
    cropped_image = image.crop((x - offset, y - offset, x + w + offset, y + h + offset))

    # Draw red box using PIL
    if draw_box:
        draw = ImageDraw.Draw(cropped_image)
        draw.rectangle(
            [offset, offset, w + offset, h + offset], outline=(255, 0, 0), width=2
        )

    return cropped_image, (x, y, w, h)


def draw_bbox_on_image(image: Image.Image, bbox: list):
    x, y, w, h = bbox
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=10)
    return annotated_image


def calculate_iou(
    bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]
) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Get bottom-right corners
    x1_br, y1_br = x1 + w1, y1 + h1
    x2_br, y2_br = x2 + w2, y2 + h2

    # Compute coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1_br, x2_br)
    y_bottom = min(y1_br, y2_br)

    # No intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union


def nms(boxes: list[np.ndarray], scores: list[float], nms_threshold: float) -> dict:
    """
    Non-maximum suppression to remove overlapping bounding boxes
    """
    # Sort boxes by score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    boxes = [boxes[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]

    # Apply NMS
    pruned_boxes = []
    pruned_scores = []
    suppressed = [False] * len(boxes)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue

        pruned_boxes.append(boxes[i])
        pruned_scores.append(scores[i])

        # Suppress overlapping boxes with lower scores
        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > nms_threshold:
                suppressed[j] = True  # Suppress the lower-confidence box

    return {"boxes": pruned_boxes, "scores": pruned_scores}

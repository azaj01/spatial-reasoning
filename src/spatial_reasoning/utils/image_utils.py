import base64
import math
from io import BytesIO
from typing import List, Tuple

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

def calculate_iou_with_offset(
    bbox_pred: tuple[int, int, int, int],
    bbox_gt: tuple[int, int, int, int],
    image_size: tuple[int, int],
    offset_pct: float = 0.02,
) -> float:
    """
    IoU with tolerance. Ground-truth box is expanded by offset_pct of image area.
    offset_pct=0.02 means expand area by 2% of total image area.
    This is useful because IoU is a harsh metric and is very sensitive if pred box is slightly off from gt box.
    Instead, we can smoothen the ground truth box by a certain percentage of the image area.
    """
    img_w, img_h = image_size
    offset = int((img_w * img_h * offset_pct) ** 0.5)  # offset in pixels

    x, y, w, h = bbox_gt
    x_exp = max(0, x - offset)
    y_exp = max(0, y - offset)
    w_exp = min(img_w - x_exp, w + 2 * offset)
    h_exp = min(img_h - y_exp, h + 2 * offset)

    expanded_gt = (x_exp, y_exp, w_exp, h_exp)
    return calculate_iou(bbox_pred, expanded_gt)


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


def calculate_modified_diou(bbox1, bbox2, alpha: float = 0.3, beta: float = 0.5, use_sigmoid: bool = True):
    """
    Modified Distance-IoU (DIoU) metric. Inspired from https://arxiv.org/pdf/1911.08287.
    Raw score = 1 - IoU + center_distance_penalty
    - Lower is better (0 = perfect overlap).
    - Range is [0, 2].

    center_distance_penalty = (center_distance) / (enclosing_box_diagonal), where:
    - center_distance is the squared distance (MSE) between the centers of the two boxes
    - enclosing_box_diagonal is the squared diagonal length of the smallest enclosing box containing both boxes

    Optional sigmoid normalization:
    - Converts raw score into a probability-like [0, 1] scale.
    - Formula: sigmoid(-scaled_alpha * (raw_score - 1)) * beta
      where scaled_alpha = 1 + 9 * alpha  (maps user \alpha \in [0,1] → practical range [1,10]).
    
    Low values of alpha make for a peaky curve, high values of alpha make for a flat curve.
    Low values of beta focus more on IoU while high values of beta focus more on penalty

    Why 9?
    ------
    The logistic needs a steeper slope than 0-1 to be useful. 
    In practice, k ~ 1-10 gives a good spread: k≈1 is soft; k≈10 is peaky.
    Mapping [0,1] -> [1,10] with 1 + 9*alpha keeps \alpha intuitive while covering the useful region.

    Here's a chart I made showcasing the dynamics between IoU and unscaled DIoU scores: https://imgur.com/a/zoW3XpY

    """
    # error checking
    if not bbox1 or not bbox2:
        return 0.0 if use_sigmoid else 1.0  # safe numeric default

    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # IoU calculation
    iou = calculate_iou(bbox1, bbox2)
    iou = float(iou) if iou is not None else 0.0

    # get centers
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
    d2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # calculate the normalizing factor
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1 + w1, x2 + w2)
    bottom = max(y1 + h1, y2 + h2)
    c2 = (right - left) ** 2 + (bottom - top) ** 2

    penalty = (d2 / c2) if c2 > 0 else 0.0

    # beta scaling of the penalty
    # note: do this in the logit (pre-sigmoid), not the output.
    raw_score = (1.0 - iou) + float(beta) * penalty  # in [0, 1+beta]

    if not use_sigmoid:
        return raw_score

    # mapping trick
    a = max(0.0, min(1.0, float(alpha)))
    k = 1.0 + 9.0 * a

    # apply normalization, output range is [0, 1]
    return 1.0 / (1.0 + math.exp(k * (raw_score - 1.0)))


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


def calculate_focal_diou(
    gt: Tuple[int, int, int, int],
    zoom_boxes: List[Tuple[int, int, int, int]]
) -> float:
    """
    Compute the maximum "Focal DIoU" between the ground-truth box and any zoom box.
    
    Normalization accounts for the fact that zoom windows rarely fit the object perfectly:
    - Small crops may fully contain the object but yield low DIoU.
    - Large crops may overlap with the object but get unfairly rewarded for covering lots of background.
    
    Formula:
        raw_score = DIoU / (1 + crop_area / gt_area)
        focal_diou = sqrt(raw_score)

    - Taking sqrt at the end smooths harsh DIoU penalties while keeping the area penalty active.
    - Output is in [0, 1]. Higher = better focus on object.

    Why is this useful?
    -------------------
    Max DIoU alone punishes small crops harshly. This metric rewards crops that 
    "preserve focus" on the object even if perfect overlap is impossible.

    Intuition:
    --------
    Suppose ground-truth box is tiny (50x50 pixels).
    - Crop A: Large region 500x500 covering the object -> DIoU = 0.1
        raw_score = 0.1 / (1 + 500*500 / 50*50) = 0.1 / (1 + 100) = 0.001
        focal_diou = sqrt(0.001) = 0.03
    - Crop B: Tighter 80x80 region around the object -> DIoU = 0.4
        raw_score = 0.4 / (1 + 80*80 / 50*50) = 0.4 / (1 + 2.56) = 0.112
        focal_diou = sqrt(0.112) = 0.33

    Crop B is clearly better, and focal IoU reflects it.
    """
    if not zoom_boxes or gt is None:
        return 0.0

    _, _, w_gt, h_gt = gt
    gt_area = w_gt * h_gt
    if gt_area <= 0:
        return 0.0

    scores = []
    for crop in zoom_boxes:
        diou = calculate_modified_diou(crop, gt)
        _, _, w_c, h_c = crop
        crop_area = w_c * h_c

        raw_score = diou / (1.0 + crop_area / gt_area)
        focal_diou = math.sqrt(raw_score)   # compute geometric mean
        scores.append(focal_diou)

    return max(scores) if scores else 0.0

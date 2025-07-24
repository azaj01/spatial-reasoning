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


def resize_image(
    img: Image.Image, max_size: tuple = (1024, 1024)
) -> Image.Image:
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
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop image
    cropped_image = image.crop((x - offset, y - offset, x + w + offset, y + h + offset))
    
    # Draw red box using PIL
    if draw_box:
        draw = ImageDraw.Draw(cropped_image)
        draw.rectangle([offset, offset, w + offset, h + offset], outline=(255, 0, 0), width=2)
    
    return cropped_image, (x, y, w, h)


def draw_bbox_on_image(image: Image.Image, bbox: list):
    x, y, w, h = bbox
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=10)
    return annotated_image

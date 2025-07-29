import json
import os
import time
from typing import Dict, List, Optional, Union

from agents.agent_factory import AgentFactory
from data import BaseDataset, Cell
from dotenv import load_dotenv
from PIL import Image
from tasks import (AdvancedReasoningModelTask, GeminiTask,
                   MultiAdvancedReasoningModelTask, VanillaReasoningModelTask,
                   VisionModelTask)
from utils.io_utils import (convert_list_of_cells_to_list_of_bboxes,
                            download_image, get_timestamp)

# Load environment variables
load_dotenv()


def detect(
    image_path: str,
    object_of_interest: str,
    task_type: str,
    task_kwargs: Optional[Dict] = None,
    save_outputs: bool = False,
    output_folder_path: Optional[str] = None,
    return_overlay_images: bool = True
) -> Dict[str, Union[List, float, Image.Image]]:
    """
    Main detection function that processes an image and returns bounding boxes
    for objects of interest.
    
    Args:
        image_path (str): Path to the image file or URL
        object_of_interest (str): Description of what to detect in the image
        task_type (str): Type of detection task to run. Options:
            - "advanced_reasoning_model"
            - "vanilla_reasoning_model"
            - "vision_model"
            - "gemini"
            - "multi_advanced_reasoning_model"
        task_kwargs (dict, optional): Additional parameters for the task
            Example: {"nms_threshold": 0.7, "multiple_predictions": True}
        save_outputs (bool): Whether to save output files to disk
        output_folder_path (str, optional): Where to save outputs if save_outputs=True
        return_overlay_images (bool): Whether to return overlay images in the result
    
    Returns:
        dict: Dictionary containing:
            - "bboxs": List of bounding boxes [[x1, y1, x2, y2], ...]
            - "visualized_image": PIL Image with bounding boxes drawn
            - "original_image": Original PIL Image
            - "overlay_images": List of overlay images (if any)
            - "total_time": Processing time in seconds
            - "object_of_interest": The object that was searched for
            - "task_type": The task type that was used
            - "task_kwargs": The task parameters that were used
    """
    
    # Initialize task_kwargs if not provided
    if task_kwargs is None:
        task_kwargs = {}
    
    # Start timing
    start_time = time.perf_counter()
    
    # Load the image
    print(f"Loading image from: {image_path}")
    if image_path.startswith("http"):
        image = download_image(image_path)
    else:
        image = Image.open(image_path).convert("RGB")
    
    print(f"Image loaded successfully. Size: {image.size}")
    
    # Create the appropriate agent based on task type
    print(f"Initializing task type: {task_type}")
    
    if task_type in ["advanced_reasoning_model", "vanilla_reasoning_model", "vision_model", "multi_advanced_reasoning_model"]:
        agent = AgentFactory.create_agent(model="o4-mini", platform_name="openai")
    elif task_type == "gemini":
        agent = AgentFactory.create_agent(model="gemini-2.5-flash", platform_name="gemini")
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Create the appropriate task
    if task_type == "advanced_reasoning_model":
        task = AdvancedReasoningModelTask(agent)
    elif task_type == "multi_advanced_reasoning_model":
        task = MultiAdvancedReasoningModelTask(agent)
        raise NotImplementedError("Multi advanced reasoning model task is not fully implemented yet")
    elif task_type == "gemini":
        task = GeminiTask(agent)
    elif task_type == "vanilla_reasoning_model":
        task = VanillaReasoningModelTask(agent)
    elif task_type == "vision_model":
        task = VisionModelTask(agent)
    else:
        raise ValueError(f"Task type {task_type} not supported")
    
    # Execute the detection task
    print(f"Executing detection for object: '{object_of_interest}'")
    print(f"Task kwargs: {task_kwargs}")
    
    output = task.execute(
        image=image,
        prompt=object_of_interest,
        **task_kwargs
    )
    
    # Convert Cell objects to bounding boxes if needed
    if len(output['bboxs']) > 0 and isinstance(output['bboxs'][0], Cell):
        print(f"Converting {len(output['bboxs'])} cells to bboxes")
        output['bboxs'] = convert_list_of_cells_to_list_of_bboxes(output['bboxs'])
    
    # Calculate total processing time
    total_time = time.perf_counter() - start_time
    print(f"Detection completed in {total_time:.2f} seconds")
    print(f"Found {len(output['bboxs'])} bounding boxes")
    
    # Create visualized image with bounding boxes
    visualized_image = BaseDataset.visualize_image(
        image,
        output['bboxs'],
        return_image=True
    )
    
    # Prepare the result dictionary
    result = {
        "bboxs": output['bboxs'],
        "visualized_image": visualized_image,
        "original_image": image,
        "overlay_images": output.get('overlay_images', []),
        "total_time": total_time,
        "object_of_interest": object_of_interest,
        "task_type": task_type,
        "task_kwargs": task_kwargs
    }

    # Save outputs if requested
    if save_outputs:
        if output_folder_path is None:
            output_folder_path = f"./output/{get_timestamp()}"

        print(f"Saving outputs to: {output_folder_path}")
        _save_detection_outputs(
            output_folder_path=output_folder_path,
            result=result
        )

    # Remove overlay images from result if not requested
    if not return_overlay_images:
        result.pop('overlay_images', None)

    return result


def _save_detection_outputs(
    output_folder_path: str,
    result: Dict
) -> None:
    """
    Helper function to save detection outputs to disk.
    
    Args:
        output_folder_path (str): Directory to save outputs
        result (dict): Detection results dictionary
    """
    # Create output directory
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Save original image
    result['original_image'].save(
        os.path.join(output_folder_path, "original_image.jpg")
    )
    
    # Save visualized image with bounding boxes
    result['visualized_image'].save(
        os.path.join(output_folder_path, "visualized_image.jpg")
    )
    
    # Save detection results as JSON
    output_dict = {
        "object_of_interest": result['object_of_interest'],
        "task_type": result['task_type'],
        "task_kwargs": result['task_kwargs'],
        "bboxs": result['bboxs'],
        "total_time": result['total_time']
    }
    
    with open(os.path.join(output_folder_path, "output.json"), "w") as f:
        json.dump(output_dict, f, indent=2)

    # Save overlay images if available
    for i, overlay_image in enumerate(result.get('overlay_images', [])):
        if overlay_image is not None:
            overlay_image.save(
                os.path.join(output_folder_path, f"overlay_image_{i}.jpg")
            )

    print(f"All outputs saved to: {output_folder_path}")

import json
import os
import time
from typing import Dict, Generator, List, Optional, Union

from dotenv import load_dotenv
from PIL import Image

from .agents.agent_factory import AgentFactory
from .data import BaseDataset, Cell
from .tasks import (AdvancedReasoningModelTask, GeminiTask,
                    StreamAdvancedReasoningModelTask,
                    VanillaReasoningModelTask, VisionModelTask)
from .utils.io_utils import (convert_list_of_cells_to_list_of_bboxes,
                             download_image, get_timestamp)

# Load environment variables
load_dotenv()

"""
In order to run the API, you need to have the following environment variables set:
- OPENAI_API_KEY: OpenAI API key
- GEMINI_API_KEY: Gemini API key

You can get the API keys from the following links:
- OpenAI: https://platform.openai.com/
- Gemini: https://console.cloud.google.com/apis/credentials

You can set the environment variables in the following way:
```
export OPENAI_API_KEY=your_openai_api_key
export GEMINI_API_KEY=your_gemini_api_key
```

Follow instructions in the README.md to run the API.

If you only want to use the OpenAI or Gemini model, you can comment out the other models in the `_initialize_if_needed` method.

"""


class OptimizedDetectionAPI:
    """
    Optimized API that loads agents and models only once to prevent CUDA memory issues.
    """

    def __init__(self):
        self._agents = {}
        self._tasks = {}
        self._initialized = False
        self._initialize_if_needed()

    def _initialize_if_needed(self):
        """Lazy initialization of agents and tasks."""
        if not self._initialized:
            print("Initializing agents and models...")

            try:
                self._agents["openai"] = AgentFactory.create_agent(
                    model="o4-mini", platform_name="openai"
                )
                self._tasks["advanced_reasoning_model"] = AdvancedReasoningModelTask(
                    self._agents["openai"]
                )
                self._tasks["stream_advanced_reasoning_model"] = (
                    StreamAdvancedReasoningModelTask(self._agents["openai"])
                )
                self._tasks["vanilla_reasoning_model"] = VanillaReasoningModelTask(
                    self._agents["openai"],
                    prompt_type="vanilla",  # NOTE: Change to "advanced" if you'd like the model to incorporate a visual grid (Set-of-Marks style) as part of reasoning
                )

            except Exception as e:
                print(f"Error initializing OpenAI agent: {e}")
                self._agents["openai"] = None
                self._tasks["advanced_reasoning_model"] = None
                self._tasks["stream_advanced_reasoning_model"] = None
                self._tasks["vanilla_reasoning_model"] = None

                print("OpenAI agent initialization failed. Possible due to API key not being set.")

            # Initialize Gemini agent and corresponding tasks
            try:
                self._agents["gemini"] = AgentFactory.create_agent(
                        model="gemini-2.5-flash", platform_name="gemini"
                    )
                self._tasks["gemini"] = GeminiTask(self._agents["gemini"])
            except Exception as e:
                print(f"Error initializing Gemini agent: {e}")
                self._agents["gemini"] = None
                self._tasks["gemini"] = None

                print("Gemini agent initialization failed. Possible due to API key not being set.")

            self._tasks["vision_model"] = VisionModelTask(self._agents["openai"])

            self._initialized = True
            print("Initialization complete!")

    def detect_stream(
        self,
        image_path: str,
        object_of_interest: str,
        task_type: str = "stream_advanced_reasoning_model",
        task_kwargs: Optional[Dict] = None,
    ) -> Generator[Dict, None, None]:
        """
        Streaming detection function that yields intermediate results.
        Currently only supports 'stream_advanced_reasoning_model' task type.

        Args:
            image_path (str): Path to the image file or URL
            object_of_interest (str): Description of what to detect in the image
            task_type (str): Must be "stream_advanced_reasoning_model" for streaming
            task_kwargs (dict, optional): Additional parameters for the task

        Yields:
            dict: Intermediate results with 'type' field indicating:
                - 'intermediate': Intermediate crop results
                - 'final': Final detection results
                - 'error': Error occurred during processing
        """

        # Initialize agents and tasks if not already done
        self._initialize_if_needed()

        # Currently only support streaming for stream_advanced_reasoning_model
        if task_type != "stream_advanced_reasoning_model":
            yield {
                "type": "error",
                "error": f"Streaming not supported for task type: {task_type}",
            }
            return

        # Initialize task_kwargs if not provided
        if task_kwargs is None:
            task_kwargs = {}

        try:
            # Start timing
            start_time = time.perf_counter()

            # Load the image
            print(f"Loading image from: {image_path}")
            if image_path.startswith("http"):
                image = download_image(image_path)
            else:
                image = Image.open(image_path).convert("RGB")

            print(f"Image loaded successfully. Size: {image.size}")

            # Get the streaming task
            task = self._tasks["stream_advanced_reasoning_model"]

            # Execute the detection task with streaming
            print(f"Executing streaming detection for object: '{object_of_interest}'")
            print(f"Task kwargs: {task_kwargs}")

            # Yield each result from the streaming task
            for result in task.execute_streaming(
                image=image, prompt=object_of_interest, **task_kwargs
            ):
                # Add timing information
                result["elapsed_time"] = time.perf_counter() - start_time

                # For final results, add additional processing
                if result["type"] == "final":
                    # Convert Cell objects to bounding boxes if needed
                    if (
                        "bboxs" in result
                        and len(result["bboxs"]) > 0
                        and isinstance(result["bboxs"][0], Cell)
                    ):
                        print(f"Converting {len(result['bboxs'])} cells to bboxes")
                        result["bboxs"] = convert_list_of_cells_to_list_of_bboxes(
                            result["bboxs"]
                        )

                    # Add metadata
                    result["object_of_interest"] = object_of_interest
                    result["task_type"] = task_type
                    result["task_kwargs"] = task_kwargs
                    result["total_time"] = time.perf_counter() - start_time

                    print(f"Detection completed in {result['total_time']:.2f} seconds")
                    print(f"Found {len(result.get('bboxs', []))} bounding boxes")

                yield result

        except Exception as e:
            print(f"Error during streaming detection: {str(e)}")
            yield {
                "type": "error",
                "error": str(e),
                "elapsed_time": time.perf_counter() - start_time,
            }

    def detect(
        self,
        image_path: str,
        object_of_interest: str,
        task_type: str,
        task_kwargs: Optional[Dict] = None,
        save_outputs: bool = False,
        output_folder_path: Optional[str] = None,
        return_overlay_images: bool = True,
    ) -> Dict[str, Union[List, float, Image.Image]]:
        """
        Main detection function that processes an image and returns bounding boxes
        for objects of interest.

        Args:
            image_path (str): Path to the image file or URL
            object_of_interest (str): Description of what to detect in the image
            task_type (str): Type of detection task to run. Options:
                - "advanced_reasoning_model"
                - "stream_advanced_reasoning_model"
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

        # Initialize agents and tasks if not already done
        self._initialize_if_needed()

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

        # Check if task type is supported
        if task_type not in self._tasks:
            if task_type == "multi_advanced_reasoning_model":
                raise NotImplementedError(
                    "Multi advanced reasoning model task is not fully implemented yet"
                )
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

        # Get the appropriate task
        task = self._tasks[task_type]

        # Execute the detection task
        print(f"Executing detection for object: '{object_of_interest}'")
        print(f"Task kwargs: {task_kwargs}")

        output = task.execute(image=image, prompt=object_of_interest, **task_kwargs)

        # Convert Cell objects to bounding boxes if needed
        if len(output["bboxs"]) > 0 and isinstance(output["bboxs"][0], Cell):
            print(f"Converting {len(output['bboxs'])} cells to bboxes")
            output["bboxs"] = convert_list_of_cells_to_list_of_bboxes(output["bboxs"])

        # Calculate total processing time
        total_time = time.perf_counter() - start_time
        print(f"Detection completed in {total_time:.2f} seconds")
        print(f"Found {len(output['bboxs'])} bounding boxes")

        # Create visualized image with bounding boxes
        visualized_image = BaseDataset.visualize_image(
            image, output["bboxs"], return_image=True
        )

        # Prepare the result dictionary
        result = {
            "bboxs": output["bboxs"],
            "visualized_image": visualized_image,
            "original_image": image,
            "overlay_images": output.get("overlay_images", []),
            "total_time": total_time,
            "object_of_interest": object_of_interest,
            "task_type": task_type,
            "task_kwargs": task_kwargs,
        }

        # Save outputs if requested
        if save_outputs:
            if output_folder_path is None:
                output_folder_path = f"./output/{get_timestamp()}"

            print(f"Saving outputs to: {output_folder_path}")
            _save_detection_outputs(
                output_folder_path=output_folder_path, result=result
            )

        # Remove overlay images from result if not requested
        if not return_overlay_images:
            result.pop("overlay_images", None)

        return result


# Create a singleton instance. Important to avoid reloading models on every call, burns GPU memory.
_api_instance = OptimizedDetectionAPI()


def detect(
    image_path: str,
    object_of_interest: str,
    task_type: str,
    task_kwargs: Optional[Dict] = None,
    save_outputs: bool = False,
    output_folder_path: Optional[str] = None,
    return_overlay_images: bool = True,
) -> Dict[str, Union[List, float, Image.Image]]:
    """
    Main detection function that processes an image and returns bounding boxes
    for objects of interest. This function uses a singleton instance to avoid
    reloading models on every call.

    Args:
        image_path (str): Path to the image file or URL
        object_of_interest (str): Description of what to detect in the image
        task_type (str): Type of detection task to run. Options:
            - "advanced_reasoning_model"
            - "stream_advanced_reasoning_model"
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
    return _api_instance.detect(
        image_path=image_path,
        object_of_interest=object_of_interest,
        task_type=task_type,
        task_kwargs=task_kwargs,
        save_outputs=save_outputs,
        output_folder_path=output_folder_path,
        return_overlay_images=return_overlay_images,
    )


def detect_stream(
    image_path: str,
    object_of_interest: str,
    task_type: str = "stream_advanced_reasoning_model",
    task_kwargs: Optional[Dict] = None,
) -> Generator[Dict, None, None]:
    """
    Streaming detection function that yields intermediate results.
    Currently only supports 'stream_advanced_reasoning_model' task type.

    Args:
        image_path (str): Path to the image file or URL
        object_of_interest (str): Description of what to detect in the image
        task_type (str): Must be "stream_advanced_reasoning_model" for streaming
        task_kwargs (dict, optional): Additional parameters for the task

    Yields:
        dict: Intermediate results with 'type' field indicating:
            - 'intermediate': Intermediate crop results
            - 'final': Final detection results
            - 'error': Error occurred during processing
    """
    return _api_instance.detect_stream(
        image_path=image_path,
        object_of_interest=object_of_interest,
        task_type=task_type,
        task_kwargs=task_kwargs,
    )


def _save_detection_outputs(output_folder_path: str, result: Dict) -> None:
    """
    Helper function to save detection outputs to disk.

    Args:
        output_folder_path (str): Directory to save outputs
        result (dict): Detection results dictionary
    """
    # Create output directory
    os.makedirs(output_folder_path, exist_ok=True)

    # Save original image
    result["original_image"].save(
        os.path.join(output_folder_path, "original_image.jpg")
    )

    # Save visualized image with bounding boxes
    result["visualized_image"].save(
        os.path.join(output_folder_path, "visualized_image.jpg")
    )

    # Save detection results as JSON
    output_dict = {
        "object_of_interest": result["object_of_interest"],
        "task_type": result["task_type"],
        "task_kwargs": result["task_kwargs"],
        "bboxs": result["bboxs"],
        "total_time": result["total_time"],
    }

    with open(os.path.join(output_folder_path, "output.json"), "w") as f:
        json.dump(output_dict, f, indent=2)

    # Save overlay images if available
    for i, overlay_image in enumerate(result.get("overlay_images", [])):
        if overlay_image is not None:
            overlay_image.save(
                os.path.join(output_folder_path, f"overlay_image_{i}.jpg")
            )

    print(f"All outputs saved to: {output_folder_path}")

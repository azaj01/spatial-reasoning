import json
import os
import random
import time
from argparse import ArgumentParser

from agents.agent_factory import AgentFactory
from data import Cell, DetectionDataset
from dotenv import load_dotenv
from PIL import Image
from tasks import (AdvancedReasoningModelTask, GeminiTask,
                   MultiAdvancedReasoningModelTask, VanillaReasoningModelTask,
                   VisionModelTask)
from utils.io_utils import (convert_list_of_cells_to_list_of_bboxes,
                            get_timestamp)

load_dotenv()

def save_output(
    output_folder_path: str,
    image: Image.Image,
    object_of_interest: str, 
    task_type: str, 
    task_kwargs: dict, 
    output: dict, 
    total_time: float
) -> None:

    os.makedirs(output_folder_path, exist_ok=True)
    image.save(os.path.join(output_folder_path, "original_image.jpg"))
    visualized_image = DetectionDataset.visualize_image(image, output['bboxs'], return_image=True)
    visualized_image.save(os.path.join(output_folder_path, "visualized_image.jpg"))
    
    output_dict = {
        "object_of_interest": object_of_interest,
        "task_type": task_type,
        "task_kwargs": task_kwargs,
        "bboxs": output['bboxs'],
        "total_time": total_time
        # TODO: add compute cost
    }

    with open(os.path.join(output_folder_path, "output.json"), "w") as f:
        json.dump(output_dict, f)

    for i, overlay_image in enumerate(output['overlay_images']):
        if overlay_image is not None:
            overlay_image.save(os.path.join(output_folder_path, f"overlay_image_{i}.jpg"))
        
if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--image-path", type=str, required=False)
    args.add_argument("--object-of-interest", type=str, required=True)
    # Task type
    args.add_argument("--task-type", type=str, required=False, default="advanced_reasoning_model")
    args.add_argument("--task-kwargs", type=dict, required=False, default={})

    # Dataset arguments
    args.add_argument("--dataset-path", type=str, required=False)
    args.add_argument("--dataset-split", type=str, required=False, default="validation")  # assumes dataset is a huggingface dataset
    args.add_argument("--dataset-visualize", type=bool, required=False, default=False)
    args.add_argument("--trust-remote-code", type=bool, required=False, default=True)
    args.add_argument("--dataset-kwargs", type=dict, required=False, default={})
    
    # Output arguments
    args.add_argument("--output-folder-path", type=str, required=False, default=f"/home/qasim/code/exp/vision_evals/output/{get_timestamp()}")

    args = args.parse_args()
    
    object_of_interest = args.object_of_interest

    if args.image_path is not None:
        image = Image.open(args.image_path).convert("RGB")
    elif args.dataset_path is not None:
        dataset = DetectionDataset(tf=None,
                                   path=args.dataset_path,
                                   split=args.dataset_split,
                                   visualize=args.dataset_visualize,
                                   trust_remote_code=args.trust_remote_code,
                                   **args.dataset_kwargs)
        random_index = random.randint(0, len(dataset) - 1)
        data = dataset[random_index]  # TODO: make this more flexible
        image = data['pixel_values']
        gt_bbox = data['bboxs'][0]
        object_of_interest = list(data['unique_labels'].keys())[0]
        # Save the ground truth annotation
        gt_image = DetectionDataset.visualize_image(image, [gt_bbox])
        os.makedirs(args.output_folder_path, exist_ok=True)
        gt_image.save(os.path.join(args.output_folder_path, "gt_image.jpg"))

    else:
        raise ValueError("Either image-path or dataset-path must be provided")
    
    # Run the task
    openai_agent = AgentFactory.create_agent(model="o4-mini", platform_name="openai")
    gemini_agent = AgentFactory.create_agent(model="gemini-2.5-flash", platform_name="gemini")
    
    if args.task_type == "advanced_reasoning_model":
        task = AdvancedReasoningModelTask(openai_agent, **args.task_kwargs)
    elif args.task_type == "multi_advanced_reasoning_model":
        task = MultiAdvancedReasoningModelTask(openai_agent, **args.task_kwargs)
    elif args.task_type == "gemini":
        task = GeminiTask(gemini_agent)
    elif args.task_type == "vanilla_reasoning_model":
        task = VanillaReasoningModelTask(openai_agent)
    elif args.task_type == "vision_model":
        task = VisionModelTask(openai_agent)
    else:
        raise ValueError(f"Task type {args.task_type} not supported")
    
    start_time = time.perf_counter()
    output: list[Cell] = task.execute(image=image, prompt=object_of_interest)
    if len(output['bboxs']) > 0 and isinstance(output['bboxs'][0], Cell):
        print(f"Converting {len(output['bboxs'])} cells to bboxes")
        output['bboxs'] = convert_list_of_cells_to_list_of_bboxes(output['bboxs'])
    
    save_output(
        args.output_folder_path,
        image,
        object_of_interest,
        args.task_type,
        args.task_kwargs,
        output,
        time.perf_counter() - start_time)

# Example usage:
# Option 1. Supply your own image: python main.py --image-path /path/to/image.jpg --object-of-interest "object of interest"
# Option 2. Supply a dataset:python main.py --dataset-path /path/to/dataset.json --object-of-interest "object of interest" --dataset-split "train" --dataset-visualize True --dataset-trust-remote-code True --dataset-kwargs '{"task": "coco", "year": "2017"}'
# python main.py --image-path /home/data/unitx/images/1645672134274-IR5J98FHAW.png --object-of-interest "defect chip. could be golden in color, like a tintish look. or just anything that looks hella abnormal"
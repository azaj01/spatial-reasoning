import json
from argparse import ArgumentParser

from api import detect
from utils.io_utils import get_timestamp

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--image-path", type=str, required=True)
    args.add_argument("--object-of-interest", type=str, required=True)
    
    # Task type
    args.add_argument("--task-type", type=str, required=False, default="advanced_reasoning_model")
    args.add_argument('--task-kwargs', type=lambda x: json.loads(x), help='Task kwargs as JSON')

    # Output arguments
    args.add_argument("--output-folder-path", type=str, required=False, default=f"./output/{get_timestamp()}")

    args = args.parse_args()

    object_of_interest = args.object_of_interest

    result = detect(
        image_path=args.image_path,
        object_of_interest=object_of_interest,
        task_type=args.task_type,
        task_kwargs=args.task_kwargs,
        save_outputs=True,
        output_folder_path=args.output_folder_path
    )

    print(f"Found {len(result['bboxs'])} objects")
    print(f"Bounding boxes: {result['bboxs']}")


# Example usage:
# python main.py --image-path https://www.shutterstock.com/shutterstock/photos/1015857448/display_1500/stock-photo-detailed-photo-of-shoes-with-holes-in-them-and-toes-sticking-out-1015857448.jpg --object-of-interest "holes in shoes" --task-kwargs '{"nms_threshold": 0.7, "multiple_predictions": true}'
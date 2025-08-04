import json
from argparse import ArgumentParser

from .api import detect
from .utils.io_utils import get_timestamp

# Example usage:
# spatial-reasoning --image-path https://ix-cdn.b2e5.com/images/27094/27094_3063d356a3a54cc3859537fd23c5ba9d_1539205710.jpeg --object-of-interest "farthest scooter in the image" --task-kwargs '{"nms_threshold": 0.7, "multiple_predictions": false}' --task-type "gemini"

def main():
    """Main entry point for the CLI."""
    args = ArgumentParser()
    args.add_argument("--image-path", type=str, required=True)
    args.add_argument("--object-of-interest", type=str, required=True)

    # Task type
    args.add_argument(
        "--task-type", type=str, required=False, default="advanced_reasoning_model"
    )
    args.add_argument(
        "--task-kwargs", type=lambda x: json.loads(x), help="Task kwargs as JSON"
    )

    # Output arguments
    args.add_argument(
        "--output-folder-path",
        type=str,
        required=False,
        default=f"./output/{get_timestamp()}",
    )

    args = args.parse_args()

    object_of_interest = args.object_of_interest

    result = detect(
        image_path=args.image_path,
        object_of_interest=object_of_interest,
        task_type=args.task_type,
        task_kwargs=args.task_kwargs,
        save_outputs=True,
        output_folder_path=args.output_folder_path,
    )

    print(f"Found {len(result['bboxs'])} objects")
    print(f"Bounding boxes: {result['bboxs']}")


if __name__ == "__main__":
    main()

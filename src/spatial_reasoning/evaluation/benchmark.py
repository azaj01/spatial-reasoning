from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List

from ..api import detect
from ..utils.image_utils import (calculate_focal_diou,
                                 calculate_iou_with_offset,
                                 calculate_modified_diou)

SUPPORTED_TASKS = {
    "gemini",
    "xai_vanilla_reasoning",
    "xai_advanced_reasoning",
    "openai_vanilla_reasoning",
    "openai_advanced_reasoning",
    "openai_stream_reasoning",
    "vision_model",
}

TASK_ALIASES: Dict[str, str] = {name: name for name in SUPPORTED_TASKS}
TASK_ALIASES.update({
    "grounding_dino": "vision_model",
})

SERIAL_TASKS = {"vision_model"}  # gdino can be expensive to run in parallel


def _clean_result(result: Dict) -> Dict:
    return {
        key: value
        for key, value in result.items()
        if key not in {"visualized_image", "original_image"}
    }


def _compute_metrics(payload: Dict, sample: Dict) -> Dict[str, float]:
    metrics = {"iou": 0.0, "modified_diou": 0.0, "max_focal_diou": 0.0}

    if not isinstance(payload, dict):
        return metrics

    annotations = sample.get("annotations")
    if not annotations:
        return metrics

    ground_truths = [
        tuple(map(float, ann[:4]))
        for ann in annotations
        if isinstance(ann, (list, tuple)) and len(ann) >= 4
    ]
    if not ground_truths:
        return metrics

    crop_metadata = payload.get("crop_metadata") or {}
    zoom_source = crop_metadata.get("global_crops") or payload.get("global_crops") or []
    zoom_boxes = [
        tuple(map(float, crop[:4]))
        for crop in zoom_source
        if isinstance(crop, (list, tuple)) and len(crop) >= 4
    ]

    confidences = crop_metadata.get("confidences") or payload.get("confidences") or []  # one confidence value for each zoom level
    if confidences:
        metrics["average_confidence"] = float(
            sum(confidences[1:]) / len(confidences[1:]) # first one is the original image and it's confidence level is nonsensical
        ) if len(confidences) > 1 else 0.0

    if zoom_boxes:
        metrics["max_focal_diou"] = float(
            max(calculate_focal_diou(gt_box, zoom_boxes) for gt_box in ground_truths)
        )

    predictions = payload.get("bboxs")
    if not predictions:
        return metrics

    resolution = sample.get("metadata", {}).get("resolution")
    if not (isinstance(resolution, (list, tuple)) and len(resolution) == 2):
        return metrics
    image_size = (int(resolution[0]), int(resolution[1]))

    best_iou = -1.0
    best_diou = None

    for pred in predictions:
        if not (isinstance(pred, (list, tuple)) and len(pred) == 4):
            continue
        pred_box = tuple(map(float, pred))
        for gt_box in ground_truths:
            iou = calculate_iou_with_offset(pred_box, gt_box, image_size)
            diou = calculate_modified_diou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_diou = diou

    if best_iou >= 0:
        metrics["iou"] = float(best_iou)
        if best_diou is not None:
            metrics["modified_diou"] = float(best_diou)

    return metrics


def run_benchmark(
    agents: List[str],
    data_path: str,
    save_location: str,
    num_workers: int | None = None,
) -> None:
    with open(data_path) as fh:
        samples = json.load(fh)

    output_root = Path(save_location)
    output_root.mkdir(parents=True, exist_ok=True)

    task_types = {agent: TASK_ALIASES.get(agent, agent) for agent in agents}
    outputs: Dict[str, List[Dict]] = {agent: [None] * len(samples) for agent in agents}

    def _execute(agent: str, sample: Dict, idx: int) -> tuple[str, int, Dict]:
        task_type = task_types[agent]
        prompt = sample.get("metadata", {}).get("prompt", "")
        task_kwargs = sample.get("metadata", {}).get("task_kwargs")
        kwargs = dict(task_kwargs) if isinstance(task_kwargs, dict) else task_kwargs
        try:
            result = detect(
                image_path=sample["image_url"],
                object_of_interest=prompt,
                task_type=task_type,
                task_kwargs=kwargs,
                return_overlay_images=False,
            )
            payload = _clean_result(result)
        except Exception as exc:  # noqa: BLE001
            payload = {"error": str(exc)}

        record = {
            "image_url": sample.get("image_url"),
            "metadata": sample.get("metadata"),
            "annotations": sample.get("annotations"),
            "model_outputs": payload,
            "metrics": _compute_metrics(payload, sample),
        }
        return agent, idx, record

    parallel_agents = [
        agent for agent in agents if task_types[agent] not in SERIAL_TASKS
    ]
    serial_agents = [agent for agent in agents if agent not in parallel_agents]
    worker_count = max(
        1,
        num_workers
        or min(32, len(samples) * max(1, len(parallel_agents))),
    )

    if parallel_agents:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_execute, agent, sample, idx)
                for agent in parallel_agents
                for idx, sample in enumerate(samples)
            ]
            for future in as_completed(futures):
                agent, idx, record = future.result()
                outputs[agent][idx] = record

    for agent in serial_agents:
        for idx, sample in enumerate(samples):
            _, _, record = _execute(agent, sample, idx)
            outputs[agent][idx] = record

    for agent, entries in outputs.items():
        agent_folder = output_root / agent
        agent_folder.mkdir(parents=True, exist_ok=True)
        with open(agent_folder / "output.json", "w") as fh:
            json.dump(entries, fh, indent=2)


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Run spatial reasoning benchmark")
    parser.add_argument("--agents", nargs="+", required=True, help="Agent names to evaluate")
    parser.add_argument("--data", required=True, help="Path to dataset JSON file")
    parser.add_argument(
        "--save-location",
        required=True,
        help="Directory to store per-agent benchmark outputs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker threads to use for parallel agents (auto if omitted)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_benchmark(args.agents, args.data, args.save_location, args.num_workers)


if __name__ == "__main__":
    main()

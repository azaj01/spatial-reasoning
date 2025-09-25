#!/usr/bin/env python3
"""Aggregate evaluation metrics and generate comparison plots for multiple agents."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required to run this analysis. Install it with 'pip install matplotlib'."
    ) from exc


MetricMap = Dict[str, Dict[str, Optional[float]]]


AGENT_DISPLAY_NAMES = {
    "openai_advanced_reasoning": "OpenAI with Tool-Use",
    "openai_vanilla_reasoning": "OpenAI default o4-mini",
    "xai_advanced_reasoning": "Grok4 with Tool-Use",
    "xai_vanilla_reasoning": "Grok4 default fast-reasoning",
    "gemini": "Gemini 2.5 Flash (no thinking)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create diagnostic plots for agent evaluation outputs."
    )
    parser.add_argument(
        "benchmark_dir",
        type=str,
        help="Path to the folder that contains per-agent subdirectories with output.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write plots (defaults to the benchmark directory).",
    )
    return parser.parse_args()


def load_agent_results(root: str) -> Dict[str, List[dict]]:
    agent_data: Dict[str, List[dict]] = {}
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        output_path = os.path.join(entry.path, "output.json")
        if not os.path.isfile(output_path):
            continue
        with open(output_path, "r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:  # pragma: no cover
                raise ValueError(f"Failed to parse {output_path}: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {output_path}")
        agent_data[entry.name] = payload
    if not agent_data:
        raise ValueError(f"No agent output.json files found under {root}")
    return agent_data


def normalise_task_types(raw: object) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if item is not None]
    return []


def bucket_resolution(resolution: object) -> Optional[str]:
    if not resolution:
        return None
    if isinstance(resolution, (list, tuple)) and resolution:
        try:
            dims = [int(v) for v in resolution]
        except (TypeError, ValueError):
            return None
        major_axis = max(dims)
        if major_axis < 1024:
            return "standard"
        if major_axis <= 2048:
            return "medium"
        return "high"
    return None


def slugify(label: str) -> str:
    safe = label.lower().strip().replace(" ", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-"})
    return safe or "unknown"


def display_agent(agent: str) -> str:
    return AGENT_DISPLAY_NAMES.get(agent, agent)


def bucket_confidence(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if value < 0.7:
        return "<70%"
    if value < 0.8:
        return "70-80%"
    if value < 0.9:
        return "80-90%"
    if value <= 1.0:
        return "90-100%"
    return "90-100%"


def build_rows(agent_data: Dict[str, List[dict]]):
    rows = []
    agent_to_urls: Dict[str, set] = {}
    for agent, entries in agent_data.items():
        urls = set()
        for entry in entries:
            url = entry.get("image_url")
            if not url:
                continue
            urls.add(url)
            metadata = entry.get("metadata") or {}
            metrics = entry.get("metrics") or {}
            model_outputs = entry.get("model_outputs") or {}
            rows.append(
                {
                    "agent": agent,
                    "image_url": url,
                    "iou": metrics.get("iou"),
                    "diou": metrics.get("modified_diou") or metrics.get("diou"),
                    "max_focal_diou": metrics.get("max_focal_diou"),
                    "average_confidence": metrics.get("average_confidence"),
                    "total_time": model_outputs.get("total_time"),
                    "task_types": normalise_task_types(metadata.get("task_type")),
                    "is_relational": metadata.get("is_relational"),
                    "resolution_bucket": bucket_resolution(metadata.get("resolution")),
                    "confidence_bucket": bucket_confidence(metrics.get("average_confidence")),
                }
            )
        agent_to_urls[agent] = urls
    url_sets = [urls for urls in agent_to_urls.values() if urls]
    if not url_sets:
        raise ValueError("No image_url entries found across agents")
    shared_urls = set.intersection(*url_sets)
    if not shared_urls:
        raise ValueError("Agents do not share any common image_url entries")
    if any(len(urls) != len(shared_urls) for urls in url_sets):
        print(
            "[analysis] Warning: Restricting analysis to",
            len(shared_urls),
            "shared image_urls across agents.",
            file=sys.stderr,
        )
    filtered_rows = [row for row in rows if row["image_url"] in shared_urls]
    return filtered_rows, sorted(agent_data.keys()), shared_urls


def compute_means(
    rows: List[dict],
    metric_keys: Iterable[str],
    agents: Iterable[str],
) -> MetricMap:
    values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        agent = row["agent"]
        for key in metric_keys:
            value = row.get(key)
            if value is not None:
                values[agent][key].append(value)
    averages: MetricMap = {}
    for agent in agents:
        agent_metrics: Dict[str, Optional[float]] = {}
        for key in metric_keys:
            vals = values.get(agent, {}).get(key, [])
            agent_metrics[key] = sum(vals) / len(vals) if vals else None
        averages[agent] = agent_metrics
    return averages


def grouped_means(
    rows: List[dict],
    category_getter,
    metric_keys: Iterable[str],
    agents: Iterable[str],
) -> Dict[str, MetricMap]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        categories = category_getter(row)
        if not categories:
            continue
        if isinstance(categories, (str, bool)):
            categories = [categories]
        for category in categories:
            if category is None:
                continue
            grouped[str(category)].append(row)
    return {
        label: compute_means(group_rows, metric_keys, agents)
        for label, group_rows in grouped.items()
        if group_rows
    }


def plot_grouped_metrics(
    data: MetricMap,
    title: str,
    ylabel: str,
    output_path: str,
    metric_labels: Optional[Dict[str, str]] = None,
) -> None:
    metric_labels = metric_labels or {"iou": "IoU", "diou": "DIoU"}
    agents = list(data.keys())
    if not agents:
        return
    metrics = list(metric_labels.keys())
    x_positions = list(range(len(agents)))
    width = 0.35 if len(metrics) == 2 else 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(agents) * 1.3), 4.5))
    for idx, metric in enumerate(metrics):
        offsets = [x - width * (len(metrics) - 1) / 2 + idx * width for x in x_positions]
        values = [
            data[agent].get(metric) if data[agent].get(metric) is not None else math.nan
            for agent in agents
        ]
        ax.bar(offsets, values, width, label=metric_labels.get(metric, metric))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([display_agent(agent) for agent in agents], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_time_to_completion(data: MetricMap, output_path: str) -> None:
    agents = list(data.keys())
    values = [
        data[agent].get("total_time")
        if data[agent].get("total_time") is not None
        else math.nan
        for agent in agents
    ]
    fig, ax = plt.subplots(figsize=(max(6, len(agents) * 1.3), 4.5))
    ax.bar(range(len(agents)), values, color="#2a9d8f")
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels([display_agent(agent) for agent in agents], rotation=20, ha="right")
    ax.set_ylabel("Average completion time (s)")
    ax.set_title("Average time to completion by agent")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confidence_accuracy(
    bucket_agent_metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    agents: List[str],
    output_path: str,
    title: str = "Model confidence vs Accuracy Analysis",
) -> None:
    bucket_sequence = ["<70%", "70-80%", "80-90%", "90-100%"]
    buckets = []
    for bucket in bucket_sequence:
        agent_metrics = bucket_agent_metrics.get(bucket)
        if not agent_metrics:
            continue
        if any(
            agent_metrics.get(agent, {}).get(metric) is not None
            for agent in agents
            for metric in ("iou", "diou")
        ):
            buckets.append(bucket)
    if not buckets:
        return

    agent_labels = {agent: display_agent(agent) for agent in agents}

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(buckets) * 3.2), 4.8), sharey=True)
    metrics = [("iou", "(a) IoU Scores"), ("diou", "(b) DIoU Scores")]
    bar_width = 0.35
    x_positions = list(range(len(buckets)))

    for axis, (metric_key, subplot_title) in zip(axes, metrics):
        for idx, agent in enumerate(agents):
            offsets = [
                x - bar_width / 2 + idx * bar_width
                for x in x_positions
            ]
            values = []
            for bucket in buckets:
                value = (
                    bucket_agent_metrics.get(bucket, {})
                    .get(agent, {})
                    .get(metric_key)
                )
                values.append(value if value is not None else math.nan)
            axis.bar(
                offsets,
                values,
                bar_width,
                label=agent_labels.get(agent, agent),
                alpha=0.85,
            )

        axis.set_xticks(x_positions)
        axis.set_xticklabels(buckets)
        axis.set_xlabel("Confidence bucket")
        axis.set_title(subplot_title)
        axis.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Accuracy")
    fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(agents))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_focal_diou_breakdown(
    rows: List[dict],
    agents: List[str],
    output_path: str,
) -> None:
    metric_key = "max_focal_diou"
    resolution_order = ["standard", "medium", "high"]
    category_order = ["cua", "general", "tiny"]
    resolution_labels = {
        "standard": "Standard",
        "medium": "Medium",
        "high": "High",
    }
    category_labels = {
        "cua": "CUA",
        "general": "General",
        "tiny": "Tiny",
    }

    resolution_values: Dict[str, Dict[str, List[float]]] = {
        bucket: {agent: [] for agent in agents} for bucket in resolution_order
    }
    category_values: Dict[str, Dict[str, List[float]]] = {
        bucket: {agent: [] for agent in agents} for bucket in category_order
    }

    for row in rows:
        agent = row.get("agent")
        if agent not in agents:
            continue
        value = row.get(metric_key)
        if value is None:
            continue

        resolution_bucket = row.get("resolution_bucket")
        if resolution_bucket in resolution_values:
            resolution_values[resolution_bucket][agent].append(value)

        for task_type in row.get("task_types", []) or []:
            if task_type in category_values:
                category_values[task_type][agent].append(value)

    resolution_buckets = [
        bucket
        for bucket in resolution_order
        if any(resolution_values[bucket][agent] for agent in agents)
    ]
    category_buckets = [
        bucket
        for bucket in category_order
        if any(category_values[bucket][agent] for agent in agents)
    ]

    if not resolution_buckets and not category_buckets:
        return

    agent_labels = {agent: display_agent(agent) for agent in agents}
    bar_width = 0.35
    effective_resolution = max(len(resolution_buckets), 1)
    effective_category = max(len(category_buckets), 1)
    fig_width = max(10, (effective_resolution + effective_category) * 2.2)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.8), sharey=True)

    def plot_axis(axis, buckets, value_map, title, xlabel, label_map):
        if not buckets:
            axis.axis("off")
            return
        x_positions = list(range(len(buckets)))
        for idx, agent in enumerate(agents):
            offsets = [x - bar_width / 2 + idx * bar_width for x in x_positions]
            values = []
            for bucket in buckets:
                bucket_values = value_map[bucket][agent]
                if bucket_values:
                    values.append(sum(bucket_values) / len(bucket_values))
                else:
                    values.append(math.nan)
            axis.bar(offsets, values, bar_width, label=agent_labels.get(agent, agent))
        axis.set_xticks(range(len(buckets)))
        axis.set_xticklabels([label_map.get(bucket, bucket.title()) for bucket in buckets])
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.set_ylim(bottom=0)

    plot_axis(
        axes[0],
        resolution_buckets,
        resolution_values,
        "(a) Max Focal DIoU by Resolution",
        "Resolution",
        resolution_labels,
    )
    plot_axis(
        axes[1],
        category_buckets,
        category_values,
        "(b) Max Focal DIoU by Category",
        "Category",
        category_labels,
    )

    axes[0].set_ylabel("Max Focal DIoU")
    legend_handles = []
    legend_labels: List[str] = []
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            legend_handles = handles
            legend_labels = labels
            break
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(agents))
    fig.suptitle("Max Focal DIoU breakdown for selected agents")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    benchmark_dir = os.path.abspath(args.benchmark_dir)
    if not os.path.isdir(benchmark_dir):
        raise SystemExit(f"Benchmark directory not found: {benchmark_dir}")
    output_dir = os.path.abspath(args.output_dir or benchmark_dir)
    os.makedirs(output_dir, exist_ok=True)

    agent_data = load_agent_results(benchmark_dir)
    rows, agents, shared_urls = build_rows(agent_data)
    print(
        f"[analysis] Loaded {len(shared_urls)} shared samples across {len(agents)} agents.",
        file=sys.stderr,
    )

    overall_metrics = compute_means(rows, ("iou", "diou"), agents)
    plot_grouped_metrics(
        overall_metrics,
        "Average IoU and DIoU by agent",
        "Score",
        os.path.join(output_dir, "overall_performance.png"),
    )

    task_type_metrics = grouped_means(
        rows,
        lambda row: row["task_types"],
        ("iou", "diou"),
        agents,
    )
    for task_type, metrics in task_type_metrics.items():
        title = {
            "cua": "Performance on CUA (computer use)",
            "standard": "Performance on common in-the-wild object detection",
            "general": "Performance on common in-the-wild object detection",
            "tiny": "Performance on small object detection",
        }.get(task_type, f"Performance for task type {task_type}")
        plot_grouped_metrics(
            metrics,
            title,
            "Score",
            os.path.join(
                output_dir,
                f"performance_task_type_{slugify(task_type)}.png",
            ),
        )

    relational_metrics = grouped_means(
        rows,
        lambda row: [row["is_relational"]] if row["is_relational"] is not None else [],
        ("iou", "diou"),
        agents,
    )
    for relational_flag, metrics in relational_metrics.items():
        if relational_flag in {"True", True}:
            title = "Performance on relative instructions"
            filename = "performance_relative_instructions.png"
        else:
            title = "Performance on specific instructions"
            filename = "performance_specific_instructions.png"
        plot_grouped_metrics(
            metrics,
            title,
            "Score",
            os.path.join(output_dir, filename),
        )

    resolution_metrics = grouped_means(
        rows,
        lambda row: [row["resolution_bucket"]] if row["resolution_bucket"] else [],
        ("iou", "diou"),
        agents,
    )
    for resolution_bucket, metrics in resolution_metrics.items():
        plot_grouped_metrics(
            metrics,
            f"Performance for {resolution_bucket} resolution",
            "Score",
            os.path.join(
                output_dir,
                f"performance_resolution_{slugify(resolution_bucket)}.png",
            ),
        )

    time_metrics = compute_means(rows, ("total_time",), agents)
    plot_time_to_completion(
        time_metrics,
        os.path.join(output_dir, "average_time_to_completion.png"),
    )

    zoom_agents = ["xai_advanced_reasoning", "openai_advanced_reasoning"]
    if all(agent in agents for agent in zoom_agents):
        zoom_rows = [row for row in rows if row["agent"] in zoom_agents]
        zoom_metrics = compute_means(zoom_rows, ("max_focal_diou",), zoom_agents)
        plot_grouped_metrics(
            zoom_metrics,
            "Evaluation of agent's ability to zoom correctly",
            "Max Focal DIoU",
            os.path.join(output_dir, "max_focal_diou_xai_vs_openai.png"),
            {"max_focal_diou": "Max Focal DIoU"},
        )
        plot_focal_diou_breakdown(
            zoom_rows,
            zoom_agents,
            os.path.join(output_dir, "max_focal_diou_resolution_category.png"),
        )

        confidence_rows = [
            row
            for row in zoom_rows
            if row.get("confidence_bucket") and row.get("iou") is not None
        ]
        if confidence_rows:
            bucket_order = [
                "<70%",
                "70-80%",
                "80-90%",
                "90-100%",
            ]
            bucket_values: Dict[str, Dict[str, Dict[str, List[float]]]] = {
                bucket: {agent: {"iou": [], "diou": []} for agent in zoom_agents}
                for bucket in bucket_order
            }
            for row in confidence_rows:
                bucket = row["confidence_bucket"]
                agent = row["agent"]
                if bucket not in bucket_values or agent not in bucket_values[bucket]:
                    continue
                for metric in ("iou", "diou"):
                    value = row.get(metric)
                    if value is not None:
                        bucket_values[bucket][agent][metric].append(value)

            confidence_metrics: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
            for bucket in bucket_order:
                agent_metrics: Dict[str, Dict[str, Optional[float]]] = {}
                for agent in zoom_agents:
                    metric_avgs = {}
                    for metric in ("iou", "diou"):
                        values = bucket_values[bucket][agent][metric]
                        metric_avgs[metric] = (
                            sum(values) / len(values) if values else None
                        )
                    agent_metrics[agent] = metric_avgs
                if any(
                    metric_avgs[metric] is not None
                    for metric_avgs in agent_metrics.values()
                    for metric in ("iou", "diou")
                ):
                    confidence_metrics[bucket] = agent_metrics

            if confidence_metrics:
                plot_confidence_accuracy(
                    confidence_metrics,
                    zoom_agents,
                    os.path.join(output_dir, "confidence_vs_accuracy.png"),
                )


if __name__ == "__main__":
    main()

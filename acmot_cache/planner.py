"""Dependency-aware AC-MOT cache planner."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .hashing import cache_id, config_hash


DEFAULT_EXPERIMENT = {
    "dataset": {"name": "VisDrone2019-MOT-test-dev", "sequence_count": 17, "frame_count": 6635},
    "detector": {
        "model": "YOLOv8n",
        "weights": "yolov8n.pt",
        "precision": "FP16",
        "imgsz": [640, 736, 832],
        "classes": [0, 2, 5, 7],
    },
    "tracker": {"implementation": "ByteTrack", "match_thresh": 0.86},
    "sci": {"scene_analyzer_stride": 10, "rolling_window": 7, "resolutions": [640, 736, 832]},
    "gt_filter": {"classes": [1, 4, 5, 6, 9], "score": 1, "occlusion_lt": 2, "truncation_lt": 2},
    "trackeval": {"metrics": ["HOTA", "CLEAR", "Identity"], "threshold": 0.5},
}


@dataclass
class StageDecision:
    stage: str
    action: str
    reason: str
    expensive_gpu: bool = False
    cache_id: str | None = None


def load_experiment_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_EXPERIMENT
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    merged = json.loads(json.dumps(DEFAULT_EXPERIMENT))
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def expected_ids(config: dict[str, Any]) -> dict[str, Any]:
    dataset_id = cache_id("dataset", config["dataset"])
    det_ids = {
        str(size): cache_id("det", {**config["detector"], "imgsz": size, "dataset_cache_id": dataset_id})
        for size in config["detector"]["imgsz"]
    }
    tracker_id = cache_id("tracker", {"detections": det_ids, "tracker": config["tracker"]})
    replay_id = cache_id("replay", {"detections": det_ids, "tracker": config["tracker"], "sci": config["sci"]})
    eval_id = cache_id("eval", {"prediction": replay_id, "gt_filter": config["gt_filter"], "trackeval": config["trackeval"]})
    return {
        "dataset": dataset_id,
        "detections": det_ids,
        "tracker": tracker_id,
        "replay": replay_id,
        "evaluation": eval_id,
        "config_hashes": {
            "detector": config_hash(config["detector"]),
            "tracker": config_hash(config["tracker"]),
            "sci": config_hash(config["sci"]),
            "gt_filter": config_hash(config["gt_filter"]),
            "trackeval": config_hash(config["trackeval"]),
        },
    }


def _has_valid(registry: dict, category: str, key: str) -> bool:
    item = registry.get(category, {}).get(key)
    return bool(item and item.get("validation_status") in {"VALID", "FROZEN_FINAL"} and item.get("completion_status") == "complete")


def plan(config: dict[str, Any], registry: dict, mode: str = "auto", discovery_completed: bool = False) -> dict[str, Any]:
    ids = expected_ids(config)
    decisions: list[StageDecision] = []

    if _has_valid(registry, "datasets", ids["dataset"]):
        decisions.append(StageDecision("dataset", "REUSE", "compatible dataset cache found", cache_id=ids["dataset"]))
    else:
        decisions.append(StageDecision("dataset", "VERIFY_OR_STAGE", "compatible dataset cache not registered", cache_id=ids["dataset"]))

    missing_dets = []
    for size, det_id in ids["detections"].items():
        if _has_valid(registry, "detections", det_id):
            decisions.append(StageDecision(f"YOLOv8n {config['detector']['precision']} {size}", "REUSE", "compatible detector cache found", cache_id=det_id))
        else:
            missing_dets.append(size)
            action = "REFUSE_MISSING" if mode == "replay" else "COMPUTE_IF_APPROVED"
            decisions.append(
                StageDecision(
                    f"YOLOv8n {config['detector']['precision']} {size}",
                    action,
                    "required detector cache missing",
                    expensive_gpu=(mode in {"auto", "live"}),
                    cache_id=det_id,
                )
            )

    if mode == "live":
        decisions.append(StageDecision("live_detector_timing", "RUN_LIVE_INFERENCE", "live FPS cannot use cached detector outputs", expensive_gpu=True))
    elif missing_dets and mode == "replay":
        decisions.append(StageDecision("replay_guard", "ABORT", "replay mode never runs YOLO", expensive_gpu=False))
    elif missing_dets and not discovery_completed:
        decisions.append(StageDecision("expensive_guard", "ABORT", "cache discovery must complete before YOLO inference", expensive_gpu=False))

    if _has_valid(registry, "trackers", ids["tracker"]):
        decisions.append(StageDecision("tracker", "REUSE", "compatible tracker output found", cache_id=ids["tracker"]))
    else:
        decisions.append(StageDecision("tracker", "RECOMPUTE", "tracker/Sci output not registered or detector dependency changed", cache_id=ids["tracker"]))

    if _has_valid(registry, "evaluation", ids["evaluation"]):
        decisions.append(StageDecision("TrackEval", "REUSE", "compatible official evaluation found", cache_id=ids["evaluation"]))
    else:
        decisions.append(StageDecision("TrackEval", "RECOMPUTE", "evaluation missing or config changed", cache_id=ids["evaluation"]))

    expensive = any(d.expensive_gpu for d in decisions)
    return {
        "mode": mode,
        "discovery_completed": discovery_completed,
        "ids": ids,
        "decisions": [asdict(d) for d in decisions],
        "expensive_yolo_inference": expensive,
        "missing_detector_resolutions": missing_dets,
    }


def format_plan(plan_data: dict[str, Any]) -> str:
    lines = ["AC-MOT CACHE PLAN", "", f"Mode: {plan_data['mode']}", ""]
    for d in plan_data["decisions"]:
        lines.append(f"{d['stage']}: {d['action']}")
        lines.append(f"Reason: {d['reason']}")
        if d.get("cache_id"):
            lines.append(f"Cache ID: {d['cache_id']}")
        lines.append(f"Expensive GPU inference: {'YES' if d.get('expensive_gpu') else 'NO'}")
        lines.append("")
    lines.append("EXPENSIVE YOLO INFERENCE:")
    lines.append("YES" if plan_data["expensive_yolo_inference"] else "NO")
    if plan_data["missing_detector_resolutions"]:
        lines.append("")
        lines.append("Missing detector resolutions:")
        for size in plan_data["missing_detector_resolutions"]:
            lines.append(f"- {size}")
    return "\n".join(lines)

"""Broad, read-only discovery of legacy AC-MOT artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .hashing import cache_id


KEYWORDS = [
    "ACMOT",
    "AC-MOT",
    "VisDrone",
    "VisDrone_Results",
    "ACMOT_IDS",
    "cache",
    "detection_cache",
    "detections",
    "replay",
    "TrackEval",
    "trackeval",
    "predictions",
    "FP16",
    "FP32",
    "YOLO",
    "ByteTrack",
    "SCI",
    "v10",
    "v10_p3",
    "v10_p4",
    "v17",
    "ROUND2",
    "IDS",
    "benchmark",
    "results",
    "metrics",
]

KNOWN_ARTIFACTS = [
    "detection_cache_v10_p3_yolov8n_fp32_640_736_832",
    "detection_cache_v1",
    "ACMOT_IDS",
    "ROUND2_replay_20260906_103513",
    "ROUND2_trackeval_20260906_103513",
    "ROUND2_replay_20260906_103411",
    "ROUND2_replay_20260906_103134",
    "IDS_replay_20260906_101949",
    "IDS_trackeval_20260906_101949",
    "V10_P4_FP16_FAIR",
    "codex_v10_p4_fp16_20260907_134125",
    "realtime_fp16_20260906_142626",
    "ACMOT_V10STYLE_SCI_IDS_GUARD",
]

INTERESTING_SUFFIXES = {".json", ".csv", ".txt", ".md", ".ipynb", ".npz", ".gz", ".yaml", ".yml", ".pt"}


@dataclass
class ArtifactRecord:
    cache_id: str
    name: str
    logical_path: str
    drive_file_id: str | None
    owner_account: str | None
    shared: bool | None
    artifact_type: str
    project_version: str | None
    dataset: str | None
    sequence_count: int | None
    frame_count: int | None
    model: str | None
    weights: str | None
    precision: str | None
    imgsz: list[int] | None
    classes: list[int] | None
    detector_settings: dict | None
    tracker: str | None
    tracker_configuration: dict | None
    sci_configuration: dict | None
    calibrator_configuration: dict | None
    gt_filtering: dict | None
    metric_configuration: dict | None
    trackeval_configuration: dict | None
    environment: dict | None
    git_commit: str | None
    creation_time: str | None
    completion_status: str
    validation_status: str
    safe_reuse_purposes: list[str]
    unsafe_reuse_purposes: list[str]


def classify_type(path: Path) -> str:
    name = path.name.lower()
    full = str(path).lower()
    if "detection_cache" in name or "detections" in full:
        return "detection_cache"
    if "trackeval" in full:
        return "trackeval_output"
    if "prediction" in full or path.suffix == ".txt" and "trackers" in full:
        return "tracker_predictions"
    if "visdrone2019-mot-test-dev" in full or name == "sequences" or name == "annotations":
        return "dataset"
    if path.suffix == ".ipynb":
        return "notebook"
    if path.suffix == ".csv":
        return "result_csv"
    if "run" in name or "benchmark" in name or "v10_p4" in full or "v17" in full:
        return "experiment_run"
    return "legacy_artifact"


def infer_precision(path: Path) -> str | None:
    low = str(path).lower()
    if "fp16" in low:
        return "FP16"
    if "fp32" in low:
        return "FP32"
    return None


def infer_imgsz(path: Path) -> list[int] | None:
    low = str(path).lower()
    found = [size for size in (640, 736, 832) if str(size) in low]
    return found or None


def infer_status(path: Path) -> tuple[str, str]:
    low = str(path).lower()
    if "codex_v10_p4_fp16_20260907_134125" in low:
        return "FROZEN_FINAL", "complete"
    if path.is_dir() and any(path.glob("*.complete.json")):
        return "VALID", "complete"
    if path.is_dir() and any(path.rglob("*summary*")):
        return "UNKNOWN", "complete_or_partial"
    if path.is_file() and path.stat().st_size > 0:
        return "UNKNOWN", "file_present"
    return "UNKNOWN", "unknown"


def reuse_notes(artifact_type: str, precision: str | None) -> tuple[list[str], list[str]]:
    if artifact_type == "detection_cache":
        safe = ["tracker tuning", "SCI replay", "calibrator experiments", "ablation replay"]
        unsafe = ["live FPS evidence", "different precision evidence"]
        if precision == "FP32":
            unsafe.append("FP16 detector inference claims")
        return safe, unsafe
    if artifact_type == "trackeval_output":
        return ["paper table reuse if config matches", "result comparison"], ["new metric config", "different GT filter"]
    if artifact_type == "dataset":
        return ["dataset validation", "local staging/archive creation"], ["different split without validation"]
    if artifact_type == "tracker_predictions":
        return ["TrackEval rerun if GT and metric config match"], ["detector FPS evidence"]
    return ["historical provenance"], ["scientific reuse without inspection"]


def is_candidate(path: Path) -> bool:
    name = path.name
    haystack = str(path)
    if any(k.lower() in haystack.lower() for k in KEYWORDS + KNOWN_ARTIFACTS):
        return True
    if path.is_file() and path.suffix.lower() in INTERESTING_SUFFIXES:
        return True
    return False


def discover_paths(roots: Iterable[Path], max_items: int = 5000) -> list[ArtifactRecord]:
    records = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if len(records) >= max_items:
                return records
            if any(part in {".git", "__pycache__", ".ipynb_checkpoints"} for part in path.parts):
                continue
            if not is_candidate(path):
                continue
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            artifact_type = classify_type(path)
            validation_status, completion_status = infer_status(path)
            precision = infer_precision(path)
            safe, unsafe = reuse_notes(artifact_type, precision)
            stat = path.stat()
            records.append(
                ArtifactRecord(
                    cache_id=cache_id("legacy", {"path": str(path), "size": stat.st_size, "mtime": int(stat.st_mtime)}),
                    name=path.name,
                    logical_path=str(path),
                    drive_file_id=None,
                    owner_account=None,
                    shared=None,
                    artifact_type=artifact_type,
                    project_version=None,
                    dataset="VisDrone2019-MOT-test-dev" if "visdrone" in str(path).lower() else None,
                    sequence_count=None,
                    frame_count=None,
                    model="YOLOv8n" if "yolo" in str(path).lower() or "detection_cache" in str(path).lower() else None,
                    weights=None,
                    precision=precision,
                    imgsz=infer_imgsz(path),
                    classes=None,
                    detector_settings=None,
                    tracker="ByteTrack" if "bytetrack" in str(path).lower() or "track" in str(path).lower() else None,
                    tracker_configuration=None,
                    sci_configuration=None,
                    calibrator_configuration=None,
                    gt_filtering=None,
                    metric_configuration=None,
                    trackeval_configuration=None,
                    environment=None,
                    git_commit=None,
                    creation_time=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    completion_status=completion_status,
                    validation_status=validation_status,
                    safe_reuse_purposes=safe,
                    unsafe_reuse_purposes=unsafe,
                )
            )
    return records


def records_to_json(records: list[ArtifactRecord]) -> list[dict]:
    return [asdict(r) for r in records]


def write_inventory(records: list[ArtifactRecord], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records_to_json(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Legacy AC-MOT Cache Inventory",
        "",
        f"Discovered artifacts: {len(records)}",
        "",
        "| status | type | precision | name | path | safe reuse |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        safe = ", ".join(r.safe_reuse_purposes[:3])
        lines.append(
            f"| {r.validation_status} | {r.artifact_type} | {r.precision or ''} | {r.name} | `{r.logical_path}` | {safe} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

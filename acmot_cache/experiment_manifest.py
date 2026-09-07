"""Experiment manifest creation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .environment import snapshot
from .hashing import config_hash
from .planner import expected_ids


def build_manifest(
    experiment_name: str,
    mode: str,
    config: dict,
    stages: dict,
    output_paths: dict | None = None,
    metrics: dict | None = None,
    cwd: Path | None = None,
) -> dict:
    ids = expected_ids(config)
    return {
        "experiment_id": config_hash({"name": experiment_name, "config": config, "mode": mode}),
        "name": experiment_name,
        "date": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "git_commit": snapshot(cwd).get("git_commit"),
        "dataset_cache_id": ids["dataset"],
        "detector_cache_ids": ids["detections"],
        "tracker_config_hash": ids["config_hashes"]["tracker"],
        "sci_config_hash": ids["config_hashes"]["sci"],
        "calibrator_config_hash": config_hash(config.get("calibrator", {})),
        "gt_filter_hash": ids["config_hashes"]["gt_filter"],
        "evaluation_config_hash": ids["config_hashes"]["trackeval"],
        "environment": snapshot(cwd),
        "precision": config["detector"].get("precision"),
        "gpu": None,
        "frame_count": config["dataset"].get("frame_count"),
        "sequence_count": config["dataset"].get("sequence_count"),
        "output_paths": output_paths or {},
        "metrics": metrics or {},
        "stages": stages,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

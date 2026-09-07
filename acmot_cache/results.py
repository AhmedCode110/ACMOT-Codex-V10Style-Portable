"""Result table collection with compatibility warnings."""

from __future__ import annotations

import csv
import json
from pathlib import Path


RESULT_KEYS = ["experiment", "mode", "precision", "HOTA", "MOTA", "IDF1", "IDS", "FPS", "mean_imgsz", "git_commit"]


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_manifest_results(root: Path) -> list[dict]:
    rows = []
    for path in root.rglob("RUN_MANIFEST.json"):
        data = load_manifest(path)
        metrics = data.get("metrics", {})
        rows.append(
            {
                "experiment": data.get("experiment_id") or data.get("name") or path.parent.name,
                "mode": data.get("mode"),
                "precision": data.get("precision"),
                "HOTA": metrics.get("HOTA"),
                "MOTA": metrics.get("MOTA"),
                "IDF1": metrics.get("IDF1"),
                "IDS": metrics.get("IDS"),
                "FPS": metrics.get("FPS") or metrics.get("processing_fps"),
                "mean_imgsz": metrics.get("mean_imgsz"),
                "dataset": data.get("dataset_cache_id") or data.get("dataset"),
                "gt_filter": data.get("gt_filter_hash"),
                "trackeval": data.get("evaluation_config_hash"),
                "hardware": data.get("gpu") or data.get("environment", {}).get("gpu_name"),
                "git_commit": data.get("git_commit"),
                "source": str(path),
            }
        )
    return rows


def compatibility_warnings(rows: list[dict]) -> list[str]:
    warnings = []
    checks = [
        ("precision", "FP32 vs FP16 FPS or metric comparison"),
        ("dataset", "different datasets or sequence subsets"),
        ("gt_filter", "different GT filters"),
        ("trackeval", "different TrackEval settings"),
        ("hardware", "different hardware"),
    ]
    for key, message in checks:
        values = {r.get(key) for r in rows if r.get(key)}
        if len(values) > 1:
            warnings.append(f"WARNING: {message}: {sorted(values)}")
    return warnings


def write_results_csv(rows: list[dict], out: Path, sort_by: str | None = None) -> None:
    if sort_by:
        reverse = sort_by.upper() not in {"IDS"}
        rows = sorted(rows, key=lambda r: float(r.get(sort_by) or 0), reverse=reverse)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(RESULT_KEYS + [k for r in rows for k in r.keys()]))
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

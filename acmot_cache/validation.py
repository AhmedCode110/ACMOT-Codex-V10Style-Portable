"""Scientific cache validation helpers."""

from __future__ import annotations

import json
from pathlib import Path


VISDRONE_TEST_DEV = {
    "name": "VisDrone2019-MOT-test-dev",
    "expected_sequences": 17,
    "expected_frames": 6635,
    "gt_filter": {"classes": [1, 4, 5, 6, 9], "score": 1, "occlusion_lt": 2, "truncation_lt": 2},
}


def image_number(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 10**12


def validate_dataset_root(dataset: Path, expected_frames: int | None = 6635) -> dict:
    seq_dir = dataset / "sequences"
    ann_dir = dataset / "annotations"
    if not seq_dir.is_dir() or not ann_dir.is_dir():
        return {"status": "INCOMPATIBLE", "reason": "missing sequences or annotations folder"}
    seqs = sorted(p.name for p in seq_dir.iterdir() if p.is_dir())
    missing_ann = [s for s in seqs if not (ann_dir / f"{s}.txt").exists()]
    frame_count = 0
    seq_frames = {}
    for seq in seqs:
        frames = sorted((seq_dir / seq).glob("*.jpg"), key=image_number)
        seq_frames[seq] = len(frames)
        frame_count += len(frames)
    problems = []
    if len(seqs) != 17:
        problems.append(f"expected 17 sequences, found {len(seqs)}")
    if missing_ann:
        problems.append(f"missing annotations: {', '.join(missing_ann)}")
    if expected_frames is not None and frame_count != expected_frames:
        problems.append(f"expected {expected_frames} frames, found {frame_count}")
    return {
        "status": "VALID" if not problems else "PARTIAL",
        "dataset": VISDRONE_TEST_DEV["name"],
        "sequence_count": len(seqs),
        "frame_count": frame_count,
        "missing_annotations": missing_ann,
        "sequence_frames": seq_frames,
        "problems": problems,
    }


def read_json_if_exists(path: Path) -> dict | None:
    if path.exists() and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def validate_detection_manifest(manifest: dict, required: dict) -> dict:
    mismatches = {}
    for key in ["dataset_cache_id", "model", "weights_hash", "precision", "imgsz", "classes"]:
        if key in required and manifest.get(key) != required[key]:
            mismatches[key] = {"required": required[key], "found": manifest.get(key)}
    status = "VALID" if not mismatches and manifest.get("completion_status") == "complete" else "INCOMPATIBLE"
    return {"status": status, "mismatches": mismatches}


def validate_complete_markers(folder: Path, expected_sequences: list[str] | None = None) -> dict:
    markers = list(folder.rglob("*.complete.json")) if folder.exists() else []
    if expected_sequences:
        marker_names = {m.stem.replace(".complete", "") for m in markers}
        missing = [s for s in expected_sequences if s not in marker_names]
    else:
        missing = []
    return {"status": "VALID" if folder.exists() and not missing else "PARTIAL", "markers": len(markers), "missing": missing}

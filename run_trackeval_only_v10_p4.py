"""Run official TrackEval on an EXISTING v10_p4 run without rerunning YOLO/ByteTrack.

This script reuses:
  <run_dir>/trackeval_gt
  <run_dir>/trackeval_trackers
  <run_dir>/seqmap.txt

It intentionally does not touch detector/tracker predictions or timing results.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SYSTEMS = ["Baseline_Default", "Baseline_TunedTracker", "ACMOT_V10STYLE_SCI"]
TRACK_EVAL_REPO = "https://github.com/JonathonLuiten/TrackEval.git"


def ensure_trackeval_repo(root: Path) -> Path:
    script = root / "scripts" / "run_mot_challenge.py"
    if script.exists():
        return script
    if root.exists():
        shutil.rmtree(root)
    print("Cloning TrackEval once...")
    subprocess.run(["git", "clone", "--depth", "1", TRACK_EVAL_REPO, str(root)], check=True)
    if not script.exists():
        raise RuntimeError(f"TrackEval runner not found after clone: {script}")
    return script


def validate_run(run_dir: Path):
    required = [
        run_dir / "trackeval_gt" / "VisDroneACMOT-test",
        run_dir / "trackeval_trackers" / "VisDroneACMOT-test",
        run_dir / "seqmap.txt",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing existing TrackEval input: {p}")
    for system in SYSTEMS:
        data = run_dir / "trackeval_trackers" / "VisDroneACMOT-test" / system / "data"
        if not data.exists():
            raise FileNotFoundError(f"Missing tracker data for {system}: {data}")
        txts = list(data.glob("*.txt"))
        if len(txts) != 17:
            raise RuntimeError(f"Expected 17 prediction files for {system}, found {len(txts)}")


def parse_summary_file(path: Path):
    """Best-effort parser for TrackEval summary text files."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    header = re.split(r"\s+", lines[0])
    values = re.split(r"\s+", lines[1])
    if len(values) < len(header):
        return {}
    out = {}
    for k, v in zip(header, values):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def collect_official(run_dir: Path):
    base = run_dir / "trackeval_trackers" / "VisDroneACMOT-test"
    rows = []
    raw = {}
    for system in SYSTEMS:
        sysdir = base / system
        summaries = sorted(sysdir.rglob("*_summary.txt"))
        raw[system] = {}
        merged = {}
        for p in summaries:
            parsed = parse_summary_file(p)
            raw[system][str(p.relative_to(run_dir))] = parsed
            merged.update(parsed)
        row = {"system": system}
        for key in ["HOTA", "MOTA", "IDF1", "IDSW", "IDs", "IDS"]:
            if key in merged:
                row[key] = merged[key]
        rows.append(row)
    (run_dir / "official_trackeval_parsed.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    csv_path = run_dir / "official_trackeval_v10_p4.csv"
    keys = sorted({k for r in rows for k in r.keys()}, key=lambda x: (x != "system", x))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    return csv_path, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--trackeval-root", default="/content/TrackEval", type=Path)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    validate_run(run_dir)
    script = ensure_trackeval_repo(args.trackeval_root)

    cmd = [
        sys.executable, str(script),
        "--GT_FOLDER", str(run_dir / "trackeval_gt"),
        "--TRACKERS_FOLDER", str(run_dir / "trackeval_trackers"),
        "--BENCHMARK", "VisDroneACMOT",
        "--SPLIT_TO_EVAL", "test",
        "--SEQMAP_FILE", str(run_dir / "seqmap.txt"),
        "--TRACKERS_TO_EVAL", *SYSTEMS,
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--DO_PREPROC", "False",
        "--USE_PARALLEL", "False",
    ]
    print("TRACK-EVAL ONLY. NO YOLO/ByteTrack RERUN.")
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path = run_dir / "trackeval_stdout_stderr_FIXED.txt"
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"TrackEval-only run failed. See {log_path}")

    csv_path, rows = collect_official(run_dir)
    print("\nOFFICIAL TRACKEVAL COMPLETE")
    print("Log:", log_path)
    print("Parsed CSV:", csv_path)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()

"""Run official TrackEval only on an existing v10_p4 run folder.

This reuses already-saved predictions/TrackEval layout and does NOT rerun YOLO or ByteTrack.
It uses TrackEval's Python API directly to avoid the CLI SEQMAP_FILE parser turning a single path into a list.
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

SYSTEMS = ["Baseline_Default", "Baseline_TunedTracker", "ACMOT_V10STYLE_SCI"]


def ensure_trackeval(root: Path):
    if not (root / "trackeval" / "__init__.py").exists():
        if root.exists():
            shutil.rmtree(root)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/JonathonLuiten/TrackEval.git",
            str(root),
        ], check=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--trackeval-root", default=Path("/content/TrackEval"), type=Path)
    a = p.parse_args()

    run = a.run_dir
    gt_parent = run / "trackeval_gt"
    tracker_parent = run / "trackeval_trackers"
    seqmap = run / "seqmap.txt"

    for path in [gt_parent, tracker_parent, seqmap]:
        if not path.exists():
            raise RuntimeError(f"Missing required cached TrackEval input: {path}")

    ensure_trackeval(a.trackeval_root)
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": False,
        "PRINT_CONFIG": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
    })

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(gt_parent),
        "TRACKERS_FOLDER": str(tracker_parent),
        "TRACKERS_TO_EVAL": SYSTEMS,
        "BENCHMARK": "VisDroneACMOT",
        "SPLIT_TO_EVAL": "test",
        "SEQMAP_FILE": str(seqmap),
        "DO_PREPROC": False,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "PRINT_CONFIG": True,
    })

    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}

    print("Running TrackEval only via Python API. No YOLO/ByteTrack inference will run.")
    print("Run dir:", run)
    print("Seqmap:", seqmap)

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]

    results = evaluator.evaluate(dataset_list, metrics_list)

    marker = run / "TRACKEVAL_ONLY_COMPLETE.txt"
    marker.write_text(
        "Official TrackEval completed from cached v10_p4 predictions. No YOLO/ByteTrack rerun.\n",
        encoding="utf-8",
    )
    print("TRACK EVAL ONLY COMPLETE")
    print("Results saved under:", tracker_parent)
    print("Marker:", marker)
    return results


if __name__ == "__main__":
    main()

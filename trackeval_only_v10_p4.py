"""Run official TrackEval only on an existing v10_p4 run folder.

This script reuses already-saved predictions/TrackEval layout and does NOT rerun YOLO or ByteTrack.
"""
from pathlib import Path
import argparse, subprocess, sys, shutil

SYSTEMS = ["Baseline_Default", "Baseline_TunedTracker", "ACMOT_V10STYLE_SCI"]


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

    # TrackEval's CLI scripts live in the repository's top-level scripts/ directory,
    # not in the installed Python package namespace.
    if not (a.trackeval_root / "scripts" / "run_mot_challenge.py").exists():
        if a.trackeval_root.exists():
            shutil.rmtree(a.trackeval_root)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/JonathonLuiten/TrackEval.git",
            str(a.trackeval_root),
        ], check=True)

    script = a.trackeval_root / "scripts" / "run_mot_challenge.py"
    cmd = [
        sys.executable, str(script),
        "--GT_FOLDER", str(gt_parent),
        "--TRACKERS_FOLDER", str(tracker_parent),
        "--BENCHMARK", "VisDroneACMOT",
        "--SPLIT_TO_EVAL", "test",
        "--SEQMAP_FILE", str(seqmap),
        "--TRACKERS_TO_EVAL", *SYSTEMS,
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--DO_PREPROC", "False",
        "--USE_PARALLEL", "False",
        "--PLOT_CURVES", "False",
    ]
    print("Running TrackEval only. No YOLO/ByteTrack inference will run.")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = run / "trackeval_only_stdout_stderr.txt"
    log.write_text(proc.stdout or "", encoding="utf-8")
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"TrackEval-only run failed. See {log}")
    print("TRACK EVAL ONLY COMPLETE")
    print("Results saved under:", tracker_parent)
    print("Log:", log)


if __name__ == "__main__":
    main()

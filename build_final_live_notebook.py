"""Build the final live benchmark Colab notebook.

This notebook is intentionally gated by ALLOW_LIVE_RUN.  It is for the last
validation stage after ACMOT_MASTER_SWEEP.ipynb selects the best cached-replay
combination.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks" / "ACMOT_FINAL_LIVE_BENCHMARK.ipynb"


def cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(True),
    }


cells: list[dict] = []

cells.append(cell(r'''# CELL 1 - Setup, Drive mount, repo refresh
from pathlib import Path
from datetime import datetime
import json, os, sys, subprocess, time, math, shutil, textwrap

# SAFETY GATE:
# Keep False while preparing. Change to True only after the master sweep chooses the final combination.
ALLOW_LIVE_RUN = False

REPO_URL = "https://github.com/AhmedCode110/ACMOT-Codex-V10Style-Portable.git"
REPO = Path("/content/ACMOT-Codex-V10Style-Portable")

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    print("Drive mount skipped or unavailable:", repr(exc))

if REPO.exists():
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)

os.chdir(REPO)
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("Repo:", REPO)
print("Commit:", COMMIT)
print("Datetime:", datetime.now().isoformat(timespec="seconds"))
print("Drive mounted:", Path("/content/drive/MyDrive").exists())
print("ALLOW_LIVE_RUN:", ALLOW_LIVE_RUN)
'''))

cells.append(cell(r'''# CELL 2 - Install live benchmark dependencies
import subprocess, sys

PKGS = [
    "ultralytics==8.3.200",
    "motmetrics",
    "opencv-python-headless",
    "pandas",
    "numpy",
    "scipy",
    "lap",
    "pyyaml",
    "tqdm",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PKGS], check=True)

import numpy as np
for name, value in {"float": float, "int": int, "bool": bool}.items():
    if name not in np.__dict__:
        setattr(np, name, value)

print("Dependencies ready")
'''))

cells.append(cell(r'''# CELL 3 - Select latest sweep and finalist
import pandas as pd

SWEEP_ROOT = Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_SWEEPS")

# Optional manual overrides:
SWEEP_DIR_OVERRIDE = None   # example: "/content/drive/MyDrive/VisDrone_Results/ACMOT_SWEEPS/master_sweep_20260907_123456"
FINALIST_TRIAL_NAME = None  # example: "COMBINED_MATCH88_IDS_GUARD"; None = auto-pick best LIVE_CANDIDATE/PARETO

def latest_sweep(root):
    candidates = sorted([p for p in root.glob("master_sweep_*") if p.is_dir()])
    if not candidates:
        raise RuntimeError(f"No master_sweep_* folders found under {root}")
    return candidates[-1]

SWEEP_DIR = Path(SWEEP_DIR_OVERRIDE) if SWEEP_DIR_OVERRIDE else latest_sweep(SWEEP_ROOT)
LEADERBOARD = SWEEP_DIR / "LEADERBOARD.csv"
if not LEADERBOARD.exists():
    raise RuntimeError(f"Missing leaderboard: {LEADERBOARD}")

lb = pd.read_csv(LEADERBOARD)
candidate_mask = lb["status"].astype(str).str.contains("LIVE_CANDIDATE|PARETO|KEEP", regex=True, na=False)
candidate_df = lb[candidate_mask].copy()
if candidate_df.empty:
    candidate_df = lb.copy()

if FINALIST_TRIAL_NAME is None:
    selected_row = candidate_df.sort_values("rank").iloc[0]
    FINALIST_TRIAL_NAME = str(selected_row["trial"])
else:
    hit = lb[lb["trial"] == FINALIST_TRIAL_NAME]
    if hit.empty:
        raise RuntimeError(f"Requested finalist not found in leaderboard: {FINALIST_TRIAL_NAME}")
    selected_row = hit.iloc[0]

TRIAL_CONFIG = SWEEP_DIR / "trial_outputs" / FINALIST_TRIAL_NAME / "trial_config.json"
if not TRIAL_CONFIG.exists():
    raise RuntimeError(f"Missing trial_config.json: {TRIAL_CONFIG}")

trial = json.loads(TRIAL_CONFIG.read_text())
print("Sweep dir:", SWEEP_DIR)
print("Selected finalist:", FINALIST_TRIAL_NAME)
print("Replay rank:", selected_row.get("rank", "NA"))
print("Replay HOTA approx:", selected_row.get("HOTA_or_hota_approx", "NA"))
print("Replay MOTA:", selected_row.get("MOTA", "NA"))
print("Replay IDF1:", selected_row.get("IDF1", "NA"))
print("Replay IDS:", selected_row.get("IDS", "NA"))
print("Trial config:", TRIAL_CONFIG)
'''))

cells.append(cell(r'''# CELL 4 - Verify dataset, GT, and Tesla T4
import torch

DRIVE_DATASET = Path("/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev")
LOCAL_DATASET = Path("/content/VisDrone2019-MOT-test-dev")
OUTPUT_ROOT = Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_LIVE_FINAL")

if not (DRIVE_DATASET / "sequences").is_dir() or not (DRIVE_DATASET / "annotations").is_dir():
    raise RuntimeError(f"Drive dataset folders missing: {DRIVE_DATASET}")

# Copy once before timing. This keeps Drive I/O out of the measured live FPS.
if not (LOCAL_DATASET / "sequences").is_dir() or not (LOCAL_DATASET / "annotations").is_dir():
    if LOCAL_DATASET.exists():
        shutil.rmtree(LOCAL_DATASET)
    print("Copying dataset from Drive to local Colab SSD. This is outside live timing...")
    shutil.copytree(DRIVE_DATASET, LOCAL_DATASET)
else:
    print("Local dataset already staged:", LOCAL_DATASET)

DATASET = LOCAL_DATASET
SEQ_DIR = DATASET / "sequences"
ANN_DIR = DATASET / "annotations"

SEQS = sorted([p.name for p in SEQ_DIR.iterdir() if p.is_dir()])
ANNS = sorted([p.stem for p in ANN_DIR.glob("*.txt")])
missing_ann = [s for s in SEQS if s not in set(ANNS)]
frame_counts = {s: len(list((SEQ_DIR / s).glob("*.jpg"))) for s in SEQS}
total_frames = sum(frame_counts.values())

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA"
t4_ok = torch.cuda.is_available() and "T4" in gpu_name

print("Drive dataset:", DRIVE_DATASET)
print("Timed local dataset:", DATASET)
print("Total sequences:", len(SEQS))
print("Total annotations:", len(ANNS))
print("Missing annotations:", missing_ann or "None")
print("Total frames:", total_frames)
print("GPU:", gpu_name)
print("Tesla T4 confirmed:", t4_ok)

if len(SEQS) != 17 or missing_ann or total_frames != 6635:
    raise RuntimeError("Dataset is not complete/final. Stop before live benchmark.")
if not t4_ok:
    raise RuntimeError("Tesla T4 is required for publishable live FPS comparison.")
'''))

cells.append(cell(r'''# CELL 5 - Core live functions: SCI, YOLO once per frame, ByteTrack, metrics
import cv2, numpy as np, pandas as pd, time, json, math
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
from tqdm.auto import tqdm
import motmetrics as mm
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes

VISDRONE_GT_CLASSES = [1, 4, 5, 6, 9]
COCO_CLASSES = [0, 2, 5, 7]

@dataclass
class SceneState:
    sci: float = 0.0
    scene: str = "clear"
    tiny_ratio: float = 0.0
    object_count: int = 0

class LiveSceneAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hist = deque(maxlen=int(cfg.get("window", 7)))
        self.last = SceneState()
    def reset(self):
        self.hist.clear()
        self.last = SceneState()
    def maybe(self, frame, img, prev_boxes):
        stride = int(self.cfg.get("stride", 10))
        if frame != 1 and frame % stride != 1:
            return self.last
        small = cv2.resize(img, None, fx=0.25, fy=0.25)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edge_density = float(cv2.Canny(gray, 50, 120).mean() / 255.0)
        n = len(prev_boxes)
        crowd = min(n / float(self.cfg.get("crowd_divisor", 30)), 1.0)
        if n:
            areas = (prev_boxes[:, 2] - prev_boxes[:, 0]) * (prev_boxes[:, 3] - prev_boxes[:, 1])
            tiny = float(np.mean(areas < 32 * 32))
        else:
            tiny = 0.0
        raw = (
            float(self.cfg.get("crowd_w", 0.30)) * crowd
            + float(self.cfg.get("tiny_w", 0.30)) * tiny
            + float(self.cfg.get("edge_w", 0.20)) * min(edge_density / float(self.cfg.get("edge_norm", 0.14)), 1.0)
            + float(self.cfg.get("night_w", 0.10)) * (brightness < 80)
            + float(self.cfg.get("blur_w", 0.05)) * (blur < 180)
        )
        self.hist.append(float(np.clip(raw, 0.0, 1.0)))
        sci = float(np.mean(self.hist))
        scene = "night" if brightness < 80 else "blur" if blur < 180 else "tiny" if tiny > 0.50 else "crowded" if crowd > 0.65 or edge_density > 0.13 else "clear"
        self.last = SceneState(sci=sci, scene=scene, tiny_ratio=tiny, object_count=n)
        return self.last

class LiveCalibrator:
    def __init__(self, cfg):
        self.cfg = cfg
    def params(self, state):
        conf = float(self.cfg.get("conf_base", 0.245)) - float(self.cfg.get("conf_slope", 0.050)) * state.sci
        iou = float(self.cfg.get("iou_base", 0.490)) - float(self.cfg.get("iou_slope", 0.050)) * state.sci
        if state.scene in ["crowded", "tiny", "night"]:
            conf -= float(self.cfg.get("scene_conf_nudge", 0.012))
        if state.scene == "blur":
            iou -= float(self.cfg.get("blur_iou_nudge", 0.012))
        if self.cfg.get("ids_guard") and state.object_count > 25:
            conf += float(self.cfg.get("ids_guard_conf_add", 0.008))
            iou += float(self.cfg.get("ids_guard_iou_add", 0.010))
        imgsz = 832 if state.sci > float(self.cfg.get("sci_high", 0.60)) or state.tiny_ratio > float(self.cfg.get("tiny_gate", 0.50)) else 736 if state.sci > float(self.cfg.get("sci_mid", 0.35)) or state.scene in ["crowded", "tiny"] else 640
        return {
            "imgsz": int(imgsz),
            "conf": float(np.clip(conf, float(self.cfg.get("conf_floor", 0.19)), float(self.cfg.get("conf_ceil", 0.28)))),
            "iou": float(np.clip(iou, float(self.cfg.get("iou_floor", 0.40)), float(self.cfg.get("iou_ceil", 0.52)))),
        }

def make_tracker(cfg):
    return BYTETracker(SimpleNamespace(
        track_high_thresh=float(cfg.get("high", 0.18)),
        track_low_thresh=float(cfg.get("low", 0.04)),
        new_track_thresh=float(cfg.get("new", 0.20)),
        track_buffer=int(cfg.get("buffer", 45)),
        match_thresh=float(cfg.get("match", 0.86)),
        fuse_score=bool(cfg.get("fuse", True)),
    ), frame_rate=30)

def sync_cuda():
    torch.cuda.synchronize()

def load_gt(seq):
    cols = ["frame","id","x","y","w","h","score","class","truncation","occlusion"]
    df = pd.read_csv(ANN_DIR / f"{seq}.txt", header=None, names=cols)
    return df[(df["class"].isin(VISDRONE_GT_CLASSES)) & (df["score"] == 1) & (df["occlusion"] < 2) & (df["truncation"] < 2)].copy()

def iou_distance(pred, gt):
    pred = np.asarray(pred, dtype=float).reshape(-1, 4)
    gt = np.asarray(gt, dtype=float).reshape(-1, 4)
    if len(pred) == 0 or len(gt) == 0:
        return np.empty((len(gt), len(pred)))
    ix1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    iy1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    ix2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    iy2 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    gp = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    pp = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union = gp[:, None] + pp[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    dist = 1.0 - iou
    dist[iou < 0.5] = np.nan
    return dist

def evaluate_acc(acc, name):
    mh = mm.metrics.create()
    row = mh.compute(acc, metrics=["mota","idf1","num_switches","recall","precision","num_misses","num_false_positives","num_matches"], name=name).iloc[0]
    tp = int(row["num_matches"]); fp = int(row["num_false_positives"]); fn = int(row["num_misses"]); ids = int(row["num_switches"])
    det_a = tp / max(tp + fp + fn, 1)
    ass_a = max(0.0, 1.0 - ids / max(tp, 1))
    return {
        "MOTA": float(row["mota"]) * 100,
        "IDF1": float(row["idf1"]) * 100,
        "IDS": ids,
        "recall": float(row["recall"]) * 100,
        "precision": float(row["precision"]) * 100,
        "hota_approx_only_until_trackeval": math.sqrt(det_a * ass_a) * 100,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

print("Live functions ready")
'''))

cells.append(cell(r'''# CELL 6 - Safety gate before expensive live benchmark
print("Selected live finalist:", FINALIST_TRIAL_NAME)
print("ALLOW_LIVE_RUN:", ALLOW_LIVE_RUN)
print("This cell protects your GPU time.")

if not ALLOW_LIVE_RUN:
    raise RuntimeError(
        "Live benchmark is ready but blocked safely. "
        "After the sweep chooses the final combination, change ALLOW_LIVE_RUN=True in Cell 1 and Run all."
    )

print("Expensive GPU inference: YES")
print("Proceeding with exactly one YOLOv8n inference per frame.")
'''))

cells.append(cell(r'''# CELL 7 - Run final live benchmark on 17 sequences
RUN_TAG = "final_live_" + datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_ROOT / RUN_TAG
PRED_DIR = RUN_DIR / "predictions_mot" / FINALIST_TRIAL_NAME
RUN_DIR.mkdir(parents=True, exist_ok=False)
PRED_DIR.mkdir(parents=True, exist_ok=True)

(RUN_DIR / "trial_config.json").write_text(json.dumps(trial, indent=2), encoding="utf-8")
(RUN_DIR / "run_context.json").write_text(json.dumps({
    "repo": str(REPO),
    "commit": COMMIT,
    "sweep_dir": str(SWEEP_DIR),
    "finalist_trial": FINALIST_TRIAL_NAME,
    "drive_dataset": str(DRIVE_DATASET),
    "timed_local_dataset": str(DATASET),
    "gpu": torch.cuda.get_device_name(0),
    "protocol": "end-to-end live timing includes local SSD image loading, SceneAnalyzer, controller, YOLOv8n FP16, CUDA synchronization, and ByteTrack",
    "one_yolo_inference_per_frame": True,
    "precision_requested": "FP16",
}, indent=2), encoding="utf-8")

model = YOLO("yolov8n.pt")
ACTUAL_MODEL_FP16 = None
rows = []
total = sum(frame_counts.values())
global_start = time.perf_counter()

with tqdm(total=total, desc=FINALIST_TRIAL_NAME, dynamic_ncols=True) as pbar:
    for seq in SEQS:
        tracker = make_tracker(trial["tracker"])
        analyzer = LiveSceneAnalyzer(trial["sci"])
        calibrator = LiveCalibrator(trial["calib"])
        gt = load_gt(seq)
        acc = mm.MOTAccumulator(auto_id=True)
        prev_boxes = np.empty((0, 4))
        sizes, confs, scenes = [], [], []
        pred_lines = []
        seq_start = time.perf_counter()
        frames = sorted((SEQ_DIR / seq).glob("*.jpg"))
        for frame_path in frames:
            frame_idx = int(frame_path.stem)
            start = time.perf_counter()
            img = cv2.imread(str(frame_path))
            if img is None:
                raise RuntimeError(f"Unreadable frame: {frame_path}")
            if trial.get("adaptive", True):
                state = analyzer.maybe(frame_idx, img, prev_boxes)
                params = calibrator.params(state)
            else:
                state = SceneState()
                params = {"imgsz": 640, "conf": 0.25, "iou": 0.45}
            sync_cuda()
            result = model.predict(
                img,
                conf=float(params["conf"]),
                iou=float(params["iou"]),
                imgsz=int(params["imgsz"]),
                classes=COCO_CLASSES,
                max_det=1000,
                half=True,
                device=0,
                verbose=False,
            )[0]
            sync_cuda()
            if ACTUAL_MODEL_FP16 is None:
                ACTUAL_MODEL_FP16 = bool(getattr(getattr(model, "predictor", None).model, "fp16", False))
                print("actual_model_fp16:", ACTUAL_MODEL_FP16)
                if not ACTUAL_MODEL_FP16:
                    raise RuntimeError("FP16 was requested but Ultralytics did not report an FP16 model. Stop before publishing FPS.")
            dets = result.boxes.data.detach().cpu().numpy()
            if dets.size:
                dets = dets.reshape(-1, dets.shape[-1]).astype(float)
            else:
                dets = np.empty((0, 6), dtype=float)
            tracks = np.asarray(tracker.update(Boxes(dets, tuple(img.shape[:2]))), dtype=float).reshape(-1, 8)
            if len(tracks):
                ids = tracks[:, 4].astype(int)
                pred_boxes = tracks[:, :4].astype(float)
                pred_scores = tracks[:, 5].astype(float)
            else:
                ids = np.array([], dtype=int)
                pred_boxes = np.empty((0, 4), dtype=float)
                pred_scores = np.array([], dtype=float)
            prev_boxes = pred_boxes.copy()
            for tid, box, score in zip(ids, pred_boxes, pred_scores):
                x1, y1, x2, y2 = box.tolist()
                pred_lines.append(f"{frame_idx},{int(tid)},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{float(score):.6f},-1,-1,-1\n")
            gf = gt[gt.frame == frame_idx]
            gids = gf.id.values
            gboxes = np.column_stack([gf.x, gf.y, gf.x + gf.w, gf.y + gf.h]) if len(gf) else np.empty((0, 4))
            acc.update(gids, ids, iou_distance(pred_boxes, gboxes))
            sizes.append(params["imgsz"]); confs.append(params["conf"]); scenes.append(state.scene)
            elapsed = time.perf_counter() - global_start
            pbar.update(1)
            pbar.set_postfix(
                pct=f"{100*pbar.n/max(total,1):.1f}%",
                completed=f"{pbar.n}/{total}",
                elapsed=f"{elapsed/60:.1f}m",
                ETA=f"{((elapsed/max(pbar.n,1))*(total-pbar.n))/60:.1f}m",
                fps=f"{pbar.n/max(elapsed,1e-9):.2f}",
                seq=seq[-12:],
                imgsz=params["imgsz"],
            )
        (PRED_DIR / f"{seq}.txt").write_text("".join(pred_lines), encoding="utf-8")
        metrics = evaluate_acc(acc, seq)
        seq_seconds = time.perf_counter() - seq_start
        rows.append({
            "trial": FINALIST_TRIAL_NAME,
            "sequence": seq,
            "frames": len(frames),
            "live_seconds": seq_seconds,
            "live_fps": len(frames) / max(seq_seconds, 1e-9),
            "mean_imgsz": float(np.mean(sizes)),
            "mean_conf": float(np.mean(confs)),
            "dominant_scene": pd.Series(scenes).mode().iloc[0] if scenes else "NA",
            **metrics,
        })

per_seq = pd.DataFrame(rows)
per_seq.to_csv(RUN_DIR / "live_per_sequence_metrics.csv", index=False)

total_seconds = time.perf_counter() - global_start
summary = {
    "trial": FINALIST_TRIAL_NAME,
    "sequences": len(SEQS),
    "frames": int(per_seq["frames"].sum()),
    "live_seconds": total_seconds,
    "live_fps": float(per_seq["frames"].sum() / max(total_seconds, 1e-9)),
    "MOTA": float(np.average(per_seq["MOTA"], weights=per_seq["frames"])),
    "IDF1": float(np.average(per_seq["IDF1"], weights=per_seq["frames"])),
    "IDS": int(per_seq["IDS"].sum()),
    "hota_approx_only_until_trackeval": float(np.average(per_seq["hota_approx_only_until_trackeval"], weights=per_seq["frames"])),
    "mean_imgsz": float(np.average(per_seq["mean_imgsz"], weights=per_seq["frames"])),
    "precision": "YOLOv8n FP16",
    "actual_model_fp16": bool(ACTUAL_MODEL_FP16),
    "hardware": torch.cuda.get_device_name(0),
    "status": "LIVE_COMPLETE",
}
pd.DataFrame([summary]).to_csv(RUN_DIR / "live_summary.csv", index=False)
(RUN_DIR / "live_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Live run complete:", RUN_DIR)
print(pd.DataFrame([summary]).to_string(index=False))
'''))

cells.append(cell(r'''# CELL 8 - Optional official TrackEval placeholder for the single finalist
TRACK_EVAL_FINALIST = False

print("Prediction files:", PRED_DIR)
print("If you need official HOTA, set TRACK_EVAL_FINALIST=True and run a TrackEval export/evaluation cell.")
print("Current live_summary uses motmetrics + HOTA approximation only.")

if TRACK_EVAL_FINALIST:
    raise NotImplementedError(
        "TrackEval should be run only for the final selected output. "
        "Use the existing TrackEval export helper or ask Codex for a dedicated single-finalist TrackEval cell."
    )
'''))

cells.append(cell(r'''# CELL 9 - Final live report
summary_path = RUN_DIR / "live_summary.json"
summary = json.loads(summary_path.read_text())
report = f"""# AC-MOT Final Live Benchmark

Run folder: `{RUN_DIR}`

## Finalist

- Trial: `{FINALIST_TRIAL_NAME}`
- Source sweep: `{SWEEP_DIR}`
- Repo commit: `{COMMIT}`
- Drive dataset source: `{DRIVE_DATASET}`
- Timed local dataset: `{DATASET}`
- Hardware: `{summary['hardware']}`
- Precision: `{summary['precision']}`
- Actual model FP16 reported: `{summary.get('actual_model_fp16')}`

## Live result

- HOTA approx: {summary['hota_approx_only_until_trackeval']:.3f}
- MOTA: {summary['MOTA']:.3f}
- IDF1: {summary['IDF1']:.3f}
- IDS: {summary['IDS']}
- Live FPS: {summary['live_fps']:.2f}
- Mean image size: {summary['mean_imgsz']:.1f}

## Timing protocol

End-to-end timing includes image loading from local Colab SSD, SceneAnalyzer, controller, exactly one YOLOv8n FP16 inference per frame, CUDA synchronization, and ByteTrack. Dataset copy from Drive is outside timing.

## Safety note

This notebook runs expensive YOLO only when `ALLOW_LIVE_RUN=True`. It stages the dataset to `/content` first and stops if FP16 is not actually reported.
"""
(RUN_DIR / "FINAL_LIVE_REPORT.md").write_text(report, encoding="utf-8")
print(report)
print("Saved report:", RUN_DIR / "FINAL_LIVE_REPORT.md")
'''))


nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(f"Wrote {OUT} with {len(cells)} cells")

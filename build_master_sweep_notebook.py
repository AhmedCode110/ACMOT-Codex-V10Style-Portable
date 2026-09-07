"""Build the master cache/replay sweep Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks" / "ACMOT_MASTER_SWEEP.ipynb"


def cell(source: str, cell_type: str = "code") -> dict:
    data = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(True)}
    if cell_type == "code":
        data.update({"execution_count": None, "outputs": []})
    return data


cells: list[dict] = []

cells.append(cell(r'''# CELL 1 - Mount Drive and clone repo
from pathlib import Path
from datetime import datetime
import json, os, sys, subprocess, time, gzip, math, shutil, textwrap, hashlib

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
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
drive_mounted = Path("/content/drive/MyDrive").exists()

print("repo path:", REPO)
print("git commit:", commit)
print("current date/time:", datetime.now().isoformat(timespec="seconds"))
print("Drive mounted status:", drive_mounted)
print("ALLOW_LIVE_RUN:", ALLOW_LIVE_RUN)
'''))

cells.append(cell(r'''# CELL 2 - Install minimal dependencies
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

try:
    import trackeval
    print("TrackEval already importable:", trackeval.__file__)
except Exception:
    print("TrackEval not importable yet. It will be installed only in finalist TrackEval cell if needed.")

print("Minimal replay dependencies ready")
print("Expensive GPU inference: NO")
'''))

cells.append(cell(r'''# CELL 3 - Cache discovery and verification
import subprocess, sys

commands = [
    [sys.executable, "cache_manager_v10_p4.py", "discover"],
    [sys.executable, "cache_manager_v10_p4.py", "verify"],
    [sys.executable, "cache_manager_v10_p4.py", "status"],
]
for cmd in commands:
    print("\n$", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

print("Expensive GPU inference: NO")
'''))

cells.append(cell(r'''# CELL 4 - Experiment dry-run
import subprocess, sys

for mode in ["replay", "auto"]:
    cmd = [sys.executable, "experiment.py", "--mode", mode, "--dry-run"]
    print("\n$", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if proc.returncode != 0:
        print("Dry-run returned non-zero but notebook will continue to its own replay sweep code.")

print("Expensive GPU inference: NO")
'''))

cells.append(cell(r'''# CELL 5 - Locate cached detections
from pathlib import Path
import json, gzip, os

KNOWN_CACHE_NAMES = [
    "detection_cache_v10_p3_yolov8n_fp16_640_736_832",
    "detection_cache_v10_p3_yolov8n_fp32_640_736_832",
    "detection_cache_v1",
    "ACMOT_IDS",
    "ROUND2_replay",
    "IDS_replay",
    "V10_P4_FP16_FAIR",
    "codex_v10_p4_fp16_20260907_134125",
    "fp16",
    "half",
    "yolov8n_fp16",
]

COMMON_ROOTS = [
    Path("/content/drive/MyDrive"),
    Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_CODEX_V10STYLE"),
    Path("/content/drive/MyDrive/VisDrone_Results"),
    Path("/content/drive/MyDrive/AC-MOT-results"),
    Path("/content/drive/MyDrive/visdrone"),
    Path("/content/drive/MyDrive/ACMOT_MASTER"),
]

def candidate_paths():
    direct = [
        Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_CODEX_V10STYLE/detection_cache_v10_p3_yolov8n_fp16_640_736_832"),
        Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_CODEX_V10STYLE/V10_P4_FP16_FAIR/detection_cache_v10_p3_yolov8n_fp16_640_736_832"),
        Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_CODEX_V10STYLE/codex_v10_p4_fp16_20260907_134125/detection_cache_v10_p3_yolov8n_fp16_640_736_832"),
        Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_CODEX_V10STYLE/detection_cache_v10_p3_yolov8n_fp32_640_736_832")
    ]
    for p in direct:
        if p.exists():
            yield p
    seen = set()
    for root in COMMON_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if ".ipynb_checkpoints" in path.parts:
                continue
            low = str(path).lower()
            looks_named = any(name.lower() in low for name in KNOWN_CACHE_NAMES) or "detection_cache" in low or "detections" in low
            looks_structural = path.is_dir() and (any(path.glob("*.jsonl.gz")) or any(path.glob("*.complete.json")))
            if looks_named or looks_structural:
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                yield path

def read_json_if_exists(path):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}

def infer_precision(path, meta):
    hay = (str(path) + " " + json.dumps(meta, sort_keys=True, default=str)).lower()
    for key in ["precision", "detector_precision", "dtype"]:
        val = str(meta.get(key, "")).lower()
        if "fp16" in val or "half" in val:
            return "FP16"
        if "fp32" in val or "float32" in val:
            return "FP32"
    if "fp16" in hay or "half" in hay:
        return "FP16"
    if "fp32" in hay or "float32" in hay:
        return "FP32"
    return "UNKNOWN"

def validate_detection_cache(path):
    if not path.exists() or not path.is_dir():
        return {"status": "MISSING", "path": str(path)}
    meta_path = path / "cache_meta.json"
    alt_meta_path = path / "cache.json"
    meta = read_json_if_exists(meta_path) or read_json_if_exists(alt_meta_path)
    jsonl = sorted(path.glob("*.jsonl.gz"))
    complete = sorted(path.glob("*.complete.json"))
    frames = 0
    sample_ok = False
    if jsonl:
        with gzip.open(jsonl[0], "rt", encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        sample_ok = all(str(k) in first.get("bank", {}) for k in [640, 736, 832])
    for done in complete:
        try:
            frames += int(json.loads(done.read_text()).get("frames", 0))
        except Exception:
            pass
    precision = infer_precision(path, meta)
    status = "VALID" if len(jsonl) == 17 and len(complete) == 17 and sample_ok else "PARTIAL"
    return {
        "status": status,
        "path": str(path),
        "cache_meta_exists": meta_path.exists() or alt_meta_path.exists(),
        "jsonl_gz_files": len(jsonl),
        "complete_files": len(complete),
        "frame_count_from_complete": frames or None,
        "sample_has_640_736_832": sample_ok,
        "precision": precision,
    }

cache_reports = [validate_detection_cache(p) for p in dict.fromkeys(candidate_paths())]
valid_reports = [r for r in cache_reports if r["status"] == "VALID"]
if not cache_reports:
    raise RuntimeError("No candidate detection caches found. No YOLO was run.")

CACHE_REPORTS = cache_reports
def cache_rank(report):
    precision_score = {"FP16": 0, "FP32": 1, "UNKNOWN": 2}.get(report["precision"], 2)
    frame_score = 0 if report.get("frame_count_from_complete") in (6635, None) else 1
    return (precision_score, frame_score, report["path"])

valid_reports = sorted(valid_reports, key=cache_rank)
selected_report = valid_reports[0] if valid_reports else sorted(cache_reports, key=cache_rank)[0]
DETECTION_CACHE = Path(selected_report["path"])
REPLAY_CACHE_PRECISION = selected_report["precision"]

print("Cache candidates:")
for r in cache_reports:
    print(json.dumps(r, indent=2))

if not valid_reports:
    raise RuntimeError("No complete valid detection cache found. Stop before replay.")

print("Selected cache:", DETECTION_CACHE)
print("Replay cache precision:", REPLAY_CACHE_PRECISION)
print("Selection policy: prefer complete FP16 cache, then complete FP32 cache, never run YOLO here.")
print("Valid for development replay: YES")
print("Valid for FP16 final evidence:", "YES" if REPLAY_CACHE_PRECISION == "FP16" else "NO")
print("Valid for live FPS: NO")
print("Expensive GPU inference: NO")
'''))

cells.append(cell(r'''# CELL 6 - Dataset / GT path verification
from pathlib import Path
import pandas as pd

EXPECTED_DATASET = Path("/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev")
DATASET = EXPECTED_DATASET
SEQ_DIR = DATASET / "sequences"
ANN_DIR = DATASET / "annotations"

if not SEQ_DIR.is_dir() or not ANN_DIR.is_dir():
    raise RuntimeError(f"Dataset layout missing: {DATASET}")

SEQS = sorted([p.name for p in SEQ_DIR.iterdir() if p.is_dir()])
ANNS = sorted([p.stem for p in ANN_DIR.glob("*.txt")])
missing_ann = [s for s in SEQS if s not in set(ANNS)]
frame_counts = {s: len(list((SEQ_DIR / s).glob("*.jpg"))) for s in SEQS}
total_frames = sum(frame_counts.values())

print("Dataset:", DATASET)
print("Total sequences:", len(SEQS))
print("Total annotation txt files:", len(ANNS))
print("Total frames:", total_frames)
print("Missing annotations:", missing_ann or "None")

if len(SEQS) != 17 or missing_ann:
    raise RuntimeError("Dataset/GT is incomplete. Stop before replay.")
if total_frames != 6635:
    print("WARNING: expected 6635 frames; verify before final claims.")

GT_FILTER = {"classes": [1, 4, 5, 6, 9], "score": 1, "occlusion_lt": 2, "truncation_lt": 2}
print("GT filter:", GT_FILTER)
print("Expensive GPU inference: NO")
'''))

cells.append(cell(r'''# CELL 7 - Core replay functions
import cv2, yaml, time, gzip, json, math
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tqdm.auto import tqdm
import motmetrics as mm

COCO_CLASSES = [0, 2, 5, 7]
VISDRONE_GT_CLASSES = [1, 4, 5, 6, 9]

@dataclass
class SceneState:
    sci: float
    scene: str
    brightness: float
    blur: float
    edge_density: float
    crowd: float
    tiny_ratio: float
    n_dets: int

class CachedSceneAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hist = deque(maxlen=int(cfg.get("window", 7)))
        self.state = SceneState(0.0, "clear", 128, 500, 0.0, 0.0, 0.0, 0)
    def reset(self):
        self.hist.clear()
        self.state = SceneState(0.0, "clear", 128, 500, 0.0, 0.0, 0.0, 0)
    def maybe(self, frame_id, visual, prev_boxes):
        stride = int(self.cfg.get("stride", 10))
        if frame_id != 1 and (frame_id - 1) % stride != 0:
            return self.state
        brightness = float(visual.get("brightness", 128.0))
        blur = float(visual.get("blur", 500.0))
        edge = float(visual.get("edge_density", 0.0))
        n = len(prev_boxes)
        crowd = min(n / float(self.cfg.get("crowd_divisor", 30.0)), 1.0)
        tiny = 0.0
        if n:
            areas = (prev_boxes[:, 2] - prev_boxes[:, 0]) * (prev_boxes[:, 3] - prev_boxes[:, 1])
            tiny = float(np.mean(areas < float(self.cfg.get("tiny_area", 32 * 32))))
        raw = float(self.cfg.get("crowd_w", 0.30)) * crowd
        raw += float(self.cfg.get("tiny_w", 0.30)) * tiny
        raw += float(self.cfg.get("edge_w", 0.20)) * min(edge / float(self.cfg.get("edge_norm", 0.14)), 1.0)
        raw += float(self.cfg.get("night_w", 0.10)) * float(brightness < float(self.cfg.get("night_brightness", 80)))
        raw += float(self.cfg.get("blur_w", 0.05)) * float(blur < float(self.cfg.get("blur_thresh", 180)))
        self.hist.append(float(np.clip(raw, 0.0, 1.0)))
        sci = float(np.mean(self.hist))
        if brightness < 80:
            scene = "night"
        elif blur < 180:
            scene = "blur"
        elif tiny > 0.50:
            scene = "tiny"
        elif crowd > 0.65 or edge > 0.13:
            scene = "crowded"
        else:
            scene = "clear"
        self.state = SceneState(sci, scene, brightness, blur, edge, crowd, tiny, n)
        return self.state

class SmartCalibrator:
    def __init__(self, cfg):
        self.cfg = cfg
    def params(self, state):
        conf = float(self.cfg.get("conf_base", 0.245)) - float(self.cfg.get("conf_slope", 0.050)) * state.sci
        iou = float(self.cfg.get("iou_base", 0.490)) - float(self.cfg.get("iou_slope", 0.050)) * state.sci
        if state.scene in {"crowded", "tiny", "night"}:
            conf -= float(self.cfg.get("scene_conf_nudge", 0.012))
        if state.scene == "blur":
            iou -= float(self.cfg.get("blur_iou_nudge", 0.012))
        if self.cfg.get("ids_guard") and state.crowd > 0.65:
            conf += float(self.cfg.get("ids_guard_conf_add", 0.008))
            iou += float(self.cfg.get("ids_guard_iou_add", 0.010))
        if state.sci > float(self.cfg.get("sci_high", 0.60)) or state.tiny_ratio > float(self.cfg.get("tiny_gate", 0.50)):
            imgsz = 832
        elif state.sci > float(self.cfg.get("sci_mid", 0.35)) or state.scene in {"crowded", "tiny"}:
            imgsz = 736
        else:
            imgsz = 640
        return {
            "conf": float(np.clip(conf, float(self.cfg.get("conf_floor", 0.19)), float(self.cfg.get("conf_ceil", 0.28)))),
            "iou": float(np.clip(iou, float(self.cfg.get("iou_floor", 0.40)), float(self.cfg.get("iou_ceil", 0.52)))),
            "imgsz": int(imgsz),
        }

def load_gt(seq):
    cols = ["frame", "id", "x", "y", "w", "h", "score", "cat", "trunc", "occ"]
    df = pd.read_csv(ANN_DIR / f"{seq}.txt", header=None, names=cols)
    df = df[df["cat"].isin(VISDRONE_GT_CLASSES)]
    df = df[(df["score"] == 1) & (df["occ"] < 2) & (df["trunc"] < 2)]
    return df.reset_index(drop=True)

def iou_distance(pred_xyxy, gt_xyxy):
    pred = np.asarray(pred_xyxy, dtype=float).reshape(-1, 4)
    gt = np.asarray(gt_xyxy, dtype=float).reshape(-1, 4)
    if not len(pred) or not len(gt):
        return np.empty((len(gt), len(pred)))
    ix1 = np.maximum(pred[:, 0][None, :], gt[:, 0][:, None])
    iy1 = np.maximum(pred[:, 1][None, :], gt[:, 1][:, None])
    ix2 = np.minimum(pred[:, 2][None, :], gt[:, 2][:, None])
    iy2 = np.minimum(pred[:, 3][None, :], gt[:, 3][:, None])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = ap[None, :] + ag[:, None] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    dist = 1.0 - iou
    dist[iou < 0.5] = np.nan
    return dist

def cached_dets(rec, params):
    size = str(int(params["imgsz"]))
    if size not in rec["bank"]:
        raise RuntimeError(f"Required imgsz={size} missing from cache. No approximation allowed.")
    arr = np.asarray(rec["bank"][size], dtype=float)
    if arr.size == 0:
        return np.empty((0,4)), np.array([], dtype=float), np.array([], dtype=int)
    arr = arr.reshape(-1, arr.shape[-1])
    arr = arr[arr[:, 4] >= float(params["conf"])]
    boxes = arr[:, :4]
    scores = arr[:, 4]
    classes = arr[:, 5].astype(int) if arr.shape[1] > 5 else np.zeros(len(arr), dtype=int)
    return boxes, scores, classes

from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes

def make_byte_tracker(tracker_cfg):
    return BYTETracker(SimpleNamespace(
        track_high_thresh=float(tracker_cfg.get("high", 0.18)),
        track_low_thresh=float(tracker_cfg.get("low", 0.04)),
        new_track_thresh=float(tracker_cfg.get("new", 0.20)),
        track_buffer=int(tracker_cfg.get("buffer", 45)),
        match_thresh=float(tracker_cfg.get("match", 0.86)),
        fuse_score=bool(tracker_cfg.get("fuse", True)),
    ), frame_rate=30)

def track_from_cache(tracker, boxes, scores, classes, shape, params, tracker_cfg):
    if len(boxes):
        dets = np.column_stack([boxes, scores, classes]).astype(float)
    else:
        dets = np.empty((0, 6), dtype=float)
    tracker.args.track_high_thresh = float(tracker_cfg.get("high", 0.18))
    tracker.args.track_low_thresh = float(tracker_cfg.get("low", 0.04))
    tracker.args.new_track_thresh = float(tracker_cfg.get("new", 0.20))
    tracker.args.match_thresh = float(tracker_cfg.get("match", 0.86))
    tracker.args.track_buffer = int(tracker_cfg.get("buffer", 45))
    tracker.args.fuse_score = bool(tracker_cfg.get("fuse", True))
    out = np.asarray(tracker.update(Boxes(dets, tuple(shape))), dtype=float).reshape(-1, 8)
    if len(out):
        ids = out[:, 4].astype(int)
        pred_boxes = out[:, :4].astype(float)
        pred_scores = out[:, 5].astype(float)
    else:
        ids = np.array([], dtype=int)
        pred_boxes = np.empty((0, 4), dtype=float)
        pred_scores = np.array([], dtype=float)
    return ids, pred_boxes, pred_scores

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
    }

print("Core replay functions ready")
print("No YOLO import or inference is used in replay cells")
'''))

cells.append(cell(r'''# CELL 8 - Trial definitions
FROZEN_REFERENCE = {"HOTA": 22.856, "MOTA": 11.607, "IDF1": 21.963, "IDS": 270, "Live_FP16_FPS": 25.99}

SCI_CURRENT = {
    "crowd_w": 0.30, "tiny_w": 0.30, "edge_w": 0.20, "night_w": 0.10, "blur_w": 0.05,
    "crowd_divisor": 30, "edge_norm": 0.14, "window": 7, "stride": 10,
}
CALIB_CURRENT = {
    "conf_base": 0.245, "conf_slope": 0.050, "conf_floor": 0.19, "conf_ceil": 0.28,
    "iou_base": 0.490, "iou_slope": 0.050, "iou_floor": 0.40, "iou_ceil": 0.52,
    "scene_conf_nudge": 0.012, "blur_iou_nudge": 0.012, "sci_mid": 0.35, "sci_high": 0.60,
    "tiny_gate": 0.50,
}

TRACKER_TRIALS = [
    {"suffix": "CURRENT", "high": 0.18, "low": 0.04, "new": 0.20, "buffer": 45, "match": 0.86},
    {"suffix": "MATCH_084", "high": 0.18, "low": 0.04, "new": 0.20, "buffer": 45, "match": 0.84},
    {"suffix": "MATCH_088", "high": 0.18, "low": 0.04, "new": 0.20, "buffer": 45, "match": 0.88},
    {"suffix": "MATCH_090", "high": 0.18, "low": 0.04, "new": 0.20, "buffer": 45, "match": 0.90},
    {"suffix": "NEW22_MATCH88", "high": 0.18, "low": 0.04, "new": 0.22, "buffer": 45, "match": 0.88},
    {"suffix": "NEW24_MATCH88", "high": 0.18, "low": 0.04, "new": 0.24, "buffer": 45, "match": 0.88},
    {"suffix": "BUFFER60", "high": 0.18, "low": 0.04, "new": 0.20, "buffer": 60, "match": 0.86},
    {"suffix": "BUFFER60_NEW22_MATCH88", "high": 0.18, "low": 0.04, "new": 0.22, "buffer": 60, "match": 0.88},
    {"suffix": "BUFFER75_NEW22_MATCH88", "high": 0.18, "low": 0.04, "new": 0.22, "buffer": 75, "match": 0.88},
]

SCI_TRIALS = [
    ("SCI_CURRENT", {}),
    ("SCI_TINY_PLUS", {"tiny_w": 0.36, "crowd_w": 0.27}),
    ("SCI_CROWD_PLUS", {"crowd_w": 0.36, "tiny_w": 0.27}),
    ("SCI_EDGE_MINUS", {"edge_w": 0.14}),
    ("SCI_WINDOW_5", {"window": 5}),
    ("SCI_WINDOW_9", {"window": 9}),
    ("SCI_STRIDE_5", {"stride": 5}),
    ("SCI_STRIDE_15", {"stride": 15}),
]

CALIB_TRIALS = [
    ("CALIB_CURRENT", {}),
    ("CALIB_IDS_GUARD", {"ids_guard": True, "ids_guard_conf_add": 0.008, "ids_guard_iou_add": 0.010}),
    ("CALIB_CONF_FLOOR_020", {"conf_floor": 0.20}),
    ("CALIB_SCI_MID_040", {"sci_mid": 0.40}),
    ("CALIB_SCI_HIGH_065", {"sci_high": 0.65}),
    ("CALIB_DENSITY_GATE", {"ids_guard": True, "conf_floor": 0.20, "sci_mid": 0.40}),
    ("CALIB_TINY_GATE", {"tiny_gate": 0.45, "sci_high": 0.65}),
]

def merge(base, patch):
    out = dict(base); out.update(patch); return out

baseline_trials = [
    {"trial": "Baseline_Default", "group": "baseline", "tracker": {"high": 0.25, "low": 0.10, "new": 0.25, "buffer": 30, "match": 0.80, "fuse": True}, "sci": SCI_CURRENT, "calib": CALIB_CURRENT, "adaptive": False},
    {"trial": "Baseline_TunedTracker", "group": "baseline", "tracker": TRACKER_TRIALS[0], "sci": SCI_CURRENT, "calib": CALIB_CURRENT, "adaptive": False},
    {"trial": "ACMOT_V10STYLE_SCI", "group": "baseline", "tracker": TRACKER_TRIALS[0], "sci": SCI_CURRENT, "calib": CALIB_CURRENT, "adaptive": True},
]
tracker_trials = [{"trial": "TRK_" + t["suffix"], "group": "tracker", "tracker": t, "sci": SCI_CURRENT, "calib": CALIB_CURRENT, "adaptive": True} for t in TRACKER_TRIALS]
sci_trials = [{"trial": name, "group": "sci", "tracker": TRACKER_TRIALS[0], "sci": merge(SCI_CURRENT, patch), "calib": CALIB_CURRENT, "adaptive": True} for name, patch in SCI_TRIALS]
calibrator_trials = [{"trial": name, "group": "calibrator", "tracker": TRACKER_TRIALS[0], "sci": SCI_CURRENT, "calib": merge(CALIB_CURRENT, patch), "adaptive": True} for name, patch in CALIB_TRIALS]
density_gate_trials = [t for t in calibrator_trials if "GATE" in t["trial"] or "GUARD" in t["trial"]]
combined_trials = [
    {"trial": "COMBINED_MATCH88_IDS_GUARD", "group": "combined", "tracker": TRACKER_TRIALS[2], "sci": SCI_CURRENT, "calib": merge(CALIB_CURRENT, {"ids_guard": True}), "adaptive": True},
    {"trial": "COMBINED_BUFFER60_CONF020", "group": "combined", "tracker": TRACKER_TRIALS[6], "sci": SCI_CURRENT, "calib": merge(CALIB_CURRENT, {"conf_floor": 0.20}), "adaptive": True},
    {"trial": "COMBINED_TINY_PLUS_GATE", "group": "combined", "tracker": TRACKER_TRIALS[0], "sci": merge(SCI_CURRENT, {"tiny_w": 0.36}), "calib": merge(CALIB_CURRENT, {"tiny_gate": 0.45}), "adaptive": True},
]

TRIALS = baseline_trials + tracker_trials + sci_trials + calibrator_trials + combined_trials
print("Trials:", len(TRIALS))
print("Groups:", sorted(set(t["group"] for t in TRIALS)))
'''))

cells.append(cell(r'''# CELL 9 - Dry-run the sweep
from datetime import datetime

SWEEP_ROOT = Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_SWEEPS")
SWEEP_DIR = SWEEP_ROOT / ("master_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
TRIAL_OUTPUTS = SWEEP_DIR / "trial_outputs"
PLOTS_DIR = SWEEP_DIR / "plots"
FINALISTS_DIR = SWEEP_DIR / "finalists"

print("Number of trials:", len(TRIALS))
print("Cache path:", DETECTION_CACHE)
print("GT path:", ANN_DIR)
print("Precision:", REPLAY_CACHE_PRECISION)
print("Will YOLO run? NO")
print("Will live FPS be measured? NO")
print("Will replay metrics be generated? YES")
print("Sweep output:", SWEEP_DIR)

required_cache = all((DETECTION_CACHE / f"{seq}.jsonl.gz").exists() and (DETECTION_CACHE / f"{seq}.complete.json").exists() for seq in SEQS)
if not required_cache:
    missing = [seq for seq in SEQS if not (DETECTION_CACHE / f"{seq}.jsonl.gz").exists()]
    raise RuntimeError("Required cache files missing. No YOLO was run. Missing: " + ", ".join(missing[:10]))

print("Replay cache precision:", REPLAY_CACHE_PRECISION)
print("Valid for development replay: YES")
print("Valid for FP16 final evidence:", "YES" if REPLAY_CACHE_PRECISION == "FP16" else "NO")
print("Valid for live FPS: NO")
print("Expensive GPU inference: NO")
'''))

cells.append(cell(r'''# CELL 10 - Run all replay trials
SWEEP_DIR.mkdir(parents=True, exist_ok=False)
TRIAL_OUTPUTS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
FINALISTS_DIR.mkdir(parents=True, exist_ok=True)

def run_trial(trial):
    t0 = time.perf_counter()
    rows = []
    pred_root = TRIAL_OUTPUTS / trial["trial"] / "predictions_mot"
    pred_root.mkdir(parents=True, exist_ok=True)
    (TRIAL_OUTPUTS / trial["trial"] / "trial_config.json").write_text(json.dumps(trial, indent=2), encoding="utf-8")
    for seq in tqdm(SEQS, desc=trial["trial"], leave=False):
        gt = load_gt(seq)
        acc = mm.MOTAccumulator(auto_id=True)
        analyzer = CachedSceneAnalyzer(trial["sci"])
        calibrator = SmartCalibrator(trial["calib"])
        tracker = make_byte_tracker(trial["tracker"])
        analyzer.reset()
        prev_boxes = np.empty((0, 4))
        size_log = []
        lines = []
        seq_start = time.perf_counter()
        with gzip.open(DETECTION_CACHE / f"{seq}.jsonl.gz", "rt", encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                frame = int(rec["frame"])
                if trial["adaptive"]:
                    state = analyzer.maybe(frame, rec.get("visual", {}), prev_boxes)
                    params = calibrator.params(state)
                else:
                    params = {"conf": 0.25, "iou": 0.45, "imgsz": 640}
                boxes, scores, classes = cached_dets(rec, params)
                ids, pred_boxes, pred_scores = track_from_cache(tracker, boxes, scores, classes, rec.get("shape", (1080, 1920)), params, trial["tracker"])
                prev_boxes = pred_boxes.copy()
                size_log.append(params["imgsz"])
                for tid, box, score in zip(ids, pred_boxes, pred_scores):
                    x1, y1, x2, y2 = box.tolist()
                    lines.append(f"{frame},{int(tid)},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{float(score):.6f},-1,-1,-1\n")
                gf = gt[gt.frame == frame]
                gids = gf.id.values
                gboxes = np.column_stack([gf.x, gf.y, gf.x + gf.w, gf.y + gf.h]) if len(gf) else np.empty((0,4))
                acc.update(gids, ids, iou_distance(pred_boxes, gboxes))
        (pred_root / f"{seq}.txt").write_text("".join(lines), encoding="utf-8")
        metrics = evaluate_acc(acc, seq)
        rows.append({"trial": trial["trial"], "group": trial["group"], "sequence": seq, "frames": frame_counts[seq], "mean_imgsz": float(np.mean(size_log)), **metrics, "seq_replay_seconds": time.perf_counter() - seq_start})
    df = pd.DataFrame(rows)
    df.to_csv(TRIAL_OUTPUTS / trial["trial"] / "per_sequence_metrics.csv", index=False)
    frames = int(df["frames"].sum())
    return {
        "trial": trial["trial"], "group": trial["group"], "frames": frames,
        "HOTA_or_hota_approx": float(np.average(df["hota_approx_only_until_trackeval"], weights=df["frames"])),
        "MOTA": float(np.average(df["MOTA"], weights=df["frames"])),
        "IDF1": float(np.average(df["IDF1"], weights=df["frames"])),
        "IDS": int(df["IDS"].sum()),
        "mean_imgsz": float(np.average(df["mean_imgsz"], weights=df["frames"])),
        "replay_time": time.perf_counter() - t0,
        "replay_fps_not_live": frames / max(time.perf_counter() - t0, 1e-9),
        "status": "UNCLASSIFIED",
        "notes": "motmetrics replay estimate; HOTA is approximate until official TrackEval",
    }

all_results = []
jsonl_path = SWEEP_DIR / "SWEEP_RESULTS.jsonl"
with jsonl_path.open("w", encoding="utf-8") as jf:
    for trial in tqdm(TRIALS, desc="Master replay sweep"):
        result = run_trial(trial)
        all_results.append(result)
        jf.write(json.dumps(result) + "\n")
        pd.DataFrame(all_results).to_csv(SWEEP_DIR / "SWEEP_RESULTS.csv", index=False)

SWEEP_RESULTS = pd.DataFrame(all_results)
print("Saved:", SWEEP_DIR / "SWEEP_RESULTS.csv")
display(SWEEP_RESULTS.sort_values(["HOTA_or_hota_approx","IDF1"], ascending=False).head(10))
print("No YOLO inference was run.")
'''))

cells.append(cell(r'''# CELL 11 - Build leaderboard
df = SWEEP_RESULTS.copy()
ref = FROZEN_REFERENCE
df["delta_HOTA"] = df["HOTA_or_hota_approx"] - ref["HOTA"]
df["delta_MOTA"] = df["MOTA"] - ref["MOTA"]
df["delta_IDF1"] = df["IDF1"] - ref["IDF1"]
df["delta_IDS"] = df["IDS"] - ref["IDS"]

def norm(s, inverse=False):
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series([0.5] * len(s), index=s.index)
    v = (s - s.min()) / (s.max() - s.min())
    return 1 - v if inverse else v

df["balanced_score"] = (
    0.40 * norm(df["HOTA_or_hota_approx"]) +
    0.35 * norm(df["IDF1"]) +
    0.15 * norm(df["MOTA"]) +
    0.10 * norm(df["IDS"], inverse=True)
)
df = df.sort_values("balanced_score", ascending=False).reset_index(drop=True)
df.insert(0, "rank", range(1, len(df) + 1))

cols = ["rank","trial","group","HOTA_or_hota_approx","MOTA","IDF1","IDS","delta_HOTA","delta_MOTA","delta_IDF1","delta_IDS","mean_imgsz","replay_fps_not_live","status","notes","balanced_score"]
LEADERBOARD = df[cols]
LEADERBOARD.to_csv(SWEEP_DIR / "LEADERBOARD.csv", index=False)
(SWEEP_DIR / "LEADERBOARD.md").write_text(LEADERBOARD.to_markdown(index=False), encoding="utf-8")

print("Saved:", SWEEP_DIR / "LEADERBOARD.csv")
print("Saved:", SWEEP_DIR / "LEADERBOARD.md")
display(LEADERBOARD.head(15))
print("HOTA label: hota_approx_only_until_trackeval")
'''))

cells.append(cell(r'''# CELL 12 - Selection rules
def classify(row):
    if row["HOTA_or_hota_approx"] > ref["HOTA"] and row["IDF1"] > ref["IDF1"] and row["MOTA"] >= ref["MOTA"] and row["IDS"] < ref["IDS"]:
        return "LIVE_CANDIDATE"
    if (row["HOTA_or_hota_approx"] > ref["HOTA"] or row["IDF1"] > ref["IDF1"]) and row["IDS"] <= 270:
        return "LIVE_CANDIDATE"
    if row["IDS"] < ref["IDS"] - 20 and row["IDF1"] >= ref["IDF1"] - 1.0:
        return "PARETO"
    if row["HOTA_or_hota_approx"] < ref["HOTA"] - 3 or row["IDF1"] < ref["IDF1"] - 3:
        return "REJECT"
    return "KEEP"

LEADERBOARD["status"] = LEADERBOARD.apply(classify, axis=1)
LEADERBOARD.to_csv(SWEEP_DIR / "LEADERBOARD.csv", index=False)
(SWEEP_DIR / "LEADERBOARD.md").write_text(LEADERBOARD.to_markdown(index=False), encoding="utf-8")

display(LEADERBOARD.groupby("status").size().reset_index(name="count"))
display(LEADERBOARD[LEADERBOARD["status"].isin(["LIVE_CANDIDATE","PARETO"])].head(20))
'''))

cells.append(cell(r'''# CELL 13 - Compare top results
top_balanced = LEADERBOARD.sort_values("balanced_score", ascending=False).head(10)
top_idf1 = LEADERBOARD.sort_values("IDF1", ascending=False).head(10)
top_hota = LEADERBOARD.sort_values("HOTA_or_hota_approx", ascending=False).head(10)
lowest_ids = LEADERBOARD[LEADERBOARD["HOTA_or_hota_approx"] >= ref["HOTA"] - 1.0].sort_values("IDS").head(10)

pareto = []
for _, row in LEADERBOARD.iterrows():
    dominated = ((LEADERBOARD["HOTA_or_hota_approx"] >= row["HOTA_or_hota_approx"]) & (LEADERBOARD["IDF1"] >= row["IDF1"]) & (LEADERBOARD["MOTA"] >= row["MOTA"]) & (LEADERBOARD["IDS"] <= row["IDS"]) & ((LEADERBOARD["HOTA_or_hota_approx"] > row["HOTA_or_hota_approx"]) | (LEADERBOARD["IDF1"] > row["IDF1"]) | (LEADERBOARD["MOTA"] > row["MOTA"]) | (LEADERBOARD["IDS"] < row["IDS"]))).any()
    if not dominated:
        pareto.append(row)
PARETO_FRONT = pd.DataFrame(pareto).sort_values("balanced_score", ascending=False)

comparison = {
    "top_balanced": top_balanced["trial"].tolist(),
    "top_idf1": top_idf1["trial"].tolist(),
    "top_hota": top_hota["trial"].tolist(),
    "lowest_ids_acceptable_hota": lowest_ids["trial"].tolist(),
    "pareto_front": PARETO_FRONT["trial"].tolist(),
}
(SWEEP_DIR / "top_comparisons.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

print("Top 10 by balanced score")
display(top_balanced)
print("Top 10 by IDF1")
display(top_idf1)
print("Top 10 by HOTA")
display(top_hota)
print("Top 10 lowest IDS with acceptable HOTA")
display(lowest_ids)
print("Pareto front")
display(PARETO_FRONT)
'''))

cells.append(cell(r'''# CELL 14 - Plots
import matplotlib.pyplot as plt

plot_specs = [
    ("HOTA_or_hota_approx", "HOTA approximate vs trial"),
    ("IDF1", "IDF1 vs trial"),
    ("IDS", "IDS vs trial"),
    ("MOTA", "MOTA vs trial"),
]
for col, title in plot_specs:
    plt.figure(figsize=(12, 5))
    plt.bar(LEADERBOARD["trial"], LEADERBOARD[col])
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    out = PLOTS_DIR / f"{col}_vs_trial.png"
    plt.savefig(out, dpi=160)
    plt.close()

scatter_specs = [
    ("IDS", "IDF1", "IDF1_vs_IDS.png"),
    ("IDS", "HOTA_or_hota_approx", "HOTA_vs_IDS.png"),
    ("mean_imgsz", "IDF1", "mean_imgsz_vs_IDF1.png"),
]
for x, y, name in scatter_specs:
    plt.figure(figsize=(7, 5))
    plt.scatter(LEADERBOARD[x], LEADERBOARD[y])
    for _, r in LEADERBOARD.head(10).iterrows():
        plt.annotate(r["trial"], (r[x], r[y]), fontsize=7)
    plt.xlabel(x); plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name, dpi=160)
    plt.close()

print("Plots saved:", PLOTS_DIR)
print("No seaborn used")
'''))

cells.append(cell(r'''# CELL 15 - Optional official TrackEval for finalists
TRACK_EVAL_FINALISTS_ONLY = True
RUN_OFFICIAL_TRACKEVAL = False

FINALISTS = LEADERBOARD[LEADERBOARD["status"].isin(["LIVE_CANDIDATE", "PARETO"])].head(5)
if FINALISTS.empty:
    FINALISTS = LEADERBOARD.head(3)

FINALISTS.to_csv(FINALISTS_DIR / "finalists.csv", index=False)
print("Finalists selected:")
display(FINALISTS)

if RUN_OFFICIAL_TRACKEVAL:
    print("Official TrackEval requested for finalists only. No YOLO inference will run.")
    print("This cell needs TrackEval setup and MOTChallenge layout export per finalist.")
    print("Implement TrackEval call here only after inspecting finalist prediction paths.")
else:
    print("RUN_OFFICIAL_TRACKEVAL=False")
    print("Skipping official TrackEval for now.")
    print("motmetrics replay estimate remains separate from official TrackEval result.")

print("Expensive GPU inference: NO")
'''))

cells.append(cell(r'''# CELL 16 - Live benchmark command generation only
LIVE_COMMANDS = []
for _, r in FINALISTS.iterrows():
    safe_name = str(r["trial"]).replace(" ", "_")
    out = f"/content/drive/MyDrive/VisDrone_Results/ACMOT_CODEX_V10STYLE/live_finalist_{safe_name}"
    cmd = (
        "python experiment.py --mode live --allow-expensive -- "
        "--dataset /content/visdrone_v10_p4_local/VisDrone2019-MOT-test-dev "
        f"--output {out}"
    )
    LIVE_COMMANDS.append({"trial": r["trial"], "command": cmd, "expensive_gpu_inference": "YES"})

(SWEEP_DIR / "LIVE_BENCHMARK_COMMANDS.json").write_text(json.dumps(LIVE_COMMANDS, indent=2), encoding="utf-8")

print("Expensive GPU inference: YES")
print("Do not run unless final validation is needed.")
print("ALLOW_LIVE_RUN:", ALLOW_LIVE_RUN)
for item in LIVE_COMMANDS:
    print("\nTrial:", item["trial"])
    print(item["command"])

if not ALLOW_LIVE_RUN:
    print("\nLive benchmark was NOT launched because ALLOW_LIVE_RUN=False.")
else:
    raise RuntimeError("This notebook cell only generates commands. Run live benchmarks manually after final review.")
'''))

cells.append(cell(r'''# CELL 17 - Final report
best_balanced = LEADERBOARD.sort_values("balanced_score", ascending=False).iloc[0]
best_hota = LEADERBOARD.sort_values("HOTA_or_hota_approx", ascending=False).iloc[0]
best_idf1 = LEADERBOARD.sort_values("IDF1", ascending=False).iloc[0]
lowest_ids_ok = LEADERBOARD[LEADERBOARD["HOTA_or_hota_approx"] >= ref["HOTA"] - 1.0].sort_values("IDS")
lowest_ids_row = lowest_ids_ok.iloc[0] if len(lowest_ids_ok) else LEADERBOARD.sort_values("IDS").iloc[0]
rejected = LEADERBOARD[LEADERBOARD["status"] == "REJECT"]["trial"].tolist()

report = f"""# MASTER SWEEP REPORT

cache used: `{DETECTION_CACHE}`
dataset used: `{DATASET}`
replay cache precision: `{REPLAY_CACHE_PRECISION}`
number of trials: {len(LEADERBOARD)}

best balanced trial: `{best_balanced['trial']}`
best HOTA trial: `{best_hota['trial']}`
best IDF1 trial: `{best_idf1['trial']}`
lowest IDS acceptable trial: `{lowest_ids_row['trial']}`

trials rejected: {len(rejected)}
finalists for live benchmark: {', '.join(FINALISTS['trial'].astype(str).tolist())}

Important labels:

- HOTA in this sweep is `hota_approx_only_until_trackeval`.
- Replay FPS is not live FPS.
- {REPLAY_CACHE_PRECISION} replay cache is valid for development replay only unless final evidence uses the same precision and live protocol.

Live benchmark commands:

```bash
{chr(10).join(item['command'] for item in LIVE_COMMANDS)}
```

Expensive GPU inference: YES for live commands only.
"""

(SWEEP_DIR / "MASTER_SWEEP_REPORT.md").write_text(report, encoding="utf-8")
print(report)
print("Saved:", SWEEP_DIR / "MASTER_SWEEP_REPORT.md")
print("The master sweep notebook does not run YOLO unless ALLOW_LIVE_RUN is manually changed.")
'''))


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {OUT} with {len(cells)} cells")

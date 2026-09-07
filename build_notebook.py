"""Build the portable AC-MOT Codex v10-style Colab notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "AC_MOT_Codex_v10style_Portable.ipynb"


def cell(source, cell_type="code"):
    c = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(True),
        "outputs": [],
    }
    if cell_type == "code":
        c["execution_count"] = None
    else:
        c.pop("outputs", None)
    return c


cells = []

cells.append(cell("""# AC-MOT Codex v10-style Portable

This notebook returns to the clean v10 / presentation story:

1. Baseline default YOLOv8n + ByteTrack
2. Tuned ByteTrack only
3. AC-MOT v10-style = tuned ByteTrack + SCI adaptive threshold + SCI adaptive resolution

It is designed for Google Colab T4 and saves reproducible outputs to Google Drive.

It does **not** silently skip missing dataset files.
""", "markdown"))

cells.append(cell(r"""# CELL 1 — INSTALL, MOUNT DRIVE, PATHS
from pathlib import Path
from datetime import datetime
import json, os, sys, subprocess, time, shutil, hashlib, textwrap, math, gc

def pip_install(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install([
    "ultralytics==8.3.200",
    "motmetrics",
    "opencv-python-headless",
    "pandas",
    "numpy",
    "tqdm",
    "scipy",
    "lap",
    "pyyaml",
])

try:
    import trackeval  # noqa
except Exception:
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/JonathonLuiten/TrackEval.git"
    ], check=False)

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

WORK = Path("/content/acmot_codex_v10style")
DATASET = Path("/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev")
OUTPUT_ROOT = Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_CODEX_V10STYLE")
WEIGHTS = Path("/content/yolov8n.pt")

WORK.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

RUN_TAG = "codex_v10style_live17_" + datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_ROOT / RUN_TAG
RUN_DIR.mkdir(parents=True, exist_ok=False)

print("WORK =", WORK)
print("DATASET =", DATASET)
print("OUTPUT_ROOT =", OUTPUT_ROOT)
print("RUN_DIR =", RUN_DIR)
"""))

cells.append(cell(r"""# CELL 2 — STRICT DATASET PREFLIGHT
import pandas as pd
import numpy as np

SEQ_DIR = DATASET / "sequences"
ANN_DIR = DATASET / "annotations"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def image_number(path):
    try:
        return int(path.stem)
    except Exception:
        return None

def inspect_dataset(dataset):
    seq_dir = dataset / "sequences"
    ann_dir = dataset / "annotations"
    seqs = sorted([p.name for p in seq_dir.iterdir() if p.is_dir()]) if seq_dir.exists() else []
    anns = sorted([p.stem for p in ann_dir.glob("*.txt")]) if ann_dir.exists() else []
    missing_ann = [s for s in seqs if s not in set(anns)]
    duplicate_names = sorted([s for s in set(seqs) if seqs.count(s) > 1])
    rows = []
    missing_frame_sequences = []
    for s in seqs:
        frames = sorted((seq_dir / s).glob("*.jpg"), key=image_number)
        nums = [image_number(f) for f in frames]
        nums = [n for n in nums if n is not None]
        ann = ann_dir / f"{s}.txt"
        gt_max_frame = None
        gt_rows = 0
        if ann.exists():
            gt = pd.read_csv(ann, header=None)
            gt_rows = len(gt)
            if len(gt) and gt.shape[1] >= 1:
                gt_max_frame = int(gt.iloc[:, 0].max())
        expected = gt_max_frame if gt_max_frame else (max(nums) if nums else 0)
        present = len(frames)
        contiguous = nums == list(range(1, present + 1))
        missing_count = max(0, expected - present)
        if missing_count or not contiguous:
            missing_frame_sequences.append(s)
        rows.append({
            "sequence": s,
            "frames_found": present,
            "gt_max_frame": gt_max_frame,
            "missing_frame_count_vs_gt": missing_count,
            "contiguous_from_1": contiguous,
            "annotation_exists": ann.exists(),
            "annotation_rows": gt_rows,
            "annotation_sha256": sha256_file(ann) if ann.exists() else None,
        })
    return seqs, anns, missing_ann, duplicate_names, rows, missing_frame_sequences

seqs, anns, missing_ann, duplicate_names, rows, missing_frame_sequences = inspect_dataset(DATASET)

print(f"Total sequences: {len(seqs)}")
print(f"Total annotations found: {len(anns)}")
print("Missing annotation sequences:")
if missing_ann:
    for s in missing_ann:
        print(s)
else:
    print("None")

manifest_df = pd.DataFrame(rows)
manifest_path = RUN_DIR / "dataset_preflight_manifest.csv"
manifest_df.to_csv(manifest_path, index=False)
(RUN_DIR / "sequence_names.json").write_text(json.dumps(seqs, indent=2), encoding="utf-8")

print("\\nFrame/GT summary:")
display(manifest_df[["sequence","frames_found","gt_max_frame","missing_frame_count_vs_gt","contiguous_from_1","annotation_exists"]])
print("Saved manifest:", manifest_path)

if len(seqs) != 17:
    raise RuntimeError(f"Expected 17 VisDrone test-dev sequences, found {len(seqs)}")
if missing_ann:
    raise RuntimeError("Missing GT annotations. Stop before benchmark.")
if duplicate_names:
    raise RuntimeError(f"Duplicate sequence names: {duplicate_names}")
if missing_frame_sequences:
    raise RuntimeError(
        "Some sequences have missing/non-contiguous frames. Fix frames before final benchmark: "
        + ", ".join(missing_frame_sequences)
    )

print("\\n17/17 sequences found")
print("17/17 GT annotation files found")
print("No duplicate sequence names")
print("No missing GT")
print("Frame folders are contiguous and compatible with GT max frame")
"""))

cells.append(cell(r"""# CELL 3 — AC-MOT V10-STYLE CORE
from dataclasses import dataclass, asdict
from collections import deque, Counter
from types import SimpleNamespace
import cv2, yaml, torch
from tqdm.auto import tqdm
from ultralytics import YOLO
import motmetrics as mm

COCO_CLASSES = [0, 2, 5, 7]  # person, car, bus, truck; no fake van class.
VISDRONE_GT_CLASSES = [1, 4, 5, 6, 9]  # pedestrian, car, van, truck, bus.

@dataclass
class SceneState:
    sci: float = 0.0
    scene: str = "clear"
    brightness: float = 128.0
    blur: float = 500.0
    edge_density: float = 0.0
    crowd: float = 0.0
    tiny_ratio: float = 0.0
    n_dets: int = 0

class SceneAnalyzer:
    def __init__(self, window=7):
        self.hist = deque(maxlen=window)
    def reset(self):
        self.hist.clear()
    def analyze(self, img, prev_boxes):
        small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edge_density = float(cv2.Canny(gray, 50, 120).mean() / 255.0)
        n = len(prev_boxes)
        crowd = min(n / 30.0, 1.0)
        if n:
            areas = (prev_boxes[:, 2] - prev_boxes[:, 0]) * (prev_boxes[:, 3] - prev_boxes[:, 1])
            tiny_ratio = float(np.mean(areas < 32 * 32))
        else:
            tiny_ratio = 0.0
        raw = 0.30 * crowd + 0.20 * min(edge_density / 0.14, 1.0) + 0.30 * tiny_ratio
        raw += 0.10 * (brightness < 80) + 0.05 * (blur < 180)
        self.hist.append(float(np.clip(raw, 0, 1)))
        sci = float(np.mean(self.hist))
        if brightness < 80:
            scene = "night"
        elif blur < 180:
            scene = "blur"
        elif tiny_ratio > 0.50:
            scene = "tiny"
        elif crowd > 0.65 or edge_density > 0.13:
            scene = "crowded"
        else:
            scene = "clear"
        return SceneState(sci, scene, brightness, blur, edge_density, crowd, tiny_ratio, n)

class V10StyleCalibrator:
    def __init__(self, adaptive_threshold=True, adaptive_resolution=True):
        self.adaptive_threshold = adaptive_threshold
        self.adaptive_resolution = adaptive_resolution
    def params(self, state):
        conf, iou, imgsz = 0.25, 0.45, 640
        if self.adaptive_threshold:
            conf = 0.245 - 0.050 * state.sci
            iou = 0.490 - 0.050 * state.sci
            if state.scene in ["crowded", "tiny", "night"]:
                conf -= 0.012
            if state.scene == "blur":
                iou -= 0.012
        if self.adaptive_resolution:
            if state.sci > 0.60 or state.tiny_ratio > 0.50:
                imgsz = 832
            elif state.sci > 0.35 or state.scene in ["crowded", "tiny"]:
                imgsz = 736
        return {
            "conf": float(np.clip(conf, 0.19, 0.28)),
            "iou": float(np.clip(iou, 0.40, 0.52)),
            "imgsz": int(imgsz),
        }

def build_tracker_yaml(path, high, low, new, buffer, match, fuse=True):
    data = {
        "tracker_type": "bytetrack",
        "track_high_thresh": float(high),
        "track_low_thresh": float(low),
        "new_track_thresh": float(new),
        "track_buffer": int(buffer),
        "match_thresh": float(match),
        "fuse_score": bool(fuse),
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(path)

TRACKER_DEFAULT = "bytetrack.yaml"
TRACKER_TUNED = build_tracker_yaml(WORK / "bytetrack_v10style_tuned.yaml", high=0.18, low=0.04, new=0.20, buffer=45, match=0.86)

SYSTEMS = [
    {"name": "Baseline_Default", "tracker": TRACKER_DEFAULT, "scene": False, "adapt_thresh": False, "adapt_res": False},
    {"name": "Baseline_TunedTracker", "tracker": TRACKER_TUNED, "scene": False, "adapt_thresh": False, "adapt_res": False},
    {"name": "ACMOT_V10STYLE_SCI", "tracker": TRACKER_TUNED, "scene": True, "adapt_thresh": True, "adapt_res": True},
]

def load_gt(path):
    cols = ["frame","id","x","y","w","h","score","cat","trunc","occ"]
    df = pd.read_csv(path, header=None, names=cols)
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
    return 1.0 - np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

def eval_motmetrics(acc, name):
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["mota","idf1","num_switches","recall","precision","num_misses","num_false_positives","num_matches"], name=name)
    r = s.iloc[0]
    tp, fp, fn, ids = int(r["num_matches"]), int(r["num_false_positives"]), int(r["num_misses"]), int(r["num_switches"])
    det_a = tp / max(tp + fp + fn, 1)
    ass_a = max(0.0, 1.0 - ids / max(tp, 1))
    return {
        "mota": float(r["mota"]),
        "idf1": float(r["idf1"]),
        "recall": float(r["recall"]),
        "precision": float(r["precision"]),
        "ids": ids,
        "fn": fn,
        "fp": fp,
        "matches": tp,
        "hota_approx_only_until_trackeval": float(math.sqrt(det_a * ass_a)),
    }

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def require_t4():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Runtime -> Change runtime type -> T4 GPU.")
    name = torch.cuda.get_device_name(0)
    print("GPU:", name)
    if "T4" not in name:
        raise RuntimeError(f"This final live run is locked to Tesla T4, found: {name}")

print("Systems ready:")
for s in SYSTEMS:
    print(" -", s["name"])
print("Tuned tracker:", TRACKER_TUNED)
"""))

cells.append(cell(r"""# CELL 4 — REAL LIVE FULL-17 RUN
require_t4()

def reset_yolo_tracker(model):
    if getattr(model, "predictor", None) is not None:
        model.predictor = None

def run_one_system(system):
    model = YOLO(str(WEIGHTS))
    if hasattr(model, "model") and hasattr(model.model, "float"):
        model.model.float()
    analyzer = SceneAnalyzer()
    calibrator = V10StyleCalibrator(system["adapt_thresh"], system["adapt_res"])
    rows = []
    pred_root = RUN_DIR / "predictions_mot" / system["name"]
    pred_root.mkdir(parents=True, exist_ok=True)
    total_frames = int(manifest_df["frames_found"].sum())
    pbar = tqdm(total=total_frames, desc=system["name"], dynamic_ncols=True)
    sys_wall_start = time.perf_counter()
    for seq_name in seqs:
        frames = sorted((SEQ_DIR / seq_name).glob("*.jpg"), key=image_number)
        gt = load_gt(ANN_DIR / f"{seq_name}.txt")
        reset_yolo_tracker(model)
        analyzer.reset()
        state = SceneState()
        prev_boxes = np.empty((0, 4))
        acc = mm.MOTAccumulator(auto_id=True)
        times = []
        imgsz_log, conf_log, scene_log = [], [], []
        pred_lines = []
        for idx, fp in enumerate(frames, start=1):
            cuda_sync()
            t0 = time.perf_counter()
            img = cv2.imread(str(fp))
            if img is None:
                raise RuntimeError(f"Unreadable image: {fp}")
            if system["scene"] and (idx == 1 or idx % 10 == 1):
                state = analyzer.analyze(img, prev_boxes)
            params = calibrator.params(state)
            res = model.track(
                source=img,
                tracker=system["tracker"],
                conf=params["conf"],
                iou=params["iou"],
                imgsz=params["imgsz"],
                persist=True,
                verbose=False,
                device=0,
                half=False,
                classes=COCO_CLASSES,
                max_det=1000,
            )[0]
            cuda_sync()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if res.boxes.id is not None:
                pred_ids = res.boxes.id.cpu().numpy().astype(int)
                pred_boxes = res.boxes.xyxy.cpu().numpy()
                pred_scores = res.boxes.conf.cpu().numpy()
            else:
                pred_ids = np.array([], dtype=int)
                pred_boxes = np.empty((0, 4))
                pred_scores = np.array([], dtype=float)
            prev_boxes = pred_boxes.copy()
            for tid, box, score in zip(pred_ids, pred_boxes, pred_scores):
                x1, y1, x2, y2 = box.tolist()
                pred_lines.append(f"{idx},{int(tid)},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{float(score):.6f},-1,-1,-1\n")
            gt_f = gt[gt["frame"] == idx]
            gt_ids = gt_f["id"].values
            gt_boxes = np.column_stack([gt_f["x"], gt_f["y"], gt_f["x"] + gt_f["w"], gt_f["y"] + gt_f["h"]]) if len(gt_f) else np.empty((0, 4))
            dist = iou_distance(pred_boxes, gt_boxes)
            acc.update(gt_ids, pred_ids, dist if dist.size else np.empty((len(gt_ids), len(pred_ids))))
            imgsz_log.append(params["imgsz"])
            conf_log.append(params["conf"])
            scene_log.append(state.scene)
            pbar.update(1)
            if idx == 1 or idx == len(frames) or idx % 50 == 0:
                fps_now = len(times) / max(sum(times), 1e-9)
                pbar.set_postfix(seq=seq_name[-12:], frame=f"{idx}/{len(frames)}", fps=f"{fps_now:.2f}", imgsz=params["imgsz"])
        (pred_root / f"{seq_name}.txt").write_text("".join(pred_lines), encoding="utf-8")
        metrics = eval_motmetrics(acc, seq_name)
        fps = len(times) / max(sum(times), 1e-9)
        rows.append({
            "run_tag": RUN_TAG,
            "system": system["name"],
            "sequence": seq_name,
            "frames": len(frames),
            "fps": fps,
            "mean_latency_ms": 1000.0 / max(fps, 1e-9),
            "mean_imgsz": float(np.mean(imgsz_log)),
            "mean_conf": float(np.mean(conf_log)),
            "dominant_scene": Counter(scene_log).most_common(1)[0][0] if scene_log else "unknown",
            **metrics,
        })
        pd.DataFrame(rows).to_csv(RUN_DIR / f"{system['name']}_per_sequence_live.csv", index=False)
    pbar.close()
    print(f"{system['name']} wall time: {(time.perf_counter()-sys_wall_start)/60:.2f} min")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return pd.DataFrame(rows)

all_rows = []
for system in SYSTEMS:
    df = run_one_system(system)
    all_rows.append(df)

live_df = pd.concat(all_rows, ignore_index=True)
live_df.to_csv(RUN_DIR / "live_per_sequence_motmetrics.csv", index=False)

summary_rows = []
for system, g in live_df.groupby("system", sort=False):
    summary_rows.append({
        "system": system,
        "sequences": len(g),
        "mota": g["mota"].mean(),
        "idf1": g["idf1"].mean(),
        "recall": g["recall"].mean(),
        "precision": g["precision"].mean(),
        "ids": int(g["ids"].sum()),
        "fn": int(g["fn"].sum()),
        "fp": int(g["fp"].sum()),
        "matches": int(g["matches"].sum()),
        "fps": g["fps"].mean(),
        "mean_imgsz": g["mean_imgsz"].mean(),
        "hota_approx_only_until_trackeval": g["hota_approx_only_until_trackeval"].mean(),
    })
summary = pd.DataFrame(summary_rows)
base = summary.iloc[0]
for col in ["mota","idf1","recall","precision","fps","hota_approx_only_until_trackeval"]:
    summary[col + "_delta_vs_default"] = summary[col] - float(base[col])
summary["ids_delta_vs_default"] = summary["ids"] - int(base["ids"])
summary.to_csv(RUN_DIR / "live_summary_motmetrics.csv", index=False)

print("\\nLIVE MOTMETRICS SUMMARY — HOTA here is approximate only until CELL 5 TrackEval")
display(summary)
print("Saved:", RUN_DIR)
"""))

cells.append(cell(r"""# CELL 5 — OFFICIAL TRACKEVAL HOTA/CLEAR/IDENTITY
# This cell exports a MOTChallenge-style folder and runs official TrackEval.
# If TrackEval API changes, the previous cell's CSVs remain valid for MOTA/IDF1/IDS,
# but HOTA should only be quoted from this cell's TrackEval output.

def make_trackeval_layout():
    gt_root = RUN_DIR / "trackeval_gt" / "VisDroneACMOT-test"
    tr_root = RUN_DIR / "trackeval_trackers" / "VisDroneACMOT-test"
    seqmap = RUN_DIR / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n", encoding="utf-8")
    for seq_name in seqs:
        seq_folder = gt_root / seq_name / "gt"
        seq_folder.mkdir(parents=True, exist_ok=True)
        gt = pd.read_csv(ANN_DIR / f"{seq_name}.txt", header=None)
        gt = gt[gt.iloc[:, 7].isin(VISDRONE_GT_CLASSES)]
        gt = gt[(gt.iloc[:, 6] == 1) & (gt.iloc[:, 8] < 2) & (gt.iloc[:, 9] < 2)]
        # MOTChallenge gt: frame,id,x,y,w,h,mark,class,visibility
        mot_gt = pd.DataFrame({
            0: gt.iloc[:, 0].astype(int),
            1: gt.iloc[:, 1].astype(int),
            2: gt.iloc[:, 2],
            3: gt.iloc[:, 3],
            4: gt.iloc[:, 4],
            5: gt.iloc[:, 5],
            6: 1,
            7: 1,
            8: 1,
        })
        mot_gt.to_csv(seq_folder / "gt.txt", header=False, index=False)
        (gt_root / seq_name / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq_name}\nimDir=img1\nframeRate=30\nseqLength={int(manifest_df[manifest_df.sequence==seq_name].frames_found.iloc[0])}\nimWidth=0\nimHeight=0\nimExt=.jpg\n",
            encoding="utf-8"
        )
    for system in [s["name"] for s in SYSTEMS]:
        data_dir = tr_root / system / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for seq_name in seqs:
            shutil.copyfile(RUN_DIR / "predictions_mot" / system / f"{seq_name}.txt", data_dir / f"{seq_name}.txt")
    return gt_root.parent, tr_root.parent, seqmap

gt_parent, trackers_parent, seqmap = make_trackeval_layout()
print("TrackEval GT:", gt_parent)
print("TrackEval trackers:", trackers_parent)
print("Seqmap:", seqmap)

cmd = [
    sys.executable, "-m", "trackeval.scripts.run_mot_challenge",
    "--GT_FOLDER", str(gt_parent),
    "--TRACKERS_FOLDER", str(trackers_parent),
    "--BENCHMARK", "VisDroneACMOT",
    "--SPLIT_TO_EVAL", "test",
    "--SEQMAP_FILE", str(seqmap),
    "--TRACKERS_TO_EVAL", *[s["name"] for s in SYSTEMS],
    "--METRICS", "HOTA", "CLEAR", "Identity",
    "--DO_PREPROC", "False",
    "--USE_PARALLEL", "False",
]

print("Running TrackEval:")
print(" ".join(cmd))
trackeval_log = RUN_DIR / "trackeval_stdout_stderr.txt"
proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
trackeval_log.write_text(proc.stdout, encoding="utf-8")
print(proc.stdout[-6000:])
print("TrackEval return code:", proc.returncode)
print("Full log:", trackeval_log)
if proc.returncode != 0:
    raise RuntimeError("TrackEval failed. Do not quote HOTA until this is fixed.")

print("Official TrackEval finished. Use TrackEval output files under:", trackers_parent)
"""))

cells.append(cell(r"""# CELL 6 — REPRODUCIBILITY PACK
config = {
    "run_tag": RUN_TAG,
    "dataset": str(DATASET),
    "output_root": str(OUTPUT_ROOT),
    "run_dir": str(RUN_DIR),
    "work": str(WORK),
    "weights": str(WEIGHTS),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "systems": SYSTEMS,
    "protocol": {
        "detector": "YOLOv8n FP32",
        "tracker": "ByteTrack",
        "timing": "end-to-end live frame read + scene analysis + YOLO + ByteTrack with CUDA sync",
        "cache": "not used for live timing",
        "hota": "quote only from official TrackEval cell output",
        "dataset_requirement": "17/17 sequences, 17/17 GT, contiguous frames matching GT",
    }
}
(RUN_DIR / "run_configuration.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
print("Reproducibility pack saved:", RUN_DIR / "run_configuration.json")
print("Main files to keep:")
for p in [
    RUN_DIR / "dataset_preflight_manifest.csv",
    RUN_DIR / "live_per_sequence_motmetrics.csv",
    RUN_DIR / "live_summary_motmetrics.csv",
    RUN_DIR / "trackeval_stdout_stderr.txt",
    RUN_DIR / "run_configuration.json",
]:
    print(" -", p, "exists=", p.exists())
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(OUT)

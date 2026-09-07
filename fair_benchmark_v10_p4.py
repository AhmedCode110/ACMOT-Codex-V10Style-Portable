"""AC-MOT v10_p4 FP16 fair benchmark.

Runs the three presentation-scope systems on the same 17-sequence VisDrone split:
1) Baseline_Default
2) Baseline_TunedTracker
3) ACMOT_V10STYLE_SCI

Timing protocol:
- Tesla T4 required
- YOLOv8n FP16 for every system
- input frames must be on local /content for final Colab timing
- JPEG decode is measured separately and excluded from processing FPS
- processing FPS includes scene analysis/controller + YOLO + ByteTrack
- warm-up is excluded
- CUDA synchronization brackets measured processing

The script saves MOT predictions, provisional motmetrics, timing summaries, and a
TrackEval-ready layout. HOTA should be quoted only from official TrackEval output.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO

COCO_CLASSES = [0, 2, 5, 7]
VISDRONE_GT_CLASSES = [1, 4, 5, 6, 9]


class SceneAnalyzer:
    def __init__(self, window=7, stride=10):
        self.hist = deque(maxlen=window)
        self.stride = stride
        self.reset()

    def reset(self):
        self.hist.clear()
        self.state = dict(sci=0.0, scene="clear", tiny=0.0)

    def maybe(self, frame_id, image, prev_boxes):
        if frame_id != 1 and (frame_id - 1) % self.stride != 0:
            return self.state
        small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edge = float(cv2.Canny(gray, 50, 120).mean() / 255.0)
        n = len(prev_boxes)
        crowd = min(n / 30.0, 1.0)
        tiny = 0.0
        if n:
            areas = (prev_boxes[:, 2] - prev_boxes[:, 0]) * (prev_boxes[:, 3] - prev_boxes[:, 1])
            tiny = float(np.mean(areas < 32 * 32))
        raw = 0.30 * crowd + 0.30 * tiny + 0.20 * min(edge / 0.14, 1.0)
        raw += 0.10 * float(brightness < 80) + 0.05 * float(blur < 180)
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
        self.state = dict(sci=sci, scene=scene, tiny=tiny)
        return self.state


class Calibrator:
    @staticmethod
    def fixed():
        return dict(conf=0.25, iou=0.45, imgsz=640)

    @staticmethod
    def adaptive(state):
        conf = 0.245 - 0.050 * state["sci"]
        iou = 0.490 - 0.050 * state["sci"]
        if state["scene"] in {"crowded", "tiny", "night"}:
            conf -= 0.012
        if state["scene"] == "blur":
            iou -= 0.012
        conf = float(np.clip(conf, 0.19, 0.28))
        iou = float(np.clip(iou, 0.40, 0.52))
        if state["sci"] > 0.60 or state["tiny"] > 0.50:
            imgsz = 832
        elif state["sci"] > 0.35 or state["scene"] in {"crowded", "tiny"}:
            imgsz = 736
        else:
            imgsz = 640
        return dict(conf=conf, iou=iou, imgsz=imgsz)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_tracker(path: Path, high, low, new, buffer, match):
    cfg = dict(
        tracker_type="bytetrack",
        track_high_thresh=float(high),
        track_low_thresh=float(low),
        new_track_thresh=float(new),
        track_buffer=int(buffer),
        match_thresh=float(match),
        fuse_score=True,
    )
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return str(path)


def image_number(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return 10**12


def load_gt(path: Path):
    cols = ["frame", "id", "x", "y", "w", "h", "score", "cat", "trunc", "occ"]
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
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    dist = 1.0 - iou
    dist[iou < 0.5] = np.nan
    return dist


def provisional(acc, name):
    mh = mm.metrics.create()
    row = mh.compute(
        acc,
        metrics=["mota", "idf1", "num_switches", "recall", "precision", "num_misses", "num_false_positives", "num_matches"],
        name=name,
    ).iloc[0]
    return dict(
        mota=float(row["mota"]),
        idf1=float(row["idf1"]),
        ids=int(row["num_switches"]),
        recall=float(row["recall"]),
        precision=float(row["precision"]),
        fn=int(row["num_misses"]),
        fp=int(row["num_false_positives"]),
        matches=int(row["num_matches"]),
    )


def warmup(model, image, sizes):
    for size in sizes:
        for _ in range(3):
            model.predict(
                image,
                conf=0.19,
                iou=0.45,
                imgsz=size,
                classes=COCO_CLASSES,
                max_det=1000,
                device=0,
                half=True,
                verbose=False,
            )
    cuda_sync()


def validate_dataset(dataset: Path):
    seq_dir = dataset / "sequences"
    ann_dir = dataset / "annotations"
    seqs = sorted([p.name for p in seq_dir.iterdir() if p.is_dir()])
    if len(seqs) != 17:
        raise RuntimeError(f"Expected 17 sequences, found {len(seqs)}")
    rows = []
    for seq in seqs:
        frames = sorted((seq_dir / seq).glob("*.jpg"), key=image_number)
        ann = ann_dir / f"{seq}.txt"
        if not ann.exists():
            raise RuntimeError(f"Missing GT: {seq}")
        gt = pd.read_csv(ann, header=None)
        gt_max = int(gt.iloc[:, 0].max()) if len(gt) else 0
        nums = [image_number(x) for x in frames]
        if len(frames) != gt_max or nums != list(range(1, len(frames) + 1)):
            raise RuntimeError(f"Frame/GT mismatch: {seq}")
        rows.append(dict(sequence=seq, frames=len(frames), gt_max_frame=gt_max))
    return seqs, pd.DataFrame(rows)


def make_trackeval_layout(dataset, output, systems, seqs, manifest):
    gt_root = output / "trackeval_gt" / "VisDroneACMOT-test"
    trackers_root = output / "trackeval_trackers" / "VisDroneACMOT-test"
    seqmap = output / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n", encoding="utf-8")
    for seq in seqs:
        gtdir = gt_root / seq / "gt"
        gtdir.mkdir(parents=True, exist_ok=True)
        gt = load_gt(dataset / "annotations" / f"{seq}.txt")
        mot = pd.DataFrame({
            0: gt["frame"].astype(int), 1: gt["id"].astype(int), 2: gt["x"], 3: gt["y"], 4: gt["w"], 5: gt["h"],
            6: 1, 7: 1, 8: 1,
        })
        mot.to_csv(gtdir / "gt.txt", header=False, index=False)
        seq_len = int(manifest.loc[manifest.sequence == seq, "frames"].iloc[0])
        (gt_root / seq / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=30\nseqLength={seq_len}\nimWidth=0\nimHeight=0\nimExt=.jpg\n",
            encoding="utf-8",
        )
    for system in systems:
        data_dir = trackers_root / system / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for seq in seqs:
            shutil.copyfile(output / "predictions_mot" / system / f"{seq}.txt", data_dir / f"{seq}.txt")
    return gt_root.parent, trackers_root.parent, seqmap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path, help="LOCAL staged VisDrone test-dev root")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--run-trackeval", action="store_true")
    a = p.parse_args()

    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Final v10_p4 benchmark requires Tesla T4")
    a.output.mkdir(parents=True, exist_ok=False)
    seqs, manifest = validate_dataset(a.dataset)
    manifest.to_csv(a.output / "dataset_manifest.csv", index=False)

    tuned = build_tracker(a.output / "bytetrack_v10_p4_tuned.yaml", 0.18, 0.04, 0.20, 45, 0.86)
    systems = [
        dict(name="Baseline_Default", tracker="bytetrack.yaml", adaptive=False),
        dict(name="Baseline_TunedTracker", tracker=tuned, adaptive=False),
        dict(name="ACMOT_V10STYLE_SCI", tracker=tuned, adaptive=True),
    ]

    config = dict(
        version="v10_p4",
        detector="YOLOv8n FP16",
        gpu="Tesla T4",
        tracker="ByteTrack",
        systems=systems,
        timing="processing FPS = controller/scene analysis + YOLO + ByteTrack; JPEG decode, warm-up and storage excluded",
        gt_filter=dict(categories=VISDRONE_GT_CLASSES, score=1, occlusion_lt=2, truncation_lt=2),
        analyzer=dict(stride=10, smoothing_window=7),
    )
    (a.output / "run_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_rows = []
    for system in systems:
        model = YOLO(a.weights)
        analyzer = SceneAnalyzer()
        pred_root = a.output / "predictions_mot" / system["name"]
        pred_root.mkdir(parents=True, exist_ok=True)
        sys_proc, sys_decode = [], []
        size_log, conf_log, iou_log = [], [], []
        for seq in tqdm(seqs, desc=system["name"]):
            frames = sorted((a.dataset / "sequences" / seq).glob("*.jpg"), key=image_number)
            gt = load_gt(a.dataset / "annotations" / f"{seq}.txt")
            analyzer.reset()
            prev = np.empty((0, 4))
            acc = mm.MOTAccumulator(auto_id=True)
            if getattr(model, "predictor", None) is not None:
                model.predictor = None
            first = cv2.imread(str(frames[0]))
            warmup(model, first, [640, 736, 832] if system["adaptive"] else [640])
            if getattr(model, "predictor", None) is not None:
                model.predictor = None
            seq_proc, seq_decode, seq_sizes, seq_conf, seq_iou = [], [], [], [], []
            lines = []
            for idx, fp in enumerate(frames, 1):
                d0 = time.perf_counter()
                image = cv2.imread(str(fp))
                decode_s = time.perf_counter() - d0
                if image is None:
                    raise RuntimeError(f"Unreadable image: {fp}")

                cuda_sync(); t0 = time.perf_counter()
                if system["adaptive"]:
                    state = analyzer.maybe(idx, image, prev)
                    prm = Calibrator.adaptive(state)
                else:
                    prm = Calibrator.fixed()
                result = model.track(
                    source=image,
                    tracker=system["tracker"],
                    conf=prm["conf"], iou=prm["iou"], imgsz=prm["imgsz"],
                    persist=True, verbose=False, device=0, half=True,
                    classes=COCO_CLASSES, max_det=1000,
                )[0]
                cuda_sync(); process_s = time.perf_counter() - t0

                if result.boxes.id is not None:
                    ids = result.boxes.id.cpu().numpy().astype(int)
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                else:
                    ids = np.array([], dtype=int)
                    boxes = np.empty((0, 4))
                    scores = np.array([], dtype=float)
                prev = boxes.copy()
                for tid, box, score in zip(ids, boxes, scores):
                    x1, y1, x2, y2 = box.tolist()
                    lines.append(f"{idx},{int(tid)},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{float(score):.6f},-1,-1,-1\n")

                gf = gt[gt.frame == idx]
                gids = gf.id.values
                gboxes = np.column_stack([gf.x, gf.y, gf.x + gf.w, gf.y + gf.h]) if len(gf) else np.empty((0, 4))
                acc.update(gids, ids, iou_distance(boxes, gboxes))

                seq_proc.append(process_s); seq_decode.append(decode_s)
                seq_sizes.append(prm["imgsz"]); seq_conf.append(prm["conf"]); seq_iou.append(prm["iou"])
                sys_proc.append(process_s); sys_decode.append(decode_s)
                size_log.append(prm["imgsz"]); conf_log.append(prm["conf"]); iou_log.append(prm["iou"])

            (pred_root / f"{seq}.txt").write_text("".join(lines), encoding="utf-8")
            m = provisional(acc, seq)
            all_rows.append(dict(
                system=system["name"], sequence=seq, frames=len(frames),
                processing_fps=len(seq_proc)/max(sum(seq_proc),1e-9),
                decode_fps=len(seq_decode)/max(sum(seq_decode),1e-9),
                mean_imgsz=float(np.mean(seq_sizes)), mean_conf=float(np.mean(seq_conf)), mean_nms_iou=float(np.mean(seq_iou)),
                **m,
            ))

        print(f"{system['name']}: processing FPS={sum(r['frames'] for r in all_rows if r['system']==system['name'])/max(sum(sys_proc),1e-9):.2f}")
        del model
        torch.cuda.empty_cache()

    per_seq = pd.DataFrame(all_rows)
    per_seq.to_csv(a.output / "per_sequence_v10_p4.csv", index=False)
    summary = []
    for system, g in per_seq.groupby("system", sort=False):
        frames = int(g.frames.sum())
        # weighted harmonic-style reconstruction from per-sequence FPS to total measured time
        proc_seconds = float(np.sum(g.frames / g.processing_fps))
        decode_seconds = float(np.sum(g.frames / g.decode_fps))
        summary.append(dict(
            system=system, sequences=len(g), frames=frames,
            processing_fps=frames/max(proc_seconds,1e-9),
            decode_fps=frames/max(decode_seconds,1e-9),
            mota_provisional=float(np.average(g.mota, weights=g.frames)),
            idf1_provisional=float(np.average(g.idf1, weights=g.frames)),
            ids_provisional=int(g.ids.sum()),
            mean_imgsz=float(np.average(g.mean_imgsz, weights=g.frames)),
            strict_realtime_25fps=(frames/max(proc_seconds,1e-9)) >= 25.0,
            acceptable_realtime_20fps=(frames/max(proc_seconds,1e-9)) >= 20.0,
        ))
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(a.output / "summary_v10_p4.csv", index=False)
    print(summary_df.to_string(index=False))

    gt_parent, tracker_parent, seqmap = make_trackeval_layout(a.dataset, a.output, [x["name"] for x in systems], seqs, manifest)
    if a.run_trackeval:
        cmd = [
            sys.executable, "-m", "trackeval.scripts.run_mot_challenge",
            "--GT_FOLDER", str(gt_parent), "--TRACKERS_FOLDER", str(tracker_parent),
            "--BENCHMARK", "VisDroneACMOT", "--SPLIT_TO_EVAL", "test",
            "--SEQMAP_FILE", str(seqmap), "--TRACKERS_TO_EVAL", *[x["name"] for x in systems],
            "--METRICS", "HOTA", "CLEAR", "Identity", "--DO_PREPROC", "False", "--USE_PARALLEL", "False",
        ]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (a.output / "trackeval_stdout_stderr.txt").write_text(proc.stdout, encoding="utf-8")
        print(proc.stdout[-6000:])
        if proc.returncode != 0:
            raise RuntimeError("TrackEval failed; do not quote HOTA until fixed")


if __name__ == "__main__":
    main()

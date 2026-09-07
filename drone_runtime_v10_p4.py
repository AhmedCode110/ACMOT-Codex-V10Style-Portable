"""AC-MOT v10_p4 deployment-ready real-time runtime.

Purpose
-------
Run the same presentation-aligned AC-MOT controller on:
- a video file,
- a webcam / USB camera index,
- an RTSP/GStreamer-style camera URL exposed through OpenCV.

The runtime outputs tracked IDs + boxes in real time and can optionally save:
- annotated video,
- per-frame JSONL telemetry,
- a final runtime summary JSON.

Research rules
--------------
- v10_p3 remains the frozen FP32 reference.
- v10_p4 uses FP16 on CUDA for deployment and fair timing.
- SceneAnalyzer: every 10 frames, 7-reading rolling mean.
- SmartCalibrator: adaptive conf + NMS IoU + 640/736/832 resolution.
- ByteTrack tuned profile is fixed.
- No ReID, recovery controller, adaptive birth, or detector feedback.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

COCO_CLASSES = [0, 2, 5, 7]  # person, car, bus, truck


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
    def __init__(self, window: int = 7, stride: int = 10):
        self.hist = deque(maxlen=window)
        self.stride = int(stride)
        self.state = SceneState()

    def reset(self):
        self.hist.clear()
        self.state = SceneState()

    def maybe_analyze(self, frame_id: int, frame: np.ndarray, prev_boxes: np.ndarray) -> SceneState:
        if frame_id != 1 and (frame_id - 1) % self.stride != 0:
            return self.state

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
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

        raw = (
            0.30 * crowd
            + 0.30 * tiny_ratio
            + 0.20 * min(edge_density / 0.14, 1.0)
            + 0.10 * float(brightness < 80)
            + 0.05 * float(blur < 180)
        )
        self.hist.append(float(np.clip(raw, 0.0, 1.0)))
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

        self.state = SceneState(
            sci=sci,
            scene=scene,
            brightness=brightness,
            blur=blur,
            edge_density=edge_density,
            crowd=crowd,
            tiny_ratio=tiny_ratio,
            n_dets=n,
        )
        return self.state


class SmartCalibrator:
    def params(self, state: SceneState) -> dict[str, Any]:
        conf = 0.245 - 0.050 * state.sci
        iou = 0.490 - 0.050 * state.sci
        if state.scene in {"crowded", "tiny", "night"}:
            conf -= 0.012
        if state.scene == "blur":
            iou -= 0.012
        conf = float(np.clip(conf, 0.19, 0.28))
        iou = float(np.clip(iou, 0.40, 0.52))

        if state.sci > 0.60 or state.tiny_ratio > 0.50:
            imgsz = 832
        elif state.sci > 0.35 or state.scene in {"crowded", "tiny"}:
            imgsz = 736
        else:
            imgsz = 640

        return {"conf": conf, "iou": iou, "imgsz": imgsz}


def build_tuned_tracker_yaml(path: Path) -> Path:
    cfg = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.18,
        "track_low_thresh": 0.04,
        "new_track_thresh": 0.20,
        "track_buffer": 45,
        "match_thresh": 0.86,
        "fuse_score": True,
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def parse_source(value: str):
    if value.isdigit():
        return int(value)
    return value


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def warmup(model: YOLO, frame: np.ndarray, device: str, half: bool):
    sizes = [640, 736, 832]
    for size in sizes:
        for _ in range(2):
            model.predict(
                frame,
                imgsz=size,
                conf=0.19,
                iou=0.45,
                classes=COCO_CLASSES,
                max_det=1000,
                device=device,
                half=half,
                verbose=False,
            )
    sync_cuda()


def draw_tracks(frame: np.ndarray, ids: np.ndarray, boxes: np.ndarray, state: SceneState, params: dict, fps: float):
    out = frame.copy()
    for tid, box in zip(ids, boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"ID {int(tid)}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    text = f"FPS {fps:.1f} | SCI {state.sci:.3f} | {state.scene} | conf {params['conf']:.3f} | IoU {params['iou']:.3f} | {params['imgsz']}"
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="video path, webcam index e.g. 0, or RTSP URL")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--output-dir", default="runs/drone_v10_p4")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--max-frames", type=int, default=0, help="0 means run until source ends")
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker_yaml = build_tuned_tracker_yaml(out_dir / "bytetrack_v10_p4_tuned.yaml")

    source = parse_source(a.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {a.source}")

    ok, first = cap.read()
    if not ok or first is None:
        raise RuntimeError("Source opened but first frame could not be read")

    use_cuda = torch.cuda.is_available() and str(a.device) != "cpu"
    half = bool(use_cuda)
    model = YOLO(a.weights)
    analyzer = SceneAnalyzer(window=7, stride=10)
    calibrator = SmartCalibrator()

    warmup(model, first, a.device, half)
    if getattr(model, "predictor", None) is not None:
        model.predictor = None

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or first.shape[1])
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or first.shape[0])
    if not fps_src or fps_src <= 0:
        fps_src = 25.0

    writer = None
    if a.save_video:
        writer = cv2.VideoWriter(
            str(out_dir / "annotated_output.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps_src),
            (width, height),
        )

    telemetry = (out_dir / "telemetry.jsonl").open("w", encoding="utf-8")
    prev_boxes = np.empty((0, 4), dtype=float)
    frame_id = 0
    process_times = []
    decode_times = []
    size_log = []
    scene_log = []

    frame = first
    while True:
        frame_id += 1
        state = analyzer.maybe_analyze(frame_id, frame, prev_boxes)
        params = calibrator.params(state)

        sync_cuda()
        t0 = time.perf_counter()
        result = model.track(
            source=frame,
            tracker=str(tracker_yaml),
            conf=params["conf"],
            iou=params["iou"],
            imgsz=params["imgsz"],
            persist=True,
            classes=COCO_CLASSES,
            max_det=1000,
            device=a.device,
            half=half,
            verbose=False,
        )[0]
        sync_cuda()
        process_s = time.perf_counter() - t0
        process_times.append(process_s)

        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
        else:
            ids = np.array([], dtype=int)
            boxes = np.empty((0, 4), dtype=float)
            scores = np.array([], dtype=float)
        prev_boxes = boxes.copy()

        fps_now = 1.0 / max(process_s, 1e-9)
        size_log.append(params["imgsz"])
        scene_log.append(state.scene)

        telemetry.write(json.dumps({
            "frame": frame_id,
            "timestamp_s": time.time(),
            "processing_ms": process_s * 1000.0,
            "processing_fps": fps_now,
            "sci": state.sci,
            "scene": state.scene,
            "conf": params["conf"],
            "nms_iou": params["iou"],
            "imgsz": params["imgsz"],
            "tracks": [
                {"id": int(tid), "box_xyxy": [float(x) for x in box], "score": float(score)}
                for tid, box, score in zip(ids, boxes, scores)
            ],
        }) + "\n")

        annotated = draw_tracks(frame, ids, boxes, state, params, fps_now)
        if writer is not None:
            writer.write(annotated)
        if a.show:
            cv2.imshow("AC-MOT v10_p4", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if a.max_frames and frame_id >= a.max_frames:
            break

        d0 = time.perf_counter()
        ok, frame = cap.read()
        decode_times.append(time.perf_counter() - d0)
        if not ok or frame is None:
            break

    telemetry.close()
    cap.release()
    if writer is not None:
        writer.release()
    if a.show:
        cv2.destroyAllWindows()

    total_process = sum(process_times)
    summary = {
        "version": "v10_p4",
        "precision": "FP16" if half else "FP32_CPU_FALLBACK",
        "device": a.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source": str(a.source),
        "frames": frame_id,
        "processing_fps": frame_id / max(total_process, 1e-9),
        "mean_processing_ms": 1000.0 * total_process / max(frame_id, 1),
        "p95_processing_ms": float(np.percentile(process_times, 95) * 1000.0) if process_times else None,
        "decode_fps_separate": (len(decode_times) / max(sum(decode_times), 1e-9)) if decode_times else None,
        "mean_imgsz": float(np.mean(size_log)) if size_log else None,
        "dominant_scene": Counter(scene_log).most_common(1)[0][0] if scene_log else None,
        "strict_realtime_25fps": (frame_id / max(total_process, 1e-9)) >= 25.0,
        "acceptable_realtime_20fps": (frame_id / max(total_process, 1e-9)) >= 20.0,
    }
    (out_dir / "runtime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

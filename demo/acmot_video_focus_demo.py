#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from acmot_video_demo import (
    COCO_CLASSES,
    BASELINE_TRACKER,
    FINAL_TRACKER,
    OFFICIAL,
    SceneAnalyzer,
    SmartCalibrator,
    DemoIDSwitchCounter,
    validate_dataset,
    load_gt,
    tracks_from_result,
    draw_panel,
    resize_width,
    add_footer,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline vs Final AC-MOT demo with automatic 2-second focus pauses"
    )
    p.add_argument(
        "--dataset",
        default="/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev",
    )
    p.add_argument("--sequence", default="uav0000249_00001_v")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--output", default="/content/ACMOT_Baseline_vs_Final_FOCUS.mp4")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=600, help="0 = full sequence")
    p.add_argument("--output-fps", type=float, default=20.0)
    p.add_argument("--panel-width", type=int, default=960)
    p.add_argument("--hold-seconds", type=float, default=2.0)
    p.add_argument("--save-json", action="store_true")
    return p.parse_args()


def write_yaml(path, cfg):
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def switched_boxes(tracks, switched_ids):
    wanted = set(int(x) for x in switched_ids)
    return [tr for tr in tracks if int(tr["id"]) in wanted]


def draw_arrow(img, start, end, color, thickness=4):
    cv2.arrowedLine(img, start, end, color, thickness, cv2.LINE_AA, tipLength=0.12)


def add_focus_overlay(canvas, baseline_tracks, switched_ids, panel_width, original_width, panel_height):
    """Highlight the exact same image region on Baseline and AC-MOT panels."""
    out = canvas.copy()
    hits = switched_boxes(baseline_tracks, switched_ids)
    if not hits:
        return out

    scale = panel_width / float(original_width)
    accent = (0, 220, 255)
    green = (70, 230, 100)

    # Large freeze banner.
    cv2.rectangle(out, (0, 0), (out.shape[1], 78), (0, 0, 0), -1)
    cv2.putText(
        out,
        "PAUSE: BASELINE ID SWITCH - AC-MOT KEEPS THE ASSOCIATION",
        (28, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        accent,
        3,
        cv2.LINE_AA,
    )

    for tr in hits[:3]:
        x1, y1, x2, y2 = tr["xyxy"]
        sx1, sy1, sx2, sy2 = [int(round(v * scale)) for v in (x1, y1, x2, y2)]
        sx1 = max(0, min(panel_width - 1, sx1))
        sx2 = max(0, min(panel_width - 1, sx2))
        sy1 = max(80, min(panel_height - 1, sy1))
        sy2 = max(80, min(panel_height - 1, sy2))
        cx = (sx1 + sx2) // 2
        cy = (sy1 + sy2) // 2

        # Left panel: exact Baseline switch location.
        pad = 14
        cv2.rectangle(
            out,
            (max(0, sx1 - pad), max(80, sy1 - pad)),
            (min(panel_width - 1, sx2 + pad), min(panel_height - 1, sy2 + pad)),
            accent,
            6,
        )
        radius = max(28, min(75, int(max(sx2 - sx1, sy2 - sy1) * 1.25)))
        cv2.circle(out, (cx, cy), radius, accent, 5, cv2.LINE_AA)
        label_y = max(105, cy - radius - 15)
        cv2.putText(
            out,
            "ID SWITCH HERE",
            (max(15, cx - 95), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            accent,
            2,
            cv2.LINE_AA,
        )
        arrow_start = (max(30, cx - 160), max(100, cy - 110))
        draw_arrow(out, arrow_start, (cx, cy), accent, 4)

        # Right panel: same source-image location, to make the comparison immediate.
        rcx = panel_width + cx
        rsx1 = panel_width + sx1
        rsx2 = panel_width + sx2
        cv2.rectangle(
            out,
            (max(panel_width, rsx1 - pad), max(80, sy1 - pad)),
            (min(out.shape[1] - 1, rsx2 + pad), min(panel_height - 1, sy2 + pad)),
            green,
            5,
        )
        cv2.circle(out, (rcx, cy), radius, green, 4, cv2.LINE_AA)
        cv2.putText(
            out,
            "SAME OBJECT - ASSOCIATION KEPT",
            (max(panel_width + 15, rcx - 150), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            green,
            2,
            cv2.LINE_AA,
        )
        draw_arrow(out, (min(out.shape[1] - 30, rcx + 160), max(100, cy - 110)), (rcx, cy), green, 4)

    return out


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. In Colab choose T4 GPU.")

    print("GPU:", torch.cuda.get_device_name(0))
    root = Path(args.dataset)
    ann, frames = validate_dataset(root, args.sequence)
    start = max(1, args.start_frame)
    selected = frames[start - 1 :]
    if args.max_frames > 0:
        selected = selected[: args.max_frames]
    gt = load_gt(ann)

    work = Path("/content/acmot_video_demo")
    work.mkdir(parents=True, exist_ok=True)
    by = work / "baseline_bytetrack.yaml"
    fy = work / "final_acmot_bytetrack.yaml"
    write_yaml(by, BASELINE_TRACKER)
    write_yaml(fy, FINAL_TRACKER)

    baseline_model = YOLO(args.weights)
    acmot_model = YOLO(args.weights)
    analyzer = SceneAnalyzer()
    calibrator = SmartCalibrator()
    prev_boxes = np.empty((0, 4), np.float32)
    base_ids = DemoIDSwitchCounter()
    ac_ids = DemoIDSwitchCounter()

    warm = cv2.imread(str(selected[0]))
    for m in (baseline_model, acmot_model):
        m.predict(
            warm,
            conf=0.25,
            iou=0.5,
            imgsz=640,
            classes=COCO_CLASSES,
            device=0,
            half=True,
            verbose=False,
        )
    torch.cuda.synchronize()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = []
    focus_events = []
    t_all = time.perf_counter()
    written_frames = 0
    hold_frames = max(0, int(round(args.hold_seconds * args.output_fps)))

    for frame_idx, fp in enumerate(selected, start=start):
        frame = cv2.imread(str(fp))
        if frame is None:
            raise RuntimeError(f"Could not read {fp}")
        orig_h, orig_w = frame.shape[:2]

        t = time.perf_counter()
        rb = baseline_model.track(
            frame,
            persist=True,
            tracker=str(by),
            conf=0.25,
            iou=0.70,
            imgsz=640,
            classes=COCO_CLASSES,
            device=0,
            half=True,
            verbose=False,
        )[0]
        torch.cuda.synchronize()
        bms = (time.perf_counter() - t) * 1000
        bt = tracks_from_result(rb)

        state = analyzer.maybe(frame_idx, frame, prev_boxes)
        p = calibrator.params(state)
        t = time.perf_counter()
        ra = acmot_model.track(
            frame,
            persist=True,
            tracker=str(fy),
            conf=p["conf"],
            iou=p["iou"],
            imgsz=p["imgsz"],
            classes=COCO_CLASSES,
            device=0,
            half=True,
            verbose=False,
        )[0]
        torch.cuda.synchronize()
        ams = (time.perf_counter() - t) * 1000
        at = tracks_from_result(ra)
        prev_boxes = (
            np.stack([x["xyxy"] for x in at]) if at else np.empty((0, 4), np.float32)
        )

        g = gt.get(frame_idx, [])
        sb = base_ids.update(g, bt)
        sa = ac_ids.update(g, at)

        left = draw_panel(
            frame,
            bt,
            "BASELINE_DEFAULT",
            [
                "Detector: conf=.25  NMS IoU=.70  imgsz=640",
                f"Tracks={len(bt)}  Demo IDS={base_ids.total}  frame={bms:.1f} ms",
                "ByteTrack: high=.25 low=.10 new=.25 buffer=30 match=.80",
            ],
            sb,
            (70, 80, 255),
        )
        right = draw_panel(
            frame,
            at,
            "FINAL AC-MOT",
            [
                f"SCI={state.sci:.3f} scene={state.scene} tiny={state.tiny_ratio:.2f} prev_objects={state.object_count}",
                f"Detector: conf={p['conf']:.3f}  NMS IoU={p['iou']:.3f}  imgsz={p['imgsz']}",
                f"Tracks={len(at)}  Demo IDS={ac_ids.total}  frame={ams:.1f} ms",
                "ByteTrack: high=.18 low=.04 new=.24 buffer=45 match=.88",
            ],
            sa,
            (70, 230, 100),
        )

        left = resize_width(left, args.panel_width)
        right = resize_width(right, args.panel_width)
        ph = min(left.shape[0], right.shape[0])
        canvas = np.concatenate([left[:ph], right[:ph]], axis=1)

        visible_improvement = bool(sb and not sa)
        if visible_improvement:
            cv2.rectangle(canvas, (0, ph - 44), (canvas.shape[1], ph), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                "VISIBLE IMPROVEMENT: baseline changed ID while AC-MOT kept association",
                (24, ph - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

        normal_canvas = add_footer(canvas, frame_idx)

        if writer is None:
            h, w = normal_canvas.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.output_fps,
                (w, h),
            )
            if not writer.isOpened():
                raise RuntimeError("Could not open MP4 writer")

        writer.write(normal_canvas)
        written_frames += 1

        # Whenever the Baseline switches ID but AC-MOT does not, freeze for 2 seconds
        # and explicitly point to the exact same object region on both panels.
        if visible_improvement and hold_frames > 0:
            focus_canvas = add_focus_overlay(
                normal_canvas,
                bt,
                sb,
                args.panel_width,
                orig_w,
                ph,
            )
            for _ in range(hold_frames):
                writer.write(focus_canvas)
                written_frames += 1
            focus_events.append(
                {
                    "frame": frame_idx,
                    "baseline_switched_ids": [int(x) for x in sb],
                    "hold_seconds": float(args.hold_seconds),
                }
            )
            print(
                f"FOCUS frame {frame_idx}: baseline switch IDs={list(map(int, sb))}; "
                f"AC-MOT kept association; held {args.hold_seconds:.1f}s"
            )

        rows.append(
            {
                "frame": frame_idx,
                "baseline_demo_ids": base_ids.total,
                "acmot_demo_ids": ac_ids.total,
                "baseline_switch": bool(sb),
                "acmot_switch": bool(sa),
                "visible_improvement": visible_improvement,
                "sci": state.sci,
                "scene": state.scene,
                "conf": p["conf"],
                "iou": p["iou"],
                "imgsz": p["imgsz"],
                "baseline_ms": bms,
                "acmot_ms": ams,
            }
        )

        if len(rows) % 50 == 0:
            print(
                f"{len(rows)}/{len(selected)} | Demo IDS baseline={base_ids.total} "
                f"AC-MOT={ac_ids.total} | focus_events={len(focus_events)} | "
                f"SCI={state.sci:.3f} imgsz={p['imgsz']}"
            )

    writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        h264_tmp = out_path.with_name(out_path.stem + "_h264_tmp.mp4")
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(out_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(h264_tmp),
        ]
        try:
            subprocess.run(cmd, check=True)
            h264_tmp.replace(out_path)
            print("H.264 web-compatible MP4 ready.")
        except Exception as e:
            print("WARNING: H.264 transcode failed; keeping original MP4:", e)
            if h264_tmp.exists():
                h264_tmp.unlink()

    summary = {
        "sequence": args.sequence,
        "source_frames": len(selected),
        "video_frames_after_focus_holds": written_frames,
        "focus_events": len(focus_events),
        "hold_seconds_per_event": float(args.hold_seconds),
        "baseline_demo_ids": base_ids.total,
        "acmot_demo_ids": ac_ids.total,
        "output": str(out_path),
        "render_wall_fps": len(selected) / max(time.perf_counter() - t_all, 1e-9),
        "official_metrics": OFFICIAL,
        "note": "Demo IDS is GT-linked IoU>=0.50 visualization, not official TrackEval IDSW. Focus pauses are visualization-only.",
    }
    print(json.dumps(summary, indent=2))
    print("Video:", out_path)

    if args.save_json:
        jp = out_path.with_suffix(".json")
        jp.write_text(
            json.dumps(
                {"summary": summary, "focus_events": focus_events, "frames": rows},
                indent=2,
            ),
            encoding="utf-8",
        )
        print("JSON:", jp)


if __name__ == "__main__":
    main()

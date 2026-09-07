# AC-MOT v10_p4 — FP16 Fair Benchmark + Drone Deployment

## Status

`v10_p3` is frozen and remains the FP32 reference.

`v10_p4` is a new protocol/deployment version. It does not change the presentation-scope AC-MOT method.

## Main research system

- Detector: YOLOv8n
- Precision: FP16 on CUDA
- Tracker: ByteTrack
- Scene analyzer: every 10 frames
- SCI smoothing: rolling mean of 7 analyzer readings
- Adaptive detector confidence
- Adaptive NMS IoU
- Adaptive resolution: 640 / 736 / 832
- Output: track ID + bounding box + score

No ReID, detector feedback, recovery controller, adaptive birth, or other v12-style production features are included.

## Why v10_p4 exists

The previous v10_p3 run used YOLOv8n FP32. v17 used FP16. Comparing their FPS directly is therefore not a fair architecture comparison.

v10_p4 standardizes the real-time deployment precision to FP16 and defines a measurement protocol that separates image decode from the actual AC-MOT processing pipeline.

## Required fair benchmark

Run all three systems under the same conditions:

1. `Baseline_Default`
2. `Baseline_TunedTracker`
3. `ACMOT_V10STYLE_SCI`

Hold constant:

- Tesla T4 GPU
- YOLOv8n weights
- FP16 precision
- VisDrone 17-sequence test-dev split
- same custom GT filter
- same frame order
- same class filter
- same CUDA synchronization policy
- same warm-up policy
- same decode exclusion rule

### Timing definition

`processing FPS` includes:

- scene analysis/controller when invoked
- YOLO inference
- ByteTrack

It excludes:

- Google Drive staging
- JPEG decode
- warm-up
- file export / logging

JPEG decode must be reported separately.

### Real-time labels

- `>= 25 FPS`: strict real-time for the 25 FPS camera target
- `20 to 24.99 FPS`: acceptable / near-real-time
- `< 20 FPS`: below the chosen real-time target

## Deployment runtime

File: `drone_runtime_v10_p4.py`

The same controller can run on:

- saved flight video
- USB/webcam index
- RTSP camera URL
- any camera source OpenCV can open

Examples:

```bash
python drone_runtime_v10_p4.py --source flight.mp4 --weights yolov8n.pt --save-video
```

```bash
python drone_runtime_v10_p4.py --source 0 --weights yolov8n.pt --show
```

```bash
python drone_runtime_v10_p4.py --source rtsp://CAMERA_STREAM --weights yolov8n.pt --save-video
```

Outputs:

- `annotated_output.mp4` when `--save-video` is enabled
- `telemetry.jsonl`
- `runtime_summary.json`
- tuned ByteTrack YAML used by the run

Each telemetry line includes:

- frame number
- timestamp
- processing latency / FPS
- SCI
- scene label
- conf
- NMS IoU
- imgsz
- list of `{id, box_xyxy, score}` tracks

## Simulation ladder before real drone flight

Use the same runtime in this order:

### Stage 1 — Dataset benchmark

Prove HOTA/MOTA/IDF1/IDS and FP16 processing FPS on the verified VisDrone split.

### Stage 2 — Recorded flight video

Run a representative UAV video through `drone_runtime_v10_p4.py` and save annotated output + telemetry.

Check:

- ID continuity
- dropped frames
- p95 latency
- adaptive resolution switching
- scene transitions

### Stage 3 — Live camera simulation

Run from a USB/webcam or network stream. Confirm the system processes indefinitely without state leaks or memory growth.

### Stage 4 — Drone camera stream

Feed the actual drone camera stream to the same `--source` interface. The AC-MOT process remains the same; only the frame source changes.

### Stage 5 — Onboard / companion-computer deployment

Copy:

- `drone_runtime_v10_p4.py`
- `yolov8n.pt`
- Python environment / pinned dependencies

onto the NVIDIA-capable companion computer.

The runtime should emit tracking telemetry to the rest of the drone stack through a thin adapter (for example a local socket, ROS2 node, MAVLink companion process, or other project-specific message layer). The tracking algorithm itself should not be rewritten for the transport layer.

## Deployment acceptance criteria

A deployment candidate is acceptable only when it has:

- reproducible 17-sequence quality metrics
- measured FP16 processing FPS
- p95 latency
- no silent frame skipping
- stable IDs on representative flight footage
- verified live stream operation
- saved runtime configuration

For the thesis/paper, keep the numerical benchmark claim separate from a live-demo claim. A real-time demo proves deployability; TrackEval proves tracking quality.

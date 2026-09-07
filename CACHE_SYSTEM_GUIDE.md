# AC-MOT Cache System Guide

Policy:

```text
DISCOVER -> VALIDATE -> PLAN -> REUSE -> COMPUTE ONLY MISSING STAGES
```

The cache system exists so AC-MOT development can reuse valid existing work before running expensive YOLO, tracking, or TrackEval stages.

## Main Commands

```bash
python cache_manager_v10_p4.py discover
```

Expensive GPU inference: NO

```bash
python cache_manager_v10_p4.py verify
```

Expensive GPU inference: NO

```bash
python cache_manager_v10_p4.py list
```

Expensive GPU inference: NO

```bash
python cache_manager_v10_p4.py plan --mode replay
```

Expensive GPU inference: NO

```bash
python experiment.py --mode auto --dry-run
```

Expensive GPU inference: NO

```bash
python experiment.py --mode live --allow-expensive
```

Expensive GPU inference: YES

## Cache Root

Default persistent root:

```text
/content/drive/MyDrive/VisDrone_Results/ACMOT_CACHE/
```

Large legacy folders do not need to move. The registry can reference them in place.

## Cache Levels

- Dataset cache: validates VisDrone2019-MOT-test-dev, 17 sequences, 6635 frames, and 17 annotations.
- Detector cache: raw YOLOv8n detections before tracking, separated by precision and imgsz.
- Tracker cache: ByteTrack outputs derived from a detector cache and tracker config.
- Replay cache: SCI and calibrator decisions using existing 640/736/832 detector outputs.
- Evaluation cache: official TrackEval results tied to predictions, GT filtering, and metric config.
- Completed runs: frozen or historical experiment folders.

## FP32 and FP16

FP32 replay caches can support tracker tuning, SCI policy tests, ablations, and debugging.

FP32 replay caches must not be used as FP16 detector evidence.

Replay throughput must not be reported as live FPS.

Final publishable timing must use live inference on the target hardware and precision.

## Dependency Rules

If only ByteTrack parameters change, reuse dataset and YOLO caches, then rerun tracker and TrackEval.

If only SCI weights or resolution policy change, reuse 640/736/832 detections, then rerun replay/tracker/evaluation.

If only GT filtering changes, reuse detections and predictions, then rerun GT preparation and TrackEval.

If plot styling changes, reuse metrics and regenerate plots only.

If detector weights, precision, classes, or imgsz change, invalidate affected detector caches and downstream artifacts.

## Frozen Final Run

The completed FP16 run:

```text
codex_v10_p4_fp16_20260907_134125
```

is imported as `FROZEN_FINAL`. Do not overwrite it. Do not rerun it just to test the cache system.

## Colab Startup

Recommended fresh runtime flow:

```text
Cell 1: Mount Google Drive
Cell 2: Clone or pull this repo
Cell 3: python cache_manager_v10_p4.py discover
Cell 4: python cache_manager_v10_p4.py verify
Cell 5: python cache_manager_v10_p4.py status
Cell 6: python experiment.py --mode auto --dry-run
Cell 7: run replay or live only after reviewing the plan
```

## TrackEval

Use the Python API wrapper for TrackEval. Runtime compatibility patches may handle NumPy aliases such as `np.float`, but they must not modify metric formulas.

Record TrackEval commit, NumPy version, and compatibility patch state in experiment manifests.

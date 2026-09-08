# Baseline vs Final AC-MOT Video Demo

This demo renders the same VisDrone sequence with two independent live pipelines:

- **Left:** `Baseline_Default` — YOLOv8n FP16, detector `conf=0.25`, NMS IoU `0.70`, `imgsz=640`, stock ByteTrack-style settings.
- **Right:** **Final AC-MOT / `TRK_NEW24_MATCH88`** — 5-cue SCI, adaptive detector confidence/NMS IoU/resolution, and final ByteTrack settings `high=.18`, `low=.04`, `new=.24`, `buffer=45`, `match=.88`.

The video overlays bounding boxes, track IDs, the current SCI/controller values, per-frame processing time, and a live **Demo IDS** counter. When the baseline changes an ID while AC-MOT keeps the association, the video displays a visible improvement banner.

## Important metric note

`Demo IDS` is a visualization-only counter. It greedily links valid VisDrone ground-truth objects to current tracker boxes at IoU >= 0.50 and counts a changed tracker ID. It is **not** the official TrackEval IDSW implementation. The footer separately shows the final official 17-sequence TrackEval results:

- Baseline: MOTA 19.718, HOTA 28.418, IDF1 32.716, IDS 1238, processing FPS 44.181.
- Final AC-MOT: MOTA 22.999, HOTA 33.017, IDF1 40.021, IDS 994, processing FPS 37.686.

## Easiest run: Google Colab

Open `notebooks/ACMOT_BASELINE_VS_FINAL_VIDEO_DEMO.ipynb`, select a T4 GPU, and Run All.

Default dataset path:

`/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev`

Default dense demo sequence:

`uav0000249_00001_v`

Another useful dense sequence to try:

`uav0000306_00230_v`

The notebook writes the MP4 and JSON log to:

`/content/drive/MyDrive/VisDrone_Results/ACMOT_VIDEO_DEMOS/`

## Direct command

```bash
python demo/acmot_video_demo.py \
  --dataset /path/to/VisDrone2019-MOT-test-dev \
  --sequence uav0000249_00001_v \
  --output /content/ACMOT_Baseline_vs_Final.mp4 \
  --max-frames 600 \
  --output-fps 20 \
  --save-json
```

Use `--max-frames 0` for the full sequence.

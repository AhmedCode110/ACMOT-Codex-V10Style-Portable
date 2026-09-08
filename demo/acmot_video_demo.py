#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, time, shutil, subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

COCO_CLASSES = [0, 2, 5, 7]
VISDRONE_GT_CLASSES = [1, 4, 5, 6, 9]
BASELINE_DET = {"conf": 0.25, "iou": 0.70, "imgsz": 640}
BASELINE_TRACKER = {
    "tracker_type": "bytetrack", "track_high_thresh": 0.25,
    "track_low_thresh": 0.10, "new_track_thresh": 0.25,
    "track_buffer": 30, "match_thresh": 0.80, "fuse_score": True,
}
FINAL_TRACKER = {
    "tracker_type": "bytetrack", "track_high_thresh": 0.18,
    "track_low_thresh": 0.04, "new_track_thresh": 0.24,
    "track_buffer": 45, "match_thresh": 0.88, "fuse_score": True,
}
OFFICIAL = {
    "baseline": {"MOTA": 19.718, "HOTA": 28.418, "IDF1": 32.716, "IDS": 1238, "FPS": 44.181},
    "acmot": {"MOTA": 22.999, "HOTA": 33.017, "IDF1": 40.021, "IDS": 994, "FPS": 37.686},
}

@dataclass
class SceneState:
    sci: float = 0.0
    scene: str = "clear"
    tiny_ratio: float = 0.0
    object_count: int = 0

class SceneAnalyzer:
    def __init__(self, stride=10, window=7):
        self.stride = stride
        self.hist = deque(maxlen=window)
        self.last = SceneState()

    def maybe(self, frame_idx, frame, prev_boxes):
        if frame_idx != 1 and frame_idx % self.stride != 1:
            return self.last
        small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edge = float(cv2.Canny(gray, 50, 120).mean() / 255.0)
        n = len(prev_boxes)
        crowd = min(n / 30.0, 1.0)
        if n:
            areas = (prev_boxes[:, 2] - prev_boxes[:, 0]) * (prev_boxes[:, 3] - prev_boxes[:, 1])
            tiny = float(np.mean(areas < 32 * 32))
        else:
            tiny = 0.0
        raw = 0.30*crowd + 0.30*tiny + 0.20*min(edge/0.14, 1.0) + 0.10*(brightness < 80) + 0.05*(blur_score < 180)
        self.hist.append(float(np.clip(raw, 0.0, 1.0)))
        sci = float(np.mean(self.hist))
        if brightness < 80: scene = "night"
        elif blur_score < 180: scene = "blur"
        elif tiny > 0.50: scene = "tiny"
        elif crowd > 0.65 or edge > 0.13: scene = "crowded"
        else: scene = "clear"
        self.last = SceneState(sci, scene, tiny, n)
        return self.last

class SmartCalibrator:
    @staticmethod
    def params(state):
        conf = 0.245 - 0.050*state.sci
        iou = 0.490 - 0.050*state.sci
        if state.scene in ("crowded", "tiny", "night"): conf -= 0.012
        if state.scene == "blur": iou -= 0.012
        conf = float(np.clip(conf, 0.19, 0.28))
        iou = float(np.clip(iou, 0.40, 0.52))
        if state.sci > 0.60 or state.tiny_ratio > 0.50: imgsz = 832
        elif state.sci > 0.35 or state.scene in ("crowded", "tiny"): imgsz = 736
        else: imgsz = 640
        return {"conf": conf, "iou": iou, "imgsz": imgsz}

def parse_args():
    p = argparse.ArgumentParser(description="Render Baseline vs Final AC-MOT side-by-side VisDrone demo video")
    p.add_argument("--dataset", default="/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev")
    p.add_argument("--sequence", default="uav0000249_00001_v")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--output", default="/content/ACMOT_Baseline_vs_Final.mp4")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=600, help="0 = full sequence")
    p.add_argument("--output-fps", type=float, default=20.0)
    p.add_argument("--panel-width", type=int, default=960)
    p.add_argument("--save-json", action="store_true")
    return p.parse_args()

def validate_dataset(root, seq):
    seq_dir = root / "sequences" / seq
    ann = root / "annotations" / f"{seq}.txt"
    if not seq_dir.is_dir(): raise FileNotFoundError(f"Missing sequence: {seq_dir}")
    if not ann.is_file(): raise FileNotFoundError(f"Missing annotation: {ann}")
    frames = sorted(seq_dir.glob("*.jpg"))
    if not frames: raise RuntimeError(f"No JPG frames in {seq_dir}")
    return ann, frames

def write_yaml(path, cfg):
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

def load_gt(path):
    cols = ["frame","id","x","y","w","h","score","cat","trunc","occ"]
    df = pd.read_csv(path, header=None, names=cols)
    df = df[df.cat.isin(VISDRONE_GT_CLASSES) & (df.score == 1) & (df.occ < 2) & (df.trunc < 2)]
    out = defaultdict(list)
    for r in df.itertuples(index=False):
        out[int(r.frame)].append({"id": int(r.id), "xyxy": np.array([r.x, r.y, r.x+r.w, r.y+r.h], np.float32)})
    return out

def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)), np.float32)
    a = a[:,None,:]; b = b[None,:,:]
    x1 = np.maximum(a[...,0], b[...,0]); y1 = np.maximum(a[...,1], b[...,1])
    x2 = np.minimum(a[...,2], b[...,2]); y2 = np.minimum(a[...,3], b[...,3])
    inter = np.maximum(0,x2-x1) * np.maximum(0,y2-y1)
    aa = np.maximum(0,a[...,2]-a[...,0]) * np.maximum(0,a[...,3]-a[...,1])
    bb = np.maximum(0,b[...,2]-b[...,0]) * np.maximum(0,b[...,3]-b[...,1])
    return inter / (aa + bb - inter + 1e-9)

class DemoIDSwitchCounter:
    def __init__(self, thr=0.50):
        self.thr = thr
        self.last = {}
        self.total = 0
    def update(self, gt_items, tracks):
        if not gt_items or not tracks: return []
        g = np.stack([x["xyxy"] for x in gt_items]); t = np.stack([x["xyxy"] for x in tracks])
        m = iou_matrix(g, t).copy(); switched = []
        while m.size:
            gi, ti = np.unravel_index(np.argmax(m), m.shape)
            if float(m[gi,ti]) < self.thr: break
            gid = int(gt_items[gi]["id"]); tid = int(tracks[ti]["id"])
            if gid in self.last and self.last[gid] != tid:
                self.total += 1; switched.append(tid)
            self.last[gid] = tid
            m[gi,:] = -1; m[:,ti] = -1
        return switched

def tracks_from_result(result):
    b = result.boxes
    if b is None or len(b) == 0 or b.id is None: return []
    xyxy = b.xyxy.detach().cpu().numpy(); ids = b.id.detach().cpu().numpy().astype(int)
    conf = b.conf.detach().cpu().numpy(); cls = b.cls.detach().cpu().numpy().astype(int)
    return [{"xyxy": xyxy[i].astype(np.float32), "id": int(ids[i]), "conf": float(conf[i]), "cls": int(cls[i])} for i in range(len(ids))]

def draw_panel(frame, tracks, title, lines, switched_ids, color):
    img = frame.copy(); h,w = img.shape[:2]
    overlay = img.copy(); band_h = 150
    cv2.rectangle(overlay, (0,0), (w,band_h), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.putText(img, title, (18,34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 3, cv2.LINE_AA)
    y = 64
    for line in lines:
        cv2.putText(img, line, (18,y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245,245,245), 1, cv2.LINE_AA); y += 25
    switched = set(switched_ids)
    for tr in tracks:
        x1,y1,x2,y2 = np.round(tr["xyxy"]).astype(int); tid = tr["id"]
        x1=max(0,min(w-1,x1)); y1=max(0,min(h-1,y1)); x2=max(0,min(w-1,x2)); y2=max(0,min(h-1,y2))
        c = (0,220,255) if tid in switched else color
        th = 5 if tid in switched else 3
        cv2.rectangle(img,(x1,y1),(x2,y2),c,th)
        text = f"ID {tid}" + ("  ID SWITCH" if tid in switched else "")
        yy = max(band_h+22, y1)
        cv2.putText(img,text,(x1,yy),cv2.FONT_HERSHEY_SIMPLEX,0.55,c,2,cv2.LINE_AA)
    return img

def resize_width(img, width):
    h,w = img.shape[:2]; return cv2.resize(img,(width,int(round(h*width/w))),interpolation=cv2.INTER_AREA)

def add_footer(canvas, frame_idx):
    h,w = canvas.shape[:2]; out = np.zeros((h+92,w,3),np.uint8); out[:h] = canvas
    b=OFFICIAL["baseline"]; a=OFFICIAL["acmot"]
    l1=f"Official 17-seq | Baseline: MOTA {b['MOTA']:.3f} HOTA {b['HOTA']:.3f} IDF1 {b['IDF1']:.3f} IDS {b['IDS']} Core FPS {b['FPS']:.2f}"
    l2=f"Final AC-MOT: MOTA {a['MOTA']:.3f} HOTA {a['HOTA']:.3f} IDF1 {a['IDF1']:.3f} IDS {a['IDS']} Core FPS {a['FPS']:.2f}"
    l3=f"Frame {frame_idx} | Live 'Demo IDS' is GT-linked IoU>=0.50 visualization, not official TrackEval IDSW."
    for y,text,fs,col in [(h+25,l1,.55,(225,225,225)),(h+52,l2,.55,(225,225,225)),(h+78,l3,.48,(170,170,170))]:
        cv2.putText(out,text,(18,y),cv2.FONT_HERSHEY_SIMPLEX,fs,col,1,cv2.LINE_AA)
    return out

def main():
    args = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required. In Colab choose T4 GPU.")
    print("GPU:", torch.cuda.get_device_name(0))
    root = Path(args.dataset); ann, frames = validate_dataset(root, args.sequence)
    start = max(1,args.start_frame); selected = frames[start-1:]
    if args.max_frames > 0: selected = selected[:args.max_frames]
    gt = load_gt(ann)
    work = Path("/content/acmot_video_demo"); work.mkdir(parents=True,exist_ok=True)
    by = work/"baseline_bytetrack.yaml"; fy = work/"final_acmot_bytetrack.yaml"
    write_yaml(by,BASELINE_TRACKER); write_yaml(fy,FINAL_TRACKER)
    baseline_model = YOLO(args.weights); acmot_model = YOLO(args.weights)
    analyzer = SceneAnalyzer(); calibrator = SmartCalibrator(); prev_boxes = np.empty((0,4),np.float32)
    base_ids = DemoIDSwitchCounter(); ac_ids = DemoIDSwitchCounter()
    warm = cv2.imread(str(selected[0]))
    for m in (baseline_model,acmot_model): m.predict(warm,conf=.25,iou=.5,imgsz=640,classes=COCO_CLASSES,device=0,half=True,verbose=False)
    torch.cuda.synchronize()
    out_path = Path(args.output); out_path.parent.mkdir(parents=True,exist_ok=True)
    writer = None; rows=[]; t_all=time.perf_counter()
    for frame_idx, fp in enumerate(selected,start=start):
        frame = cv2.imread(str(fp));
        if frame is None: raise RuntimeError(f"Could not read {fp}")
        t=time.perf_counter()
        rb = baseline_model.track(frame,persist=True,tracker=str(by),conf=.25,iou=.70,imgsz=640,classes=COCO_CLASSES,device=0,half=True,verbose=False)[0]
        torch.cuda.synchronize(); bms=(time.perf_counter()-t)*1000
        bt = tracks_from_result(rb)
        state = analyzer.maybe(frame_idx,frame,prev_boxes); p = calibrator.params(state)
        t=time.perf_counter()
        ra = acmot_model.track(frame,persist=True,tracker=str(fy),conf=p["conf"],iou=p["iou"],imgsz=p["imgsz"],classes=COCO_CLASSES,device=0,half=True,verbose=False)[0]
        torch.cuda.synchronize(); ams=(time.perf_counter()-t)*1000
        at = tracks_from_result(ra); prev_boxes = np.stack([x["xyxy"] for x in at]) if at else np.empty((0,4),np.float32)
        g = gt.get(frame_idx,[]); sb=base_ids.update(g,bt); sa=ac_ids.update(g,at)
        left=draw_panel(frame,bt,"BASELINE_DEFAULT",[
            "Detector: conf=.25  NMS IoU=.70  imgsz=640",
            f"Tracks={len(bt)}  Demo IDS={base_ids.total}  frame={bms:.1f} ms",
            "ByteTrack: high=.25 low=.10 new=.25 buffer=30 match=.80",
        ],sb,(70,80,255))
        right=draw_panel(frame,at,"FINAL AC-MOT",[
            f"SCI={state.sci:.3f} scene={state.scene} tiny={state.tiny_ratio:.2f} prev_objects={state.object_count}",
            f"Detector: conf={p['conf']:.3f}  NMS IoU={p['iou']:.3f}  imgsz={p['imgsz']}",
            f"Tracks={len(at)}  Demo IDS={ac_ids.total}  frame={ams:.1f} ms",
            "ByteTrack: high=.18 low=.04 new=.24 buffer=45 match=.88",
        ],sa,(70,230,100))
        left=resize_width(left,args.panel_width); right=resize_width(right,args.panel_width)
        ph=min(left.shape[0],right.shape[0]); canvas=np.concatenate([left[:ph],right[:ph]],axis=1)
        if sb and not sa:
            cv2.rectangle(canvas,(0,ph-44),(canvas.shape[1],ph),(0,0,0),-1)
            cv2.putText(canvas,"VISIBLE IMPROVEMENT: baseline changed ID while AC-MOT kept association",(24,ph-14),cv2.FONT_HERSHEY_SIMPLEX,.72,(0,220,255),2,cv2.LINE_AA)
        canvas=add_footer(canvas,frame_idx)
        if writer is None:
            h,w=canvas.shape[:2]; writer=cv2.VideoWriter(str(out_path),cv2.VideoWriter_fourcc(*"mp4v"),args.output_fps,(w,h))
            if not writer.isOpened(): raise RuntimeError("Could not open MP4 writer")
        writer.write(canvas)
        rows.append({"frame":frame_idx,"baseline_demo_ids":base_ids.total,"acmot_demo_ids":ac_ids.total,"baseline_switch":bool(sb),"acmot_switch":bool(sa),"sci":state.sci,"scene":state.scene,"conf":p["conf"],"iou":p["iou"],"imgsz":p["imgsz"],"baseline_ms":bms,"acmot_ms":ams})
        if len(rows)%50==0: print(f"{len(rows)}/{len(selected)} | Demo IDS baseline={base_ids.total} AC-MOT={ac_ids.total} | SCI={state.sci:.3f} imgsz={p['imgsz']}")
    writer.release()

    # Re-encode to H.264 when ffmpeg is available so the MP4 plays reliably in browsers/Colab.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        h264_tmp = out_path.with_name(out_path.stem + "_h264_tmp.mp4")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(out_path),
               "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(h264_tmp)]
        try:
            subprocess.run(cmd, check=True)
            h264_tmp.replace(out_path)
            print("H.264 web-compatible MP4 ready.")
        except Exception as e:
            print("WARNING: H.264 transcode failed; keeping original MP4:", e)
            if h264_tmp.exists(): h264_tmp.unlink()

    summary={"sequence":args.sequence,"frames":len(selected),"baseline_demo_ids":base_ids.total,"acmot_demo_ids":ac_ids.total,"output":str(out_path),"render_wall_fps":len(selected)/max(time.perf_counter()-t_all,1e-9),"official_metrics":OFFICIAL,"note":"Demo IDS is GT-linked IoU>=0.50 visualization, not official TrackEval IDSW."}
    print(json.dumps(summary,indent=2)); print("Video:",out_path)
    if args.save_json:
        jp=out_path.with_suffix(".json"); jp.write_text(json.dumps({"summary":summary,"frames":rows},indent=2),encoding="utf-8"); print("JSON:",jp)

if __name__ == "__main__":
    main()

from pathlib import Path
from datetime import datetime
import os, sys, json, shutil, subprocess, re, time

REPO_URL = "https://github.com/AhmedCode110/ACMOT-Codex-V10Style-Portable.git"
REPO = Path("/content/ACMOT-Codex-V10Style-Portable")
SWEEP_NAME = "master_sweep_20260907_172533"
SYSTEMS = [
    "Baseline_Default",
    "TRK_MATCH_090",
    "TRK_BUFFER60_NEW22_MATCH88",
    "TRK_NEW24_MATCH88",
]
CANDIDATES = SYSTEMS[1:]
FOLDER_IDS = {
    "Baseline_Default": "13vIUAbW-GGG8gvvvF-zH-OOSaKtAUCix",
    "TRK_MATCH_090": "1Ew_UNtK3M1RBm5ajmJOAn2ggDS9Hl_o5",
    "TRK_BUFFER60_NEW22_MATCH88": "1N2ZMIWzp5LomuKRj0uIcBel6GJWAvCM5",
    "TRK_NEW24_MATCH88": "1jFVmxHFgquXWbOTRmRVCuQVZnIEQ9QWu",
}
REALTIME_FPS = 25.0
VISDRONE_GT_CLASSES = [1, 4, 5, 6, 9]
COCO_CLASSES = [0, 2, 5, 7]


def run(cmd):
    print("$", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True)


def setup():
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    if REPO.exists():
        run(["git", "-C", str(REPO), "pull", "--ff-only"])
    else:
        run(["git", "clone", REPO_URL, str(REPO)])
    os.chdir(REPO)
    run([sys.executable, "-m", "pip", "install", "-q",
         "ultralytics==8.3.200", "opencv-python-headless", "pandas",
         "numpy", "scipy", "lap", "pyyaml", "tqdm", "gdown"])
    import numpy as np
    for name, value in {"float": float, "int": int, "bool": bool}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print("Repo commit:", commit)
    return commit


def locate_or_download_sweep():
    roots = [
        Path("/content/drive/MyDrive/VisDrone_Results_SHARED/ACMOT_SWEEPS") / SWEEP_NAME,
        Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_SWEEPS") / SWEEP_NAME,
        Path("/content/drive/MyDrive/ACMOT_SWEEPS_SHARED") / SWEEP_NAME,
        Path("/content/drive/MyDrive/ACMOT_SWEEPS") / SWEEP_NAME,
    ]
    for p in roots:
        if p.is_dir() and all(
            (p / "trial_outputs" / s / "predictions_mot").is_dir()
            and (p / "trial_outputs" / s / "trial_config.json").is_file()
            for s in SYSTEMS
        ):
            print("Using mounted sweep:", p)
            return p

    print("Sweep not mounted. Downloading only baseline + 3 finalists...")
    import gdown
    local = Path("/content") / SWEEP_NAME
    if local.exists():
        shutil.rmtree(local)
    (local / "trial_outputs").mkdir(parents=True)
    for system, folder_id in FOLDER_IDS.items():
        dest = local / "trial_outputs" / system
        dest.mkdir(parents=True, exist_ok=True)
        files = gdown.download_folder(
            id=folder_id, output=str(dest), quiet=False,
            use_cookies=False, remaining_ok=True
        )
        if not files:
            raise RuntimeError(f"Could not download {system}")
        nested = dest / system
        if nested.is_dir():
            for item in nested.iterdir():
                shutil.move(str(item), dest / item.name)
            nested.rmdir()
        if not (dest / "trial_config.json").is_file():
            raise RuntimeError(f"Missing trial_config.json for {system}")
        if len(list((dest / "predictions_mot").glob("*.txt"))) != 17:
            raise RuntimeError(f"Expected 17 replay prediction files for {system}")
    return local


def stage_dataset():
    drive_ds = Path("/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev")
    local_ds = Path("/content/VisDrone2019-MOT-test-dev")
    if not (drive_ds / "sequences").is_dir() or not (drive_ds / "annotations").is_dir():
        raise RuntimeError(f"Dataset missing: {drive_ds}")
    if not (local_ds / "sequences").is_dir() or not (local_ds / "annotations").is_dir():
        if local_ds.exists():
            shutil.rmtree(local_ds)
        print("Copying dataset to local Colab SSD. Copy is outside timing...")
        shutil.copytree(drive_ds, local_ds)
    seqs = sorted(p.name for p in (local_ds / "sequences").iterdir() if p.is_dir())
    frames = sum(len(list((local_ds / "sequences" / s).glob("*.jpg"))) for s in seqs)
    anns = {p.stem for p in (local_ds / "annotations").glob("*.txt")}
    if len(seqs) != 17 or frames != 6635 or any(s not in anns for s in seqs):
        raise RuntimeError(f"Expected 17 sequences / 6635 frames, got {len(seqs)} / {frames}")
    print("Dataset verified:", len(seqs), "sequences /", frames, "frames")
    return local_ds, seqs


def ensure_t4():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    gpu = torch.cuda.get_device_name(0)
    print("GPU:", gpu)
    if "T4" not in gpu:
        raise RuntimeError("Final benchmark requires Tesla T4")
    return gpu


def ensure_trackeval():
    root = Path("/content/TrackEval")
    if not (root / "trackeval" / "__init__.py").exists():
        if root.exists():
            shutil.rmtree(root)
        run(["git", "clone", "--depth", "1", "https://github.com/JonathonLuiten/TrackEval.git", str(root)])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import trackeval
    return trackeval


def make_gt(gt_parent, dataset, seqs):
    import pandas as pd
    gt_root = gt_parent / "VisDroneACMOT-test"
    for seq in seqs:
        cols = ["frame", "id", "x", "y", "w", "h", "score", "cat", "trunc", "occ"]
        gt = pd.read_csv(dataset / "annotations" / f"{seq}.txt", header=None, names=cols)
        gt = gt[
            gt.cat.isin(VISDRONE_GT_CLASSES)
            & (gt.score == 1)
            & (gt.occ < 2)
            & (gt.trunc < 2)
        ]
        d = gt_root / seq / "gt"
        d.mkdir(parents=True, exist_ok=True)
        mot = pd.DataFrame({
            0: gt.frame.astype(int), 1: gt.id.astype(int),
            2: gt.x, 3: gt.y, 4: gt.w, 5: gt.h,
            6: 1, 7: 1, 8: 1
        })
        mot.to_csv(d / "gt.txt", header=False, index=False)
        n = len(list((dataset / "sequences" / seq).glob("*.jpg")))
        (gt_root / seq / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=30\n"
            f"seqLength={n}\nimWidth=0\nimHeight=0\nimExt=.jpg\n",
            encoding="utf-8",
        )


def parse_summary(path):
    lines = [x.strip() for x in path.read_text(errors="replace").splitlines() if x.strip()]
    if len(lines) < 2:
        return {}
    h = re.split(r"\s+", lines[0])
    v = re.split(r"\s+", lines[1])
    out = {}
    for k, x in zip(h, v):
        try:
            out[k] = float(x)
        except ValueError:
            out[k] = x
    return out


def official_trackeval(tag, prediction_roots, dataset, seqs, output):
    import pandas as pd
    import numpy as np
    trackeval = ensure_trackeval()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    gt_parent = output / "trackeval_gt"
    trk_parent = output / "trackeval_trackers"
    make_gt(gt_parent, dataset, seqs)
    seqmap = output / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n", encoding="utf-8")

    for name, pred_root in prediction_roots.items():
        d = trk_parent / "VisDroneACMOT-test" / name / "data"
        d.mkdir(parents=True, exist_ok=True)
        for seq in seqs:
            src = Path(pred_root) / f"{seq}.txt"
            if not src.is_file():
                raise RuntimeError(f"Missing prediction: {src}")
            shutil.copy2(src, d / src.name)

    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update({
        "USE_PARALLEL": False, "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": True, "PRINT_CONFIG": False,
        "OUTPUT_SUMMARY": True, "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
    })
    dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dc.update({
        "GT_FOLDER": str(gt_parent),
        "TRACKERS_FOLDER": str(trk_parent),
        "TRACKERS_TO_EVAL": list(prediction_roots),
        "BENCHMARK": "VisDroneACMOT",
        "SPLIT_TO_EVAL": "test",
        "SEQMAP_FILE": str(seqmap),
        "DO_PREPROC": False,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "PRINT_CONFIG": False,
    })
    mc = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
    evaluator = trackeval.Evaluator(ec)
    evaluator.evaluate(
        [trackeval.datasets.MotChallenge2DBox(dc)],
        [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)],
    )

    rows = []
    base = trk_parent / "VisDroneACMOT-test"
    for name in prediction_roots:
        merged = {}
        for p in (base / name).rglob("*_summary.txt"):
            merged.update(parse_summary(p))
        rows.append({
            "system": name,
            "HOTA": float(merged.get("HOTA", np.nan)),
            "MOTA": float(merged.get("MOTA", np.nan)),
            "IDF1": float(merged.get("IDF1", np.nan)),
            "IDS": float(merged.get("IDSW", merged.get("IDs", merged.get("IDS", np.nan)))),
        })
    df = pd.DataFrame(rows)
    if df[["HOTA", "MOTA", "IDF1", "IDS"]].isna().any().any():
        raise RuntimeError(f"Could not parse TrackEval results\n{df}")
    df.to_csv(output / f"{tag}_OFFICIAL.csv", index=False)
    print("\nOFFICIAL", tag)
    print(df.to_string(index=False))
    return df


def load_trial(sweep, system):
    p = sweep / "trial_outputs" / system / "trial_config.json"
    if not p.is_file():
        raise RuntimeError(f"Missing config: {p}")
    return json.loads(p.read_text())


def build_live_components():
    import cv2
    import numpy as np
    from collections import deque
    from dataclasses import dataclass
    from types import SimpleNamespace
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.engine.results import Boxes

    @dataclass
    class SceneState:
        sci: float = 0.0
        scene: str = "clear"
        tiny_ratio: float = 0.0
        object_count: int = 0

    class Analyzer:
        def __init__(self, cfg):
            self.cfg = cfg
            self.hist = deque(maxlen=int(cfg.get("window", 7)))
            self.last = SceneState()

        def maybe(self, frame, img, prev_boxes):
            stride = int(self.cfg.get("stride", 10))
            if frame != 1 and frame % stride != 1:
                return self.last
            small = cv2.resize(img, None, fx=0.25, fy=0.25)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            edge = float(cv2.Canny(gray, 50, 120).mean() / 255.0)
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
                + float(self.cfg.get("edge_w", 0.20))
                * min(edge / float(self.cfg.get("edge_norm", 0.14)), 1.0)
                + float(self.cfg.get("night_w", 0.10)) * (brightness < 80)
                + float(self.cfg.get("blur_w", 0.05)) * (blur < 180)
            )
            self.hist.append(float(np.clip(raw, 0.0, 1.0)))
            sci = float(np.mean(self.hist))
            scene = (
                "night" if brightness < 80 else
                "blur" if blur < 180 else
                "tiny" if tiny > 0.50 else
                "crowded" if crowd > 0.65 or edge > 0.13 else
                "clear"
            )
            self.last = SceneState(sci=sci, scene=scene, tiny_ratio=tiny, object_count=n)
            return self.last

    class Calibrator:
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
            imgsz = (
                832 if state.sci > float(self.cfg.get("sci_high", 0.60))
                or state.tiny_ratio > float(self.cfg.get("tiny_gate", 0.50))
                else 736 if state.sci > float(self.cfg.get("sci_mid", 0.35))
                or state.scene in ["crowded", "tiny"]
                else 640
            )
            return {
                "imgsz": int(imgsz),
                "conf": float(np.clip(
                    conf,
                    float(self.cfg.get("conf_floor", 0.19)),
                    float(self.cfg.get("conf_ceil", 0.28)),
                )),
                "iou": float(np.clip(
                    iou,
                    float(self.cfg.get("iou_floor", 0.40)),
                    float(self.cfg.get("iou_ceil", 0.52)),
                )),
            }

    def make_tracker(cfg):
        return BYTETracker(
            SimpleNamespace(
                track_high_thresh=float(cfg.get("high", 0.18)),
                track_low_thresh=float(cfg.get("low", 0.04)),
                new_track_thresh=float(cfg.get("new", 0.20)),
                track_buffer=int(cfg.get("buffer", 45)),
                match_thresh=float(cfg.get("match", 0.86)),
                fuse_score=bool(cfg.get("fuse", True)),
            ),
            frame_rate=30,
        )

    return SceneState, Analyzer, Calibrator, make_tracker, Boxes


def warmup_model(model, dataset, seqs):
    import cv2
    import torch
    first = sorted((dataset / "sequences" / seqs[0]).glob("*.jpg"))[0]
    img = cv2.imread(str(first))
    if img is None:
        raise RuntimeError(f"Unreadable warmup frame: {first}")
    print("Warming YOLOv8n FP16 at 640 / 736 / 832 outside timing...")
    for size in (640, 736, 832):
        _ = model.predict(
            img, conf=0.20, iou=0.45, imgsz=size,
            classes=COCO_CLASSES, max_det=1000,
            half=True, device=0, verbose=False,
        )[0]
        torch.cuda.synchronize()
    fp16 = bool(getattr(getattr(model, "predictor", None).model, "fp16", False))
    print("actual_model_fp16:", fp16)
    if not fp16:
        raise RuntimeError("Ultralytics did not report FP16")
    return fp16


def run_live_system(system, trial, model, dataset, seqs, output_root):
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from tqdm.auto import tqdm

    SceneState, Analyzer, Calibrator, make_tracker, Boxes = build_live_components()

    run_dir = output_root / system
    if run_dir.exists():
        shutil.rmtree(run_dir)
    pred_dir = run_dir / "predictions_mot"
    pred_dir.mkdir(parents=True, exist_ok=True)

    process_s = 0.0
    decode_s = 0.0
    total_frames = 0
    per_seq_rows = []
    total_expected = sum(len(list((dataset / "sequences" / s).glob("*.jpg"))) for s in seqs)

    with tqdm(total=total_expected, desc=system, dynamic_ncols=True) as pbar:
        for seq in seqs:
            tracker = make_tracker(trial.get("tracker", {}))
            analyzer = Analyzer(trial.get("sci", {}))
            calibrator = Calibrator(trial.get("calib", {}))
            prev_boxes = np.empty((0, 4), dtype=float)
            pred_lines = []
            seq_proc = 0.0
            seq_decode = 0.0
            frames = sorted((dataset / "sequences" / seq).glob("*.jpg"))

            for frame_path in frames:
                frame_idx = int(frame_path.stem)

                d0 = time.perf_counter()
                img = cv2.imread(str(frame_path))
                d1 = time.perf_counter()
                if img is None:
                    raise RuntimeError(f"Unreadable frame: {frame_path}")
                dec = d1 - d0
                decode_s += dec
                seq_decode += dec

                torch.cuda.synchronize()
                p0 = time.perf_counter()

                if trial.get("adaptive", True):
                    state = analyzer.maybe(frame_idx, img, prev_boxes)
                    params = calibrator.params(state)
                else:
                    state = SceneState()
                    params = {"imgsz": 640, "conf": 0.25, "iou": 0.45}

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
                torch.cuda.synchronize()

                dets = result.boxes.data.detach().cpu().numpy()
                if dets.size:
                    dets = dets.reshape(-1, dets.shape[-1]).astype(float)
                else:
                    dets = np.empty((0, 6), dtype=float)

                tracks = np.asarray(
                    tracker.update(Boxes(dets, tuple(img.shape[:2]))),
                    dtype=float
                ).reshape(-1, 8)

                if len(tracks):
                    ids = tracks[:, 4].astype(int)
                    pred_boxes = tracks[:, :4].astype(float)
                    pred_scores = tracks[:, 5].astype(float)
                else:
                    ids = np.array([], dtype=int)
                    pred_boxes = np.empty((0, 4), dtype=float)
                    pred_scores = np.array([], dtype=float)

                prev_boxes = pred_boxes.copy()

                torch.cuda.synchronize()
                p1 = time.perf_counter()
                proc = p1 - p0
                process_s += proc
                seq_proc += proc

                for tid, box, score in zip(ids, pred_boxes, pred_scores):
                    x1, y1, x2, y2 = box.tolist()
                    pred_lines.append(
                        f"{frame_idx},{int(tid)},{x1:.2f},{y1:.2f},"
                        f"{x2-x1:.2f},{y2-y1:.2f},{float(score):.6f},-1,-1,-1\n"
                    )

                total_frames += 1
                pbar.update(1)
                if total_frames % 100 == 0:
                    pbar.set_postfix(
                        processing_fps=f"{total_frames/max(process_s,1e-9):.2f}",
                        e2e_fps=f"{total_frames/max(process_s+decode_s,1e-9):.2f}",
                    )

            (pred_dir / f"{seq}.txt").write_text("".join(pred_lines), encoding="utf-8")
            per_seq_rows.append({
                "sequence": seq,
                "frames": len(frames),
                "processing_seconds": seq_proc,
                "decode_seconds": seq_decode,
                "processing_fps": len(frames) / max(seq_proc, 1e-9),
                "decode_inclusive_fps": len(frames) / max(seq_proc + seq_decode, 1e-9),
            })

    summary = {
        "system": system,
        "frames": total_frames,
        "processing_seconds": process_s,
        "decode_seconds": decode_s,
        "processing_fps": total_frames / max(process_s, 1e-9),
        "decode_inclusive_fps": total_frames / max(process_s + decode_s, 1e-9),
        "actual_model_fp16": bool(getattr(getattr(model, "predictor", None).model, "fp16", False)),
        "gpu": torch.cuda.get_device_name(0),
        "timing_protocol": "processing=SCI/controller+YOLOv8n FP16+ByteTrack; decode measured separately; Drive writes and TrackEval excluded",
    }
    pd.DataFrame(per_seq_rows).to_csv(run_dir / "TIMING_PER_SEQUENCE.csv", index=False)
    (run_dir / "TIMING_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTIMING", system)
    print(json.dumps(summary, indent=2))
    return run_dir, pred_dir, summary


def main():
    import pandas as pd
    from ultralytics import YOLO

    commit = setup()
    gpu = ensure_t4()
    sweep = locate_or_download_sweep()
    dataset, seqs = stage_dataset()

    root = Path("/content/drive/MyDrive/ACMOT_LIVE_FINAL") / (
        "priority_realtime_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    root.mkdir(parents=True, exist_ok=False)

    model = YOLO("yolov8n.pt")
    fp16 = warmup_model(model, dataset, seqs)

    all_rows = []

    baseline_trial = load_trial(sweep, "Baseline_Default")
    b_run, b_pred, b_timing = run_live_system(
        "Baseline_Default", baseline_trial, model, dataset, seqs, root
    )
    b_eval = official_trackeval(
        "BASELINE_LIVE_FP16",
        {"Baseline_Default": b_pred},
        dataset, seqs,
        b_run / "official_trackeval",
    ).iloc[0]
    baseline = {
        "system": "Baseline_Default",
        "HOTA": float(b_eval.HOTA),
        "MOTA": float(b_eval.MOTA),
        "IDF1": float(b_eval.IDF1),
        "IDS": int(b_eval.IDS),
        **b_timing,
    }
    all_rows.append(baseline)

    for system in CANDIDATES:
        trial = load_trial(sweep, system)
        c_run, c_pred, c_timing = run_live_system(system, trial, model, dataset, seqs, root)
        c_eval = official_trackeval(
            f"{system}_LIVE_FP16",
            {system: c_pred},
            dataset, seqs,
            c_run / "official_trackeval",
        ).iloc[0]
        row = {
            "system": system,
            "HOTA": float(c_eval.HOTA),
            "MOTA": float(c_eval.MOTA),
            "IDF1": float(c_eval.IDF1),
            "IDS": int(c_eval.IDS),
            **c_timing,
        }
        row["realtime_pass"] = row["processing_fps"] >= REALTIME_FPS
        row["mota_better_than_baseline"] = row["MOTA"] > baseline["MOTA"]
        row["ids_better_than_baseline"] = row["IDS"] < baseline["IDS"]
        row["eligible"] = (
            row["realtime_pass"]
            and row["mota_better_than_baseline"]
            and row["ids_better_than_baseline"]
        )
        row["mota_gain_vs_baseline"] = row["MOTA"] - baseline["MOTA"]
        row["ids_reduction_vs_baseline"] = baseline["IDS"] - row["IDS"]
        row["processing_fps_margin_vs_25"] = row["processing_fps"] - REALTIME_FPS
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    candidates = df[df.system.isin(CANDIDATES)].copy()

    valid = candidates[candidates["eligible"] == True].copy()
    if not valid.empty:
        ranked = valid.sort_values(
            ["MOTA", "IDS", "processing_fps"],
            ascending=[False, True, False]
        ).reset_index(drop=True)
        winner = ranked.iloc[0].to_dict()
        status = "PASS_PUBLISHABLE_SELECTION"
    else:
        ranked = candidates.sort_values(
            ["realtime_pass", "MOTA", "IDS", "processing_fps"],
            ascending=[False, False, True, False]
        ).reset_index(drop=True)
        winner = None
        status = "NO_CANDIDATE_MET_ALL_CONSTRAINTS"

    df.to_csv(root / "FINAL_LIVE_COMPARISON.csv", index=False)

    final = {
        "status": status,
        "selection_rule": {
            "priority_1": f"processing_fps >= {REALTIME_FPS}",
            "priority_2": "MOTA > live Baseline_Default MOTA",
            "priority_3": "IDS < live Baseline_Default IDS",
            "tie_break_after_all_pass": "higher MOTA, then lower IDS, then higher processing_fps",
        },
        "hardware": gpu,
        "precision": "YOLOv8n FP16",
        "actual_model_fp16": fp16,
        "repo_commit": commit,
        "run_root": str(root),
        "baseline": {
            "HOTA": baseline["HOTA"],
            "MOTA": baseline["MOTA"],
            "IDF1": baseline["IDF1"],
            "IDS": baseline["IDS"],
            "processing_fps": baseline["processing_fps"],
            "decode_inclusive_fps": baseline["decode_inclusive_fps"],
        },
        "winner": None if winner is None else {
            "system": winner["system"],
            "HOTA": float(winner["HOTA"]),
            "MOTA": float(winner["MOTA"]),
            "IDF1": float(winner["IDF1"]),
            "IDS": int(winner["IDS"]),
            "processing_fps": float(winner["processing_fps"]),
            "decode_inclusive_fps": float(winner["decode_inclusive_fps"]),
            "mota_gain_vs_baseline": float(winner["mota_gain_vs_baseline"]),
            "ids_reduction_vs_baseline": int(winner["ids_reduction_vs_baseline"]),
            "processing_fps_margin_vs_25": float(winner["processing_fps_margin_vs_25"]),
        },
    }
    (root / "FINAL_PRIORITY_RESULT.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    report_lines = [
        "# AC-MOT Final Real-Time Priority Result",
        "",
        f"Status: **{status}**",
        "",
        "Selection priority:",
        f"1. Real-time processing >= {REALTIME_FPS:.1f} FPS.",
        "2. MOTA must beat the live Baseline_Default under the same protocol.",
        "3. IDS must be lower than the live Baseline_Default.",
        "4. Among systems passing all gates: highest MOTA, then lowest IDS, then highest processing FPS.",
        "",
        "Timing excludes JPEG decode, Drive writes, metric computation, CSV/report generation, and TrackEval.",
        "Decode-inclusive FPS is reported separately.",
        "",
        "## Baseline",
        f"- HOTA: {baseline['HOTA']:.3f}",
        f"- MOTA: {baseline['MOTA']:.3f}",
        f"- IDF1: {baseline['IDF1']:.3f}",
        f"- IDS: {baseline['IDS']}",
        f"- Processing FPS: {baseline['processing_fps']:.3f}",
        f"- Decode-inclusive FPS: {baseline['decode_inclusive_fps']:.3f}",
        "",
    ]
    if winner is not None:
        report_lines += [
            "## Winner",
            f"- System: {winner['system']}",
            f"- HOTA: {winner['HOTA']:.3f}",
            f"- MOTA: {winner['MOTA']:.3f}",
            f"- IDF1: {winner['IDF1']:.3f}",
            f"- IDS: {int(winner['IDS'])}",
            f"- Processing FPS: {winner['processing_fps']:.3f}",
            f"- Decode-inclusive FPS: {winner['decode_inclusive_fps']:.3f}",
            f"- MOTA gain vs baseline: {winner['mota_gain_vs_baseline']:+.3f}",
            f"- IDS reduction vs baseline: {int(winner['ids_reduction_vs_baseline'])}",
            f"- FPS margin vs 25: {winner['processing_fps_margin_vs_25']:+.3f}",
        ]
    else:
        report_lines += [
            "## Winner",
            "- None. No candidate simultaneously met >=25 processing FPS, higher MOTA than baseline, and lower IDS than baseline.",
            "- Do not claim a publishable real-time winner from this run.",
        ]
    (root / "FINAL_PRIORITY_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("FINAL REAL-TIME PRIORITY COMPARISON")
    print("=" * 80)
    cols = ["system", "processing_fps", "decode_inclusive_fps", "HOTA", "MOTA", "IDF1", "IDS"]
    extra = ["realtime_pass", "mota_better_than_baseline", "ids_better_than_baseline", "eligible"]
    shown = [c for c in cols + extra if c in df.columns]
    print(df[shown].to_string(index=False))
    print("\nFINAL RESULT")
    print(json.dumps(final, indent=2))
    print("\nSaved:", root / "FINAL_PRIORITY_RESULT.json")


if __name__ == "__main__":
    main()

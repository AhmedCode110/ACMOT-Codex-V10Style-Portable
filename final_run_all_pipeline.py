from pathlib import Path
from datetime import datetime
import os, sys, json, shutil, subprocess, urllib.request, re

REPO_URL = "https://github.com/AhmedCode110/ACMOT-Codex-V10Style-Portable.git"
REPO = Path("/content/ACMOT-Codex-V10Style-Portable")
FINALISTS = ["TRK_MATCH_090", "TRK_BUFFER60_NEW22_MATCH88", "TRK_NEW24_MATCH88"]
TRIAL_FOLDER_IDS = {
    "TRK_MATCH_090": "1Ew_UNtK3M1RBm5ajmJOAn2ggDS9Hl_o5",
    "TRK_BUFFER60_NEW22_MATCH88": "1N2ZMIWzp5LomuKRj0uIcBel6GJWAvCM5",
    "TRK_NEW24_MATCH88": "1jFVmxHFgquXWbOTRmRVCuQVZnIEQ9QWu",
}
SWEEP_NAME = "master_sweep_20260907_172533"
VISDRONE_GT_CLASSES = [1, 4, 5, 6, 9]


def run(cmd, **kwargs):
    print("$", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, **kwargs)


def setup():
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    if REPO.exists():
        run(["git", "-C", str(REPO), "pull", "--ff-only"])
    else:
        run(["git", "clone", REPO_URL, str(REPO)])
    os.chdir(REPO)
    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.200", "motmetrics", "opencv-python-headless", "pandas", "numpy", "scipy", "lap", "pyyaml", "tqdm", "gdown"])
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
    ]
    for p in roots:
        if p.is_dir() and all((p / "trial_outputs" / t / "predictions_mot").is_dir() for t in FINALISTS):
            print("Using mounted sweep:", p)
            return p

    print("Mounted sweep not visible. Downloading ONLY the 3 finalist trial folders from their Drive IDs...")
    import gdown
    local = Path("/content") / SWEEP_NAME
    if local.exists():
        shutil.rmtree(local)
    (local / "trial_outputs").mkdir(parents=True)
    for trial, folder_id in TRIAL_FOLDER_IDS.items():
        dest = local / "trial_outputs" / trial
        dest.mkdir(parents=True, exist_ok=True)
        files = gdown.download_folder(id=folder_id, output=str(dest), quiet=False, use_cookies=False, remaining_ok=True)
        if not files:
            raise RuntimeError(f"Could not download finalist folder {trial}")
        if not (dest / "predictions_mot").is_dir() or not (dest / "trial_config.json").is_file():
            # gdown may create one nested folder named after the Drive folder.
            nested = dest / trial
            if nested.is_dir():
                for item in nested.iterdir():
                    shutil.move(str(item), dest / item.name)
                nested.rmdir()
        if len(list((dest / "predictions_mot").glob("*.txt"))) != 17:
            raise RuntimeError(f"Expected 17 prediction files for {trial}")
        if not (dest / "trial_config.json").is_file():
            raise RuntimeError(f"Missing trial_config.json for {trial}")
    print("Local finalist sweep ready:", local)
    return local


def dataset_root():
    p = Path("/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev")
    if not (p / "sequences").is_dir() or not (p / "annotations").is_dir():
        raise RuntimeError(f"Dataset missing: {p}")
    seqs = sorted(x.name for x in (p / "sequences").iterdir() if x.is_dir())
    frames = sum(len(list((p / "sequences" / s).glob("*.jpg"))) for s in seqs)
    if len(seqs) != 17 or frames != 6635:
        raise RuntimeError(f"Expected 17 sequences / 6635 frames, got {len(seqs)} / {frames}")
    print("Dataset verified: 17 sequences / 6635 frames")
    return p, seqs


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
        gt = gt[(gt.cat.isin(VISDRONE_GT_CLASSES)) & (gt.score == 1) & (gt.occ < 2) & (gt.trunc < 2)]
        d = gt_root / seq / "gt"
        d.mkdir(parents=True, exist_ok=True)
        mot = pd.DataFrame({0: gt.frame.astype(int), 1: gt.id.astype(int), 2: gt.x, 3: gt.y, 4: gt.w, 5: gt.h, 6: 1, 7: 1, 8: 1})
        mot.to_csv(d / "gt.txt", header=False, index=False)
        n = len(list((dataset / "sequences" / seq).glob("*.jpg")))
        (gt_root / seq / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=30\nseqLength={n}\nimWidth=0\nimHeight=0\nimExt=.jpg\n",
            encoding="utf-8",
        )


def parse_summary(path):
    lines = [x.strip() for x in path.read_text(errors="replace").splitlines() if x.strip()]
    if len(lines) < 2:
        return {}
    h = re.split(r"\s+", lines[0]); v = re.split(r"\s+", lines[1]); out = {}
    for k, x in zip(h, v):
        try: out[k] = float(x)
        except ValueError: out[k] = x
    return out


def official_trackeval(tag, prediction_roots, dataset, seqs, output):
    import pandas as pd, numpy as np
    trackeval = ensure_trackeval()
    if output.exists():
        shutil.rmtree(output)
    gt_parent = output / "trackeval_gt"
    trk_parent = output / "trackeval_trackers"
    make_gt(gt_parent, dataset, seqs)
    seqmap = output / "seqmap.txt"
    output.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n", encoding="utf-8")
    for name, pred_root in prediction_roots.items():
        d = trk_parent / "VisDroneACMOT-test" / name / "data"
        d.mkdir(parents=True, exist_ok=True)
        for seq in seqs:
            src = Path(pred_root) / f"{seq}.txt"
            if not src.is_file(): raise RuntimeError(f"Missing prediction: {src}")
            shutil.copy2(src, d / src.name)

    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update({"USE_PARALLEL": False, "PRINT_RESULTS": True, "PRINT_ONLY_COMBINED": True, "PRINT_CONFIG": False, "OUTPUT_SUMMARY": True, "OUTPUT_DETAILED": True, "PLOT_CURVES": False})
    dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dc.update({"GT_FOLDER": str(gt_parent), "TRACKERS_FOLDER": str(trk_parent), "TRACKERS_TO_EVAL": list(prediction_roots), "BENCHMARK": "VisDroneACMOT", "SPLIT_TO_EVAL": "test", "SEQMAP_FILE": str(seqmap), "DO_PREPROC": False, "TRACKER_SUB_FOLDER": "data", "OUTPUT_SUB_FOLDER": "", "PRINT_CONFIG": False})
    mc = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
    evaluator = trackeval.Evaluator(ec)
    evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(dc)], [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)])

    rows = []
    base = trk_parent / "VisDroneACMOT-test"
    for name in prediction_roots:
        merged = {}
        for p in (base / name).rglob("*_summary.txt"):
            merged.update(parse_summary(p))
        row = {"trial": name, "HOTA": float(merged.get("HOTA", np.nan)), "MOTA": float(merged.get("MOTA", np.nan)), "IDF1": float(merged.get("IDF1", np.nan)), "IDS": float(merged.get("IDSW", merged.get("IDs", merged.get("IDS", np.nan))))}
        rows.append(row)
    df = pd.DataFrame(rows)
    if df[["HOTA", "MOTA", "IDF1", "IDS"]].isna().any().any():
        raise RuntimeError(f"Could not parse official TrackEval metrics for {tag}\n{df}")
    df.to_csv(output / f"{tag}_OFFICIAL.csv", index=False)
    print("\nOFFICIAL", tag)
    print(df.to_string(index=False))
    return df


def run_live_notebook(winner, sweep_dir):
    live_url = "https://raw.githubusercontent.com/AhmedCode110/ACMOT-Codex-V10Style-Portable/main/notebooks/ACMOT_FINAL_LIVE_BENCHMARK.ipynb"
    print("Loading current valid live notebook:", live_url)
    with urllib.request.urlopen(live_url) as r:
        nb = json.loads(r.read().decode("utf-8"))
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not code_cells:
        raise RuntimeError("No code cells in current live notebook")
    for i, cell in enumerate(code_cells, 1):
        src = "".join(cell.get("source", []))
        if "# CELL 3 - Select latest sweep and finalist" in src:
            src = re.sub(r'^SWEEP_ROOT = Path\(.+?\)$', f'SWEEP_ROOT = Path({str(sweep_dir.parent)!r})', src, flags=re.M)
            src = re.sub(r'^SWEEP_DIR_OVERRIDE = .+?$', f'SWEEP_DIR_OVERRIDE = {str(sweep_dir)!r}', src, flags=re.M)
            src = re.sub(r'^FINALIST_TRIAL_NAME = .+?$', f'FINALIST_TRIAL_NAME = {winner!r}', src, flags=re.M)
        if "# CELL 4 - Verify dataset, GT, and Tesla T4" in src:
            src = re.sub(r'^OUTPUT_ROOT = Path\(.+?\)$', 'OUTPUT_ROOT = Path("/content/drive/MyDrive/ACMOT_LIVE_FINAL")', src, flags=re.M)
        if "# CELL 8 - Optional official TrackEval placeholder" in src:
            print("Skipping old TrackEval placeholder; official final TrackEval runs after live predictions.")
            continue
        print(f"\n===== LIVE NOTEBOOK CELL {i}/{len(code_cells)} =====")
        exec(compile(src, f"<live notebook cell {i}>", "exec"), globals(), globals())
    if "RUN_DIR" not in globals() or "PRED_DIR" not in globals():
        raise RuntimeError("Live notebook did not produce RUN_DIR/PRED_DIR")
    return Path(globals()["RUN_DIR"]), Path(globals()["PRED_DIR"])


def main():
    commit = setup()
    sweep = locate_or_download_sweep()
    dataset, seqs = dataset_root()

    replay_preds = {t: sweep / "trial_outputs" / t / "predictions_mot" for t in FINALISTS}
    finalist_eval = official_trackeval("FINALISTS", replay_preds, dataset, seqs, Path("/content/acmot_official_finalists"))
    ranked = finalist_eval.sort_values(["HOTA", "IDF1", "MOTA", "IDS"], ascending=[False, False, False, True]).reset_index(drop=True)
    winner = str(ranked.iloc[0].trial)
    print("\nTRUE OFFICIAL WINNER:", winner)
    print(ranked.to_string(index=False))

    run_dir, pred_dir = run_live_notebook(winner, sweep)
    final_eval = official_trackeval("FINAL_LIVE_FP16", {winner: pred_dir}, dataset, seqs, run_dir / "official_trackeval")

    live_summary = json.loads((run_dir / "live_summary.json").read_text())
    r = final_eval.iloc[0]
    final = {
        "winner": winner,
        "HOTA": float(r.HOTA), "MOTA": float(r.MOTA), "IDF1": float(r.IDF1), "IDS": int(r.IDS),
        "processing_fps": live_summary.get("processing_fps", live_summary.get("fps")),
        "decode_inclusive_fps": live_summary.get("decode_inclusive_fps"),
        "actual_model_fp16": live_summary.get("actual_model_fp16"),
        "gpu": live_summary.get("gpu"),
        "run_dir": str(run_dir),
        "repo_commit": commit,
    }
    (run_dir / "FINAL_PUBLISHABLE_RESULT.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("\n" + "="*72)
    print("FINAL PUBLISHABLE RESULT")
    print("="*72)
    print(json.dumps(final, indent=2))
    print("Saved:", run_dir / "FINAL_PUBLISHABLE_RESULT.json")


if __name__ == "__main__":
    main()

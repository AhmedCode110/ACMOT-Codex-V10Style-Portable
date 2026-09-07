"""Persistent dataset cache for AC-MOT v10_p4 Colab.

Hosted Colab /content is ephemeral. This helper lets us pay the slow many-file
Google Drive staging cost once, save the verified local dataset as one TAR file
on Drive, then restore that single archive into /content on future runtimes.

The benchmark itself still runs only from /content.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd

EXPECTED_SEQS = 17


def image_number(p: Path) -> int:
    try:
        return int(p.stem)
    except ValueError:
        return 10**12


def verify_local(local: Path) -> dict:
    seq_dir = local / "sequences"
    ann_dir = local / "annotations"
    if not seq_dir.is_dir() or not ann_dir.is_dir():
        raise RuntimeError(f"Local dataset is incomplete: {local}")
    seqs = sorted([p.name for p in seq_dir.iterdir() if p.is_dir()])
    if len(seqs) != EXPECTED_SEQS:
        raise RuntimeError(f"Expected {EXPECTED_SEQS} sequences, found {len(seqs)}")
    total_frames = 0
    for s in seqs:
        frames = sorted((seq_dir / s).glob("*.jpg"), key=image_number)
        ann = ann_dir / f"{s}.txt"
        if not ann.exists():
            raise RuntimeError(f"Missing annotation: {s}")
        gt = pd.read_csv(ann, header=None)
        gt_max = int(gt.iloc[:, 0].max()) if len(gt) else 0
        nums = [image_number(x) for x in frames]
        if len(frames) != gt_max or nums != list(range(1, len(frames) + 1)):
            raise RuntimeError(f"Frame/GT mismatch: {s}")
        total_frames += len(frames)
    meta = {"sequences": len(seqs), "total_frames": total_frames}
    (local / ".acmot_local_ready.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def is_ready(local: Path) -> bool:
    try:
        verify_local(local)
        return True
    except Exception:
        return False


def save_archive(local: Path, archive_drive: Path, temp_archive: Path) -> None:
    meta = verify_local(local)
    archive_drive.parent.mkdir(parents=True, exist_ok=True)
    if temp_archive.exists():
        temp_archive.unlink()
    print(f"Creating one local TAR from verified dataset: {temp_archive}")
    with tarfile.open(temp_archive, "w") as tf:
        tf.add(local, arcname=local.name)
    print(f"Copying single TAR to persistent Drive cache: {archive_drive}")
    shutil.copy2(temp_archive, archive_drive)
    print("PERSISTENT CACHE READY")
    print("Sequences:", meta["sequences"], "Frames:", meta["total_frames"])
    print("Archive:", archive_drive)


def restore_archive(local: Path, archive_drive: Path, temp_archive: Path) -> None:
    if not archive_drive.exists():
        raise FileNotFoundError(archive_drive)
    if local.exists():
        shutil.rmtree(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    if temp_archive.exists():
        temp_archive.unlink()
    print("Copying ONE archive from Drive -> /content ...")
    shutil.copy2(archive_drive, temp_archive)
    print("Extracting locally ...")
    with tarfile.open(temp_archive, "r") as tf:
        tf.extractall(local.parent)
    meta = verify_local(local)
    print("LOCAL DATASET RESTORED AND VERIFIED")
    print("Sequences:", meta["sequences"], "Frames:", meta["total_frames"])


def stage_from_drive(source: Path, local: Path) -> None:
    seq_dir = source / "sequences"
    ann_dir = source / "annotations"
    if not seq_dir.is_dir() or not ann_dir.is_dir():
        raise RuntimeError(f"Source dataset unavailable: {source}")
    seqs = sorted([p.name for p in seq_dir.iterdir() if p.is_dir()])
    if len(seqs) != EXPECTED_SEQS:
        raise RuntimeError(f"Expected {EXPECTED_SEQS} sequences, found {len(seqs)}")
    if local.exists():
        shutil.rmtree(local)
    (local / "sequences").mkdir(parents=True)
    (local / "annotations").mkdir(parents=True)
    for i, s in enumerate(seqs, 1):
        print(f"[{i:02d}/{EXPECTED_SEQS}] staging {s}")
        dst = local / "sequences" / s
        dst.mkdir()
        for fp in sorted((seq_dir / s).glob("*.jpg"), key=image_number):
            shutil.copy2(fp, dst / fp.name)
        shutil.copy2(ann_dir / f"{s}.txt", local / "annotations" / f"{s}.txt")
    verify_local(local)
    print("SLOW FIRST-TIME STAGING COMPLETE")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local", type=Path, default=Path("/content/visdrone_v10_p4_local/VisDrone2019-MOT-test-dev"))
    p.add_argument("--source", type=Path, default=Path("/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev"))
    p.add_argument("--archive", type=Path, default=Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_CODEX_V10STYLE/V10_P4_PERSISTENT_CACHE/VisDrone2019-MOT-test-dev_v10_p4.tar"))
    p.add_argument("--temp-archive", type=Path, default=Path("/content/VisDrone2019-MOT-test-dev_v10_p4.tar"))
    p.add_argument("--save", action="store_true", help="Save the already-staged local dataset to the persistent single-file cache")
    p.add_argument("--restore-or-stage", action="store_true", help="Use local copy if ready; otherwise restore archive; otherwise slow-stage then save archive")
    a = p.parse_args()

    if a.save:
        save_archive(a.local, a.archive, a.temp_archive)
        return

    if a.restore_or_stage:
        if is_ready(a.local):
            print("LOCAL DATASET ALREADY READY — NO DRIVE READ")
            print("Using:", a.local)
            return
        if a.archive.exists():
            restore_archive(a.local, a.archive, a.temp_archive)
            return
        stage_from_drive(a.source, a.local)
        save_archive(a.local, a.archive, a.temp_archive)
        return

    p.error("Choose --save or --restore-or-stage")


if __name__ == "__main__":
    main()

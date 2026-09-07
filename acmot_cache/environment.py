"""Environment snapshots for reproducible AC-MOT experiments."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def git_commit(cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:
        return None


def snapshot(cwd: Path | None = None) -> dict:
    info = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(cwd),
        "numpy": _version("numpy"),
        "opencv": _version("cv2"),
        "ultralytics": _version("ultralytics"),
        "scipy": _version("scipy"),
        "lap": _version("lap"),
        "lapx": _version("lapx"),
        "torch": _version("torch"),
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["cudnn"] = torch.backends.cudnn.version()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    except Exception as exc:
        info["torch_error"] = repr(exc)
    return info


def save_snapshot(path: Path, cwd: Path | None = None) -> dict:
    data = snapshot(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data

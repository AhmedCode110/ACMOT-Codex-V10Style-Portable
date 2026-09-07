"""Central AC-MOT path resolver for Colab and local dry-runs."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_CACHE_ROOT = Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_CACHE")
DEFAULT_OUTPUT_ROOT = Path("/content/drive/MyDrive/VisDrone_Results")
DEFAULT_DATASET = Path(
    "/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev"
)


class DrivePaths:
    def __init__(self, cache_root: Path | None = None, mydrive: Path | None = None):
        self.mydrive = Path(os.environ.get("ACMOT_MYDRIVE", mydrive or "/content/drive/MyDrive"))
        self.cache_root = Path(os.environ.get("ACMOT_CACHE_ROOT", cache_root or DEFAULT_CACHE_ROOT))
        self.registry_dir = self.cache_root / "registry"
        self.reports_dir = self.cache_root / "reports"
        self.audit_dir = self.reports_dir / "cache_audits"
        self.plan_dir = self.reports_dir / "experiment_plans"
        self.env_dir = self.registry_dir / "environment_snapshots"
        self.dataset_manifest_dir = self.cache_root / "dataset" / "manifests"
        self.detection_manifest_dir = self.cache_root / "detections" / "manifests"
        self.tracker_manifest_dir = self.cache_root / "trackers" / "manifests"
        self.evaluation_manifest_dir = self.cache_root / "evaluation" / "manifests"
        self.experiment_manifest_dir = self.cache_root / "experiments" / "manifests"

    @property
    def registry_path(self) -> Path:
        return self.registry_dir / "cache_registry.json"

    @property
    def legacy_inventory_json(self) -> Path:
        return self.registry_dir / "LEGACY_CACHE_INVENTORY.json"

    @property
    def legacy_inventory_md(self) -> Path:
        return self.registry_dir / "LEGACY_CACHE_INVENTORY.md"

    @property
    def drive_map_md(self) -> Path:
        return self.registry_dir / "ACMOT_MASTER_DRIVE_MAP.md"

    def ensure_dirs(self) -> None:
        for path in [
            self.registry_dir,
            self.env_dir,
            self.dataset_manifest_dir,
            self.detection_manifest_dir,
            self.tracker_manifest_dir,
            self.evaluation_manifest_dir,
            self.experiment_manifest_dir,
            self.audit_dir,
            self.plan_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def likely_search_roots(mydrive: Path | None = None) -> list[Path]:
    root = Path(os.environ.get("ACMOT_MYDRIVE", mydrive or "/content/drive/MyDrive"))
    candidates = [
        root,
        root / "VisDrone_Results",
        root / "AC-MOT-results",
        root / "visdrone",
        root / "ACMOT_MASTER",
        Path("/content/drive/Shareddrives"),
    ]
    return [p for p in candidates if p.exists()]

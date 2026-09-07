"""AC-MOT v10_p4 cache manager.

Default policy:
DISCOVER -> VALIDATE -> PLAN -> REUSE -> COMPUTE ONLY MISSING STAGES.

Discovery and dry-run commands never launch YOLO, tracking benchmarks, or
TrackEval. They only inspect accessible files and update/read manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acmot_cache.discovery import discover_paths, records_to_json, write_inventory
from acmot_cache.drive_paths import DEFAULT_DATASET, DrivePaths, likely_search_roots
from acmot_cache.environment import save_snapshot
from acmot_cache.planner import format_plan, load_experiment_config, plan
from acmot_cache.planner import expected_ids
from acmot_cache.registry import CacheRegistry
from acmot_cache.results import collect_manifest_results, compatibility_warnings, write_results_csv
from acmot_cache.validation import VISDRONE_TEST_DEV, validate_dataset_root


FROZEN_V10_P4_NAME = "codex_v10_p4_fp16_20260907_134125"
FROZEN_V10_P4_PATH = (
    "/content/drive/MyDrive/VisDrone_Results/ACMOT_CODEX_V10STYLE/"
    "codex_v10_p4_fp16_20260907_134125"
)


def load_registry(paths: DrivePaths) -> CacheRegistry:
    paths.ensure_dirs()
    return CacheRegistry.load(paths.registry_path)


def import_frozen_v10_p4(registry: CacheRegistry) -> None:
    key = "completed_run_codex_v10_p4_fp16_20260907_134125"
    registry.upsert(
        "completed_runs",
        key,
        {
            "cache_id": key,
            "name": FROZEN_V10_P4_NAME,
            "artifact_type": "completed_run",
            "storage": {
                "type": "legacy_google_drive",
                "drive_id": None,
                "path_hint": FROZEN_V10_P4_PATH,
            },
            "project_version": "v10_p4",
            "dataset": VISDRONE_TEST_DEV["name"],
            "sequence_count": 17,
            "frame_count": 6635,
            "model": "YOLOv8n",
            "precision": "FP16",
            "imgsz": [640, 736, 832],
            "tracker": "ByteTrack",
            "systems": ["Baseline_Default", "Baseline_TunedTracker", "ACMOT_V10STYLE_SCI"],
            "validation_status": "FROZEN_FINAL",
            "completion_status": "complete",
            "frozen": True,
            "safe_reuse_purposes": [
                "paper reference",
                "final FP16 result comparison if protocol matches",
                "reuse saved predictions and TrackEval outputs by reference",
            ],
            "unsafe_reuse_purposes": [
                "do not overwrite",
                "do not rerun just to test cache system",
                "do not use as FP32 evidence",
            ],
        },
    )


def command_discover(args: argparse.Namespace) -> None:
    paths = DrivePaths(cache_root=args.cache_root, mydrive=args.mydrive)
    registry = load_registry(paths)
    roots = [Path(p) for p in args.root] if args.root else likely_search_roots(args.mydrive)
    records = discover_paths(roots, max_items=args.max_items)
    write_inventory(records, paths.legacy_inventory_json, paths.legacy_inventory_md)
    for record in records:
        registry.upsert("legacy_artifacts", record.cache_id, records_to_json([record])[0])
    import_frozen_v10_p4(registry)
    registry.data["discovery_completed"] = True
    registry.data["discovery_roots"] = [str(r) for r in roots]
    registry.save()
    print("DISCOVERY COMPLETE")
    print("Drive locations searched:")
    for root in roots:
        print(f"- {root}")
    print("Legacy artifacts discovered:", len(records))
    print("Registry:", paths.registry_path)
    print("Inventory JSON:", paths.legacy_inventory_json)
    print("Inventory MD:", paths.legacy_inventory_md)
    print("Expensive GPU inference: NO")


def command_verify(args: argparse.Namespace) -> None:
    paths = DrivePaths(cache_root=args.cache_root, mydrive=args.mydrive)
    registry = load_registry(paths)
    dataset_path = args.dataset or DEFAULT_DATASET
    dataset_result = validate_dataset_root(dataset_path)
    dataset_key = expected_ids(load_experiment_config(None))["dataset"]
    registry.upsert(
        "datasets",
        dataset_key,
        {
            "cache_id": dataset_key,
            "artifact_type": "dataset",
            "storage": {"type": "google_drive_path", "drive_id": None, "path_hint": str(dataset_path)},
            "config": VISDRONE_TEST_DEV,
            "validation_status": dataset_result["status"],
            "completion_status": "complete" if dataset_result["status"] == "VALID" else "partial",
            "dataset": VISDRONE_TEST_DEV["name"],
            "sequence_count": dataset_result.get("sequence_count"),
            "frame_count": dataset_result.get("frame_count"),
            "validation_details": dataset_result,
        },
    )
    import_frozen_v10_p4(registry)
    registry.save()
    print("VERIFY COMPLETE")
    print(json.dumps(dataset_result, indent=2, sort_keys=True))
    print("Expensive GPU inference: NO")


def command_list(args: argparse.Namespace) -> None:
    paths = DrivePaths(cache_root=args.cache_root, mydrive=args.mydrive)
    registry = load_registry(paths)
    rows = registry.list_entries(args.category)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    print("AC-MOT CACHE REGISTRY")
    print("Entries:", len(rows))
    for row in rows:
        print(f"- {row.get('category')} | {row.get('validation_status')} | {row.get('name', row.get('cache_id'))}")
    print("Expensive GPU inference: NO")


def command_status(args: argparse.Namespace) -> None:
    paths = DrivePaths(cache_root=args.cache_root, mydrive=args.mydrive)
    registry = load_registry(paths)
    print("AC-MOT CACHE STATUS")
    print("Registry:", paths.registry_path)
    print("Discovery completed:", bool(registry.data.get("discovery_completed")))
    for cat in ["datasets", "detections", "trackers", "evaluation", "completed_runs", "legacy_artifacts"]:
        print(f"{cat}: {len(registry.data.get(cat, {}))}")
    print("Expensive GPU inference: NO")


def command_plan(args: argparse.Namespace) -> None:
    paths = DrivePaths(cache_root=args.cache_root, mydrive=args.mydrive)
    registry = load_registry(paths)
    config = load_experiment_config(args.config)
    plan_data = plan(config, registry.data, mode=args.mode, discovery_completed=bool(registry.data.get("discovery_completed")))
    paths.plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = paths.plan_dir / "latest_plan.json"
    plan_path.write_text(json.dumps(plan_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(format_plan(plan_data))
    print("")
    print("Plan JSON:", plan_path)
    print("No expensive work has been started.")


def command_dataset_status(args: argparse.Namespace) -> None:
    dataset_path = args.dataset or DEFAULT_DATASET
    print(json.dumps(validate_dataset_root(dataset_path), indent=2, sort_keys=True))
    print("Expensive GPU inference: NO")


def command_snapshot(args: argparse.Namespace) -> None:
    paths = DrivePaths(cache_root=args.cache_root, mydrive=args.mydrive)
    paths.ensure_dirs()
    out = args.output or (paths.env_dir / "latest_environment.json")
    data = save_snapshot(out, Path.cwd())
    print(json.dumps(data, indent=2, sort_keys=True))
    print("Environment snapshot:", out)
    print("Expensive GPU inference: NO")


def command_results(args: argparse.Namespace) -> None:
    rows = collect_manifest_results(args.root)
    write_results_csv(rows, args.output, args.sort_by)
    print("RESULT TABLE COMPLETE")
    print("Rows:", len(rows))
    print("Output:", args.output)
    for warning in compatibility_warnings(rows):
        print(warning)
    print("Expensive GPU inference: NO")


def main() -> None:
    parser = argparse.ArgumentParser(description="AC-MOT cache-first manager")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--mydrive", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("discover")
    p.add_argument("--root", action="append", default=[])
    p.add_argument("--max-items", type=int, default=5000)
    p.set_defaults(func=command_discover)

    p = sub.add_parser("verify")
    p.add_argument("--dataset", type=Path, default=None)
    p.set_defaults(func=command_verify)

    p = sub.add_parser("list")
    p.add_argument("--category", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=command_list)

    p = sub.add_parser("status")
    p.set_defaults(func=command_status)

    p = sub.add_parser("plan")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--mode", choices=["auto", "replay", "live"], default="auto")
    p.set_defaults(func=command_plan)

    p = sub.add_parser("dataset-status")
    p.add_argument("--dataset", type=Path, default=None)
    p.set_defaults(func=command_dataset_status)

    p = sub.add_parser("environment-snapshot")
    p.add_argument("--output", type=Path, default=None)
    p.set_defaults(func=command_snapshot)

    p = sub.add_parser("results-table")
    p.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/VisDrone_Results"))
    p.add_argument("--output", type=Path, default=Path("/content/drive/MyDrive/VisDrone_Results/ACMOT_CACHE/reports/cache_audits/results_table.csv"))
    p.add_argument("--sort-by", choices=["HOTA", "IDF1", "MOTA", "IDS", "FPS"], default=None)
    p.set_defaults(func=command_results)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

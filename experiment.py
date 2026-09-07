"""Cache-aware AC-MOT experiment entry point.

This wrapper prevents accidental expensive work. Use --dry-run first. Live mode
delegates to fair_benchmark_v10_p4.py only after cache discovery has completed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from acmot_cache.drive_paths import DrivePaths
from acmot_cache.planner import format_plan, load_experiment_config, plan
from acmot_cache.registry import CacheRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="AC-MOT cache-aware experiment runner")
    parser.add_argument("--mode", choices=["auto", "replay", "live"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--mydrive", type=Path, default=None)
    parser.add_argument("--allow-expensive", action="store_true", help="Required before live YOLO inference can start")
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    paths = DrivePaths(cache_root=args.cache_root, mydrive=args.mydrive)
    registry = CacheRegistry.load(paths.registry_path)
    config = load_experiment_config(args.config)
    plan_data = plan(config, registry.data, mode=args.mode, discovery_completed=bool(registry.data.get("discovery_completed")))

    print("PRE-RUN CACHE AUDIT")
    print(format_plan(plan_data))

    if args.dry_run:
        print("")
        print("DRY RUN COMPLETE")
        print("No expensive work has been started.")
        return

    if args.mode == "replay" and plan_data["missing_detector_resolutions"]:
        print("")
        print("REPLAY MODE REFUSED")
        for size in plan_data["missing_detector_resolutions"]:
            print(f"Missing: YOLOv8n {config['detector']['precision']} imgsz={size} cache")
        print("No YOLO inference was started.")
        raise SystemExit(2)

    if plan_data["expensive_yolo_inference"] and not registry.data.get("discovery_completed"):
        print("")
        print("REFUSING EXPENSIVE INFERENCE")
        print("Reason: cache discovery has not been completed for this environment.")
        print("Run: python cache_manager_v10_p4.py discover")
        raise SystemExit(2)

    if args.mode in {"auto", "live"} and plan_data["expensive_yolo_inference"] and not args.allow_expensive:
        print("")
        print("REFUSING EXPENSIVE INFERENCE")
        print("Reason: --allow-expensive is required for YOLO live inference.")
        print("Review the dry-run plan first.")
        raise SystemExit(2)

    if args.mode == "live":
        cmd = [sys.executable, "fair_benchmark_v10_p4.py", *args.benchmark_args]
        print("Launching live benchmark:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return

    print("")
    print("No compatible executable replay implementation was launched by this wrapper yet.")
    print("Use cache_manager_v10_p4.py plan/list to identify reusable inputs, then connect the replay script.")
    print("No expensive work has been started.")


if __name__ == "__main__":
    main()

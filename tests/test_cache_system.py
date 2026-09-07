from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acmot_cache.hashing import config_hash
from acmot_cache.planner import DEFAULT_EXPERIMENT, expected_ids, plan
from acmot_cache.registry import CacheRegistry
from acmot_cache.validation import validate_detection_manifest


class CacheSystemTests(unittest.TestCase):
    def test_hash_is_deterministic(self):
        self.assertEqual(config_hash({"b": 2, "a": 1}), config_hash({"a": 1, "b": 2}))

    def test_registry_atomic_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache_registry.json"
            reg = CacheRegistry.load(path)
            reg.upsert("datasets", "dataset_x", {"validation_status": "VALID", "completion_status": "complete"})
            reg.save()
            loaded = CacheRegistry.load(path)
            self.assertEqual(loaded.data["datasets"]["dataset_x"]["validation_status"], "VALID")

    def test_plan_reports_missing_detector_cache(self):
        out = plan(DEFAULT_EXPERIMENT, {"datasets": {}, "detections": {}, "trackers": {}, "evaluation": {}}, mode="auto")
        self.assertTrue(out["missing_detector_resolutions"])
        self.assertTrue(out["expensive_yolo_inference"])

    def test_replay_refuses_missing_detections(self):
        out = plan(DEFAULT_EXPERIMENT, {"datasets": {}, "detections": {}, "trackers": {}, "evaluation": {}}, mode="replay")
        actions = [d["action"] for d in out["decisions"]]
        self.assertIn("ABORT", actions)
        self.assertFalse(any(d["expensive_gpu"] for d in out["decisions"] if d["stage"] == "replay_guard"))

    def test_cache_hit_reuses_detector_cache(self):
        ids = expected_ids(DEFAULT_EXPERIMENT)
        registry = {
            "datasets": {ids["dataset"]: {"validation_status": "VALID", "completion_status": "complete"}},
            "detections": {
                det: {"validation_status": "VALID", "completion_status": "complete"}
                for det in ids["detections"].values()
            },
            "trackers": {},
            "evaluation": {},
        }
        out = plan(DEFAULT_EXPERIMENT, registry, mode="replay")
        self.assertEqual(out["missing_detector_resolutions"], [])
        self.assertFalse(out["expensive_yolo_inference"])

    def test_fp32_fp16_separation(self):
        manifest = {"precision": "FP32", "imgsz": 736, "completion_status": "complete"}
        result = validate_detection_manifest(manifest, {"precision": "FP16", "imgsz": 736})
        self.assertEqual(result["status"], "INCOMPATIBLE")
        self.assertIn("precision", result["mismatches"])

    def test_resolution_mismatch(self):
        manifest = {"precision": "FP16", "imgsz": 640, "completion_status": "complete"}
        result = validate_detection_manifest(manifest, {"precision": "FP16", "imgsz": 832})
        self.assertEqual(result["status"], "INCOMPATIBLE")
        self.assertIn("imgsz", result["mismatches"])

    def test_frozen_final_can_be_listed(self):
        data = json.loads(Path("cache_registry.json").read_text(encoding="utf-8"))
        frozen = data["completed_runs"]["completed_run_codex_v10_p4_fp16_20260907_134125"]
        self.assertTrue(frozen["frozen"])
        self.assertEqual(frozen["validation_status"], "FROZEN_FINAL")


if __name__ == "__main__":
    unittest.main()

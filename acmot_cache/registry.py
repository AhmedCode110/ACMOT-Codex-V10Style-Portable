"""Persistent cache registry with atomic writes."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_CATEGORIES = [
    "datasets",
    "detections",
    "trackers",
    "predictions",
    "evaluation",
    "completed_runs",
    "legacy_artifacts",
]


def empty_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": None,
        "datasets": {},
        "detections": {},
        "trackers": {},
        "predictions": {},
        "evaluation": {},
        "completed_runs": {},
        "legacy_artifacts": {},
    }


class CacheRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = empty_registry()

    @classmethod
    def load(cls, path: Path) -> "CacheRegistry":
        registry = cls(path)
        if registry.path.exists():
            loaded = json.loads(registry.path.read_text(encoding="utf-8"))
            base = empty_registry()
            base.update(loaded)
            for key in REGISTRY_CATEGORIES:
                base.setdefault(key, {})
            registry.data = base
        return registry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    def upsert(self, category: str, key: str, entry: dict[str, Any]) -> None:
        if category not in REGISTRY_CATEGORIES:
            raise KeyError(f"Unknown registry category: {category}")
        current = self.data.setdefault(category, {}).get(key, {})
        merged = {**current, **entry}
        merged.setdefault("cache_id", key)
        self.data[category][key] = merged

    def list_entries(self, category: str | None = None) -> list[dict[str, Any]]:
        cats = [category] if category else REGISTRY_CATEGORIES
        rows = []
        for cat in cats:
            for key, value in self.data.get(cat, {}).items():
                row = dict(value)
                row.setdefault("cache_id", key)
                row.setdefault("category", cat)
                rows.append(row)
        return rows

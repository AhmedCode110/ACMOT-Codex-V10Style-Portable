"""Deterministic cache identity helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def normalized_json(value: Any) -> str:
    """Return canonical JSON used for reproducible cache IDs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(normalized_json(value).encode("utf-8")).hexdigest()
    return digest[:length]


def cache_id(prefix: str, value: Any, length: int = 16) -> str:
    return f"{prefix}_{config_hash(value, length)}"

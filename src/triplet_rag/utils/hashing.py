"""Deterministic hashing for config dicts.

Two configs that produce the same artifacts must produce the same hash.
We canonicalize by sorting keys and using compact JSON, then take a SHA1
prefix that's short enough for paths but long enough to avoid collisions.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonicalize(obj: Any) -> Any:
    """Recursively normalize for stable hashing."""
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, float):
        # Round to avoid floating-point drift across machines
        return round(obj, 12)
    return obj


def stable_hash(obj: Any, length: int = 12) -> str:
    """Return a short hex hash that's stable across runs and machines."""
    canon = _canonicalize(obj)
    payload = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def stable_hash_str(s: str, length: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:length]

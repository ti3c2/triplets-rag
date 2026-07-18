"""IO helpers for parquet, JSONL, FAISS, JSON."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON-serialize dict-valued columns. pyarrow infers an empty `struct<>` for
    # uniformly-empty dicts (e.g. `metadata={}` across all rows), which Parquet
    # cannot encode. Strings are dtype-stable and round-trip through Parquet.
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda v: isinstance(v, dict)).any():
            df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, dict) else v)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_npy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def read_npy(path: Path) -> np.ndarray:
    return np.load(path)


def touch_success(path: Path, payload: dict[str, Any] | None = None) -> None:
    """Write a _SUCCESS marker for a phase."""
    payload = payload or {}
    write_json(payload, path)


def has_success(path: Path) -> bool:
    return path.exists()

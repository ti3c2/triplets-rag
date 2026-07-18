"""Experiment registry: a parquet index of all experiments and their headline metrics."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ..utils.io import read_json, write_parquet


def _registry_path(storage_dir: Path) -> Path:
    return storage_dir / "registry.parquet"


def register_experiment(
    storage_dir: Path,
    experiment_id: str,
    cfg_dict: dict[str, Any],
    status: str,
    aggregate_path: Path | None = None,
) -> None:
    p = _registry_path(storage_dir)
    if p.exists():
        df = pd.read_parquet(p)
    else:
        df = pd.DataFrame()

    headline: dict[str, Any] = {}
    if aggregate_path and aggregate_path.exists():
        agg = read_json(aggregate_path)
        for k in ("em", "f1", "rouge_l", "faithfulness", "answer_correctness", "answer_relevancy"):
            if k in agg:
                headline[k] = agg[k].get("mean")

    row = {
        "experiment_id": experiment_id,
        "experiment_name": cfg_dict.get("experiment_name"),
        "dataset": cfg_dict.get("dataset", {}).get("name"),
        "strategy": cfg_dict.get("inference", {}).get("strategy"),
        "indexing_strategy": cfg_dict.get("indexer", {}).get("indexing_strategy"),
        "student_kind": cfg_dict.get("student", {}).get("kind"),
        "student_model": cfg_dict.get("student", {}).get("model_name"),
        "generator_model": cfg_dict.get("generator", {}).get("model_name"),
        "embedder_model": cfg_dict.get("embedder", {}).get("model_name"),
        "status": status,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        **headline,
    }

    if df.empty:
        df = pd.DataFrame([row])
    else:
        df = df[df["experiment_id"] != experiment_id]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    write_parquet(df, p)
    logger.info(f"Registered experiment {experiment_id} ({status})")


def list_experiments(storage_dir: Path, filters: dict[str, str] | None = None) -> pd.DataFrame:
    p = _registry_path(storage_dir)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if filters:
        for k, v in filters.items():
            if k in df.columns:
                df = df[df[k].astype(str).str.contains(v.replace("*", ".*"), regex=True, na=False)]
    return df.reset_index(drop=True)

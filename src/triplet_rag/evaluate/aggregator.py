"""Aggregate metrics with bootstrap confidence intervals.

Merges retrieval, lexical, and judge metric tables into a single per-query
parquet (long format: query_id, metric, value) and an aggregate.json with
{metric: {mean, std, ci_low, ci_high, n}}.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.io import write_json, write_parquet


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means.append(sample.mean())
    arr = np.array(means)
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def aggregate_and_persist(
    *,
    out_dir: Path,
    retrieval_pq: pd.DataFrame | None,
    lexical_pq: pd.DataFrame | None,
    judge_pq: pd.DataFrame | None,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 12345,
) -> None:
    """Write metrics/per_query.parquet and metrics/aggregate.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    def _to_long(df: pd.DataFrame, fallback_metric_col: str | None = None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["query_id", "metric", "value"])
        if {"query_id", "metric", "value"}.issubset(df.columns):
            return df[["query_id", "metric", "value"]]
        # wide -> long
        long = df.melt(id_vars=["query_id"], var_name="metric", value_name="value")
        return long

    frames.append(_to_long(retrieval_pq))
    frames.append(_to_long(lexical_pq))
    frames.append(_to_long(judge_pq))
    pq = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    if pq.empty:
        logger.warning("No metrics computed; aggregate is empty")
        write_parquet(pq, out_dir / "per_query.parquet")
        write_json({}, out_dir / "aggregate.json")
        return

    pq["value"] = pd.to_numeric(pq["value"], errors="coerce")
    pq = pq.dropna(subset=["value"]).reset_index(drop=True)
    write_parquet(pq, out_dir / "per_query.parquet")

    agg: dict[str, dict[str, float]] = {}
    for metric, sub in pq.groupby("metric"):
        vals = sub["value"].to_numpy(dtype=np.float64)
        if len(vals) == 0:
            continue
        ci_low, ci_high = _bootstrap_ci(vals, bootstrap_n, bootstrap_seed)
        agg[str(metric)] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    write_json(agg, out_dir / "aggregate.json")
    logger.info(f"Wrote {len(agg)} aggregate metrics to {out_dir/'aggregate.json'}")

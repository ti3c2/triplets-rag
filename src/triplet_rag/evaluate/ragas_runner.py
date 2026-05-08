"""Re-run RAGAS on a completed experiment's predictions.

The classic flow runs RAGAS as part of `phase_compute_metrics`. This module
covers the offline case — an experiment was already run (perhaps with
`use_ragas: false`) and we want to add LLM-as-judge numbers without rerunning
inference. Multiple judge models can coexist: each run writes into
`experiments/<id>/metrics/ragas/<judge_tag>/` and the original metrics files
are left untouched.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from loguru import logger

from ..config import LLMConfig, MetricsConfig
from ..utils.io import read_json, read_jsonl, write_json
from .aggregator import aggregate_and_persist
from .judge import compute_ragas_metrics


def sanitize_judge_tag(model_name: str) -> str:
    """Make a model name safe as a directory component."""
    s = re.sub(r"[^A-Za-z0-9_.\-]", "_", model_name).strip("._-")
    return s or "judge"


def run_ragas_on_experiment(
    *,
    exp_dir: Path,
    judge_cfg: LLMConfig,
    metric_names: list[str],
    judge_tag: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 12345,
    force: bool = False,
) -> tuple[dict[str, float], Path]:
    """Run RAGAS over `exp_dir/predictions.jsonl` and persist results.

    `judge_base_url` / `judge_api_key` override the env-derived settings for this
    call only. Use them to point a vllm-kind judge at a self-hosted endpoint
    (e.g. http://localhost:7114/v1) without mutating env vars.

    Returns the aggregate dict and the output directory.
    """
    pred_path = exp_dir / "predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions.jsonl at {pred_path}")

    tag = judge_tag or sanitize_judge_tag(judge_cfg.model_name)
    out_dir = exp_dir / "metrics" / "ragas" / tag
    aggregate_path = out_dir / "aggregate.json"
    if aggregate_path.exists() and not force:
        raise FileExistsError(
            f"RAGAS results already exist for judge tag '{tag}' at {out_dir}. "
            "Pass force=True to overwrite."
        )

    predictions = list(read_jsonl(pred_path))
    if not predictions:
        raise ValueError(f"predictions.jsonl is empty at {pred_path}")

    cfg = MetricsConfig(
        retrieval_metrics=[],
        generation_lexical=[],
        use_ragas=True,
        ragas_metrics=metric_names,
        judge_model=judge_cfg,
        bootstrap_n=bootstrap_n,
        bootstrap_seed=bootstrap_seed,
    )

    endpoint_note = f" @ {judge_base_url}" if judge_base_url else ""
    logger.info(
        f"RAGAS rerun on {pred_path.name} with judge {judge_cfg.kind}:"
        f"{judge_cfg.model_name}{endpoint_note} (tag={tag}); metrics={metric_names}"
    )
    judge_agg, judge_pq = compute_ragas_metrics(
        predictions,
        cfg,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
    )

    if not judge_agg:
        raise RuntimeError("RAGAS returned no metrics — check the logs above for errors")

    aggregate_and_persist(
        out_dir=out_dir,
        retrieval_pq=None,
        lexical_pq=None,
        judge_pq=judge_pq,
        bootstrap_n=bootstrap_n,
        bootstrap_seed=bootstrap_seed,
    )
    write_json(
        {
            "judge_tag": tag,
            "judge_model": judge_cfg.model_dump(),
            "judge_base_url": judge_base_url,
            "metrics": metric_names,
            "n_queries": len(predictions),
            "ran_at": datetime.utcnow().isoformat() + "Z",
        },
        out_dir / "judge.json",
    )
    _update_runs_index(
        exp_dir / "metrics" / "ragas" / "runs.json",
        tag,
        judge_cfg,
        metric_names,
        judge_base_url,
    )
    logger.info(f"RAGAS results written to {out_dir}")
    return judge_agg, out_dir


def _update_runs_index(
    index_path: Path,
    judge_tag: str,
    judge_cfg: LLMConfig,
    metric_names: list[str],
    judge_base_url: str | None,
) -> None:
    runs: dict = {}
    if index_path.exists():
        try:
            runs = read_json(index_path) or {}
        except Exception:
            runs = {}
    runs[judge_tag] = {
        "judge_model": judge_cfg.model_dump(),
        "judge_base_url": judge_base_url,
        "metrics": metric_names,
        "ran_at": datetime.utcnow().isoformat() + "Z",
    }
    write_json(runs, index_path)

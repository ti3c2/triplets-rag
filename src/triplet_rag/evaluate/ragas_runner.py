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

_METRIC_K_RE = re.compile(r"@(\d+)\s*$")


def sanitize_judge_tag(model_name: str) -> str:
    """Make a model name safe as a directory component."""
    s = re.sub(r"[^A-Za-z0-9_.\-]", "_", model_name).strip("._-")
    return s or "judge"


def derive_context_ks_from_retrieval_metrics(metric_names: list[str]) -> list[int]:
    """Return sorted unique @k cutoffs from retrieval metric names."""
    ks: set[int] = set()
    for metric_name in metric_names:
        match = _METRIC_K_RE.search(str(metric_name).strip())
        if not match:
            continue
        k = int(match.group(1))
        if k > 0:
            ks.add(k)
    return sorted(ks)


def derive_context_ks_from_experiment(exp_dir: Path) -> list[int]:
    """Derive RAGAS context cutoffs from the saved experiment config."""
    cfg_path = exp_dir / "config.yaml.json"
    if not cfg_path.exists():
        return []
    try:
        cfg = read_json(cfg_path) or {}
    except Exception as e:
        logger.warning(f"Could not read experiment config for RAGAS @k derivation: {e}")
        return []
    metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg, dict) else {}
    retrieval_metrics = metrics_cfg.get("retrieval_metrics", [])
    if not isinstance(retrieval_metrics, list):
        return []
    return derive_context_ks_from_retrieval_metrics(retrieval_metrics)


def _normalize_context_ks(context_ks: list[int]) -> list[int]:
    normalized: set[int] = set()
    for raw_k in context_ks:
        k = int(raw_k)
        if k < 1:
            raise ValueError("context ks must be positive integers")
        normalized.add(k)
    return sorted(normalized)


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
    context_ks: list[int] | None = None,
    context_ks_source: str | None = None,
    max_workers: int | None = None,
    timeout: int | None = None,
    dump_inputs: bool = False,
    debug: bool = False,
) -> tuple[dict[str, float], Path]:
    """Run RAGAS over `exp_dir/predictions.jsonl` and persist results.

    `judge_base_url` / `judge_api_key` override the env-derived settings for this
    call only. Use them to point a vllm-kind judge at a self-hosted endpoint
    (e.g. http://localhost:7114/v1) without mutating env vars.

    Runtime controls such as @k selection, concurrency, timeout, input dumping,
    and debug logging stay outside MetricsConfig so experiment hashes are stable.

    Returns the aggregate dict and the output directory.
    """
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if timeout is not None and timeout < 1:
        raise ValueError("timeout must be >= 1")

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

    if context_ks is None:
        effective_context_ks = derive_context_ks_from_experiment(exp_dir)
        if effective_context_ks:
            effective_context_ks_source = (
                context_ks_source or "experiment.metrics.retrieval_metrics"
            )
        else:
            effective_context_ks_source = context_ks_source or "none"
    else:
        effective_context_ks = _normalize_context_ks(context_ks)
        effective_context_ks_source = context_ks_source or "override"

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
    context_note = (
        f"; context_ks={effective_context_ks} ({effective_context_ks_source})"
        if effective_context_ks
        else "; context_ks=all"
    )
    worker_note = f"; max_workers={max_workers}" if max_workers is not None else ""
    timeout_note = f"; timeout={timeout}s" if timeout is not None else ""
    logger.info(
        f"RAGAS rerun on {pred_path.name} with judge {judge_cfg.kind}:"
        f"{judge_cfg.model_name}{endpoint_note} (tag={tag}); metrics={metric_names}"
        f"{context_note}{worker_note}{timeout_note}"
    )
    input_dump_path = out_dir / "inputs.jsonl" if dump_inputs else None
    compute_kwargs = {
        "judge_base_url": judge_base_url,
        "judge_api_key": judge_api_key,
    }
    if effective_context_ks:
        compute_kwargs["context_ks"] = effective_context_ks
    if max_workers is not None:
        compute_kwargs["max_workers"] = max_workers
    if timeout is not None:
        compute_kwargs["timeout"] = timeout
    if input_dump_path is not None:
        compute_kwargs["input_dump_path"] = input_dump_path
    if debug:
        compute_kwargs["debug"] = debug
    judge_agg, judge_pq = compute_ragas_metrics(predictions, cfg, **compute_kwargs)

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
            "context_ks": effective_context_ks,
            "context_ks_source": effective_context_ks_source,
            "max_workers": max_workers,
            "timeout": timeout,
            "input_dump_path": input_dump_path.name if input_dump_path is not None else None,
            "debug": debug,
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
        effective_context_ks,
        effective_context_ks_source,
        max_workers,
        timeout,
        input_dump_path.name if input_dump_path is not None else None,
        debug,
    )
    logger.info(f"RAGAS results written to {out_dir}")
    return judge_agg, out_dir


def _update_runs_index(
    index_path: Path,
    judge_tag: str,
    judge_cfg: LLMConfig,
    metric_names: list[str],
    judge_base_url: str | None,
    context_ks: list[int],
    context_ks_source: str,
    max_workers: int | None,
    timeout: int | None,
    input_dump_path: str | None,
    debug: bool,
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
        "context_ks": context_ks,
        "context_ks_source": context_ks_source,
        "max_workers": max_workers,
        "timeout": timeout,
        "input_dump_path": input_dump_path,
        "debug": debug,
        "ran_at": datetime.utcnow().isoformat() + "Z",
    }
    write_json(runs, index_path)

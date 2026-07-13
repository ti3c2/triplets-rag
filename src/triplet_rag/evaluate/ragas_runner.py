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
from .judge import (
    CONTEXT_DEPENDENT_METRICS,
    CONTEXT_FREE_METRICS,
    compute_ragas_metrics,
)


def sanitize_judge_tag(model_name: str) -> str:
    """Make a model name safe as a directory component."""
    s = re.sub(r"[^A-Za-z0-9_.\-]", "_", model_name).strip("._-")
    return s or "judge"


def _parse_ks_from_retrieval_metrics(retrieval_metrics: list[str]) -> list[int]:
    """Extract the unique `@k` suffixes from a retrieval metrics list.

    "nDCG@10", "Recall@5", "P@1" → [1, 5, 10]; bare metrics like "RR" are
    ignored.
    """
    ks: set[int] = set()
    for m in retrieval_metrics:
        match = re.search(r"@(\d+)$", m.strip())
        if match:
            ks.add(int(match.group(1)))
    return sorted(ks)


def _derive_ks(exp_dir: Path, ks_override: list[int] | None) -> list[int]:
    """Pick the per-k replication levels.

    Precedence: explicit CLI override > experiment config's retrieval @k
    suffixes > [] (single un-suffixed pass, matches the legacy behavior).
    """
    if ks_override is not None:
        return sorted(set(int(k) for k in ks_override))
    cfg_path = exp_dir / "config.yaml.json"
    if not cfg_path.exists():
        return []
    try:
        cfg = read_json(cfg_path) or {}
        retrieval_metrics = cfg.get("metrics", {}).get("retrieval_metrics", []) or []
    except Exception as e:
        logger.warning(f"Could not parse {cfg_path} for ks derivation: {e}")
        return []
    return _parse_ks_from_retrieval_metrics(retrieval_metrics)


def _partition_metric_names(
    metric_names: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split into (context_free, context_dependent, unknown)."""
    cf, cd, unknown = [], [], []
    for name in metric_names:
        if name in CONTEXT_FREE_METRICS:
            cf.append(name)
        elif name in CONTEXT_DEPENDENT_METRICS:
            cd.append(name)
        else:
            unknown.append(name)
    return cf, cd, unknown


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
    max_workers: int | None = None,
    timeout: int | None = None,
    debug: bool = False,
    dump_inputs: bool = False,
    ks: list[int] | None = None,
) -> tuple[dict[str, float], Path]:
    """Run RAGAS over `exp_dir/predictions.jsonl` and persist results.

    `judge_base_url` / `judge_api_key` override the env-derived settings for this
    call only. Use them to point a vllm-kind judge at a self-hosted endpoint
    (e.g. http://localhost:7114/v1) without mutating env vars.

    Per-k evaluation: context-dependent metrics (faithfulness, context_*,
    nv_response_groundedness, nv_context_relevance) are replicated for each k
    in `ks` with truncated contexts; results are suffixed `<metric>@<k>`.
    A single k uses one RAGAS call for judge-backed metrics and, when requested,
    one additional local-metric call so local/string metrics do not consume
    judge worker slots. With multiple k values, context-free metrics run once
    and context-dependent rows for all k values are merged into one additional
    sequential RAGAS call so we do not recompute answer-only metrics. `ks=None`
    auto-derives from `<exp_dir>/config.yaml.json`'s `metrics.retrieval_metrics`;
    passing `ks=[]` explicitly disables per-k.

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

    cf_metrics, cd_metrics, unknown = _partition_metric_names(metric_names)
    if unknown:
        # Unknown metrics — try them anyway in a single pass; the judge layer
        # will warn and skip what it can't resolve. Treat as context-free so
        # we don't blow them up per-k.
        logger.warning(
            f"Unknown RAGAS metric(s) {unknown!r}; running in a single pass with full contexts"
        )
        cf_metrics = cf_metrics + unknown

    resolved_ks = _derive_ks(exp_dir, ks)
    # Per-k is only meaningful when we actually have context-dependent metrics.
    if not cd_metrics:
        resolved_ks = []

    endpoint_note = f" @ {judge_base_url}" if judge_base_url else ""
    logger.info(
        f"RAGAS rerun on {pred_path.name} with judge {judge_cfg.kind}:"
        f"{judge_cfg.model_name}{endpoint_note} (tag={tag}); "
        f"metrics={metric_names}; ks={resolved_ks or 'single-pass'}; "
        f"max_workers={max_workers}; timeout={timeout}; debug={debug}"
    )

    inputs_dir = out_dir / "inputs" if dump_inputs else None

    cfg = MetricsConfig(
        retrieval_metrics=[],
        generation_lexical=[],
        use_ragas=True,
        ragas_metrics=metric_names,
        judge_model=judge_cfg,
        bootstrap_n=bootstrap_n,
        bootstrap_seed=bootstrap_seed,
    )
    judge_agg, combined_pq = compute_ragas_metrics(
        predictions,
        cfg,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        context_ks=resolved_ks,
        max_workers=max_workers,
        timeout=timeout,
        debug=debug,
        dump_path=(inputs_dir / "ragas_inputs.jsonl") if inputs_dir else None,
    )

    if not judge_agg:
        raise RuntimeError("RAGAS returned no metrics — check the logs above for errors")

    aggregate_and_persist(
        out_dir=out_dir,
        retrieval_pq=None,
        lexical_pq=None,
        judge_pq=combined_pq,
        bootstrap_n=bootstrap_n,
        bootstrap_seed=bootstrap_seed,
    )
    write_json(
        {
            "judge_tag": tag,
            "judge_model": judge_cfg.model_dump(),
            "judge_base_url": judge_base_url,
            "metrics": metric_names,
            "ks": resolved_ks,
            "context_free_metrics": cf_metrics,
            "context_dependent_metrics": cd_metrics,
            "max_workers": max_workers,
            "timeout": timeout,
            "debug": debug,
            "dump_inputs": dump_inputs,
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
        resolved_ks,
    )
    logger.info(f"RAGAS results written to {out_dir}")
    return judge_agg, out_dir


def _update_runs_index(
    index_path: Path,
    judge_tag: str,
    judge_cfg: LLMConfig,
    metric_names: list[str],
    judge_base_url: str | None,
    ks: list[int],
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
        "ks": ks,
        "ran_at": datetime.utcnow().isoformat() + "Z",
    }
    write_json(runs, index_path)

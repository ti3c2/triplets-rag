"""LLM-as-judge metrics via RAGAS.

Why RAGAS:
- It implements faithfulness, answer relevancy, and answer correctness with
  validated decomposition (claims extraction + entailment); we don't have to
  reinvent these.
- It accepts a pluggable judge LLM via langchain wrappers, so we can pin a
  specific model independent of the student/teacher.

We compute these only when the metric set actually requires them. SQuAD-style
EM/F1 are still preferred for span-extractive evaluation; the judge fills the
gap on faithfulness, free-form answers, and demo leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ..config import LLMConfig, MetricsConfig
from ..settings import get_settings
from ..utils.io import write_jsonl

# Metrics that consume the `contexts` field. Splitting them off lets the
# offline rerunner replicate just these per retrieval-k while running the
# rest exactly once.
CONTEXT_DEPENDENT_METRICS: frozenset[str] = frozenset(
    {
        "faithfulness",
        "context_precision",
        "context_recall",
        "nv_response_groundedness",
        "nv_context_relevance",
    }
)
CONTEXT_FREE_METRICS: frozenset[str] = frozenset(
    {
        "answer_relevancy",
        "answer_correctness",
        "nv_accuracy",
    }
)


def _build_ragas_judge_llm(
    judge_cfg: LLMConfig | None,
    *,
    base_url_override: str | None = None,
    api_key_override: str | None = None,
) -> Any:
    """Build a langchain-compatible LLM for RAGAS to use as judge.

    If judge_cfg is None, RAGAS falls back to its env-default (OpenAI).

    For OpenAI-compatible endpoints (kind in {vllm, local_hf}, or kind=openai
    when you want to point at a non-OpenAI host), `base_url_override` and
    `api_key_override` take precedence over the env-derived settings. They're
    threaded through `compute_ragas_metrics` so the offline rerunner can target
    a self-hosted vLLM without mutating env vars.
    """
    if judge_cfg is None:
        return None

    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    s = get_settings()
    if judge_cfg.kind == "openai":
        chat = ChatOpenAI(
            model=judge_cfg.model_name,
            temperature=judge_cfg.temperature,
            api_key=api_key_override or s.openai_api_key,
            base_url=base_url_override,
        )
    elif judge_cfg.kind == "anthropic":
        from langchain_anthropic import ChatAnthropic

        chat = ChatAnthropic(
            model=judge_cfg.model_name,
            temperature=judge_cfg.temperature,
            api_key=api_key_override or s.anthropic_api_key,
        )
    elif judge_cfg.kind in ("vllm", "local_hf"):
        chat = ChatOpenAI(
            model=judge_cfg.model_name,
            temperature=judge_cfg.temperature,
            api_key=api_key_override or s.vllm_api_key,
            base_url=base_url_override or s.vllm_base_url,
        )
    else:
        raise ValueError(f"Unsupported judge kind for ragas: {judge_cfg.kind}")
    return LangchainLLMWrapper(chat)


def _build_ragas_embeddings() -> Any:
    """answer_relevancy needs embeddings. Use OpenAI ada by default."""
    try:
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        s = get_settings()
        emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=s.openai_api_key)
        return LangchainEmbeddingsWrapper(emb)
    except Exception as e:
        logger.warning(f"Could not build OpenAI embeddings for ragas: {e}")
        return None


def compute_ragas_metrics(
    predictions: list[dict],
    cfg: MetricsConfig,
    *,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    max_workers: int | None = None,
    timeout: int | None = None,
    debug: bool = False,
    dump_path: Path | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return (aggregate, per_query_df). Empty if ragas disabled.

    `judge_base_url` / `judge_api_key` override settings for this call only;
    used by the offline rerunner to target a self-hosted endpoint.

    `max_workers` / `timeout` are wired into a ragas `RunConfig` when set.
    `debug=True` enables `langchain_core.globals.set_debug(True)` so the judge
    prompts are printed (we don't use `set_verbose` because RAGAS bypasses the
    Chain layer). `dump_path`, when set, writes the dataset rows actually sent
    to ragas as JSONL — handy for inspecting per-k truncation.
    """
    if not cfg.use_ragas or not cfg.ragas_metrics:
        return {}, pd.DataFrame()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.metrics._nv_metrics import (
            AnswerAccuracy,
            ContextRelevance,
            ResponseGroundedness,
        )
        from ragas.run_config import RunConfig
    except Exception as e:
        logger.warning(f"RAGAS unavailable: {e}; skipping LLM-as-judge metrics")
        return {}, pd.DataFrame()

    metric_lookup = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "answer_correctness": answer_correctness,
        "context_precision": context_precision,
        "context_recall": context_recall,
        # NVIDIA family — instantiated per-call so they pick up the judge LLM.
        "nv_accuracy": AnswerAccuracy(),
        "nv_response_groundedness": ResponseGroundedness(),
        "nv_context_relevance": ContextRelevance(),
    }
    metrics = []
    for name in cfg.ragas_metrics:
        if name in metric_lookup:
            metrics.append(metric_lookup[name])
        else:
            logger.warning(f"Unknown RAGAS metric '{name}'; skipping")

    if not metrics:
        return {}, pd.DataFrame()

    # Build dataset rows
    rows = []
    for pred in predictions:
        text = pred["prediction"] or ""
        golds = list(pred.get("gold_answers", [])) or [pred.get("gold_answer", "") or ""]
        ctxs = list(pred.get("retrieved_texts", [])) or []
        rows.append(
            {
                "question": pred["query"],
                "answer": text,
                "contexts": ctxs,
                "ground_truth": golds[0] if golds else "",
                "query_id": pred["query_id"],
            }
        )
    if not rows:
        return {}, pd.DataFrame()

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(rows, dump_path)
        logger.info(f"Wrote RAGAS input rows to {dump_path}")

    if debug:
        try:
            from langchain_core.globals import set_debug

            set_debug(True)
            logger.info("langchain_core debug enabled — judge prompts will be printed")
        except Exception as e:
            logger.warning(f"Could not enable langchain debug mode: {e}")

    ds = Dataset.from_list(rows)

    judge_llm = _build_ragas_judge_llm(
        cfg.judge_model,
        base_url_override=judge_base_url,
        api_key_override=judge_api_key,
    )
    embeddings = _build_ragas_embeddings()

    run_config: Any = None
    if max_workers is not None or timeout is not None:
        # Use ragas defaults for fields the caller didn't override.
        defaults = RunConfig()
        run_config = RunConfig(
            timeout=timeout if timeout is not None else defaults.timeout,
            max_workers=max_workers if max_workers is not None else defaults.max_workers,
        )
        logger.info(
            f"RAGAS RunConfig: max_workers={run_config.max_workers}, "
            f"timeout={run_config.timeout}s"
        )

    logger.info(f"Running RAGAS with metrics: {[m.name for m in metrics]}")
    try:
        evaluate_kwargs: dict[str, Any] = {
            "metrics": metrics,
            "llm": judge_llm,
            "embeddings": embeddings,
            "raise_exceptions": False,
        }
        if run_config is not None:
            evaluate_kwargs["run_config"] = run_config
        result = evaluate(ds, **evaluate_kwargs)
    except Exception as e:
        logger.error(f"RAGAS evaluate failed: {e}")
        return {}, pd.DataFrame()

    df = result.to_pandas()
    df["query_id"] = [r["query_id"] for r in rows]

    metric_cols = [m.name for m in metrics if m.name in df.columns]
    long_rows = []
    for _, r in df.iterrows():
        for col in metric_cols:
            v = r[col]
            try:
                fv = float(v)
            except Exception:
                continue
            long_rows.append({"query_id": r["query_id"], "metric": col, "value": fv})

    pq = pd.DataFrame(long_rows)
    agg = {col: float(df[col].mean()) for col in metric_cols if df[col].notna().any()}
    return agg, pq

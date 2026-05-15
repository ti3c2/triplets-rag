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

CONTEXT_DEPENDENT_RAGAS_METRICS = frozenset(
    {
        "faithfulness",
        "context_precision",
        "context_recall",
        "nv_response_groundedness",
        "nv_context_relevance",
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


def _normalize_context_ks(context_ks: list[int] | None) -> list[int] | None:
    if context_ks is None:
        return None
    normalized = sorted({int(k) for k in context_ks if int(k) > 0})
    return normalized or None


def _prediction_to_ragas_row(pred: dict, *, context_k: int | None = None) -> dict[str, Any]:
    text = pred.get("prediction") or ""
    gold_answers = pred.get("gold_answers") or []
    if not isinstance(gold_answers, list):
        gold_answers = [gold_answers]
    if not gold_answers:
        gold_answers = [pred.get("gold_answer", "") or ""]
    contexts = list(pred.get("retrieved_texts") or [])
    if context_k is not None:
        contexts = contexts[:context_k]
    question = pred["query"]
    reference = gold_answers[0] if gold_answers else ""
    return {
        # RAGAS <=0.1 style columns.
        "question": question,
        "answer": text,
        "contexts": contexts,
        "ground_truth": reference,
        # RAGAS >=0.2 style columns.
        "user_input": question,
        "response": text,
        "retrieved_contexts": contexts,
        "reference": reference,
        "query_id": pred["query_id"],
    }


def _build_run_config(max_workers: int | None, timeout: int | None) -> Any:
    if max_workers is None and timeout is None:
        return None
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if timeout is not None and timeout < 1:
        raise ValueError("timeout must be >= 1")

    from ragas.run_config import RunConfig

    kwargs: dict[str, int] = {}
    if max_workers is not None:
        kwargs["max_workers"] = int(max_workers)
    if timeout is not None:
        kwargs["timeout"] = int(timeout)
    return RunConfig(**kwargs)


def _set_langchain_debug(debug: bool) -> tuple[Any | None, bool | None]:
    if not debug:
        return None, None
    try:
        from langchain_core.globals import get_debug, set_debug

        previous = bool(get_debug())
        set_debug(True)
        logger.info("LangChain debug enabled for RAGAS judge prompts")
        return set_debug, previous
    except Exception as e:
        logger.warning(f"Could not enable LangChain debug for RAGAS: {e}")
        return None, None


def compute_ragas_metrics(
    predictions: list[dict],
    cfg: MetricsConfig,
    *,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    context_ks: list[int] | None = None,
    max_workers: int | None = None,
    timeout: int | None = None,
    input_dump_path: Path | None = None,
    debug: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return (aggregate, per_query_df). Empty if ragas disabled.

    `judge_base_url` / `judge_api_key` override settings for this call only;
    used by the offline rerunner to target a self-hosted endpoint.

    Runtime controls such as `context_ks`, `max_workers`, `timeout`, input
    dumping, and debug mode are intentionally parameters rather than
    MetricsConfig fields so they do not alter experiment hashes.
    """
    if not cfg.use_ragas or not cfg.ragas_metrics:
        return {}, pd.DataFrame()
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if timeout is not None and timeout < 1:
        raise ValueError("timeout must be >= 1")

    restore_debug, previous_debug = _set_langchain_debug(debug)
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
        run_config = _build_run_config(max_workers, timeout)
    except Exception as e:
        logger.warning(f"RAGAS unavailable: {e}; skipping LLM-as-judge metrics")
        if restore_debug is not None and previous_debug is not None:
            restore_debug(previous_debug)
        return {}, pd.DataFrame()

    try:
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

        requested_names = []
        for name in cfg.ragas_metrics:
            if name in metric_lookup:
                requested_names.append(name)
            else:
                logger.warning(f"Unknown RAGAS metric '{name}'; skipping")

        if not requested_names:
            return {}, pd.DataFrame()

        normalized_ks = _normalize_context_ks(context_ks)
        if normalized_ks is None:
            eval_specs = [("all_contexts", requested_names, None)]
        else:
            context_free = [
                name for name in requested_names if name not in CONTEXT_DEPENDENT_RAGAS_METRICS
            ]
            context_dependent = [
                name for name in requested_names if name in CONTEXT_DEPENDENT_RAGAS_METRICS
            ]
            eval_specs = []
            if context_free:
                eval_specs.append(("context_free", context_free, None))
            for k in normalized_ks:
                if context_dependent:
                    eval_specs.append((f"context_at_{k}", context_dependent, k))

        if not predictions:
            return {}, pd.DataFrame()

        prepared_specs = []
        dump_records = []
        for label, metric_names, context_k in eval_specs:
            rows = [_prediction_to_ragas_row(pred, context_k=context_k) for pred in predictions]
            if not rows:
                continue
            prepared_specs.append((label, metric_names, context_k, rows))
            if input_dump_path is not None:
                for row in rows:
                    dump_records.append(
                        {
                            "ragas_run": label,
                            "ragas_metrics": metric_names,
                            "context_k": context_k,
                            **row,
                        }
                    )

        if not prepared_specs:
            return {}, pd.DataFrame()

        if input_dump_path is not None:
            write_jsonl(dump_records, input_dump_path)
            logger.info(f"Wrote RAGAS input dump to {input_dump_path}")

        judge_llm = _build_ragas_judge_llm(
            cfg.judge_model,
            base_url_override=judge_base_url,
            api_key_override=judge_api_key,
        )
        embeddings = _build_ragas_embeddings()

        long_rows = []
        for label, metric_names, context_k, rows in prepared_specs:
            metrics = [metric_lookup[name] for name in metric_names]
            metric_labels = {
                name: f"{name}@{context_k}"
                if context_k is not None and name in CONTEXT_DEPENDENT_RAGAS_METRICS
                else name
                for name in metric_names
            }
            logger.info(
                f"Running RAGAS ({label}) with metrics: {[m.name for m in metrics]}"
            )
            ds = Dataset.from_list(rows)
            evaluate_kwargs = {
                "metrics": metrics,
                "llm": judge_llm,
                "embeddings": embeddings,
                "raise_exceptions": False,
            }
            if run_config is not None:
                evaluate_kwargs["run_config"] = run_config
            result = evaluate(ds, **evaluate_kwargs)
            df = result.to_pandas()
            df["query_id"] = [r["query_id"] for r in rows]

            metric_cols = [m.name for m in metrics if m.name in df.columns]
            for _, r in df.iterrows():
                for col in metric_cols:
                    v = r[col]
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if pd.isna(fv):
                        continue
                    long_rows.append(
                        {
                            "query_id": r["query_id"],
                            "metric": metric_labels.get(col, col),
                            "value": fv,
                        }
                    )

        pq = pd.DataFrame(long_rows)
        if pq.empty:
            return {}, pq
        agg = {
            str(metric): float(sub["value"].mean())
            for metric, sub in pq.groupby("metric", sort=False)
            if sub["value"].notna().any()
        }
        return agg, pq
    except Exception as e:
        logger.error(f"RAGAS evaluate failed: {e}")
        return {}, pd.DataFrame()
    finally:
        if restore_debug is not None and previous_debug is not None:
            restore_debug(previous_debug)

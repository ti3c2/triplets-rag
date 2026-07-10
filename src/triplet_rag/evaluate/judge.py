"""LLM-as-judge and RAGAS metrics.

The experiment runner can compute judge metrics inline, and `eval-ragas` can
rerun them later on saved predictions. The latter is preferred for expensive
judges because multiple judge models can coexist under one experiment.
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
EMBEDDING_RAGAS_METRICS = frozenset({"answer_relevancy", "answer_correctness"})
LLM_RAGAS_METRICS = frozenset(
    {
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
        "context_recall",
        "nv_accuracy",
        "nv_response_groundedness",
        "nv_context_relevance",
        "factual_correctness",
    }
)
SUPPORTED_RAGAS_METRICS = frozenset(
    {
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
        "context_recall",
        "nv_accuracy",
        "nv_response_groundedness",
        "nv_context_relevance",
        "factual_correctness",
        "rouge_score",
        "bleu_score",
        "non_llm_string_similarity",
        "string_present",
        "exact_match",
    }
)
CONTEXT_DEPENDENT_METRICS = CONTEXT_DEPENDENT_RAGAS_METRICS
CONTEXT_FREE_METRICS = SUPPORTED_RAGAS_METRICS - CONTEXT_DEPENDENT_RAGAS_METRICS


def _build_ragas_judge_llm(
    judge_cfg: LLMConfig | None,
    *,
    base_url_override: str | None = None,
    api_key_override: str | None = None,
) -> Any:
    """Build a langchain-compatible LLM for RAGAS to use as judge.

    If judge_cfg is None, RAGAS falls back to its environment-default behavior.
    For OpenAI-compatible endpoints, `base_url_override` and `api_key_override`
    let `eval-ragas` target a self-hosted vLLM without mutating `.env`.
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
            base_url=base_url_override or judge_cfg.base_url,
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
            base_url=base_url_override or judge_cfg.base_url or s.vllm_base_url,
        )
    else:
        raise ValueError(f"Unsupported judge kind for ragas: {judge_cfg.kind}")
    return LangchainLLMWrapper(chat)


def _build_ragas_embeddings() -> Any:
    """Build embeddings for RAGAS metrics that require semantic similarity."""
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


def _make_ragas_metric(name: str) -> Any:
    """Instantiate a RAGAS metric by stable config name.

    RAGAS has moved metric exports across versions. Local imports keep optional
    dependencies isolated so one missing optional metric cannot break unrelated
    metrics.
    """
    if name == "faithfulness":
        from ragas.metrics import faithfulness

        return faithfulness
    if name == "answer_relevancy":
        from ragas.metrics import answer_relevancy

        return answer_relevancy
    if name == "answer_correctness":
        from ragas.metrics import answer_correctness

        return answer_correctness
    if name == "context_precision":
        from ragas.metrics import context_precision

        return context_precision
    if name == "context_recall":
        from ragas.metrics import context_recall

        return context_recall
    if name == "nv_accuracy":
        from ragas.metrics._nv_metrics import AnswerAccuracy

        return AnswerAccuracy()
    if name == "nv_response_groundedness":
        from ragas.metrics._nv_metrics import ResponseGroundedness

        return ResponseGroundedness()
    if name == "nv_context_relevance":
        from ragas.metrics._nv_metrics import ContextRelevance

        return ContextRelevance()
    if name == "factual_correctness":
        from ragas.metrics import FactualCorrectness

        return FactualCorrectness()
    if name == "rouge_score":
        from ragas.metrics import RougeScore

        return RougeScore()
    if name == "bleu_score":
        from ragas.metrics import BleuScore

        return BleuScore()
    if name == "non_llm_string_similarity":
        from ragas.metrics import NonLLMStringSimilarity

        return NonLLMStringSimilarity()
    if name == "string_present":
        from ragas.metrics import StringPresence

        return StringPresence()
    if name == "exact_match":
        from ragas.metrics import ExactMatch

        return ExactMatch()
    raise KeyError(name)


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
    dump_path: Path | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return (aggregate, per_query_df). Empty if RAGAS is disabled.

    `dump_path` is kept as a backward-compatible alias for `input_dump_path`.
    """
    if not cfg.use_ragas or not cfg.ragas_metrics:
        return {}, pd.DataFrame()
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if timeout is not None and timeout < 1:
        raise ValueError("timeout must be >= 1")
    if input_dump_path is None and dump_path is not None:
        input_dump_path = dump_path

    restore_debug, previous_debug = _set_langchain_debug(debug)
    try:
        from datasets import Dataset
        from ragas import evaluate

        run_config = _build_run_config(max_workers, timeout)
    except Exception as e:
        logger.warning(f"RAGAS unavailable: {e}; skipping LLM-as-judge metrics")
        if restore_debug is not None and previous_debug is not None:
            restore_debug(previous_debug)
        return {}, pd.DataFrame()

    try:
        metric_lookup: dict[str, Any] = {}
        requested_names: list[str] = []
        for name in cfg.ragas_metrics:
            if name not in SUPPORTED_RAGAS_METRICS:
                logger.warning(f"Unknown RAGAS metric '{name}'; skipping")
                continue
            try:
                metric_lookup[name] = _make_ragas_metric(name)
            except Exception as e:
                logger.warning(f"Could not initialize RAGAS metric '{name}': {e}; skipping")
                continue
            requested_names.append(name)

        if not requested_names or not predictions:
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

        judge_llm = (
            _build_ragas_judge_llm(
                cfg.judge_model,
                base_url_override=judge_base_url,
                api_key_override=judge_api_key,
            )
            if any(name in LLM_RAGAS_METRICS for name in requested_names)
            else None
        )
        embeddings = (
            _build_ragas_embeddings()
            if any(name in EMBEDDING_RAGAS_METRICS for name in requested_names)
            else None
        )

        long_rows = []
        for label, metric_names, context_k, rows in prepared_specs:
            metrics = [metric_lookup[name] for name in metric_names]
            metric_labels = {
                name: f"{name}@{context_k}"
                if context_k is not None and name in CONTEXT_DEPENDENT_RAGAS_METRICS
                else name
                for name in metric_names
            }
            logger.info(f"Running RAGAS ({label}) with metrics: {[m.name for m in metrics]}")
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
            df["query_id"] = [row["query_id"] for row in rows]

            metric_cols = [m.name for m in metrics if m.name in df.columns]
            for _, row in df.iterrows():
                for col in metric_cols:
                    value = row[col]
                    try:
                        score = float(value)
                    except Exception:
                        continue
                    if pd.isna(score):
                        continue
                    long_rows.append(
                        {
                            "query_id": row["query_id"],
                            "metric": metric_labels.get(col, col),
                            "value": score,
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

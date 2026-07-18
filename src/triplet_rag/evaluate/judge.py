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
from pydantic import SecretStr

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


def _secret(value: str | None) -> SecretStr | None:
    return SecretStr(value) if value else None


def _find_metric_result_column(
    df_columns: list[str], requested_name: str, metric_name: str
) -> str | None:
    """Map a configured metric name to the column returned by RAGAS.

    Some RAGAS metrics include constructor parameters in the result column
    name, e.g. ``rouge_score(mode=fmeasure)`` for ``RougeScore()``.
    """
    for name in (metric_name, requested_name):
        if name in df_columns:
            return name
    for name in (metric_name, requested_name):
        prefix = f"{name}("
        matches = [col for col in df_columns if col.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
    return None


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
    from ragas.llms.base import LangchainLLMWrapper

    s = get_settings()
    if judge_cfg.kind == "openai":
        chat = ChatOpenAI(
            model=judge_cfg.model_name,
            temperature=judge_cfg.temperature,
            max_completion_tokens=judge_cfg.max_tokens,
            top_p=judge_cfg.top_p,
            api_key=_secret(api_key_override or s.openai_api_key),
            base_url=base_url_override or judge_cfg.base_url,
        )
    elif judge_cfg.kind == "anthropic":
        from langchain_anthropic import ChatAnthropic

        chat = ChatAnthropic(
            model=judge_cfg.model_name,
            temperature=judge_cfg.temperature,
            max_tokens=judge_cfg.max_tokens,
            top_p=judge_cfg.top_p,
            api_key=_secret(api_key_override or s.anthropic_api_key),
        )
    elif judge_cfg.kind in ("vllm", "local_hf"):
        chat = ChatOpenAI(
            model=judge_cfg.model_name,
            temperature=judge_cfg.temperature,
            max_completion_tokens=judge_cfg.max_tokens,
            top_p=judge_cfg.top_p,
            api_key=_secret(api_key_override or s.vllm_api_key),
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
        emb = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=_secret(s.openai_api_key),
        )
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
        "user_input": question,
        "response": text,
        "retrieved_contexts": contexts,
        "reference": reference,
        "query_id": pred["query_id"],
    }


def resolve_ragas_max_workers(max_workers: int | None) -> int:
    """Resolve the RAGAS executor concurrency from CLI or the repo-wide limit."""
    resolved = get_settings().llm_concurrency if max_workers is None else int(max_workers)
    if resolved < 1:
        raise ValueError("max_workers must be >= 1")
    return resolved


def _make_ragas_metric(name: str) -> Any:
    """Instantiate a RAGAS metric by stable config name.

    RAGAS has moved metric exports across versions. Local imports keep optional
    dependencies isolated so one missing optional metric cannot break unrelated
    metrics.
    """
    if name == "faithfulness":
        from ragas.metrics._faithfulness import Faithfulness

        return Faithfulness()
    if name == "answer_relevancy":
        from ragas.metrics._answer_relevance import AnswerRelevancy

        return AnswerRelevancy()
    if name == "answer_correctness":
        from ragas.metrics._answer_correctness import AnswerCorrectness

        return AnswerCorrectness()
    if name == "context_precision":
        from ragas.metrics._context_precision import ContextPrecision

        return ContextPrecision()
    if name == "context_recall":
        from ragas.metrics._context_recall import ContextRecall

        return ContextRecall()
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


def _strip_internal_ragas_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("__")}


def _build_ragas_input_dump_records(
    calls: list[tuple[str, list[str], list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, metric_names, rows in calls:
        for row in rows:
            records.append(
                {
                    "ragas_run": row.get("__dump_label", label),
                    "ragas_metrics": metric_names,
                    "context_k": row.get("__context_k"),
                    **_strip_internal_ragas_fields(row),
                }
            )
    return records


def _evaluate_ragas_call(
    *,
    label: str,
    metric_names: list[str],
    rows: list[dict[str, Any]],
    dataset_cls: Any,
    evaluate_fn: Any,
    judge_llm: Any,
    embeddings: Any,
    run_config: Any,
) -> list[dict[str, Any]]:
    # RAGAS mutates metric instances during init/reset, so each evaluation
    # call needs freshly constructed metric objects.
    metrics = [_make_ragas_metric(name) for name in metric_names]
    logger.info(
        f"Running RAGAS ({label}) with {len(rows)} rows and metrics: {[m.name for m in metrics]}"
    )
    ds = dataset_cls.from_list([_strip_internal_ragas_fields(row) for row in rows])
    evaluate_kwargs = {
        "metrics": metrics,
        "llm": judge_llm,
        "embeddings": embeddings,
        "raise_exceptions": False,
        # RAGAS batches are drain barriers, not GPU inference batches.
        "batch_size": None,
        "run_config": run_config,
    }
    result = evaluate_fn(ds, **evaluate_kwargs)
    df = result.to_pandas()
    df["query_id"] = [row["query_id"] for row in rows]
    df["__metric_suffix"] = [row.get("__metric_suffix", "") for row in rows]

    result_cols: dict[str, str] = {}
    df_columns = list(df.columns)
    for requested_name, metric in zip(metric_names, metrics, strict=True):
        metric_name = getattr(metric, "name", requested_name)
        result_col = _find_metric_result_column(df_columns, requested_name, metric_name)
        if result_col is None:
            logger.warning(
                f"RAGAS result missing column for metric '{requested_name}'. "
                f"Available columns: {df_columns}"
            )
            continue
        result_cols[result_col] = requested_name

    long_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        suffix = str(row.get("__metric_suffix") or "")
        for col, metric_name in result_cols.items():
            value = row[col]
            try:
                score = float(value)
            except Exception:
                continue
            if pd.isna(score):
                continue
            label_name = (
                f"{metric_name}{suffix}"
                if suffix and metric_name in CONTEXT_DEPENDENT_RAGAS_METRICS
                else metric_name
            )
            long_rows.append(
                {
                    "query_id": row["query_id"],
                    "metric": label_name,
                    "value": score,
                }
            )
    return long_rows


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
    """Return (aggregate, per_query_df). Empty if RAGAS is disabled."""
    if not cfg.use_ragas or not cfg.ragas_metrics:
        return {}, pd.DataFrame()
    effective_max_workers = resolve_ragas_max_workers(max_workers)
    if timeout is not None and timeout < 1:
        raise ValueError("timeout must be >= 1")

    restore_debug, previous_debug = _set_langchain_debug(debug)
    try:
        from ragas import EvaluationDataset, RunConfig, evaluate

        run_config_kwargs = {"max_workers": effective_max_workers}
        if timeout is not None:
            run_config_kwargs["timeout"] = timeout
        run_config = RunConfig(**run_config_kwargs)
    except Exception as e:
        logger.warning(f"RAGAS unavailable: {e}; skipping LLM-as-judge metrics")
        if restore_debug is not None and previous_debug is not None:
            restore_debug(previous_debug)
        return {}, pd.DataFrame()

    try:
        requested_names: list[str] = []
        for name in cfg.ragas_metrics:
            if name not in SUPPORTED_RAGAS_METRICS:
                logger.warning(f"Unknown RAGAS metric '{name}'; skipping")
                continue
            requested_names.append(name)

        if not requested_names or not predictions:
            return {}, pd.DataFrame()

        normalized_ks = _normalize_context_ks(context_ks)
        ragas_calls: list[tuple[str, list[str], list[dict[str, Any]]]] = []

        context_free = [
            name for name in requested_names if name not in CONTEXT_DEPENDENT_RAGAS_METRICS
        ]
        context_dependent = [
            name for name in requested_names if name in CONTEXT_DEPENDENT_RAGAS_METRICS
        ]

        if normalized_ks is None:
            rows = [_prediction_to_ragas_row(pred) for pred in predictions]
            ragas_calls.append(("all_contexts", requested_names, rows))
        elif context_dependent and len(normalized_ks) == 1:
            k = normalized_ks[0]
            rows = []
            for pred in predictions:
                row = _prediction_to_ragas_row(pred, context_k=k)
                row["__metric_suffix"] = f"@{k}"
                row["__context_k"] = k
                rows.append(row)
            ragas_calls.append((f"all_metrics_at_{k}", requested_names, rows))
        else:
            if context_free:
                rows = [_prediction_to_ragas_row(pred, context_k=0) for pred in predictions]
                for row in rows:
                    row["__context_k"] = None
                ragas_calls.append(("context_free", context_free, rows))
            if context_dependent:
                rows = []
                for k in normalized_ks:
                    for pred in predictions:
                        row = _prediction_to_ragas_row(pred, context_k=k)
                        row["__metric_suffix"] = f"@{k}"
                        row["__context_k"] = k
                        row["__dump_label"] = f"context_at_{k}"
                        rows.append(row)
                ragas_calls.append(("context_dependent_by_k", context_dependent, rows))

        if input_dump_path is not None:
            dump_records = _build_ragas_input_dump_records(ragas_calls)
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
        logger.info(f"RAGAS max_workers={effective_max_workers}; batching=disabled")
        embeddings = (
            _build_ragas_embeddings()
            if any(name in EMBEDDING_RAGAS_METRICS for name in requested_names)
            else None
        )

        long_rows: list[dict[str, Any]] = []
        for label, metric_names, rows in ragas_calls:
            long_rows.extend(
                _evaluate_ragas_call(
                    label=label,
                    metric_names=metric_names,
                    rows=rows,
                    dataset_cls=EvaluationDataset,
                    evaluate_fn=evaluate,
                    judge_llm=judge_llm,
                    embeddings=embeddings,
                    run_config=run_config,
                )
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

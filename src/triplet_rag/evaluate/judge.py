"""LLM-as-judge and RAGAS metrics.

The experiment runner can compute judge metrics inline, and `eval-ragas` can
rerun them later on saved predictions. The latter is preferred for expensive
judges because multiple judge models can coexist under one experiment.
"""

from __future__ import annotations

import asyncio
import copy
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
REMOTE_RAGAS_METRICS = LLM_RAGAS_METRICS | EMBEDDING_RAGAS_METRICS


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
    max_concurrency: int | None = None,
) -> Any:
    """Build a langchain-compatible LLM for RAGAS to use as judge.

    If judge_cfg is None, RAGAS falls back to its environment-default behavior.
    For OpenAI-compatible endpoints, `base_url_override` and `api_key_override`
    let `eval-ragas` target a self-hosted vLLM without mutating `.env`.
    """
    if judge_cfg is None:
        return None
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    from langchain_openai import ChatOpenAI
    from ragas.llms.base import LangchainLLMWrapper

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
    return _concurrency_limited_langchain_wrapper(
        LangchainLLMWrapper,
        chat,
        max_concurrency=max_concurrency,
    )


def _concurrency_limited_langchain_wrapper(
    wrapper_cls: Any,
    langchain_llm: Any,
    *,
    max_concurrency: int | None,
) -> Any:
    """Build a RAGAS wrapper with one shared limit around actual LLM requests.

    RAGAS limits whole metric coroutines, not individual generations. A metric
    can parse one response and prepare another while occupying a worker slot.
    This request-level semaphore lets multiple RAGAS evaluations keep work
    queued without exceeding the configured judge concurrency.
    """
    if max_concurrency is None:
        return wrapper_cls(langchain_llm)
    request_limit = int(max_concurrency)

    class _ConcurrencyLimitedWrapper(wrapper_cls):
        def __init__(self, llm: Any) -> None:
            super().__init__(llm)
            self._request_semaphore = asyncio.Semaphore(request_limit)
            self._request_active = 0
            self._request_peak = 0
            self._request_total = 0
            self._request_limit = request_limit

        async def agenerate_text(self, *args: Any, **kwargs: Any) -> Any:
            self._request_total += 1
            async with self._request_semaphore:
                self._request_active += 1
                self._request_peak = max(self._request_peak, self._request_active)
                try:
                    # LangchainLLMWrapper temporarily mutates temperature and n.
                    # Give each request its own shallow wrapper/model copy so
                    # concurrent metrics cannot race on those values.
                    request_wrapper = copy.copy(self)
                    if hasattr(self.langchain_llm, "model_copy"):
                        request_wrapper.langchain_llm = self.langchain_llm.model_copy()
                    else:
                        request_wrapper.langchain_llm = copy.copy(self.langchain_llm)
                    return await wrapper_cls.agenerate_text(
                        request_wrapper,
                        *args,
                        **kwargs,
                    )
                finally:
                    self._request_active -= 1

    return _ConcurrencyLimitedWrapper(langchain_llm)


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


def resolve_ragas_max_workers(max_workers: int | None) -> int:
    """Resolve RAGAS concurrency, using the repo-wide LLM limit by default."""
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


def _append_ragas_calls(
    calls: list[tuple[str, list[str], list[dict[str, Any]]]],
    *,
    label: str,
    metric_names: list[str],
    rows: list[dict[str, Any]],
) -> None:
    remote_metrics = [name for name in metric_names if name in REMOTE_RAGAS_METRICS]
    local_metrics = [name for name in metric_names if name not in REMOTE_RAGAS_METRICS]
    if remote_metrics and local_metrics:
        calls.append((f"{label}_judge", remote_metrics, rows))
        calls.append((f"{label}_local", local_metrics, rows))
    elif remote_metrics:
        calls.append((label, remote_metrics, rows))
    elif local_metrics:
        calls.append((label, local_metrics, rows))


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


async def _evaluate_ragas_call(
    *,
    label: str,
    metric_names: list[str],
    rows: list[dict[str, Any]],
    dataset_cls: Any,
    aevaluate_fn: Any,
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
    }
    if run_config is not None:
        evaluate_kwargs["run_config"] = run_config
    result = await aevaluate_fn(ds, **evaluate_kwargs)
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


async def _evaluate_ragas_calls(
    *,
    calls: list[tuple[str, list[str], list[dict[str, Any]]]],
    dataset_cls: Any,
    aevaluate_fn: Any,
    judge_llm: Any,
    embeddings: Any,
    run_config: Any,
) -> list[dict[str, Any]]:
    judge_calls = [call for call in calls if any(name in REMOTE_RAGAS_METRICS for name in call[1])]
    local_calls = [
        call for call in calls if not any(name in REMOTE_RAGAS_METRICS for name in call[1])
    ]

    async def _run_group(
        group: list[tuple[str, list[str], list[dict[str, Any]]]],
    ) -> list[list[dict[str, Any]]]:
        return await asyncio.gather(
            *(
                _evaluate_ragas_call(
                    label=label,
                    metric_names=metric_names,
                    rows=rows,
                    dataset_cls=dataset_cls,
                    aevaluate_fn=aevaluate_fn,
                    judge_llm=judge_llm,
                    embeddings=embeddings,
                    run_config=run_config,
                )
                for label, metric_names, rows in group
            )
        )

    results: list[list[dict[str, Any]]] = []
    if local_calls:
        logger.info(f"Running {len(local_calls)} local RAGAS call(s) before judge work")
        results.extend(await _run_group(local_calls))
    if judge_calls:
        logger.info(
            f"Running {len(judge_calls)} judge-backed RAGAS call(s) concurrently "
            "with a shared request limit"
        )
        results.extend(await _run_group(judge_calls))
    if hasattr(judge_llm, "_request_peak"):
        logger.info(
            "RAGAS judge requests: "
            f"total={judge_llm._request_total}, "
            f"peak_concurrency={judge_llm._request_peak}/"
            f"{judge_llm._request_limit}"
        )
    return [row for result in results for row in result]


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
    effective_max_workers = resolve_ragas_max_workers(max_workers)
    if timeout is not None and timeout < 1:
        raise ValueError("timeout must be >= 1")
    if input_dump_path is None and dump_path is not None:
        input_dump_path = dump_path

    restore_debug, previous_debug = _set_langchain_debug(debug)
    try:
        from datasets import Dataset
        from ragas import aevaluate

        run_config = _build_run_config(effective_max_workers, timeout)
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
        ragas_calls: list[tuple[str, list[str], list[dict[str, Any]]]] = []

        context_free = [
            name for name in requested_names if name not in CONTEXT_DEPENDENT_RAGAS_METRICS
        ]
        context_dependent = [
            name for name in requested_names if name in CONTEXT_DEPENDENT_RAGAS_METRICS
        ]

        if normalized_ks is None:
            rows = [_prediction_to_ragas_row(pred) for pred in predictions]
            _append_ragas_calls(
                ragas_calls,
                label="all_contexts",
                metric_names=requested_names,
                rows=rows,
            )
        elif context_dependent and len(normalized_ks) == 1:
            k = normalized_ks[0]
            rows = []
            for pred in predictions:
                row = _prediction_to_ragas_row(pred, context_k=k)
                row["__metric_suffix"] = f"@{k}"
                row["__context_k"] = k
                rows.append(row)
            _append_ragas_calls(
                ragas_calls,
                label=f"all_metrics_at_{k}",
                metric_names=requested_names,
                rows=rows,
            )
        else:
            if context_free:
                rows = [_prediction_to_ragas_row(pred, context_k=0) for pred in predictions]
                for row in rows:
                    row["__context_k"] = None
                _append_ragas_calls(
                    ragas_calls,
                    label="context_free",
                    metric_names=context_free,
                    rows=rows,
                )
            if context_dependent:
                rows = []
                for k in normalized_ks:
                    for pred in predictions:
                        row = _prediction_to_ragas_row(pred, context_k=k)
                        row["__metric_suffix"] = f"@{k}"
                        row["__context_k"] = k
                        row["__dump_label"] = f"context_at_{k}"
                        rows.append(row)
                _append_ragas_calls(
                    ragas_calls,
                    label="context_dependent_by_k",
                    metric_names=context_dependent,
                    rows=rows,
                )

        if not ragas_calls:
            return {}, pd.DataFrame()

        if input_dump_path is not None:
            dump_records = _build_ragas_input_dump_records(ragas_calls)
            write_jsonl(dump_records, input_dump_path)
            logger.info(f"Wrote RAGAS input dump to {input_dump_path}")

        judge_llm = (
            _build_ragas_judge_llm(
                cfg.judge_model,
                base_url_override=judge_base_url,
                api_key_override=judge_api_key,
                max_concurrency=effective_max_workers,
            )
            if any(name in LLM_RAGAS_METRICS for name in requested_names)
            else None
        )
        logger.info(
            f"RAGAS concurrency={effective_max_workers} "
            "(metric workers and global judge-request limit); batching=disabled"
        )
        embeddings = (
            _build_ragas_embeddings()
            if any(name in EMBEDDING_RAGAS_METRICS for name in requested_names)
            else None
        )

        long_rows = asyncio.run(
            _evaluate_ragas_calls(
                calls=ragas_calls,
                dataset_cls=Dataset,
                aevaluate_fn=aevaluate,
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

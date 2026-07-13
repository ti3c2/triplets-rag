from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pandas as pd

from triplet_rag.config import MetricsConfig
from triplet_rag.evaluate import judge
from triplet_rag.utils.io import read_jsonl


class _FakeMetric:
    def __init__(self, name: str):
        self.name = name


class _FakeDataset(list):
    @classmethod
    def from_list(cls, rows):
        return cls(rows)


class _FakeRunConfig:
    def __init__(self, timeout=180, max_workers=16, **kwargs):
        self.timeout = timeout
        self.max_workers = max_workers
        self.kwargs = kwargs


def _install_fake_ragas(monkeypatch, calls, *, delay: float = 0.0, active=None):
    datasets_mod = types.ModuleType("datasets")
    datasets_mod.Dataset = _FakeDataset

    metrics_mod = types.ModuleType("ragas.metrics")
    metrics_mod.faithfulness = _FakeMetric("faithfulness")
    metrics_mod.answer_relevancy = _FakeMetric("answer_relevancy")
    metrics_mod.answer_correctness = _FakeMetric("answer_correctness")
    metrics_mod.context_precision = _FakeMetric("context_precision")
    metrics_mod.context_recall = _FakeMetric("context_recall")

    faithfulness_mod = types.ModuleType("ragas.metrics._faithfulness")

    class Faithfulness(_FakeMetric):
        def __init__(self):
            super().__init__("faithfulness")

    faithfulness_mod.Faithfulness = Faithfulness

    answer_relevance_mod = types.ModuleType("ragas.metrics._answer_relevance")

    class AnswerRelevancy(_FakeMetric):
        def __init__(self):
            super().__init__("answer_relevancy")

    answer_relevance_mod.AnswerRelevancy = AnswerRelevancy

    answer_correctness_mod = types.ModuleType("ragas.metrics._answer_correctness")

    class AnswerCorrectness(_FakeMetric):
        def __init__(self):
            super().__init__("answer_correctness")

    answer_correctness_mod.AnswerCorrectness = AnswerCorrectness

    context_precision_mod = types.ModuleType("ragas.metrics._context_precision")

    class ContextPrecision(_FakeMetric):
        def __init__(self):
            super().__init__("context_precision")

    context_precision_mod.ContextPrecision = ContextPrecision

    context_recall_mod = types.ModuleType("ragas.metrics._context_recall")

    class ContextRecall(_FakeMetric):
        def __init__(self):
            super().__init__("context_recall")

    context_recall_mod.ContextRecall = ContextRecall

    class FactualCorrectness(_FakeMetric):
        def __init__(self):
            super().__init__("factual_correctness")

    class RougeScore(_FakeMetric):
        def __init__(self):
            super().__init__("rouge_score")

    class BleuScore(_FakeMetric):
        def __init__(self):
            super().__init__("bleu_score")

    class NonLLMStringSimilarity(_FakeMetric):
        def __init__(self):
            super().__init__("non_llm_string_similarity")

    class StringPresence(_FakeMetric):
        def __init__(self):
            super().__init__("string_present")

    class ExactMatch(_FakeMetric):
        def __init__(self):
            super().__init__("exact_match")

    metrics_mod.FactualCorrectness = FactualCorrectness
    metrics_mod.RougeScore = RougeScore
    metrics_mod.BleuScore = BleuScore
    metrics_mod.NonLLMStringSimilarity = NonLLMStringSimilarity
    metrics_mod.StringPresence = StringPresence
    metrics_mod.ExactMatch = ExactMatch

    nv_mod = types.ModuleType("ragas.metrics._nv_metrics")

    class AnswerAccuracy(_FakeMetric):
        def __init__(self):
            super().__init__("nv_accuracy")

    class ContextRelevance(_FakeMetric):
        def __init__(self):
            super().__init__("nv_context_relevance")

    class ResponseGroundedness(_FakeMetric):
        def __init__(self):
            super().__init__("nv_response_groundedness")

    nv_mod.AnswerAccuracy = AnswerAccuracy
    nv_mod.ContextRelevance = ContextRelevance
    nv_mod.ResponseGroundedness = ResponseGroundedness

    run_config_mod = types.ModuleType("ragas.run_config")
    run_config_mod.RunConfig = _FakeRunConfig

    ragas_mod = types.ModuleType("ragas")

    async def aevaluate(dataset, metrics, llm, embeddings, raise_exceptions, run_config=None):
        if active is not None:
            active["current"] += 1
            active["max"] = max(active["max"], active["current"])
        try:
            calls.append(
                {
                    "rows": list(dataset),
                    "metrics": [m.name for m in metrics],
                    "run_config": run_config,
                }
            )
            if delay:
                await asyncio.sleep(delay)
            data = {"query_id": [row["query_id"] for row in dataset]}
            for metric in metrics:
                data[metric.name] = [0.5 for _ in dataset]
            return SimpleNamespace(to_pandas=lambda: pd.DataFrame(data))
        finally:
            if active is not None:
                active["current"] -= 1

    ragas_mod.aevaluate = aevaluate

    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    monkeypatch.setitem(sys.modules, "ragas", ragas_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics._faithfulness", faithfulness_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics._answer_relevance", answer_relevance_mod)
    monkeypatch.setitem(
        sys.modules, "ragas.metrics._answer_correctness", answer_correctness_mod
    )
    monkeypatch.setitem(sys.modules, "ragas.metrics._context_precision", context_precision_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics._context_recall", context_recall_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics._nv_metrics", nv_mod)
    monkeypatch.setitem(sys.modules, "ragas.run_config", run_config_mod)


def test_find_metric_result_column_handles_parameterized_ragas_columns():
    columns = [
        "user_input",
        "response",
        "reference",
        "rouge_score(mode=fmeasure)",
        "bleu_score",
    ]

    assert (
        judge._find_metric_result_column(columns, "rouge_score", "rouge_score")
        == "rouge_score(mode=fmeasure)"
    )
    assert judge._find_metric_result_column(columns, "bleu_score", "bleu_score") == "bleu_score"
    assert judge._find_metric_result_column(columns, "missing", "missing") is None


def test_compute_ragas_metrics_runs_context_metrics_per_k(tmp_path, monkeypatch):
    calls = []
    _install_fake_ragas(monkeypatch, calls)
    monkeypatch.setattr(judge, "_build_ragas_judge_llm", lambda *args, **kwargs: None)
    monkeypatch.setattr(judge, "_build_ragas_embeddings", lambda: None)

    predictions = [
        {
            "query_id": "q1",
            "query": "what is X?",
            "prediction": "X",
            "gold_answers": ["gold X"],
            "retrieved_texts": ["c1", "c2", "c3"],
        },
        {
            "query_id": "q2",
            "query": "what is Y?",
            "prediction": "Y",
            "gold_answers": ["gold Y"],
            "retrieved_texts": ["d1", "d2"],
        },
    ]
    cfg = MetricsConfig(
        use_ragas=True,
        ragas_metrics=["faithfulness", "answer_correctness"],
        judge_model=None,
    )
    dump_path = tmp_path / "inputs.jsonl"

    agg, per_query = judge.compute_ragas_metrics(
        predictions,
        cfg,
        context_ks=[2, 1],
        max_workers=3,
        timeout=45,
        input_dump_path=dump_path,
    )

    assert set(agg) == {"answer_correctness", "faithfulness@1", "faithfulness@2"}
    assert set(per_query["metric"].unique()) == set(agg)
    assert [call["metrics"] for call in calls] == [["answer_correctness"], ["faithfulness"]]
    assert len(calls[0]["rows"]) == 2
    assert calls[0]["rows"][0]["retrieved_contexts"] == []
    assert len(calls[1]["rows"]) == 4
    assert calls[1]["rows"][0]["retrieved_contexts"] == ["c1"]
    assert calls[1]["rows"][2]["retrieved_contexts"] == ["c1", "c2"]
    assert [call["run_config"].max_workers for call in calls] == [3, 3]
    assert all(call["run_config"].timeout == 45 for call in calls)

    dumped = list(read_jsonl(dump_path))
    assert len(dumped) == 6
    assert dumped[0]["ragas_run"] == "context_free"
    assert dumped[0]["ragas_metrics"] == ["answer_correctness"]
    assert dumped[0]["retrieved_contexts"] == []
    assert dumped[2]["ragas_run"] == "context_at_1"
    assert dumped[2]["ragas_metrics"] == ["faithfulness"]
    assert dumped[2]["retrieved_contexts"] == ["c1"]


def test_compute_ragas_metrics_uses_one_evaluate_for_single_k(monkeypatch):
    calls = []
    active = {"current": 0, "max": 0}
    _install_fake_ragas(monkeypatch, calls, delay=0.01, active=active)
    monkeypatch.setattr(judge, "_build_ragas_judge_llm", lambda *args, **kwargs: None)
    monkeypatch.setattr(judge, "_build_ragas_embeddings", lambda: None)

    predictions = [
        {
            "query_id": "q1",
            "query": "what is X?",
            "prediction": "X",
            "gold_answers": ["gold X"],
            "retrieved_texts": ["c1"],
        }
    ]
    cfg = MetricsConfig(
        use_ragas=True,
        ragas_metrics=["faithfulness", "answer_correctness"],
        judge_model=None,
    )

    agg, _ = judge.compute_ragas_metrics(predictions, cfg, context_ks=[1])

    assert set(agg) == {"answer_correctness", "faithfulness@1"}
    assert [call["metrics"] for call in calls] == [["faithfulness", "answer_correctness"]]
    assert calls[0]["rows"][0]["retrieved_contexts"] == ["c1"]
    assert active["max"] == 1


def test_compute_ragas_metrics_runs_partitioned_scopes_sequentially(monkeypatch):
    calls = []
    active = {"current": 0, "max": 0}
    _install_fake_ragas(monkeypatch, calls, delay=0.01, active=active)
    monkeypatch.setattr(judge, "_build_ragas_judge_llm", lambda *args, **kwargs: None)
    monkeypatch.setattr(judge, "_build_ragas_embeddings", lambda: None)

    predictions = [
        {
            "query_id": "q1",
            "query": "what is X?",
            "prediction": "X",
            "gold_answers": ["gold X"],
            "retrieved_texts": ["c1"],
        }
    ]
    cfg = MetricsConfig(
        use_ragas=True,
        ragas_metrics=["faithfulness", "answer_correctness"],
        judge_model=None,
    )

    agg, _ = judge.compute_ragas_metrics(predictions, cfg, context_ks=[1, 2], max_workers=1)

    assert set(agg) == {"answer_correctness", "faithfulness@1", "faithfulness@2"}
    assert [call["metrics"] for call in calls] == [["answer_correctness"], ["faithfulness"]]
    assert [call["run_config"].max_workers for call in calls] == [1, 1]
    assert active["max"] == 1


def test_compute_ragas_metrics_supports_reference_metric_names(monkeypatch):
    calls = []
    _install_fake_ragas(monkeypatch, calls)
    monkeypatch.setattr(judge, "_build_ragas_judge_llm", lambda *args, **kwargs: None)
    monkeypatch.setattr(judge, "_build_ragas_embeddings", lambda: None)

    predictions = [
        {
            "query_id": "q1",
            "query": "what is X?",
            "prediction": "X",
            "gold_answers": ["X"],
            "retrieved_texts": ["c1", "c2"],
        }
    ]
    metric_names = [
        "factual_correctness",
        "rouge_score",
        "bleu_score",
        "non_llm_string_similarity",
        "string_present",
        "exact_match",
    ]
    cfg = MetricsConfig(use_ragas=True, ragas_metrics=metric_names, judge_model=None)

    agg, per_query = judge.compute_ragas_metrics(predictions, cfg, context_ks=[1, 2])

    assert set(agg) == set(metric_names)
    assert set(per_query["metric"].unique()) == set(metric_names)
    assert [call["metrics"] for call in calls] == [
        ["factual_correctness"],
        [
            "rouge_score",
            "bleu_score",
            "non_llm_string_similarity",
            "string_present",
            "exact_match",
        ],
    ]

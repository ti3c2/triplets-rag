from __future__ import annotations

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


def _install_fake_ragas(monkeypatch, calls):
    datasets_mod = types.ModuleType("datasets")
    datasets_mod.Dataset = _FakeDataset

    metrics_mod = types.ModuleType("ragas.metrics")
    metrics_mod.faithfulness = _FakeMetric("faithfulness")
    metrics_mod.answer_relevancy = _FakeMetric("answer_relevancy")
    metrics_mod.answer_correctness = _FakeMetric("answer_correctness")
    metrics_mod.context_precision = _FakeMetric("context_precision")
    metrics_mod.context_recall = _FakeMetric("context_recall")

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

    def evaluate(dataset, metrics, llm, embeddings, raise_exceptions, run_config):
        calls.append(
            {
                "rows": list(dataset),
                "metrics": [m.name for m in metrics],
                "run_config": run_config,
            }
        )
        data = {"query_id": [row["query_id"] for row in dataset]}
        for metric in metrics:
            data[metric.name] = [0.5 for _ in dataset]
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame(data))

    ragas_mod.evaluate = evaluate

    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    monkeypatch.setitem(sys.modules, "ragas", ragas_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics._nv_metrics", nv_mod)
    monkeypatch.setitem(sys.modules, "ragas.run_config", run_config_mod)


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
    assert [call["metrics"] for call in calls] == [
        ["answer_correctness"],
        ["faithfulness"],
        ["faithfulness"],
    ]
    assert calls[0]["rows"][0]["retrieved_contexts"] == ["c1", "c2", "c3"]
    assert calls[1]["rows"][0]["retrieved_contexts"] == ["c1"]
    assert calls[2]["rows"][0]["retrieved_contexts"] == ["c1", "c2"]
    assert all(call["run_config"].max_workers == 3 for call in calls)
    assert all(call["run_config"].timeout == 45 for call in calls)

    dumped = list(read_jsonl(dump_path))
    assert len(dumped) == 6
    assert dumped[0]["ragas_run"] == "context_free"
    assert dumped[0]["ragas_metrics"] == ["answer_correctness"]
    assert dumped[2]["ragas_run"] == "context_at_1"
    assert dumped[2]["retrieved_contexts"] == ["c1"]

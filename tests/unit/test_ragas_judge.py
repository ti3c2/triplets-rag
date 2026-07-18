from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import ragas

from triplet_rag.config import LLMConfig, MetricsConfig
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
    def evaluate(
        dataset,
        metrics,
        llm,
        embeddings,
        raise_exceptions,
        run_config=None,
        batch_size=None,
    ):
        calls.append(
            {
                "rows": list(dataset),
                "metrics": [m.name for m in metrics],
                "run_config": run_config,
                "batch_size": batch_size,
            }
        )
        data = {"query_id": [row["query_id"] for row in dataset]}
        for metric in metrics:
            data[metric.name] = [0.5 for _ in dataset]
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame(data))

    monkeypatch.setattr(ragas, "evaluate", evaluate)
    monkeypatch.setattr(ragas, "EvaluationDataset", _FakeDataset)
    monkeypatch.setattr(ragas, "RunConfig", _FakeRunConfig)
    monkeypatch.setattr(judge, "_make_ragas_metric", lambda name: _FakeMetric(name))


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


def test_pinned_ragas_exports_cover_all_supported_metrics():
    assert ragas.__version__ == "0.3.2"
    for name in sorted(judge.SUPPORTED_RAGAS_METRICS):
        assert judge._make_ragas_metric(name).name == name


def test_pinned_ragas_evaluate_path_runs_without_network():
    predictions = [
        {
            "query_id": "q1",
            "query": "What is alpha?",
            "prediction": "alpha",
            "gold_answers": ["alpha"],
            "retrieved_texts": ["alpha context"],
        }
    ]
    cfg = MetricsConfig(
        use_ragas=True,
        ragas_metrics=["exact_match", "rouge_score"],
        judge_model=None,
    )

    agg, per_query = judge.compute_ragas_metrics(predictions, cfg, max_workers=3)

    assert agg == {"exact_match": 1.0, "rouge_score": 1.0}
    assert per_query.to_dict(orient="records") == [
        {"query_id": "q1", "metric": "exact_match", "value": 1.0},
        {"query_id": "q1", "metric": "rouge_score", "value": 1.0},
    ]


def test_build_ragas_judge_honors_generation_limits(monkeypatch):
    monkeypatch.setattr(
        judge,
        "get_settings",
        lambda: SimpleNamespace(vllm_api_key="EMPTY", vllm_base_url="http://localhost:8000/v1"),
    )
    cfg = LLMConfig(
        kind="vllm",
        model_name="local-judge",
        max_tokens=77,
        top_p=0.8,
    )

    wrapped = judge._build_ragas_judge_llm(cfg)

    assert wrapped.langchain_llm._default_params["max_completion_tokens"] == 77
    assert wrapped.langchain_llm._default_params["top_p"] == 0.8


def test_resolve_ragas_max_workers_uses_llm_setting(monkeypatch):
    monkeypatch.setattr(
        judge,
        "get_settings",
        lambda: SimpleNamespace(llm_concurrency=37),
    )

    assert judge.resolve_ragas_max_workers(None) == 37
    assert judge.resolve_ragas_max_workers(11) == 11


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
    assert all(call["batch_size"] is None for call in calls)

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
    _install_fake_ragas(monkeypatch, calls)
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


def test_compute_ragas_metrics_uses_two_evaluations_for_multiple_ks(monkeypatch):
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
    assert [call["metrics"] for call in calls] == [metric_names]

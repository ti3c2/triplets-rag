"""Layout-only test for the RAGAS rerunner.

`compute_ragas_metrics` is stubbed out — the test verifies that
`run_ragas_on_experiment` reads predictions, writes the per-judge subfolder
exactly where we expect, and that two different judge tags coexist without
overwriting each other.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from triplet_rag.config import LLMConfig
from triplet_rag.evaluate import ragas_runner
from triplet_rag.utils.io import read_json, write_jsonl


def _make_predictions(exp_dir: Path) -> None:
    write_jsonl(
        [
            {
                "query_id": "q1",
                "query": "what is X?",
                "strategy": "vanilla_rag",
                "retrieved_ids": ["d1", "d2"],
                "retrieved_texts": ["X is a thing.", "X has properties."],
                "prompt": "stub",
                "prediction": "X is a thing.",
                "gold_answers": ["X is a thing."],
                "gold_doc_ids": ["d1"],
                "latency_ms": 12.0,
            },
            {
                "query_id": "q2",
                "query": "what about Y?",
                "strategy": "vanilla_rag",
                "retrieved_ids": ["d3"],
                "retrieved_texts": ["Y is unrelated."],
                "prompt": "stub",
                "prediction": "Y is unrelated.",
                "gold_answers": ["Y is unrelated."],
                "gold_doc_ids": ["d3"],
                "latency_ms": 9.0,
            },
        ],
        exp_dir / "predictions.jsonl",
    )


def _stub_compute(monkeypatch, value: float) -> None:
    def _fake(
        predictions,
        cfg,
        *,
        judge_base_url=None,
        judge_api_key=None,
        context_ks=None,
        max_workers=None,
        timeout=None,
        debug=False,
        dump_path=None,
    ):
        rows = []
        agg = {}
        for m in cfg.ragas_metrics:
            agg[m] = value
            for p in predictions:
                rows.append({"query_id": p["query_id"], "metric": m, "value": value})
        return agg, pd.DataFrame(rows)

    monkeypatch.setattr(ragas_runner, "compute_ragas_metrics", _fake)


def test_layout_one_judge(tmp_path, monkeypatch):
    exp = tmp_path / "exp1"
    exp.mkdir()
    _make_predictions(exp)
    _stub_compute(monkeypatch, 0.85)

    judge = LLMConfig(kind="openai", model_name="gpt-4o")
    agg, out_dir = ragas_runner.run_ragas_on_experiment(
        exp_dir=exp,
        judge_cfg=judge,
        metric_names=["faithfulness", "answer_relevancy"],
    )

    assert out_dir == exp / "metrics" / "ragas" / "gpt-4o"
    assert (out_dir / "aggregate.json").exists()
    assert (out_dir / "per_query.parquet").exists()
    assert (out_dir / "judge.json").exists()
    runs = read_json(exp / "metrics" / "ragas" / "runs.json")
    assert "gpt-4o" in runs
    assert runs["gpt-4o"]["judge_model"]["model_name"] == "gpt-4o"
    aggregate = read_json(out_dir / "aggregate.json")
    assert "faithfulness" in aggregate and "answer_relevancy" in aggregate
    assert agg == {"faithfulness": 0.85, "answer_relevancy": 0.85}


def test_default_concurrency_is_resolved_and_persisted(tmp_path, monkeypatch):
    exp = tmp_path / "exp_concurrency"
    exp.mkdir()
    _make_predictions(exp)
    captured = {}

    def _fake(predictions, cfg, **kwargs):
        captured["max_workers"] = kwargs["max_workers"]
        rows = [
            {"query_id": p["query_id"], "metric": "faithfulness", "value": 0.5} for p in predictions
        ]
        return {"faithfulness": 0.5}, pd.DataFrame(rows)

    monkeypatch.setattr(ragas_runner, "compute_ragas_metrics", _fake)
    monkeypatch.setattr(ragas_runner, "resolve_ragas_max_workers", lambda value: 37)

    _, out_dir = ragas_runner.run_ragas_on_experiment(
        exp_dir=exp,
        judge_cfg=LLMConfig(kind="openai", model_name="gpt-4o"),
        metric_names=["faithfulness"],
    )

    metadata = read_json(out_dir / "judge.json")
    assert captured["max_workers"] == 37
    assert metadata["max_workers"] == 37
    assert metadata["max_workers_source"] == "TRIPLET_RAG_LLM_CONCURRENCY"


def test_two_judges_coexist(tmp_path, monkeypatch):
    exp = tmp_path / "exp2"
    exp.mkdir()
    _make_predictions(exp)

    _stub_compute(monkeypatch, 0.7)
    ragas_runner.run_ragas_on_experiment(
        exp_dir=exp,
        judge_cfg=LLMConfig(kind="openai", model_name="gpt-4o"),
        metric_names=["faithfulness"],
    )
    _stub_compute(monkeypatch, 0.4)
    ragas_runner.run_ragas_on_experiment(
        exp_dir=exp,
        judge_cfg=LLMConfig(kind="anthropic", model_name="claude-3-5-haiku-20241022"),
        metric_names=["faithfulness"],
    )

    a = read_json(exp / "metrics" / "ragas" / "gpt-4o" / "aggregate.json")
    b = read_json(exp / "metrics" / "ragas" / "claude-3-5-haiku-20241022" / "aggregate.json")
    assert a["faithfulness"]["mean"] == pytest.approx(0.7)
    assert b["faithfulness"]["mean"] == pytest.approx(0.4)
    runs = read_json(exp / "metrics" / "ragas" / "runs.json")
    assert set(runs.keys()) == {"gpt-4o", "claude-3-5-haiku-20241022"}


def test_refuses_overwrite_without_force(tmp_path, monkeypatch):
    exp = tmp_path / "exp3"
    exp.mkdir()
    _make_predictions(exp)
    _stub_compute(monkeypatch, 0.5)

    judge = LLMConfig(kind="openai", model_name="gpt-4o")
    ragas_runner.run_ragas_on_experiment(
        exp_dir=exp, judge_cfg=judge, metric_names=["faithfulness"]
    )
    with pytest.raises(FileExistsError):
        ragas_runner.run_ragas_on_experiment(
            exp_dir=exp, judge_cfg=judge, metric_names=["faithfulness"]
        )
    # force=True overwrites
    _stub_compute(monkeypatch, 0.99)
    ragas_runner.run_ragas_on_experiment(
        exp_dir=exp, judge_cfg=judge, metric_names=["faithfulness"], force=True
    )
    a = read_json(exp / "metrics" / "ragas" / "gpt-4o" / "aggregate.json")
    assert a["faithfulness"]["mean"] == pytest.approx(0.99)


def test_custom_judge_tag(tmp_path, monkeypatch):
    exp = tmp_path / "exp4"
    exp.mkdir()
    _make_predictions(exp)
    _stub_compute(monkeypatch, 0.5)

    ragas_runner.run_ragas_on_experiment(
        exp_dir=exp,
        judge_cfg=LLMConfig(kind="openai", model_name="gpt-4o"),
        metric_names=["faithfulness"],
        judge_tag="rerun_2026_05",
    )
    assert (exp / "metrics" / "ragas" / "rerun_2026_05" / "aggregate.json").exists()


def test_sanitize_judge_tag():
    assert ragas_runner.sanitize_judge_tag("Qwen/Qwen2.5-7B-Instruct") == "Qwen_Qwen2.5-7B-Instruct"
    assert ragas_runner.sanitize_judge_tag("gpt-4o") == "gpt-4o"
    assert ragas_runner.sanitize_judge_tag("a/b c@d") == "a_b_c_d"


def test_base_url_threaded_through(tmp_path, monkeypatch):
    """The CLI's --base-url must reach compute_ragas_metrics so a self-hosted
    vLLM endpoint can be addressed without env vars."""
    exp = tmp_path / "exp_url"
    exp.mkdir()
    _make_predictions(exp)

    captured = {}

    def _fake(
        predictions,
        cfg,
        *,
        judge_base_url=None,
        judge_api_key=None,
        context_ks=None,
        max_workers=None,
        timeout=None,
        debug=False,
        dump_path=None,
    ):
        captured["base_url"] = judge_base_url
        captured["api_key"] = judge_api_key
        captured["judge_kind"] = cfg.judge_model.kind if cfg.judge_model else None
        captured["judge_model"] = cfg.judge_model.model_name if cfg.judge_model else None
        rows = [
            {"query_id": p["query_id"], "metric": "faithfulness", "value": 0.5} for p in predictions
        ]
        return {"faithfulness": 0.5}, pd.DataFrame(rows)

    monkeypatch.setattr(ragas_runner, "compute_ragas_metrics", _fake)

    ragas_runner.run_ragas_on_experiment(
        exp_dir=exp,
        judge_cfg=LLMConfig(kind="vllm", model_name="Qwen/Qwen2.5-32B-Instruct"),
        metric_names=["faithfulness"],
        judge_base_url="http://localhost:7114/v1",
        judge_api_key="EMPTY",
    )

    assert captured["base_url"] == "http://localhost:7114/v1"
    assert captured["api_key"] == "EMPTY"
    assert captured["judge_kind"] == "vllm"
    assert captured["judge_model"] == "Qwen/Qwen2.5-32B-Instruct"

    # judge.json + runs.json record the endpoint
    judge_json = read_json(exp / "metrics" / "ragas" / "Qwen_Qwen2.5-32B-Instruct" / "judge.json")
    assert judge_json["judge_base_url"] == "http://localhost:7114/v1"
    runs = read_json(exp / "metrics" / "ragas" / "runs.json")
    assert runs["Qwen_Qwen2.5-32B-Instruct"]["judge_base_url"] == "http://localhost:7114/v1"

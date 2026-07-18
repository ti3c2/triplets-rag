from __future__ import annotations

import pandas as pd

from triplet_rag.config import DatasetConfig
from triplet_rag.data.loaders import get_loader
from triplet_rag.utils.io import read_json, read_parquet


def test_csv_qa_loader_dedupes_contexts_and_writes_normalized_tables(tmp_path):
    source = tmp_path / "squad_selected.csv"
    pd.DataFrame(
        [
            {
                "id": "q1",
                "title": "Doc",
                "text": "The same context.",
                "query": "What context?",
                "answer": "same context",
            },
            {
                "id": "q2",
                "title": "Doc",
                "text": "The same context.",
                "query": "Which context?",
                "answer": "The same context",
            },
            {
                "id": "q3",
                "title": "Other",
                "text": "A second context.",
                "query": "What is second?",
                "answer": "second context",
            },
        ]
    ).to_csv(source, index=False)

    cfg = DatasetConfig(name="squad_selected", source_path=str(source))
    loader = get_loader(cfg, tmp_path / "raw")
    loader.load()

    corpus = read_parquet(loader.corpus_path)
    queries = read_parquet(loader.queries_path)
    qrels = read_parquet(loader.qrels_path)
    marker = read_json(loader.success_path)

    assert len(corpus) == 2
    assert len(queries) == 3
    assert len(qrels) == 3
    assert queries.loc[0, "gold_answers"] == ["same context"]
    assert set(qrels["doc_id"]).issubset(set(corpus["doc_id"]))
    assert marker["n_docs"] == 2


def test_csv_qa_loader_honors_max_queries_with_nrows(tmp_path):
    source = tmp_path / "squad_selected.csv"
    pd.DataFrame(
        [
            {
                "id": f"q{i}",
                "title": "Doc",
                "text": f"Context {i}",
                "query": f"Q{i}",
                "answer": f"A{i}",
            }
            for i in range(5)
        ]
    ).to_csv(source, index=False)

    cfg = DatasetConfig(name="squad_selected", source_path=str(source), max_queries=2)
    loader = get_loader(cfg, tmp_path / "raw")
    loader.load()

    queries = read_parquet(loader.queries_path)
    assert queries["query_id"].tolist() == ["q0", "q1"]

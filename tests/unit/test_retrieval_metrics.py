"""Retrieval metric resolution: chunk_id vs doc_id, with `::` in doc_ids.

Regression: SQuAD's doc_ids have the form `<title>::<hash8>`. The previous
fallback split rid on `::` and took the first segment, which produced
unmatchable doc_ids and zeroed every retrieval metric. These tests pin the
correct resolution: chunk_ids map to their doc_id, doc_ids pass through
unchanged (even when they contain `::`).
"""

from __future__ import annotations

import pandas as pd

from triplet_rag.evaluate.retrieval_metrics import (
    build_run_from_predictions,
    compute_retrieval_metrics,
)


def _chunks_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": "Super_Bowl_50::6b0af9df::0-200::abcdef",
                "doc_id": "Super_Bowl_50::6b0af9df",
                "text": "...",
            },
            {
                "chunk_id": "Super_Bowl_50::f1b96eb2::0-200::123456",
                "doc_id": "Super_Bowl_50::f1b96eb2",
                "text": "...",
            },
            {"chunk_id": "d1::0-100::aaaaaa", "doc_id": "d1", "text": "..."},
        ]
    )


def test_doc_id_with_double_colon_passes_through():
    """retrieved_ids that are SQuAD doc_ids must hit qrels intact."""
    chunks = _chunks_df()
    chunk_to_doc = chunks.set_index("chunk_id")["doc_id"].to_dict()
    valid_doc_ids = set(chunks["doc_id"].astype(str).unique().tolist())

    preds = [
        {
            "query_id": "q1",
            "retrieved_ids": [
                "Super_Bowl_50::6b0af9df",
                "Super_Bowl_50::f1b96eb2",
            ],
        }
    ]
    run = list(
        build_run_from_predictions(preds, chunk_to_doc=chunk_to_doc, valid_doc_ids=valid_doc_ids)
    )
    doc_ids = [r.doc_id for r in run]
    assert doc_ids == ["Super_Bowl_50::6b0af9df", "Super_Bowl_50::f1b96eb2"]


def test_chunk_id_resolves_to_doc_id():
    """Triplet strategy unions chunk_ids; these must map back to doc_ids."""
    chunks = _chunks_df()
    chunk_to_doc = chunks.set_index("chunk_id")["doc_id"].to_dict()
    valid_doc_ids = set(chunks["doc_id"].astype(str).unique().tolist())

    preds = [
        {
            "query_id": "q1",
            "retrieved_ids": [
                "Super_Bowl_50::6b0af9df::0-200::abcdef",  # chunk_id
                "d1::0-100::aaaaaa",  # chunk_id with simple doc_id
            ],
        }
    ]
    run = list(
        build_run_from_predictions(preds, chunk_to_doc=chunk_to_doc, valid_doc_ids=valid_doc_ids)
    )
    doc_ids = [r.doc_id for r in run]
    assert doc_ids == ["Super_Bowl_50::6b0af9df", "d1"]


def test_dedupes_within_query():
    """Two chunks of the same doc collapse to one run row per query."""
    chunks = _chunks_df()
    chunk_to_doc = chunks.set_index("chunk_id")["doc_id"].to_dict()
    valid_doc_ids = set(chunks["doc_id"].astype(str).unique().tolist())
    # Add a second chunk for the same doc
    chunks2 = pd.concat(
        [
            chunks,
            pd.DataFrame(
                [
                    {
                        "chunk_id": "Super_Bowl_50::6b0af9df::200-400::ffffff",
                        "doc_id": "Super_Bowl_50::6b0af9df",
                        "text": "...",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    chunk_to_doc = chunks2.set_index("chunk_id")["doc_id"].to_dict()

    preds = [
        {
            "query_id": "q1",
            "retrieved_ids": [
                "Super_Bowl_50::6b0af9df::0-200::abcdef",
                "Super_Bowl_50::6b0af9df::200-400::ffffff",
                "Super_Bowl_50::f1b96eb2::0-200::123456",
            ],
        }
    ]
    run = list(
        build_run_from_predictions(preds, chunk_to_doc=chunk_to_doc, valid_doc_ids=valid_doc_ids)
    )
    doc_ids = [r.doc_id for r in run]
    assert doc_ids == ["Super_Bowl_50::6b0af9df", "Super_Bowl_50::f1b96eb2"]


def test_compute_retrieval_metrics_squad_like_nonzero():
    """End-to-end: SQuAD-shaped predictions + qrels → nonzero metrics."""
    chunks = _chunks_df()
    qrels = pd.DataFrame(
        [
            {"query_id": "q1", "doc_id": "Super_Bowl_50::6b0af9df", "relevance": 1},
            {"query_id": "q2", "doc_id": "Super_Bowl_50::f1b96eb2", "relevance": 1},
        ]
    )
    preds = [
        {
            "query_id": "q1",
            "retrieved_ids": [
                "Super_Bowl_50::6b0af9df",
                "Super_Bowl_50::f1b96eb2",
            ],
        },
        {
            "query_id": "q2",
            "retrieved_ids": [
                "Super_Bowl_50::f1b96eb2",
                "d1",
            ],
        },
    ]
    agg, pq = compute_retrieval_metrics(preds, qrels, ["R@5", "RR", "P@1"], chunks)
    assert agg.get("R@5", 0.0) == 1.0  # both queries hit at rank 1
    assert agg.get("RR", 0.0) == 1.0
    assert agg.get("P@1", 0.0) == 1.0
    assert not pq.empty


def test_unknown_id_passes_through_without_crash():
    chunks = _chunks_df()
    chunk_to_doc = chunks.set_index("chunk_id")["doc_id"].to_dict()
    valid_doc_ids = set(chunks["doc_id"].astype(str).unique().tolist())
    preds = [{"query_id": "q1", "retrieved_ids": ["nonexistent_id"]}]
    run = list(
        build_run_from_predictions(preds, chunk_to_doc=chunk_to_doc, valid_doc_ids=valid_doc_ids)
    )
    assert len(run) == 1
    assert run[0].doc_id == "nonexistent_id"

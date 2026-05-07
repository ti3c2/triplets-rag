"""Retrieval metrics via ir_measures.

We compute both per-query and aggregate metrics. Two scoring modes for triplet
retrieval are supported:
- strict: the chunks attached to the *retrieved* triplet, in the order they
  appear, are the ranked list.
- lenient: the union of all chunks across retrieved triplets is the candidate
  set, ranked by triplet score.

Both rank-on-chunk-id; qrels are at doc_id by default. We map chunks to doc_ids
using the chunks table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import ir_measures
import pandas as pd
from loguru import logger


def _parse_measures(metric_strs: list[str]) -> list:
    return [ir_measures.parse_measure(m) for m in metric_strs]


def build_qrels_for_ir_measures(qrels_df: pd.DataFrame) -> Iterable:
    for _, row in qrels_df.iterrows():
        yield ir_measures.Qrel(
            query_id=str(row["query_id"]),
            doc_id=str(row["doc_id"]),
            relevance=int(row.get("relevance", 1)),
        )


def build_run_from_predictions(
    predictions: list[dict],
    chunk_to_doc: dict[str, str] | None = None,
) -> Iterable:
    """Map each prediction's retrieved_ids to TREC-run rows.

    If retrieved_ids look like chunk_ids (contain '::'), we map them to doc_id
    using chunk_to_doc.
    """
    for pred in predictions:
        seen = set()
        for rank, rid in enumerate(pred["retrieved_ids"]):
            doc_id = rid
            if chunk_to_doc is not None and rid in chunk_to_doc:
                doc_id = chunk_to_doc[rid]
            elif "::" in rid and chunk_to_doc is not None:
                # chunk_id format
                base = rid.split("::")[0]
                doc_id = base
            if doc_id in seen:
                continue
            seen.add(doc_id)
            score = float(len(pred["retrieved_ids"]) - rank)
            yield ir_measures.ScoredDoc(
                query_id=str(pred["query_id"]), doc_id=str(doc_id), score=score
            )


def compute_retrieval_metrics(
    predictions: list[dict],
    qrels_df: pd.DataFrame,
    metric_strs: list[str],
    chunks_df: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Returns (aggregate_dict, per_query_df)."""
    measures = _parse_measures(metric_strs)
    chunk_to_doc = chunks_df.set_index("chunk_id")["doc_id"].to_dict()
    qrels = list(build_qrels_for_ir_measures(qrels_df))
    run = list(build_run_from_predictions(predictions, chunk_to_doc=chunk_to_doc))

    if not qrels or not run:
        logger.warning("No qrels or run rows for retrieval metrics")
        return {}, pd.DataFrame()

    agg = ir_measures.calc_aggregate(measures, qrels, run)
    agg_dict = {str(k): float(v) for k, v in agg.items()}

    rows = []
    for record in ir_measures.iter_calc(measures, qrels, run):
        rows.append(
            {
                "query_id": record.query_id,
                "metric": str(record.measure),
                "value": float(record.value),
            }
        )
    pq = pd.DataFrame(rows)
    return agg_dict, pq

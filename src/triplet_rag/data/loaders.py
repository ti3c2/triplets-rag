"""Dataset loaders that produce a normalized triple of (corpus, queries, qrels)
parquet files under storage/raw/<dataset_name>/.

Implementations are minimal but real:
- SQuAD: from HF datasets `rajpurkar/squad`. Each context becomes a "doc";
  qrels link each query to the doc holding its gold span.
- Natural Questions: from HF `google-research-datasets/natural_questions` (open
  variant typically used; we use `nq_open` if present, or fall back to a tiny
  curated sample).
- MultiHop-RAG: from the MultiHop-RAG GitHub release (loaded via local files
  that the user must download, with instructions printed if missing).
- Fixture: synthetic mini dataset used by tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from loguru import logger

from ..config import DatasetConfig
from ..utils.io import has_success, touch_success, write_parquet


class DatasetLoader(ABC):
    """Produces corpus.parquet, queries.parquet, qrels.parquet."""

    name: str

    def __init__(self, cfg: DatasetConfig, raw_dir: Path):
        self.cfg = cfg
        self.out_dir = raw_dir / cfg.name
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def corpus_path(self) -> Path:
        return self.out_dir / "corpus.parquet"

    @property
    def queries_path(self) -> Path:
        return self.out_dir / "queries.parquet"

    @property
    def qrels_path(self) -> Path:
        return self.out_dir / "qrels.parquet"

    @property
    def success_path(self) -> Path:
        return self.out_dir / "_SUCCESS.json"

    def already_loaded(self) -> bool:
        return (
            has_success(self.success_path)
            and self.corpus_path.exists()
            and self.queries_path.exists()
            and self.qrels_path.exists()
        )

    @abstractmethod
    def _build(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (corpus_df, queries_df, qrels_df)."""

    def load(self, force: bool = False) -> None:
        if self.already_loaded() and not force:
            logger.info(f"[dataset:{self.cfg.name}] already loaded, skipping")
            return
        logger.info(f"[dataset:{self.cfg.name}] building...")
        corpus, queries, qrels = self._build()

        if self.cfg.max_documents:
            corpus = corpus.head(self.cfg.max_documents).reset_index(drop=True)
            keep_doc_ids = set(corpus["doc_id"].tolist())
            qrels = qrels[qrels["doc_id"].isin(keep_doc_ids)].reset_index(drop=True)
            keep_q_ids = set(qrels["query_id"].unique())
            queries = queries[queries["query_id"].isin(keep_q_ids)].reset_index(drop=True)

        if self.cfg.max_queries:
            queries = queries.head(self.cfg.max_queries).reset_index(drop=True)
            keep_q_ids = set(queries["query_id"].tolist())
            qrels = qrels[qrels["query_id"].isin(keep_q_ids)].reset_index(drop=True)

        write_parquet(corpus, self.corpus_path)
        write_parquet(queries, self.queries_path)
        write_parquet(qrels, self.qrels_path)
        touch_success(
            self.success_path,
            {
                "n_docs": len(corpus),
                "n_queries": len(queries),
                "n_qrels": len(qrels),
                "config": self.cfg.model_dump(),
            },
        )
        logger.info(f"[dataset:{self.cfg.name}] wrote {len(corpus)} docs, {len(queries)} queries")


class SquadLoader(DatasetLoader):
    name = "squad"

    def _build(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        from datasets import load_dataset

        ds = load_dataset("rajpurkar/squad", split=self.cfg.split)
        if self.cfg.max_queries:
            ds = ds.select(range(min(len(ds), self.cfg.max_queries)))

        # Doc id = title + hash of context, since same title may have many contexts
        from ..utils.hashing import stable_hash_str

        seen_contexts: dict[str, str] = {}
        corpus_records = []
        qrels_records = []
        query_records = []

        for ex in ds:
            ctx = ex["context"]
            title = ex["title"]
            doc_id = f"{title}::{stable_hash_str(ctx, 8)}"
            if doc_id not in seen_contexts:
                seen_contexts[doc_id] = ctx
                corpus_records.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "text": ctx,
                        "metadata": {},
                    }
                )

            qid = ex["id"]
            ans = ex["answers"]["text"]
            query_records.append(
                {
                    "query_id": qid,
                    "query": ex["question"],
                    "gold_answer": ans[0] if ans else "",
                    "gold_answers": list(ans),
                    "gold_doc_ids": [doc_id],
                    "gold_chunk_ids": [],
                    "metadata": {"title": title},
                }
            )
            qrels_records.append({"query_id": qid, "doc_id": doc_id, "relevance": 1})

        return (
            pd.DataFrame(corpus_records),
            pd.DataFrame(query_records),
            pd.DataFrame(qrels_records),
        )


class NaturalQuestionsLoader(DatasetLoader):
    name = "natural_questions"

    def _build(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Minimal NQ-Open style; the full NQ has heavy preprocessing.
        # For research convenience we use the `nq_open` variant if available.
        try:
            from datasets import load_dataset

            ds = load_dataset("google-research-datasets/nq_open", split=self.cfg.split)
            if self.cfg.max_queries:
                ds = ds.select(range(min(len(ds), self.cfg.max_queries)))
        except Exception as e:
            logger.warning(
                f"NQ load failed: {e}. NQ-Open is questions-only and lacks gold "
                f"docs; for retrieval evaluation prefer the prepared NQ from QuOTE-style "
                f"experiments. Falling back to a tiny placeholder."
            )
            return self._tiny_placeholder()

        # NQ-Open has no document corpus; for retrieval we'd need Wikipedia-2018
        # dump. We emit a placeholder corpus per query (its own answer as a doc)
        # so the pipeline runs end to end on this dataset; users who need real
        # retrieval should use SQuAD or supply a preprocessed NQ.
        corpus_records = []
        qrels_records = []
        query_records = []

        for i, ex in enumerate(ds):
            qid = f"nq-{i}"
            answers = ex["answer"]
            doc_id = f"nq-doc-{i}"
            doc_text = answers[0] if answers else ""
            corpus_records.append(
                {
                    "doc_id": doc_id,
                    "title": f"NQ doc {i}",
                    "text": doc_text or "[empty]",
                    "metadata": {"placeholder": True},
                }
            )
            query_records.append(
                {
                    "query_id": qid,
                    "query": ex["question"],
                    "gold_answer": answers[0] if answers else "",
                    "gold_answers": list(answers),
                    "gold_doc_ids": [doc_id],
                    "gold_chunk_ids": [],
                    "metadata": {},
                }
            )
            qrels_records.append({"query_id": qid, "doc_id": doc_id, "relevance": 1})

        return (
            pd.DataFrame(corpus_records),
            pd.DataFrame(query_records),
            pd.DataFrame(qrels_records),
        )

    def _tiny_placeholder(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return FixtureLoader(self.cfg, self.out_dir.parent)._build()


class MultiHopRagLoader(DatasetLoader):
    """Expects user to place MultiHop-RAG dataset files under
    storage/raw/multihop_rag_source/ before running.

    See https://github.com/yixuantt/MultiHop-RAG
    """

    name = "multihop_rag"

    def _build(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        src = self.out_dir.parent / "multihop_rag_source"
        corpus_file = src / "corpus.json"
        queries_file = src / "MultiHopRAG.json"
        if not corpus_file.exists() or not queries_file.exists():
            raise FileNotFoundError(
                f"MultiHop-RAG source files not found under {src}. "
                f"Download corpus.json and MultiHopRAG.json from "
                f"https://github.com/yixuantt/MultiHop-RAG and place them there."
            )
        import json

        with corpus_file.open() as f:
            corpus_raw = json.load(f)
        with queries_file.open() as f:
            queries_raw = json.load(f)

        if self.cfg.max_queries:
            queries_raw = queries_raw[: self.cfg.max_queries]
        if self.cfg.max_documents:
            corpus_raw = corpus_raw[: self.cfg.max_documents]

        corpus_records = []
        for i, doc in enumerate(corpus_raw):
            text = doc.get("body") or doc.get("text") or ""
            title = doc.get("title", f"doc-{i}")
            doc_id = doc.get("url") or f"mhr-doc-{i}"
            corpus_records.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "metadata": {"published": doc.get("published_at", "")},
                }
            )

        query_records = []
        qrel_records = []
        for i, q in enumerate(queries_raw):
            qid = f"mhr-{i}"
            golds = q.get("evidence_list", [])
            gold_doc_ids = list({g.get("url") or g.get("source") or "" for g in golds if g})
            query_records.append(
                {
                    "query_id": qid,
                    "query": q["query"],
                    "gold_answer": q.get("answer", ""),
                    "gold_answers": [q.get("answer", "")],
                    "gold_doc_ids": gold_doc_ids,
                    "gold_chunk_ids": [],
                    "metadata": {"question_type": q.get("question_type", "")},
                }
            )
            for d in gold_doc_ids:
                qrel_records.append({"query_id": qid, "doc_id": d, "relevance": 1})

        return (
            pd.DataFrame(corpus_records),
            pd.DataFrame(query_records),
            pd.DataFrame(qrel_records),
        )


class FixtureLoader(DatasetLoader):
    """Tiny synthetic dataset for tests; deterministic output."""

    name = "fixture"

    def _build(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        docs = [
            {
                "doc_id": "d1",
                "title": "Photosynthesis",
                "text": (
                    "Photosynthesis is the process by which green plants use sunlight, "
                    "water and carbon dioxide to produce glucose and oxygen. It occurs "
                    "primarily in the chloroplasts of plant cells."
                ),
                "metadata": {},
            },
            {
                "doc_id": "d2",
                "title": "Mitochondria",
                "text": (
                    "Mitochondria are membrane-bound organelles found in the cytoplasm "
                    "of eukaryotic cells. They generate most of a cell's supply of ATP, "
                    "the chemical energy used to power biochemical reactions."
                ),
                "metadata": {},
            },
            {
                "doc_id": "d3",
                "title": "Eiffel Tower",
                "text": (
                    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
                    "Mars in Paris, France. It is named after the engineer Gustave "
                    "Eiffel, whose company designed and built the tower from 1887 to 1889."
                ),
                "metadata": {},
            },
        ]
        queries = [
            {
                "query_id": "q1",
                "query": "What do plants use to perform photosynthesis?",
                "gold_answer": "sunlight, water and carbon dioxide",
                "gold_answers": ["sunlight, water and carbon dioxide"],
                "gold_doc_ids": ["d1"],
                "gold_chunk_ids": [],
                "metadata": {},
            },
            {
                "query_id": "q2",
                "query": "What do mitochondria produce?",
                "gold_answer": "ATP",
                "gold_answers": ["ATP", "adenosine triphosphate"],
                "gold_doc_ids": ["d2"],
                "gold_chunk_ids": [],
                "metadata": {},
            },
            {
                "query_id": "q3",
                "query": "Who built the Eiffel Tower?",
                "gold_answer": "Gustave Eiffel",
                "gold_answers": ["Gustave Eiffel"],
                "gold_doc_ids": ["d3"],
                "gold_chunk_ids": [],
                "metadata": {},
            },
        ]
        qrels = [
            {"query_id": q["query_id"], "doc_id": q["gold_doc_ids"][0], "relevance": 1}
            for q in queries
        ]
        return pd.DataFrame(docs), pd.DataFrame(queries), pd.DataFrame(qrels)


def get_loader(cfg: DatasetConfig, raw_dir: Path) -> DatasetLoader:
    mapping = {
        "squad": SquadLoader,
        "natural_questions": NaturalQuestionsLoader,
        "multihop_rag": MultiHopRagLoader,
        "fixture": FixtureLoader,
    }
    if cfg.name not in mapping:
        raise ValueError(f"Unknown dataset: {cfg.name}")
    return mapping[cfg.name](cfg, raw_dir)

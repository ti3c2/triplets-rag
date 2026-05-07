"""Inference strategies.

Each takes (test query, retrieved items) and produces a Prediction.
For retrieval_only no LLM is invoked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from ..config import BudgetConfig, InferenceConfig, InferenceStrategy
from ..data.schemas import Prediction
from ..models import LLMClient
from ..prompts import infer_key, render
from ..retrieve.retriever import RetrievalResult, RetrievedItem


@dataclass
class _TripletForPrompt:
    question: str
    contexts: list[str]
    answer: str


def _build_vanilla_prompt(query: str, items: list[RetrievedItem], budget: BudgetConfig) -> str:
    contexts = []
    for it in items[: budget.total_context_items]:
        if it.item_type == "chunk":
            contexts.append(it.text)
        elif it.item_type == "question":
            # the chunk text is in `text` for question items in our id_map
            contexts.append(it.text)
        elif it.item_type == "triplet":
            # vanilla on triplet results: use first context only
            if it.chunk_texts:
                contexts.append(it.chunk_texts[0])
        elif it.item_type == "qa_pair":
            contexts.append(it.text)
    return render(infer_key("vanilla_rag"), question=query, contexts=contexts)


def _flatten_chunks_from_triplets(
    items: list[RetrievedItem], n_triplets: int, per_triplet: int
) -> list[_TripletForPrompt]:
    out: list[_TripletForPrompt] = []
    for it in items[:n_triplets]:
        if it.item_type != "triplet":
            continue
        out.append(
            _TripletForPrompt(
                question=it.question or "",
                contexts=list(it.chunk_texts[:per_triplet]),
                answer=it.teacher_answer or "",
            )
        )
    return out


def _build_triplet_prompt(
    query: str,
    items: list[RetrievedItem],
    budget: BudgetConfig,
    fresh_contexts: list[str] | None,
) -> tuple[str, list[_TripletForPrompt]]:
    triplets = _flatten_chunks_from_triplets(items, budget.num_triplets, budget.per_triplet_contexts)
    prompt = render(
        infer_key("triplet_rag"),
        question=query,
        triplets=[
            {"question": t.question, "contexts": t.contexts, "answer": t.answer} for t in triplets
        ],
        fresh_contexts=fresh_contexts or [],
    )
    return prompt, triplets


def _build_qa_demo_prompt(
    query: str,
    items: list[RetrievedItem],
    budget: BudgetConfig,
    fresh_contexts: list[str] | None,
) -> str:
    triplets = _flatten_chunks_from_triplets(items, budget.num_triplets, budget.per_triplet_contexts)
    return render(
        infer_key("qa_demo_rag"),
        question=query,
        triplets=[
            {"question": t.question, "answer": t.answer} for t in triplets
        ],
        fresh_contexts=fresh_contexts or [],
    )


def run_inference_for_query(
    *,
    query: str,
    query_id: str,
    gold_answers: list[str],
    gold_doc_ids: list[str],
    retrieval: RetrievalResult,
    inference_cfg: InferenceConfig,
    budget: BudgetConfig,
    student: LLMClient | None,
    fresh_retrieval: RetrievalResult | None = None,
) -> Prediction:
    items = retrieval.items
    fresh_contexts: list[str] | None = None
    if inference_cfg.include_fresh_contexts and fresh_retrieval is not None:
        fresh_contexts = [
            it.text for it in fresh_retrieval.items[: budget.total_context_items]
        ]

    triplet_ids: list[str] = []
    if inference_cfg.strategy == InferenceStrategy.RETRIEVAL_ONLY:
        prompt = ""
        prediction = ""
        latency_ms = 0.0
    elif inference_cfg.strategy == InferenceStrategy.VANILLA_RAG:
        prompt = _build_vanilla_prompt(query, items, budget)
        if student is None:
            raise ValueError("student LLM required for vanilla_rag")
        t0 = time.time()
        prediction = student.chat([{"role": "user", "content": prompt}])
        latency_ms = (time.time() - t0) * 1000
    elif inference_cfg.strategy == InferenceStrategy.TRIPLET_RAG:
        prompt, used_triplets = _build_triplet_prompt(query, items, budget, fresh_contexts)
        triplet_ids = [it.item_id for it in items if it.item_type == "triplet"][
            : budget.num_triplets
        ]
        if student is None:
            raise ValueError("student LLM required for triplet_rag")
        t0 = time.time()
        prediction = student.chat([{"role": "user", "content": prompt}])
        latency_ms = (time.time() - t0) * 1000
    elif inference_cfg.strategy == InferenceStrategy.QA_DEMO_RAG:
        prompt = _build_qa_demo_prompt(query, items, budget, fresh_contexts)
        triplet_ids = [it.item_id for it in items if it.item_type == "triplet"][
            : budget.num_triplets
        ]
        if student is None:
            raise ValueError("student LLM required for qa_demo_rag")
        t0 = time.time()
        prediction = student.chat([{"role": "user", "content": prompt}])
        latency_ms = (time.time() - t0) * 1000
    else:
        raise ValueError(f"Unknown strategy: {inference_cfg.strategy}")

    # retrieved_ids: which docs were retrieved (for retrieval metrics)
    retrieved_ids: list[str] = []
    retrieved_texts: list[str] = []
    for it in items:
        if it.item_type == "chunk":
            retrieved_ids.append(it.doc_id or it.chunk_id or it.item_id)
            retrieved_texts.append(it.text)
        elif it.item_type == "question":
            retrieved_ids.append(it.doc_id or it.chunk_id or "")
            retrieved_texts.append(it.text)
        elif it.item_type == "triplet":
            # For metrics, use the union of chunk_ids
            for cid in it.chunk_ids:
                retrieved_ids.append(cid)
                retrieved_texts.append(it.chunk_texts[it.chunk_ids.index(cid)] if cid in it.chunk_ids else "")
        elif it.item_type == "qa_pair":
            retrieved_ids.append(it.item_id)
            retrieved_texts.append(it.text)

    return Prediction(
        query_id=query_id,
        query=query,
        strategy=inference_cfg.strategy.value,
        retrieved_ids=retrieved_ids,
        retrieved_texts=retrieved_texts,
        retrieved_triplet_ids=triplet_ids,
        prompt=prompt,
        prediction=prediction.strip(),
        gold_answers=gold_answers,
        gold_doc_ids=gold_doc_ids,
        latency_ms=latency_ms,
    )

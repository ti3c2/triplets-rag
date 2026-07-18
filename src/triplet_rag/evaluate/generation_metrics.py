"""Lexical generation metrics: EM, F1 (SQuAD-style), ROUGE-L."""

from __future__ import annotations

import re
import string
from collections import Counter

import pandas as pd


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_tokens)
    r = num_same / len(gold_tokens)
    return 2 * p * r / (p + r)


def best_over_golds(metric_fn, pred: str, golds: list[str]) -> float:
    if not golds:
        return 0.0
    return max(metric_fn(pred, g) for g in golds if g)


def rouge_l(pred: str, gold: str) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(gold, pred)["rougeL"].fmeasure
    except Exception:
        return 0.0


def compute_lexical_metrics(
    predictions: list[dict],
    metric_names: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    for pred in predictions:
        text = pred["prediction"] or ""
        golds = list(pred.get("gold_answers", [])) or []
        if not golds and pred.get("gold_answer"):
            golds = [pred["gold_answer"]]
        rec: dict = {"query_id": pred["query_id"]}
        for m in metric_names:
            ml = m.lower()
            if ml == "em":
                rec["em"] = best_over_golds(exact_match, text, golds)
            elif ml == "f1":
                rec["f1"] = best_over_golds(token_f1, text, golds)
            elif ml == "rouge_l":
                rec["rouge_l"] = best_over_golds(rouge_l, text, golds)
        rows.append(rec)

    pq = pd.DataFrame(rows)
    agg: dict[str, float] = {}
    for m in metric_names:
        ml = m.lower()
        if ml in pq.columns:
            agg[ml] = float(pq[ml].mean())
    return agg, pq

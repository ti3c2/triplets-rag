"""Lexical metrics (EM, F1, ROUGE-L) on canned predictions."""

from triplet_rag.evaluate.generation_metrics import (
    best_over_golds,
    compute_lexical_metrics,
    exact_match,
    normalize_answer,
    rouge_l,
    token_f1,
)


def test_normalize_answer():
    assert normalize_answer("The cat") == "cat"
    assert normalize_answer("a Cat.") == "cat"
    assert normalize_answer("  the  cat  ") == "cat"


def test_exact_match_normalizes():
    assert exact_match("The Eiffel Tower", "Eiffel Tower") == 1
    assert exact_match("Paris", "London") == 0


def test_token_f1_partial_credit():
    # Predicting "Eiffel Tower" against gold "the Eiffel Tower" -> after normalization equal
    f = token_f1("Eiffel Tower", "the Eiffel Tower")
    assert f == 1.0
    # Half overlap
    f = token_f1("Paris is", "Paris France")
    assert 0 < f < 1


def test_token_f1_no_overlap():
    assert token_f1("apples", "oranges") == 0.0


def test_best_over_golds():
    """If any gold yields a perfect score, best is taken."""
    score = best_over_golds(exact_match, "ATP", ["adenosine triphosphate", "ATP"])
    assert score == 1


def test_rouge_l_basic():
    score = rouge_l("the cat sat", "the cat sat")
    assert score == 1.0
    score = rouge_l("nothing", "the cat sat")
    assert score == 0.0


def test_compute_lexical_metrics_full():
    preds = [
        {
            "query_id": "q1",
            "prediction": "Eiffel Tower",
            "gold_answers": ["The Eiffel Tower", "Eiffel Tower"],
        },
        {
            "query_id": "q2",
            "prediction": "I don't know.",
            "gold_answers": ["ATP"],
        },
    ]
    agg, pq = compute_lexical_metrics(preds, ["em", "f1"])
    assert "em" in agg
    assert "f1" in agg
    assert len(pq) == 2
    # q1 should be a perfect match
    q1_row = pq[pq["query_id"] == "q1"].iloc[0]
    assert q1_row["em"] == 1
    assert q1_row["f1"] == 1.0
    # q2 should be 0
    q2_row = pq[pq["query_id"] == "q2"].iloc[0]
    assert q2_row["em"] == 0

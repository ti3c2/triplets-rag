"""Prompt templates render and reject undefined variables (StrictUndefined)."""

import pytest
from jinja2 import UndefinedError

from triplet_rag.prompts import (
    PROMPTS,
    answer_gen_key,
    infer_key,
    question_gen_key,
    render,
)


def test_all_keys_render():
    """Every prompt should accept its expected variables and render without error."""
    rendered = render(question_gen_key("complex"), n=5, chunk_text="The cat sat.")
    assert "The cat sat." in rendered
    assert "5" in rendered

    rendered = render(answer_gen_key("rag_default"), question="Q?", contexts=["c1", "c2"])
    assert "c1" in rendered and "c2" in rendered and "Q?" in rendered

    rendered = render(infer_key("vanilla_rag"), question="Q?", contexts=["c1"])
    assert "c1" in rendered

    rendered = render(
        infer_key("triplet_rag"),
        question="test?",
        triplets=[{"question": "q1", "contexts": ["c1"], "answer": "a1"}],
        fresh_contexts=["fc1"],
    )
    assert "q1" in rendered and "a1" in rendered and "fc1" in rendered

    rendered = render(
        infer_key("qa_demo_rag"),
        question="test?",
        triplets=[{"question": "q1", "answer": "a1"}],
        fresh_contexts=[],
    )
    assert "q1" in rendered and "a1" in rendered


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        render("nonexistent.prompt.v99", x=1)


def test_strict_undefined_raises_on_missing():
    """StrictUndefined makes typos in template variables loud."""
    with pytest.raises(UndefinedError):
        render(infer_key("vanilla_rag"), question="Q?")  # missing contexts


def test_versioning_in_keys():
    """Prompt keys carry an explicit version so silent drift is detectable."""
    for key in PROMPTS:
        assert key.endswith(".v1"), f"prompt key {key} must end in version"

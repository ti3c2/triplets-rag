"""Hashing must be deterministic and stable across key ordering."""

from triplet_rag.utils.hashing import stable_hash, stable_hash_str


def test_hash_is_deterministic():
    a = {"a": 1, "b": [2, 3], "c": {"d": 4}}
    b = {"c": {"d": 4}, "b": [2, 3], "a": 1}
    assert stable_hash(a) == stable_hash(b)


def test_hash_changes_with_value():
    a = {"a": 1}
    b = {"a": 2}
    assert stable_hash(a) != stable_hash(b)


def test_hash_changes_with_key():
    a = {"a": 1}
    b = {"b": 1}
    assert stable_hash(a) != stable_hash(b)


def test_hash_str():
    assert stable_hash_str("abc") == stable_hash_str("abc")
    assert stable_hash_str("abc") != stable_hash_str("abd")


def test_float_rounding_stability():
    a = {"x": 1.0}
    b = {"x": 1.0 + 1e-16}
    # rounded to 12 places - should match
    assert stable_hash(a) == stable_hash(b)


def test_length_param():
    h6 = stable_hash({"a": 1}, length=6)
    h12 = stable_hash({"a": 1}, length=12)
    assert len(h6) == 6
    assert len(h12) == 12
    assert h12.startswith(h6)

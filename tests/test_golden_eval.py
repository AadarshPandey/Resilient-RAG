"""Tests for M10: Golden benchmark set, key-fact coverage scorer, and RAGAS adapter fallback."""

from pathlib import Path

import pytest

from rag.evals.golden import (
    GoldenQuestion,
    compute_key_fact_coverage,
    load_golden_set,
)
from rag.evals.ragas_adapter import evaluate_with_ragas


def test_key_fact_coverage_scorer() -> None:
    """Verify normalized key-fact coverage scoring."""
    facts = ["node agent", "port 2379", "consistent key-value"]

    # Full coverage
    ans_full = (
        "The kubelet is a Node Agent! Etcd runs on port 2379 as a consistent key-value store."
    )
    assert compute_key_fact_coverage(ans_full, facts) == 1.0

    # Partial coverage (2 out of 3)
    ans_partial = "It is a node agent and consistent key value store."
    assert pytest.approx(compute_key_fact_coverage(ans_partial, facts), abs=0.01) == 2.0 / 3.0

    # Zero coverage
    ans_none = "Completely unrelated text without any matching terms."
    assert compute_key_fact_coverage(ans_none, facts) == 0.0

    # Empty facts list
    assert compute_key_fact_coverage(ans_full, []) == 1.0


def test_golden_set_schema_and_loader() -> None:
    """Verify golden set file exists and contains exactly 50 validated questions."""
    path = Path("data/golden/golden_set.jsonl")
    assert path.is_file(), "data/golden/golden_set.jsonl must exist"

    questions = load_golden_set(path)
    assert len(questions) == 50

    for q in questions:
        assert isinstance(q, GoldenQuestion)
        assert len(q.id) > 0
        assert len(q.question) > 5
        assert len(q.reference_answer) > 10
        assert len(q.key_facts) >= 1
        assert len(q.source_chunk_ids) >= 1


def test_ragas_adapter_fallback_on_import_or_api_error() -> None:
    """RAGAS adapter must gracefully fall back to unavailable rather than crashing."""
    res = evaluate_with_ragas(
        question="What is kubelet?",
        answer="Node agent",
        contexts=["The kubelet is a node agent"],
        reference="The kubelet is a node agent",
    )
    assert res.available is False
    assert res.error_message is not None
    assert "unavailable" in res.error_message.lower()
    assert res.faithfulness is None

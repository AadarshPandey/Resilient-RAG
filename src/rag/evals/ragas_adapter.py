"""RAGAS metrics evaluation adapter with graceful degradation on failure."""

from typing import Any

from pydantic import BaseModel


class RagasEvaluationResult(BaseModel):
    """Result of RAGAS quality metric evaluation."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    available: bool = True
    error_message: str | None = None


def _safe_extract_score(scores: Any, key: str) -> float | None:
    """Extract float score from ragas result container."""
    try:
        if isinstance(scores, dict) and key in scores:
            val = scores[key]
            return float(val[0] if isinstance(val, list) else val)
        if hasattr(scores, "to_pandas"):
            df = scores.to_pandas()
            if key in df.columns:
                return float(df[key].iloc[0])
    except Exception:
        return None
    return None


def evaluate_with_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
) -> RagasEvaluationResult:
    """Evaluate generation using RAGAS metrics with defensive try/except.

    Args:
        question: User query.
        answer: Generated answer.
        contexts: List of retrieved context text passages.
        reference: Golden reference answer.

    Returns:
        RagasEvaluationResult object.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [reference],
        }
        dataset = Dataset.from_dict(data)

        scores: Any = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        return RagasEvaluationResult(
            faithfulness=_safe_extract_score(scores, "faithfulness"),
            answer_relevancy=_safe_extract_score(scores, "answer_relevancy"),
            context_precision=_safe_extract_score(scores, "context_precision"),
            context_recall=_safe_extract_score(scores, "context_recall"),
            available=True,
        )
    except Exception as e:
        return RagasEvaluationResult(
            available=False,
            error_message=f"RAGAS evaluation unavailable: {e}",
        )

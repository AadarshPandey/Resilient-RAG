"""Golden benchmark question dataset loader and key-fact coverage scorer."""

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenQuestion(BaseModel):
    """A benchmark query paired with reference facts and source provenance."""

    id: str
    question: str
    reference_answer: str
    key_facts: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


def _normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and stripping punctuation."""
    lowered = text.lower()
    return re.sub(r"[^\w\s]", " ", lowered)


def compute_key_fact_coverage(answer: str, key_facts: list[str]) -> float:
    """Compute proportion of reference key facts supported in the answer.

    Both text and facts are normalized (lowercased, punctuation-stripped)
    and matched as substrings or token sets.

    Args:
        answer: Generated answer text.
        key_facts: List of atomic factual statements/phrases.

    Returns:
        Float score between 0.0 and 1.0.
    """
    if not key_facts:
        return 1.0

    norm_answer = _normalize_text(answer)
    answer_tokens = set(norm_answer.split())

    matches = 0
    for fact in key_facts:
        norm_fact = _normalize_text(fact).strip()
        # Direct phrase match
        if norm_fact in norm_answer:
            matches += 1
            continue

        # Token set subset match (all words in fact present in answer)
        fact_tokens = set(norm_fact.split())
        if fact_tokens and fact_tokens.issubset(answer_tokens):
            matches += 1

    return float(matches) / float(len(key_facts))


def load_golden_set(path: str | Path) -> list[GoldenQuestion]:
    """Load and validate the 50-question golden set from JSONL.

    Args:
        path: Path to golden_set.jsonl.

    Returns:
        List of validated GoldenQuestion objects.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Golden set file not found: {path}")

    questions: list[GoldenQuestion] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            questions.append(GoldenQuestion.model_validate(data))

    return questions

"""LangSmith tracing and evaluation utilities.

Defensively designed: tracing is disabled by default unless LANGSMITH_TRACING=true
and LANGSMITH_API_KEY is present. Pure functions require zero network calls.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from rag.evals.golden import compute_key_fact_coverage, load_golden_set

logger = logging.getLogger(__name__)


def is_langsmith_enabled() -> bool:
    """Return True if LangSmith tracing is explicitly enabled and API key is present."""
    tracing = os.environ.get("LANGSMITH_TRACING", "").strip().lower() in (
        "true",
        "1",
    ) or os.environ.get("LANGCHAIN_TRACING_V2", "").strip().lower() in ("true", "1")
    has_key = bool(
        os.environ.get("LANGSMITH_API_KEY", "").strip()
        or os.environ.get("LANGCHAIN_API_KEY", "").strip()
    )
    return tracing and has_key


def build_run_metadata(
    condition: str | None = None,
    scenario: str | None = None,
    seed: int | None = None,
    heal_actions: list[str] | None = None,
    outcome: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct structured metadata dictionary for LangSmith run tracking."""
    metadata: dict[str, Any] = {}
    if condition is not None:
        metadata["condition"] = condition
    if scenario is not None:
        metadata["scenario"] = scenario
    if seed is not None:
        metadata["seed"] = seed
    if heal_actions is not None:
        metadata["heal_actions"] = list(heal_actions)
        metadata["heal_count"] = len(heal_actions)
    if outcome is not None:
        metadata["outcome"] = outcome
    if extra:
        metadata.update(extra)
    return metadata


def build_run_tags(
    condition: str | None = None,
    scenario: str | None = None,
    outcome: str | None = None,
    extra_tags: list[str] | None = None,
) -> list[str]:
    """Build list of tag strings for filtering in LangSmith UI."""
    tags: list[str] = []
    if condition:
        tags.append(f"condition:{condition}")
    if scenario:
        tags.append(f"scenario:{scenario}")
    if outcome:
        tags.append(f"outcome:{outcome}")
    if extra_tags:
        tags.extend(extra_tags)
    return tags


def upload_golden_dataset(
    dataset_name: str = "kubernetes-golden-set",
    file_path: Path | str = "data/golden/golden_set.jsonl",
    client: Any | None = None,
    description: str = "Resilient RAG 50-question Kubernetes golden set",
) -> bool:
    """Upload the golden set JSONL file as a dataset to LangSmith.

    Returns True on success, False if LangSmith is disabled or unavailable.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.error(f"Golden set file does not exist: {path}")
        return False

    if client is None:
        if not is_langsmith_enabled():
            logger.info("LangSmith is not enabled; skipping dataset upload.")
            return False
        try:
            from langsmith import Client

            client = Client()
        except Exception as e:
            logger.warning(f"Failed to instantiate LangSmith client: {e}")
            return False

    try:
        questions = load_golden_set(path)
        if not questions:
            logger.warning(f"No questions loaded from {path}")
            return False

        # Create dataset if missing
        if not client.has_dataset(dataset_name=dataset_name):
            client.create_dataset(
                dataset_name=dataset_name,
                description=description,
            )
            logger.info(f"Created LangSmith dataset '{dataset_name}'")

        # Upload examples
        for q in questions:
            client.create_example(
                inputs={"question": q.question},
                outputs={
                    "reference_answer": q.reference_answer,
                    "key_facts": q.key_facts,
                    "source_chunk_ids": q.source_chunk_ids,
                },
                metadata={"id": q.id},
                dataset_name=dataset_name,
            )

        logger.info(f"Successfully uploaded {len(questions)} examples to '{dataset_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to upload golden dataset to LangSmith: {e}")
        return False


def eval_key_fact_coverage(run: Any, example: Any) -> dict[str, Any]:
    """LangSmith evaluator for key-fact coverage.

    Inspects run output answer and example reference key_facts.
    """
    prediction = ""
    if hasattr(run, "outputs") and isinstance(run.outputs, dict):
        prediction = str(run.outputs.get("answer", ""))
    elif isinstance(run, dict):
        prediction = str(run.get("answer", ""))

    key_facts: list[str] = []
    if hasattr(example, "outputs") and isinstance(example.outputs, dict):
        key_facts = example.outputs.get("key_facts", [])
    elif isinstance(example, dict):
        key_facts = example.get("key_facts", [])

    coverage = compute_key_fact_coverage(prediction, key_facts)
    return {
        "key": "key_fact_coverage",
        "score": coverage,
    }


def eval_safe_outcome(run: Any, example: Any) -> dict[str, Any]:
    """LangSmith evaluator for safe outcome."""
    outcome = "UNKNOWN"
    if hasattr(run, "outputs") and isinstance(run.outputs, dict):
        outcome = str(run.outputs.get("outcome", ""))
    elif isinstance(run, dict):
        outcome = str(run.get("outcome", ""))

    safe = outcome in ("ANSWERED", "ABSTAIN", "ESCALATE")
    return {
        "key": "safe_outcome",
        "score": 1.0 if safe else 0.0,
    }

"""Pure failure detectors mapping node outcomes and exceptions to typed FailureType."""

import re
from typing import Any

from rag.errors import (
    ContextOverflowError,
    LLMError,
    MalformedOutputError,
)
from rag.healing.failures import FailureType
from rag.schemas import Chunk

# Heuristic prompt injection patterns
INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(?:all\s+)?previous\s+(?:instructions|prompts|directives)", re.IGNORECASE
    ),
    re.compile(r"you\s+are\s+now\s+(?:a|an)\s+[a-z0-9_-]+", re.IGNORECASE),
    re.compile(r"system\s+override\b", re.IGNORECASE),
    re.compile(r"(?:repeat|print|show|output)\s+(?:.*?\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"\b(?:DAN|jailbreak|developer mode)\b", re.IGNORECASE),
    re.compile(r"bypass\s+(?:all\s+)?restrictions", re.IGNORECASE),
]


def detect_prompt_injection(text: str) -> bool:
    """Check text for known prompt injection and instruction override heuristics.

    Args:
        text: Candidate chunk text or user input.

    Returns:
        True if any injection heuristic matches, False otherwise.
    """
    return any(pattern.search(text) is not None for pattern in INJECTION_PATTERNS)


def detect_retrieval_failure(
    chunks: list[Chunk],
    exc: Exception | None = None,
) -> FailureType | None:
    """Classify retrieval failures from exception and candidate chunk list.

    CRITICAL REGRESSION RULE:
    Non-LLM exceptions occurring in retrieval MUST map to RETRIEVAL_BACKEND_DOWN,
    never LLM_ERROR.
    """
    if exc is not None:
        if isinstance(exc, LLMError):
            return FailureType.LLM_ERROR
        return FailureType.RETRIEVAL_BACKEND_DOWN

    if not chunks:
        return FailureType.EMPTY_RETRIEVAL

    # Scan for prompt injection in retrieved passages
    for chunk in chunks:
        if detect_prompt_injection(chunk.text):
            return FailureType.PROMPT_INJECTION

    return None


def detect_grading_failure(
    chunks: list[Any],
    avg_relevance: float,
    relevance_threshold: float = 0.15,
    exc: Exception | None = None,
) -> FailureType | None:
    """Classify grading failures.

    Non-LLM exceptions in grading map to RETRIEVAL_BACKEND_DOWN.
    """
    if exc is not None:
        if isinstance(exc, LLMError):
            return FailureType.LLM_ERROR
        return FailureType.RETRIEVAL_BACKEND_DOWN

    if not chunks:
        return FailureType.EMPTY_RETRIEVAL

    if avg_relevance < relevance_threshold:
        return FailureType.LOW_RELEVANCE

    return None


def detect_generation_failure(
    answer: str,
    exc: Exception | None = None,
    estimated_tokens: int = 0,
    token_limit: int = 4000,
) -> FailureType | None:
    """Classify failures during generation step."""
    if exc is not None:
        if isinstance(exc, ContextOverflowError):
            return FailureType.CONTEXT_OVERFLOW
        if isinstance(exc, MalformedOutputError):
            return FailureType.MALFORMED_OUTPUT
        if isinstance(exc, LLMError):
            return FailureType.LLM_ERROR
        return FailureType.LLM_ERROR

    if estimated_tokens > token_limit:
        return FailureType.CONTEXT_OVERFLOW

    norm_ans = answer.strip().upper()
    if norm_ans == "INSUFFICIENT CONTEXT" or "INSUFFICIENT CONTEXT" in norm_ans:
        return FailureType.INSUFFICIENT_CONTEXT

    return None


def detect_verification_failure(
    is_grounded: bool,
    unsupported_claims: list[str],
    exc: Exception | None = None,
) -> FailureType | None:
    """Classify verification failures."""
    if exc is not None:
        if isinstance(exc, MalformedOutputError):
            return FailureType.MALFORMED_OUTPUT
        if isinstance(exc, LLMError):
            return FailureType.LLM_ERROR
        return FailureType.LLM_ERROR

    if not is_grounded or len(unsupported_claims) > 0:
        return FailureType.UNGROUNDED_ANSWER

    return None

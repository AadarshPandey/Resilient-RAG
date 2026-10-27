"""Typed exception hierarchy for Resilient RAG."""

from typing import Any


class RagError(Exception):
    """Base exception for all domain errors in Resilient RAG."""

    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args)
        self.message = message


class BackendError(RagError):
    """Raised when an external or storage backend (Qdrant, BM25, SQLite) fails."""


class LLMError(RagError):
    """Raised when an LLM call fails."""

    def __init__(self, message: str, retryable: bool = False, *args: Any) -> None:
        super().__init__(message, *args)
        self.retryable = retryable


class LLMRateLimitError(LLMError):
    """Raised when rate limit / HTTP 429 is encountered (retryable)."""

    def __init__(self, message: str = "LLM rate limit reached") -> None:
        super().__init__(message, retryable=True)


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out (retryable)."""

    def __init__(self, message: str = "LLM request timed out") -> None:
        super().__init__(message, retryable=True)


class MalformedOutputError(RagError):
    """Raised when LLM output cannot be parsed into expected structure."""


class ContextOverflowError(RagError):
    """Raised when context exceeds prompt or model token budget."""


class UngroundedAnswerError(RagError):
    """Raised when generated answer contains ungrounded claims."""


class PromptInjectionError(RagError):
    """Raised when a retrieved document contains a prompt injection attack."""

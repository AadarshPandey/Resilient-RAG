"""LLM base protocols and result types."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMResult:
    """Standardized output from an LLM call."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tier: int = 0


class ChatModel(Protocol):
    """Protocol for LLM chat model components."""

    def invoke(self, messages: list[Any], *, tier: int = 0) -> LLMResult:
        """Invoke model with messages on the designated tier."""
        ...

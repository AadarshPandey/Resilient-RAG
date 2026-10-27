"""Graph state definition for the Resilient RAG pipeline."""

from typing import TypedDict

from rag.schemas import (
    Chunk,
    Citation,
    FailureType,
    Outcome,
    RetrievalMode,
    RetrievedChunk,
    TraceEvent,
    VerificationResult,
)


class RagState(TypedDict, total=False):
    """Execution state for the LangGraph pipeline."""

    question: str
    rewritten_query: str | None
    retrieval_mode: RetrievalMode | str
    top_k: int
    chunks: list[Chunk]
    retrieved_chunks: list[RetrievedChunk]
    quarantined_ids: list[str]
    answer: str
    claims: list[str]
    citations: list[Citation]
    verification: VerificationResult | None
    llm_tier: int
    strict_mode: bool
    heal_attempts: int
    heal_history: list[tuple[str, str]]
    last_failure: FailureType | str | None
    outcome: Outcome | str | None
    trace: list[TraceEvent]
    token_usage: int
    llm_calls: int
    error_message: str | None

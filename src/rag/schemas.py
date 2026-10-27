"""Core domain models and schema definitions."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Outcome(StrEnum):
    """Final outcome of a RAG query execution."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"
    ESCALATED = "escalated"
    FAILED = "failed"


class FailureType(StrEnum):
    """Detectable typed failure scenarios across the RAG lifecycle."""

    RETRIEVAL_BACKEND_DOWN = "retrieval_backend_down"
    EMPTY_RETRIEVAL = "empty_retrieval"
    PROMPT_INJECTION = "prompt_injection"
    LOW_RELEVANCE = "low_relevance"
    CONTEXT_OVERFLOW = "context_overflow"
    LLM_ERROR = "llm_error"
    MALFORMED_OUTPUT = "malformed_output"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    UNGROUNDED_ANSWER = "ungrounded_answer"


class HealAction(StrEnum):
    """Bounded, deterministic healing actions selectable by the supervisor."""

    RETRY_SAME = "retry_same"
    REWRITE_QUERY = "rewrite_query"
    WIDEN_TOPK = "widen_topk"
    BM25_ONLY = "bm25_only"
    VECTOR_ONLY = "vector_only"
    REGENERATE_STRICT = "regenerate_strict"
    MODEL_FALLBACK = "model_fallback"
    COMPRESS_CONTEXT = "compress_context"
    QUARANTINE_CHUNK = "quarantine_chunk"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"


class RetrievalMode(StrEnum):
    """Operational mode for candidate retrieval."""

    HYBRID = "hybrid"
    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"


class Document(BaseModel):
    """A raw ingested document before chunking."""

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunk derived from a document with a deterministic content hash ID."""

    id: str
    doc_id: str
    text: str
    index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk retrieved by a retriever component with relevance score."""

    chunk: Chunk
    score: float
    source: str = "hybrid"


class Citation(BaseModel):
    """A source citation referencing a chunk that supported the answer."""

    chunk_id: str
    text: str
    score: float | None = None


class Claim(BaseModel):
    """An individual atomic claim extracted from a generated answer."""

    claim: str
    grounded: bool
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""


class VerificationResult(BaseModel):
    """Outcome of claim groundedness verification."""

    is_grounded: bool
    claims: list[Claim] = Field(default_factory=list)
    summary: str = ""


class TraceEvent(BaseModel):
    """A single trace event recording node entry/exit and state changes."""

    node: str
    timestamp: float
    details: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    """Comprehensive audit and experiment log for a single query run."""

    question: str
    outcome: Outcome
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    attempts: int = 1
    detected_faults: list[str] = Field(default_factory=list)
    heal_actions: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    trace: list[TraceEvent] = Field(default_factory=list)
    tokens_used: int = 0
    llm_calls: int = 0
    condition: str = "A"
    scenario: str | None = None
    seed: int | None = None

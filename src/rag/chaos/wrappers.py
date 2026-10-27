"""Chaos wrappers for pipeline components implementing deterministic fault injection."""

from typing import Any

from rag.chaos.controller import FaultController
from rag.chaos.faults import FaultAction
from rag.clock import Clock, SystemClock
from rag.errors import (
    BackendError,
    ContextOverflowError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    MalformedOutputError,
    RagError,
)
from rag.llm.base import ChatModel, LLMResult
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.embedder import Embedder
from rag.retrieval.retriever import Retriever
from rag.retrieval.vector import VectorStore
from rag.schemas import Chunk, RetrievalMode, RetrievedChunk


def _raise_error_by_name(error_type: str, message: str) -> None:
    """Instantiate and raise domain error by type name."""
    norm = error_type.lower()
    if "ratelimit" in norm:
        raise LLMRateLimitError(message)
    if "timeout" in norm:
        raise LLMTimeoutError(message)
    if "overflow" in norm:
        raise ContextOverflowError(message)
    if "malformed" in norm:
        raise MalformedOutputError(message)
    if "backend" in norm:
        raise BackendError(message)
    if "llm" in norm:
        raise LLMError(message, retryable=False)
    raise RagError(message)


class ChaosChatModel:
    """Wraps ChatModel with chaos fault injection."""

    def __init__(
        self,
        target: ChatModel,
        controller: FaultController,
        clock: Clock | None = None,
    ) -> None:
        self.target = target
        self.controller = controller
        self.clock = clock or SystemClock()

    def invoke(self, messages: list[Any], *, tier: int = 0) -> LLMResult:
        """Invoke wrapped model, intercepting with faults when scheduled."""
        # Detect node type from message content heuristics
        msg_str = str(messages[0] if messages else "")
        node = "verify" if "claims to verify" in msg_str.lower() else "generate"

        fault = self.controller.check_fault(component="llm", tier=tier, node=node)
        if fault is not None:
            if fault.action == FaultAction.DELAY:
                self.clock.advance(fault.delay_seconds)
            elif fault.action == FaultAction.ERROR:
                _raise_error_by_name(fault.error_type, fault.error_message)
            elif fault.action == FaultAction.CORRUPT:
                return LLMResult(
                    content=fault.corrupt_value,
                    prompt_tokens=10,
                    completion_tokens=10,
                    tier=tier,
                )

        return self.target.invoke(messages, tier=tier)


class ChaosEmbedder:
    """Wraps Embedder with chaos fault injection."""

    def __init__(
        self,
        target: Embedder,
        controller: FaultController,
        clock: Clock | None = None,
    ) -> None:
        self.target = target
        self.controller = controller
        self.clock = clock or SystemClock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, raising BackendError if fault scheduled."""
        fault = self.controller.check_fault(component="embedder")
        if fault is not None:
            if fault.action == FaultAction.DELAY:
                self.clock.advance(fault.delay_seconds)
            elif fault.action == FaultAction.ERROR:
                _raise_error_by_name(fault.error_type, fault.error_message)
            elif fault.action == FaultAction.EMPTY:
                return []
        return self.target.embed(texts)


class ChaosVectorStore:
    """Wraps VectorStore with chaos fault injection."""

    def __init__(
        self,
        target: VectorStore,
        controller: FaultController,
        clock: Clock | None = None,
    ) -> None:
        self.target = target
        self.controller = controller
        self.clock = clock or SystemClock()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.target.upsert(chunks, vectors)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Search vector store with fault injection."""
        fault = self.controller.check_fault(component="vector")
        if fault is not None:
            if fault.action == FaultAction.DELAY:
                self.clock.advance(fault.delay_seconds)
            elif fault.action == FaultAction.ERROR:
                _raise_error_by_name(fault.error_type, fault.error_message)
            elif fault.action == FaultAction.EMPTY:
                return []
        return self.target.search(query_vector, top_k=top_k)

    def count(self) -> int:
        return self.target.count()


class ChaosBM25Index(BM25Index):
    """Wraps BM25Index with chaos fault injection."""

    def __init__(
        self,
        target: BM25Index,
        controller: FaultController,
        clock: Clock | None = None,
    ) -> None:
        super().__init__()
        self.target = target
        self.controller = controller
        self.clock = clock or SystemClock()

    def search_chunks(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search BM25 with fault injection."""
        fault = self.controller.check_fault(component="bm25")
        if fault is not None:
            if fault.action == FaultAction.DELAY:
                self.clock.advance(fault.delay_seconds)
            elif fault.action == FaultAction.ERROR:
                _raise_error_by_name(fault.error_type, fault.error_message)
            elif fault.action == FaultAction.EMPTY:
                return []
        return self.target.search_chunks(query, top_k=top_k)


class ChaosRetriever(Retriever):
    """Wraps Retriever facade with candidate corruption and prompt injection faults."""

    def __init__(
        self,
        target: Retriever,
        controller: FaultController,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(
            embedder=target.embedder,
            vector_store=target.vector_store,
            bm25_index=target.bm25_index,
            reranker=target.reranker,
        )
        self.target = target
        self.controller = controller
        self.clock = clock or SystemClock()

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Execute retrieval with post-retrieval faults (injection, empty)."""
        results = self.target.retrieve(query=query, mode=mode, top_k=top_k, candidate_k=candidate_k)

        fault = self.controller.check_fault(component="retriever")
        if fault is not None:
            if fault.action == FaultAction.EMPTY:
                return []
            if fault.action == FaultAction.INJECT:
                poisoned = Chunk(
                    id="poisoned_chunk_inj",
                    doc_id="doc_hacked",
                    text=fault.injected_text,
                    index=0,
                )
                results.insert(0, RetrievedChunk(chunk=poisoned, score=1.0, source="poisoned"))

        return results

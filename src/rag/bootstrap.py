"""Dependency injection container and pipeline bootstrap."""

from dataclasses import dataclass
from typing import Any

from rag.audit import AuditLogger
from rag.clock import Clock, SystemClock
from rag.config import Settings
from rag.graph.build import build_graph
from rag.graph.state import RagState
from rag.llm.base import ChatModel
from rag.llm.client import LLMClient
from rag.llm.providers import build_chat_model
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.embedder import Embedder, SentenceTransformerEmbedder
from rag.retrieval.reranker import CrossEncoderReranker, Reranker
from rag.retrieval.retriever import Retriever
from rag.retrieval.vector import QdrantVectorStore, VectorStore


@dataclass
class Components:
    """Dependency injection container holding pipeline components."""

    settings: Settings
    embedder: Embedder
    vector_store: VectorStore
    bm25_index: BM25Index
    reranker: Reranker
    retriever: Retriever
    llm_client: LLMClient
    audit_logger: AuditLogger
    clock: Clock
    graph: Any

    def run(
        self,
        question: str,
        thread_id: str | None = None,
        retrieval_mode: str | None = None,
        top_k: int = 5,
        healing_enabled: bool | None = None,
    ) -> RagState:
        """Execute a query through the LangGraph pipeline."""
        initial_state: RagState = {
            "question": question,
            "retrieval_mode": retrieval_mode or "hybrid",
            "top_k": top_k,
            "heal_attempts": 0,
            "heal_history": [],
            "quarantined_ids": [],
            "strict_mode": False,
            "llm_tier": 0,
            "token_usage": 0,
            "llm_calls": 0,
            "healing_enabled": self.settings.healing_enabled
            if healing_enabled is None
            else healing_enabled,
            "trace": [],
        }

        tid = thread_id or f"run_{question[:16]}"
        config = {
            "configurable": {
                "thread_id": tid,
                "components": self,
                "healing_enabled": self.settings.healing_enabled
                if healing_enabled is None
                else healing_enabled,
            }
        }

        result = self.graph.invoke(initial_state, config=config)
        return dict(result)  # type: ignore[return-value]


def build_components(
    settings: Settings | None = None,
    overrides: dict[str, Any] | None = None,
) -> Components:
    """Construct pipeline components with dependency injection support for testing."""
    cfg = settings or Settings()
    ov = overrides or {}

    clock: Clock = ov.get("clock") or SystemClock()
    audit_logger: AuditLogger = ov.get("audit_logger") or AuditLogger(path=cfg.audit_log_path)

    # Embedder
    embedder: Embedder = ov.get("embedder") or SentenceTransformerEmbedder(
        model_name=cfg.embedding_model,
        device=cfg.device,
    )

    # Vector store
    vector_store: VectorStore = ov.get("vector_store") or QdrantVectorStore(
        location=cfg.qdrant_path if not cfg.qdrant_url else ":memory:",
        url=cfg.qdrant_url,
    )

    # BM25 index
    bm25_index: BM25Index = ov.get("bm25_index") or BM25Index()

    # Reranker
    reranker: Reranker = ov.get("reranker") or CrossEncoderReranker(
        model_name=cfg.reranker_model,
        device=cfg.device,
    )

    # Retriever facade
    retriever: Retriever = ov.get("retriever") or Retriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        reranker=reranker,
    )

    # LLM Client and Chat Models
    if "llm_client" in ov:
        llm_client = ov["llm_client"]
    elif "chat_model" in ov:
        chat_model: ChatModel = ov["chat_model"]
        llm_client = LLMClient(tiers={0: chat_model, 1: chat_model}, clock=clock)
    else:
        tiers_info = cfg.get_llm_tiers()
        tiers: dict[int, ChatModel] = {}
        for idx, (provider, model_name) in enumerate(tiers_info):
            tiers[idx] = build_chat_model(provider, model_name, cfg)
        if not tiers:
            # Fallback default gemini
            tiers[0] = build_chat_model("google", "gemini-1.5-flash", cfg)
        llm_client = LLMClient(tiers=tiers, clock=clock)

    # Graph
    graph = ov.get("graph") or build_graph()

    return Components(
        settings=cfg,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        reranker=reranker,
        retriever=retriever,
        llm_client=llm_client,
        audit_logger=audit_logger,
        clock=clock,
        graph=graph,
    )

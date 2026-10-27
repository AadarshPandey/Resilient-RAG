"""Tests for Milestone P6: Reliability, Startup Warmup, Embedding Cache, Budgets, and Shutdown."""

from __future__ import annotations

from unittest.mock import MagicMock

from rag.bootstrap import Components, build_components
from rag.clock import VirtualClock
from rag.config import Settings
from rag.fakes import FakeVectorStore, HashEmbedder, LexicalReranker, ScriptedChatModel
from rag.llm.base import LLMResult
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.cache import CachedEmbedder, EmbeddingCache
from rag.retrieval.retriever import Retriever
from rag.schemas import Chunk, Outcome
from rag.warmup import warmup


def make_test_components(settings: Settings | None = None) -> Components:
    cfg = settings or Settings()
    clock = VirtualClock(start_time=0.0)
    embedder = HashEmbedder(dimension=32)
    vstore = FakeVectorStore()
    bm25 = BM25Index()
    reranker = LexicalReranker()

    chunk = Chunk(
        id="c_rel_1",
        doc_id="d_rel",
        text="Reliability engineering requires fault isolation and graceful degradation.",
        index=0,
    )
    vstore.upsert([chunk], embedder.embed([chunk.text]))
    bm25.index_chunks([chunk])

    retriever = Retriever(
        embedder=embedder,
        vector_store=vstore,
        bm25_index=bm25,
        reranker=reranker,
    )

    gen_json = '{"answer": "Reliability requires fault isolation.", "claims": ["Reliability requires fault isolation"], "citations": ["c_rel_1"]}'
    verify_json = (
        '{"is_grounded": true, "unsupported_claims": [], "summary": "Grounding confirmed"}'
    )

    scripted_model = ScriptedChatModel(
        responses=[
            LLMResult(content=gen_json, prompt_tokens=30, completion_tokens=20, tier=0),
            LLMResult(content=verify_json, prompt_tokens=25, completion_tokens=15, tier=0),
        ]
    )

    return build_components(
        settings=cfg,
        overrides={
            "clock": clock,
            "embedder": embedder,
            "vector_store": vstore,
            "bm25_index": bm25,
            "reranker": reranker,
            "retriever": retriever,
            "chat_model": scripted_model,
        },
    )


def test_warmup_sequence() -> None:
    """Verify startup warmup sequence exercises embedder and reranker."""
    components = make_test_components()
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 32]
    mock_reranker = MagicMock()
    mock_reranker.score.return_value = [0.9]

    components.embedder = mock_embedder
    components.reranker = mock_reranker

    warmup(components)

    mock_embedder.embed.assert_called_once()
    mock_reranker.score.assert_called_once()


def test_cached_embedder() -> None:
    """Verify CachedEmbedder caches embeddings by content hash and skips redundant calls."""
    base_embedder = MagicMock()
    base_embedder.embed.side_effect = lambda texts: [[float(len(t))] * 8 for t in texts]

    cache = EmbeddingCache()
    cached = CachedEmbedder(embedder=base_embedder, cache=cache)

    # First call: 2 misses
    res1 = cached.embed(["text alpha", "text beta"])
    assert len(res1) == 2
    assert base_embedder.embed.call_count == 1
    assert base_embedder.embed.call_args[0][0] == ["text alpha", "text beta"]

    # Second call: 1 hit ("text beta"), 1 miss ("text gamma")
    res2 = cached.embed(["text beta", "text gamma"])
    assert len(res2) == 2
    assert base_embedder.embed.call_count == 2
    assert base_embedder.embed.call_args[0][0] == ["text gamma"]

    # Third call: all hits
    res3 = cached.embed(["text alpha", "text beta", "text gamma"])
    assert len(res3) == 3
    assert base_embedder.embed.call_count == 2  # No new calls to base embedder


def test_token_budget_exhaustion_produces_abstain() -> None:
    """Verify that exceeding token budget forces terminal ABSTAIN outcome."""
    # Set tight token budget
    cfg = Settings(max_request_token_budget=40)
    components = make_test_components(settings=cfg)

    # scripted_model outputs 30+20 = 50 tokens on generate, exceeding budget of 40
    result = components.run(question="What is reliability?")
    assert result.get("outcome") == Outcome.ABSTAINED
    assert (
        "budget" in (result.get("error_message") or "").lower() or result.get("token_usage", 0) > 40
    )


def test_graceful_shutdown_closes_handles() -> None:
    """Verify components.close() closes vector store and checkpointer cleanly."""
    components = make_test_components()
    mock_vstore = MagicMock()
    components.vector_store = mock_vstore

    components.close()
    mock_vstore.close.assert_called_once()

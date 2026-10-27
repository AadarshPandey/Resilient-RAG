"""Tests for M5: Graph skeleton (prompts, defensive JSON, state, nodes, routing, finalize, build, bootstrap)."""

import pytest

from rag.bootstrap import build_components
from rag.clock import VirtualClock
from rag.errors import MalformedOutputError
from rag.fakes import FakeVectorStore, HashEmbedder, LexicalReranker, ScriptedChatModel
from rag.graph.parsing import parse_defensive_json
from rag.llm.base import LLMResult
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.retriever import Retriever
from rag.schemas import Chunk, FailureType, Outcome


def test_defensive_json_parser() -> None:
    """Verify parsing clean JSON, markdown code fences, and text with noisy pre/postamble."""
    # Clean JSON
    clean = '{"answer": "A kubelet is a node agent", "claims": ["node agent"], "citations": ["c1"]}'
    res1 = parse_defensive_json(clean)
    assert res1["answer"] == "A kubelet is a node agent"
    assert res1["citations"] == ["c1"]

    # Markdown fenced JSON
    fenced = """```json
    {
        "answer": "Fenced answer",
        "claims": ["Fenced claim"]
    }
    ```"""
    res2 = parse_defensive_json(fenced)
    assert res2["answer"] == "Fenced answer"

    # Conversational chat preamble and postamble
    noisy = """Sure! Here is the JSON you requested:
    {
        "answer": "Noisy extracted answer",
        "claims": ["Extract claim"]
    }
    Hope this helps your research!"""
    res3 = parse_defensive_json(noisy)
    assert res3["answer"] == "Noisy extracted answer"

    # Broken unclosed JSON raises MalformedOutputError
    broken = '{"answer": "incomplete text without closing'
    with pytest.raises(MalformedOutputError):
        parse_defensive_json(broken)


def test_graph_happy_path_with_fakes() -> None:
    """Verify happy path through graph with deterministic fakes yields ANSWERED with citations."""
    clock = VirtualClock(start_time=100.0)
    embedder = HashEmbedder(dimension=32)
    vstore = FakeVectorStore()
    bm25 = BM25Index()
    reranker = LexicalReranker()

    # Index sample corpus
    chunk = Chunk(
        id="k8s_node_kubelet",
        doc_id="doc_k8s",
        text="The kubelet is the primary node agent that registers nodes with the apiserver.",
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

    # Scripted LLM responses for generate and verify
    gen_json = (
        '{"answer": "The kubelet is the primary node agent that registers nodes.", '
        '"claims": ["The kubelet is the primary node agent"], '
        '"citations": ["k8s_node_kubelet"]}'
    )
    verify_json = (
        '{"is_grounded": true, "unsupported_claims": [], "summary": "All claims grounded"}'
    )

    scripted_model = ScriptedChatModel(
        responses=[
            LLMResult(content=gen_json, prompt_tokens=20, completion_tokens=15, tier=0),
            LLMResult(content=verify_json, prompt_tokens=25, completion_tokens=10, tier=0),
        ]
    )

    components = build_components(
        overrides={
            "clock": clock,
            "embedder": embedder,
            "vector_store": vstore,
            "bm25_index": bm25,
            "reranker": reranker,
            "retriever": retriever,
            "chat_model": scripted_model,
        }
    )

    result_state = components.run(question="What is the role of kubelet on worker nodes?")

    assert result_state["outcome"] == Outcome.ANSWERED
    assert "kubelet" in result_state["answer"]
    assert len(result_state["citations"]) > 0
    assert result_state["citations"][0].chunk_id == "k8s_node_kubelet"
    assert len(result_state["trace"]) >= 5


def test_graph_low_relevance_routes_to_failure() -> None:
    """Verify that when retrieved chunks lack relevance, grade node emits LOW_RELEVANCE."""
    clock = VirtualClock(start_time=0.0)
    embedder = HashEmbedder(dimension=32)
    vstore = FakeVectorStore()
    bm25 = BM25Index()
    reranker = LexicalReranker()

    chunk = Chunk(
        id="c1",
        doc_id="d1",
        text="Kubernetes storage CSI driver volume attachments.",
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

    components = build_components(
        overrides={
            "clock": clock,
            "embedder": embedder,
            "vector_store": vstore,
            "bm25_index": bm25,
            "reranker": reranker,
            "retriever": retriever,
            "chat_model": ScriptedChatModel(),
        }
    )

    # Completely disjoint query with 0 lexical overlap
    result_state = components.run(
        question="Photosynthesis chlorophyll light reactions",
        thread_id="test_low_rel",
    )

    assert result_state.get("last_failure") == FailureType.LOW_RELEVANCE
    assert result_state["outcome"] in (Outcome.FAILED, Outcome.ABSTAINED, Outcome.ESCALATED)

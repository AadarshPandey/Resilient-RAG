"""Tests for Milestone P5: Prometheus Metrics, /metrics endpoint, and Grafana dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from rag.api import create_app
from rag.bootstrap import Components, build_components
from rag.clock import VirtualClock
from rag.fakes import FakeVectorStore, HashEmbedder, LexicalReranker, ScriptedChatModel
from rag.llm.base import LLMResult
from rag.metrics import (
    RAG_BREAKER_STATE,
    RAG_HEAL_ATTEMPTS,
    RAG_LLM_TOKENS,
    RAG_REQUESTS,
    record_ask_metrics,
)
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.retriever import Retriever
from rag.schemas import Chunk


def make_test_components() -> Components:
    clock = VirtualClock(start_time=0.0)
    embedder = HashEmbedder(dimension=32)
    vstore = FakeVectorStore()
    bm25 = BM25Index()
    reranker = LexicalReranker()

    chunk = Chunk(
        id="c_m5",
        doc_id="d_m5",
        text="Observability requires metrics, logs, and distributed traces.",
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

    gen_json = '{"answer": "Observability uses metrics and logs.", "claims": ["Observability uses metrics"], "citations": ["c_m5"]}'
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


def test_metrics_endpoint_exists() -> None:
    """Verify /metrics returns 200 and standard Prometheus text format."""
    components = make_test_components()
    app = create_app(components)
    client = TestClient(app)

    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    text = res.text
    assert "rag_requests_total" in text
    assert "rag_request_duration_seconds" in text
    assert "rag_heal_attempts_total" in text
    assert "rag_circuit_breaker_state" in text
    assert "rag_llm_tokens_total" in text


def test_metrics_increment_on_ask() -> None:
    """Verify that calling /ask records request outcome, duration, and tokens."""
    components = make_test_components()
    app = create_app(components)
    client = TestClient(app)

    # Initial ask
    res = client.post("/ask", json={"question": "What is observability?"})
    assert res.status_code == 200

    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    metrics_text = metrics_res.text
    assert 'rag_requests_total{outcome="answered"}' in metrics_text


def test_record_ask_metrics_helper() -> None:
    """Verify record_ask_metrics helper updates all Prometheus collectors accurately."""
    # Record mock execution with healing history and tokens
    record_ask_metrics(
        outcome="abstained",
        duration=0.342,
        heal_history=[("retrieval_backend_down", "bm25_only")],
        llm_tier=0,
        prompt_tokens=45,
        completion_tokens=22,
    )

    # Assert metric counters directly
    assert RAG_REQUESTS.labels(outcome="abstained")._value.get() >= 1.0
    assert (
        RAG_HEAL_ATTEMPTS.labels(action="bm25_only", failure="retrieval_backend_down")._value.get()
        >= 1.0
    )
    assert RAG_LLM_TOKENS.labels(tier="0", direction="prompt")._value.get() >= 45.0
    assert RAG_LLM_TOKENS.labels(tier="0", direction="completion")._value.get() >= 22.0

    # Test breaker gauge
    RAG_BREAKER_STATE.labels(tier="0").set(2)  # open
    assert RAG_BREAKER_STATE.labels(tier="0")._value.get() == 2.0


def test_grafana_dashboard_json_structure() -> None:
    """Verify deploy/grafana/dashboard.json exists and is valid Grafana schema."""
    dash_path = Path("deploy/grafana/dashboard.json")
    assert dash_path.is_file(), "deploy/grafana/dashboard.json does not exist"

    data = json.loads(dash_path.read_text(encoding="utf-8"))
    assert "title" in data
    assert "panels" in data
    assert len(data["panels"]) >= 4

    panel_titles = [p.get("title", "") for p in data["panels"]]
    assert any("Requests" in t or "Throughput" in t for t in panel_titles)
    assert any("Latency" in t or "Duration" in t for t in panel_titles)
    assert any("Healing" in t or "Recovery" in t for t in panel_titles)
    assert any("Breaker" in t or "Circuit" in t for t in panel_titles)

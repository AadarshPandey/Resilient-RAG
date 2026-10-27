"""Tests for M12: CLI, API, Streamlit and Plotting."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from typer.testing import CliRunner

import app as streamlit_app
from rag.api import create_app
from rag.bootstrap import Components, build_components
from rag.cli import app as cli_app
from rag.clock import VirtualClock
from rag.fakes import FakeVectorStore, HashEmbedder, LexicalReranker, ScriptedChatModel
from rag.llm.base import LLMResult
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.retriever import Retriever
from rag.schemas import Chunk, Outcome, RunRecord
from scripts.plot_results import plot_scorecard


def make_test_components() -> Components:
    """Create reproducible test components with fake backends."""
    clock = VirtualClock(start_time=0.0)
    embedder = HashEmbedder(dimension=32)
    vstore = FakeVectorStore()
    bm25 = BM25Index()
    reranker = LexicalReranker()

    chunk = Chunk(
        id="c_k8s_pod",
        doc_id="d1",
        text="Pods are the smallest deployable units of computing in Kubernetes.",
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

    gen_json = (
        '{"answer": "Pods are the smallest deployable units in Kubernetes.", '
        '"claims": ["Pods are smallest deployable units"], '
        '"citations": ["c_k8s_pod"]}'
    )
    verify_json = (
        '{"is_grounded": true, "unsupported_claims": [], "summary": "Grounding confirmed"}'
    )

    scripted_model = ScriptedChatModel(
        responses=[
            LLMResult(content=gen_json, prompt_tokens=20, completion_tokens=15, tier=0),
            LLMResult(content=verify_json, prompt_tokens=25, completion_tokens=10, tier=0),
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


def test_api_health_endpoint() -> None:
    test_comp = make_test_components()
    app = create_app(test_comp)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_api_ask_endpoint() -> None:
    test_comp = make_test_components()
    app = create_app(test_comp)
    client = TestClient(app)

    payload = {
        "question": "What is a pod in Kubernetes?",
        "healing_enabled": True,
        "retrieval_mode": "hybrid",
        "top_k": 3,
    }

    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["outcome"] == "answered"
    assert "Pods are the smallest" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["chunk_id"] == "c_k8s_pod"
    assert "healing_report" in data
    assert len(data["trace"]) > 0


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "ask" in result.output
    assert "chaos" in result.output
    assert "scorecard" in result.output
    assert "eval" in result.output
    assert "langsmith-upload-dataset" in result.output


def test_cli_chaos_dry_run() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        [
            "chaos",
            "--dry-run",
            "--conditions",
            "A,B",
            "--seeds",
            "42",
            "--scenario",
            "llm_rate_limit_transient",
        ],
    )
    assert result.exit_code == 0
    assert "Chaos Experiment Configuration:" in result.output
    assert "[Dry run] Exiting without executing runs." in result.output


def test_cli_scorecard(tmp_path: Path) -> None:
    runner = CliRunner()

    runs_file = tmp_path / "test_runs.jsonl"
    md_file = tmp_path / "test_scorecard.md"
    csv_file = tmp_path / "test_scorecard.csv"

    r1 = RunRecord(
        question="q1",
        outcome=Outcome.ANSWERED,
        condition="A",
        scenario="none",
        seed=1,
    )
    runs_file.write_text(json.dumps(r1.model_dump()) + "\n")

    result = runner.invoke(
        cli_app,
        [
            "scorecard",
            "--input",
            str(runs_file),
            "--output-md",
            str(md_file),
            "--output-csv",
            str(csv_file),
        ],
    )
    assert result.exit_code == 0
    assert md_file.is_file()
    assert csv_file.is_file()
    assert "Resilient RAG Chaos Scorecard" in md_file.read_text()


def test_cli_ask() -> None:
    runner = CliRunner()
    test_comp = make_test_components()

    with patch("rag.cli.build_components", return_value=test_comp):
        result = runner.invoke(cli_app, ["ask", "What are pods in Kubernetes?"])
        assert result.exit_code == 0
        assert "Outcome: answered" in result.output
        assert "Pods are the smallest" in result.output


def test_streamlit_app_import() -> None:
    """Verify Streamlit app imports cleanly and helper functions execute."""
    scenarios = streamlit_app.get_available_scenarios()
    assert "None" in scenarios
    assert len(scenarios) > 1


def test_plot_results(tmp_path: Path) -> None:
    """Verify scorecard plotting script runs and generates PNG output."""
    csv_file = tmp_path / "scorecard.csv"
    csv_file.write_text(
        "condition,total_runs,task_success_rate,task_success_ci_low,task_success_ci_high,safe_outcome_rate,safe_outcome_ci_low,safe_outcome_ci_high,recovery_rate,mttr\n"
        "A,50,0.92,0.85,0.97,0.94,0.88,0.98,1.0,0.0\n"
        "B,50,0.42,0.31,0.54,0.48,0.36,0.60,0.35,1.2\n"
        "C,50,0.88,0.79,0.94,0.92,0.84,0.96,0.85,0.8\n"
    )
    png_file = tmp_path / "scorecard.png"
    plot_scorecard(csv_file, png_file)
    assert png_file.is_file()
    assert png_file.stat().st_size > 1000

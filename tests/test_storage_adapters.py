"""Tests for Phase 2 Milestone P2: Storage Adapters."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from rag.errors import BackendError
from rag.graph.checkpointer import create_checkpointer, setup_postgres_checkpointer
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.sparse import SparseIndex
from rag.retrieval.vector import QdrantVectorStore
from rag.schemas import Chunk


def test_checkpointer_factory_memory() -> None:
    """Verify checkpointer factory returns MemorySaver when dsn is memory or None."""
    cp_none = create_checkpointer(None)
    assert isinstance(cp_none, MemorySaver)

    cp_mem = create_checkpointer(":memory:")
    assert isinstance(cp_mem, MemorySaver)


def test_checkpointer_factory_sqlite(tmp_path: Path) -> None:
    """Verify checkpointer factory creates and sets up SqliteSaver for sqlite:/// URLs."""
    db_file = tmp_path / "checkpoints.db"
    dsn = f"sqlite:///{db_file}"

    cp = create_checkpointer(dsn)
    assert isinstance(cp, SqliteSaver)
    assert db_file.is_file()

    # Verify tables were created by setup()
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    assert "checkpoints" in tables
    conn.close()


def test_checkpointer_factory_postgres_mocked() -> None:
    """Verify checkpointer factory handles PostgreSQL DSNs with PostgresSaver."""
    dsn = "postgresql://user:pass@localhost:5432/rag"

    mock_saver = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_saver

    with patch(
        "langgraph.checkpoint.postgres.PostgresSaver.from_conn_string", return_value=mock_cm
    ):
        saver = create_checkpointer(dsn)
        assert saver == mock_saver
        mock_saver.setup.assert_called_once()


def test_checkpointer_factory_unsupported() -> None:
    """Verify checkpointer factory rejects unsupported protocols."""
    with pytest.raises(ValueError, match="Unsupported checkpointer DSN scheme"):
        create_checkpointer("mongodb://localhost:27017")


def test_setup_postgres_checkpointer_runner() -> None:
    """Verify standalone setup_postgres_checkpointer migration helper."""
    dsn = "postgresql://user:pass@localhost:5432/rag"
    mock_saver = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_saver

    with patch(
        "langgraph.checkpoint.postgres.PostgresSaver.from_conn_string", return_value=mock_cm
    ):
        setup_postgres_checkpointer(dsn)
        mock_saver.setup.assert_called_once()


def test_qdrant_vector_store_remote_url_error_mapping() -> None:
    """Verify QdrantVectorStore handles invalid remote URLs by raising BackendError."""
    with pytest.raises(BackendError, match="Failed to initialize Qdrant vector store"):
        # Port 9 is discard protocol, immediate refusal without hanging
        QdrantVectorStore(url="http://127.0.0.1:9")


def test_sparse_index_protocol_compliance() -> None:
    """Verify BM25Index conforms to SparseIndex protocol."""
    bm25 = BM25Index()
    chunks = [
        Chunk(id="c1", text="Kubernetes pods run containers", doc_id="d1", index=0),
        Chunk(id="c2", text="Services route network traffic to pods", doc_id="d2", index=0),
    ]
    bm25.index(chunks)

    # Check protocol adherence
    assert isinstance(bm25, SparseIndex)
    assert bm25.count() == 2

    results = bm25.search("network traffic", top_k=1)
    assert len(results) == 1
    assert results[0][0] == "c2"

    chunk_results = bm25.search_chunks("network traffic", top_k=1)
    assert len(chunk_results) == 1
    assert chunk_results[0].chunk.id == "c2"

    bm25.close()

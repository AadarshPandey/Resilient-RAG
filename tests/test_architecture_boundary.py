"""Tests for M13: Architectural boundary purity and resource cleanup."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock

from rag.bootstrap import build_components
from rag.fakes import HashEmbedder, LexicalReranker, ScriptedChatModel
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.retriever import Retriever
from rag.retrieval.vector import QdrantVectorStore


def test_healing_pure_python_boundary() -> None:
    """Assert that importing rag.healing never imports torch, langchain, langgraph, or qdrant_client."""
    code = (
        "import sys\n"
        "import rag.healing\n"
        "banned = {'torch', 'langchain', 'langgraph', 'qdrant_client'}\n"
        "loaded = banned.intersection(sys.modules)\n"
        "if loaded:\n"
        "    print(f'VIOLATION: {loaded}')\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )

    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"Boundary violated: {proc.stdout} {proc.stderr}"


def test_components_close_releases_resources() -> None:
    """Verify components.close() forwards close to underlying storage handles."""
    mock_store = MagicMock()
    components = build_components(
        overrides={
            "vector_store": mock_store,
            "embedder": HashEmbedder(dimension=8),
            "bm25_index": BM25Index(),
            "reranker": LexicalReranker(),
            "retriever": MagicMock(spec=Retriever),
            "chat_model": ScriptedChatModel(),
        }
    )

    components.close()
    mock_store.close.assert_called_once()


def test_qdrant_vector_store_close_in_memory() -> None:
    """Verify in-memory QdrantVectorStore closes gracefully."""
    store = QdrantVectorStore(location=":memory:", collection_name="test_cleanup")
    assert store.count() == 0
    store.close()

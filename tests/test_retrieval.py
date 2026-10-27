"""Tests for M3: Retrieval (RRF fusion, reranker, and Retriever facade)."""

from unittest.mock import MagicMock, patch

import pytest

from rag.errors import BackendError
from rag.fakes import FakeVectorStore, HashEmbedder, LexicalReranker
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.hybrid import reciprocal_rank_fusion
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.retriever import Retriever
from rag.schemas import Chunk, RetrievalMode, RetrievedChunk


def make_chunk(cid: str, text: str) -> Chunk:
    """Helper to create a test Chunk."""
    return Chunk(id=cid, doc_id="d1", text=text, index=0)


def test_rrf_math_hand_computed() -> None:
    """Verify RRF formula 1 / (60 + rank) and deterministic tie-breaker by chunk ID."""
    # List 1 ranks: c1 (rank 1), c2 (rank 2)
    # List 2 ranks: c2 (rank 1), c3 (rank 2)
    list1 = [
        RetrievedChunk(chunk=make_chunk("c1", "text 1"), score=0.9, source="vector"),
        RetrievedChunk(chunk=make_chunk("c2", "text 2"), score=0.8, source="vector"),
    ]
    list2 = [
        RetrievedChunk(chunk=make_chunk("c2", "text 2"), score=15.0, source="bm25"),
        RetrievedChunk(chunk=make_chunk("c3", "text 3"), score=10.0, source="bm25"),
    ]

    # k = 60
    # c1: 1/(60+1) = 1/61 =~ 0.016393
    # c2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 =~ 0.016129 + 0.016393 = 0.032522
    # c3: 1/(60+2) = 1/62 =~ 0.016129
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(fused) == 3
    assert fused[0].chunk.id == "c2"  # c2 has highest combined score
    assert pytest.approx(fused[0].score, rel=1e-3) == (1 / 62 + 1 / 61)

    # Test deterministic tie-breaking: identical scores broken by chunk ID
    tie_list1 = [RetrievedChunk(chunk=make_chunk("chunk_b", "b"), score=1.0, source="s1")]
    tie_list2 = [RetrievedChunk(chunk=make_chunk("chunk_a", "a"), score=1.0, source="s2")]
    fused_tie = reciprocal_rank_fusion([tie_list1, tie_list2], k=60)
    assert fused_tie[0].chunk.id == "chunk_a"
    assert fused_tie[1].chunk.id == "chunk_b"


def test_hybrid_beats_single_mode() -> None:
    """A chunk ranked well by both vector and BM25 beats a chunk ranked first by only one."""
    vector_results = [
        RetrievedChunk(chunk=make_chunk("v_only", "dense only match"), score=0.99, source="vector"),
        RetrievedChunk(chunk=make_chunk("both", "matched by both"), score=0.95, source="vector"),
    ]
    bm25_results = [
        RetrievedChunk(
            chunk=make_chunk("bm25_only", "keyword only match"), score=12.0, source="bm25"
        ),
        RetrievedChunk(chunk=make_chunk("both", "matched by both"), score=11.5, source="bm25"),
    ]

    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=60)
    assert fused[0].chunk.id == "both"


class BrokenVectorStore:
    """Mock vector store that raises an error simulating an outage."""

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        raise RuntimeError("Vector store connection refused: 6333")


class BrokenBM25Index:
    """Mock BM25 index that raises an error simulating corruption or missing index."""

    def search_chunks(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        raise ValueError("BM25 index corrupted")


def test_retriever_hybrid_fails_loudly_on_backend_error() -> None:
    """HYBRID mode must raise BackendError if either backend fails (no silent fallback)."""
    embedder = HashEmbedder(dimension=32)
    reranker = LexicalReranker()

    # Broken vector backend
    retriever_broken_vector = Retriever(
        embedder=embedder,
        vector_store=BrokenVectorStore(),  # type: ignore[arg-type]
        bm25_index=BM25Index(),
        reranker=reranker,
    )
    with pytest.raises(BackendError, match="Vector backend failed"):
        retriever_broken_vector.retrieve("query", mode=RetrievalMode.HYBRID)

    # Broken BM25 backend
    retriever_broken_bm25 = Retriever(
        embedder=embedder,
        vector_store=FakeVectorStore(),
        bm25_index=BrokenBM25Index(),  # type: ignore[arg-type]
        reranker=reranker,
    )
    with pytest.raises(BackendError, match="BM25 backend failed"):
        retriever_broken_bm25.retrieve("query", mode=RetrievalMode.HYBRID)


def test_retriever_single_modes_work_when_other_broken() -> None:
    """Single modes work when the unused backend is broken."""
    embedder = HashEmbedder(dimension=32)
    reranker = LexicalReranker()

    # Working BM25, broken vector
    bm25 = BM25Index()
    test_chunk = make_chunk("c1", "kubelet node agent")
    bm25.index_chunks([test_chunk])

    retriever1 = Retriever(
        embedder=embedder,
        vector_store=BrokenVectorStore(),  # type: ignore[arg-type]
        bm25_index=bm25,
        reranker=reranker,
    )
    res_bm25 = retriever1.retrieve("kubelet", mode=RetrievalMode.BM25_ONLY, top_k=1)
    assert len(res_bm25) == 1
    assert res_bm25[0].chunk.id == "c1"

    # Working vector, broken BM25
    vstore = FakeVectorStore()
    vstore.upsert([test_chunk], embedder.embed([test_chunk.text]))
    retriever2 = Retriever(
        embedder=embedder,
        vector_store=vstore,
        bm25_index=BrokenBM25Index(),  # type: ignore[arg-type]
        reranker=reranker,
    )
    res_vector = retriever2.retrieve("kubelet", mode=RetrievalMode.VECTOR_ONLY, top_k=1)
    assert len(res_vector) == 1
    assert res_vector[0].chunk.id == "c1"


def test_lexical_reranker() -> None:
    """Verify LexicalReranker computes term overlap scores."""
    reranker = LexicalReranker()
    scores = reranker.score(
        "kubelet node", ["kubelet is a node agent", "etcd storage", "unrelated text"]
    )
    assert scores[0] > scores[1]
    assert scores[1] == 0.0


def test_cross_encoder_reranker_mocked() -> None:
    """Verify CrossEncoderReranker delegates to model predict."""
    with patch("sentence_transformers.CrossEncoder") as mock_ce_cls:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = [0.95, 0.12]
        mock_ce_cls.return_value = mock_instance

        reranker = CrossEncoderReranker(device="cpu")
        scores = reranker.score("query", ["doc1", "doc2"])
        assert scores == [0.95, 0.12]

        empty_scores = reranker.score("query", [])
        assert empty_scores == []

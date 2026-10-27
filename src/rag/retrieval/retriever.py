"""Retriever facade supporting HYBRID, VECTOR_ONLY, and BM25_ONLY modes."""

from rag.errors import BackendError
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.embedder import Embedder
from rag.retrieval.hybrid import reciprocal_rank_fusion
from rag.retrieval.reranker import Reranker
from rag.retrieval.vector import VectorStore
from rag.schemas import RetrievalMode, RetrievedChunk


class Retriever:
    """Unified retrieval facade with explicit fail-loud semantics in HYBRID mode."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.reranker = reranker

    def _search_vector(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Search vector store with query embedding."""
        try:
            query_vec = self.embedder.embed([query])[0]
            return self.vector_store.search(query_vec, top_k=top_k)
        except Exception as e:
            raise BackendError(f"Vector backend failed: {e}") from e

    def _search_bm25(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Search BM25 keyword index."""
        try:
            return self.bm25_index.search_chunks(query, top_k=top_k)
        except Exception as e:
            raise BackendError(f"BM25 backend failed: {e}") from e

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Retrieve candidate chunks using specified mode.

        In HYBRID mode, failure of either backend raises BackendError (no silent fallback).
        In single-backend modes (VECTOR_ONLY, BM25_ONLY), only that backend is queried.

        Args:
            query: User or rewritten query string.
            mode: Operational retrieval mode.
            top_k: Number of final chunks to return.
            candidate_k: Number of candidate chunks retrieved from each index before reranking.

        Returns:
            List of top_k RetrievedChunk objects.
        """
        candidates: list[RetrievedChunk] = []

        if mode == RetrievalMode.HYBRID:
            # Query both. If either raises, let BackendError bubble up.
            vector_hits = self._search_vector(query, top_k=candidate_k)
            bm25_hits = self._search_bm25(query, top_k=candidate_k)
            candidates = reciprocal_rank_fusion([vector_hits, bm25_hits])
        elif mode == RetrievalMode.VECTOR_ONLY:
            candidates = self._search_vector(query, top_k=candidate_k)
        elif mode == RetrievalMode.BM25_ONLY:
            candidates = self._search_bm25(query, top_k=candidate_k)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        if not candidates:
            return []

        # Apply reranker if configured
        if self.reranker is not None:
            passages = [c.chunk.text for c in candidates]
            rerank_scores = self.reranker.score(query, passages)
            reranked_pairs = list(zip(candidates, rerank_scores, strict=True))
            # Sort descending by cross-encoder / reranker score
            reranked_pairs.sort(key=lambda x: x[1], reverse=True)
            results: list[RetrievedChunk] = []
            for item, score in reranked_pairs[:top_k]:
                results.append(
                    RetrievedChunk(
                        chunk=item.chunk,
                        score=score,
                        source=f"{item.source}_reranked",
                    )
                )
            return results

        return candidates[:top_k]

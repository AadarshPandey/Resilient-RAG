"""Deterministic fakes for offline, zero-cost testing across all components."""

import hashlib
import math
import re
from typing import Any

from rag.llm.base import LLMResult
from rag.schemas import Chunk, RetrievedChunk


class HashEmbedder:
    """Deterministic hash-based embedding model producing normalized unit vectors.

    Produces fixed-dimension float vectors derived from MD5/SHA256 hashes of text.
    Zero network access, runs in sub-millisecond, completely reproducible.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into normalized float vectors."""
        results: list[list[float]] = []
        for text in texts:
            vec: list[float] = []
            normalized_text = text.lower().strip()
            # Generate deterministic floats across dimensions
            for i in range(self.dimension):
                seed_str = f"{normalized_text}:{i}"
                digest = hashlib.sha256(seed_str.encode("utf-8")).digest()
                # Unpack an unsigned 32-bit int into a float in [-1.0, 1.0]
                int_val = int.from_bytes(digest[:4], "big")
                float_val = (int_val / 2147483647.0) - 1.0
                vec.append(float_val)

            # L2 normalize
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            results.append(vec)
        return results


class LexicalReranker:
    """Deterministic lexical overlap reranker for testing."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score query against passages based on normalized term overlap."""
        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return [0.0] * len(passages)

        scores: list[float] = []
        for p in passages:
            passage_terms = set(re.findall(r"\w+", p.lower()))
            overlap = len(query_terms.intersection(passage_terms))
            scores.append(overlap / float(len(query_terms)))
        return scores


class FakeVectorStore:
    """In-memory vector store implementing cosine similarity search for testing."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update chunks and their embeddings."""
        for chunk, vec in zip(chunks, vectors, strict=True):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = vec

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Perform cosine similarity search against stored vectors."""
        scored: list[tuple[float, Chunk]] = []
        for chunk_id, vec in self._vectors.items():
            # Dot product (assuming normalized vectors)
            sim = sum(a * b for a, b in zip(query_vector, vec, strict=False))
            scored.append((sim, self._chunks[chunk_id]))

        # Sort descending by score
        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievedChunk] = []
        for sim, chunk in scored[:top_k]:
            results.append(RetrievedChunk(chunk=chunk, score=float(sim), source="vector"))
        return results

    def count(self) -> int:
        """Return total stored chunks."""
        return len(self._chunks)


class ScriptedChatModel:
    """Fake chat model that yields scripted responses or raises scripted errors."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self._responses: list[Any] = list(responses) if responses else []
        self._call_count: int = 0
        self.recorded_calls: list[dict[str, Any]] = []

    def set_responses(self, responses: list[Any]) -> None:
        """Replace scripted responses queue."""
        self._responses = list(responses)

    def invoke(self, messages: list[Any], *, tier: int = 0) -> Any:
        """Invoke scripted response or raise scripted error."""
        self._call_count += 1
        self.recorded_calls.append({"messages": messages, "tier": tier})

        if not self._responses:
            # Default response
            return LLMResult(
                content='{"answer": "Scripted response", "claims": ["Scripted claim"], "citations": []}',
                prompt_tokens=10,
                completion_tokens=10,
                tier=tier,
            )

        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

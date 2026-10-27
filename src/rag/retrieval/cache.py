"""In-memory SHA-256 embedding cache and CachedEmbedder decorator."""

from __future__ import annotations

import hashlib
import threading

from rag.retrieval.embedder import Embedder


class EmbeddingCache:
    """Thread-safe in-memory cache for embeddings keyed by text SHA-256 digest."""

    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = max_size
        self._lock = threading.Lock()
        self._cache: dict[str, list[float]] = {}

    @staticmethod
    def hash_text(text: str) -> str:
        """Compute stable SHA-256 digest for input text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> list[float] | None:
        """Retrieve embedding vector for text if cached, else None."""
        key = self.hash_text(text)
        with self._lock:
            return self._cache.get(key)

    def set(self, text: str, vector: list[float]) -> None:
        """Store embedding vector for text in cache with simple eviction if full."""
        key = self.hash_text(text)
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Evict oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = vector

    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()


class CachedEmbedder:
    """Embedder decorator that transparently caches vectors by text SHA-256 hash."""

    def __init__(self, embedder: Embedder, cache: EmbeddingCache | None = None) -> None:
        self.embedder = embedder
        self.cache = cache if cache is not None else EmbeddingCache()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings, utilizing cached vectors for any previously seen texts."""
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for idx, text in enumerate(texts):
            cached_vector = self.cache.get(text)
            if cached_vector is not None:
                results[idx] = cached_vector
            else:
                miss_indices.append(idx)
                miss_texts.append(text)

        if miss_texts:
            computed = self.embedder.embed(miss_texts)
            for idx, text, vec in zip(miss_indices, miss_texts, computed, strict=True):
                self.cache.set(text, vec)
                results[idx] = vec

        return [v for v in results if v is not None]

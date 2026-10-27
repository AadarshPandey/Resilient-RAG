"""Embedding models with PyTorch runtime integration."""

from typing import Protocol

import torch

from rag.device import resolve_device


class Embedder(Protocol):
    """Protocol for embedding components."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings into float vectors."""
        ...


class SentenceTransformerEmbedder:
    """Production embedder wrapping sentence-transformers with PyTorch."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.resolved_device = resolve_device(device)
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.resolved_device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute normalized vector embeddings for a list of strings."""
        if not texts:
            return []

        with torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return embeddings.tolist()  # type: ignore[no-any-return]

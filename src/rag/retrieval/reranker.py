"""Cross-encoder reranking models using PyTorch."""

from typing import Protocol

import torch

from rag.device import resolve_device


class Reranker(Protocol):
    """Protocol for candidate rerankers."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score passages against query."""
        ...


class CrossEncoderReranker:
    """Production cross-encoder reranker wrapping sentence-transformers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.resolved_device = resolve_device(device)
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=self.resolved_device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Compute cross-encoder relevance scores for query and passages."""
        if not passages:
            return []

        pairs = [[query, p] for p in passages]
        with torch.inference_mode():
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

        # Ensure python floats
        return [float(s) for s in scores]

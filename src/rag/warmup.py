"""Startup warm-up sequence for deep learning embedding and reranking models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.bootstrap import Components

logger = logging.getLogger(__name__)


def warmup(components: Components) -> None:
    """Pre-warm neural models (embedder and reranker) on startup with dummy inputs."""
    logger.info("Executing startup warm-up sequence...")

    # Warm up embedding model
    try:
        dummy_embeddings = components.embedder.embed(["resilient rag warmup query"])
        logger.debug("Embedder warmed up (%d vectors generated)", len(dummy_embeddings))
    except Exception as exc:
        logger.warning("Embedder warmup failed: %s", exc)

    # Warm up cross-encoder reranker
    try:
        dummy_scores = components.reranker.score(
            query="warmup query",
            passages=["resilient rag warmup passage for cross encoder latency reduction"],
        )
        logger.debug("Reranker warmed up (%d scores computed)", len(dummy_scores))
    except Exception as exc:
        logger.warning("Reranker warmup failed: %s", exc)

    logger.info("Startup warm-up sequence complete.")

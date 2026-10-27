"""Reciprocal Rank Fusion (RRF) for hybrid retrieval."""

from collections import defaultdict

from rag.schemas import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Formula: score(d) = sum_{m in M} (1 / (k + rank_{m}(d)))
    where rank is 1-indexed.

    Deterministic tie-breaking:
    Ties in fused scores are sorted alphabetically by chunk ID.

    Args:
        ranked_lists: List of ranked RetrievedChunk lists from different retrievers.
        k: Smoothing constant (default: 60).

    Returns:
        A list of fused RetrievedChunk objects sorted descending by score.
    """
    if not ranked_lists:
        return []

    fused_scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, Chunk] = {}

    for r_list in ranked_lists:
        for rank_0, item in enumerate(r_list):
            chunk_id = item.chunk.id
            chunk_map[chunk_id] = item.chunk
            rank_1 = rank_0 + 1  # 1-indexed rank
            fused_scores[chunk_id] += 1.0 / (k + rank_1)

    # Sort primarily by score descending, secondarily by chunk_id ascending (deterministic tie-breaker)
    sorted_items = sorted(
        fused_scores.items(),
        key=lambda entry: (-entry[1], entry[0]),
    )

    results: list[RetrievedChunk] = []
    for chunk_id, score in sorted_items:
        results.append(
            RetrievedChunk(
                chunk=chunk_map[chunk_id],
                score=score,
                source="rrf_hybrid",
            )
        )

    return results

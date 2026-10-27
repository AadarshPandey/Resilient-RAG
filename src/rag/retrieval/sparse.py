"""Protocol for sparse and lexical retrieval indexes."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class SparseIndex(Protocol):
    """Protocol for lexical and sparse vector retrieval indexes."""

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search nearest chunk IDs and scores."""
        ...

    def count(self) -> int:
        """Return total chunks count in index."""
        ...

    def close(self) -> None:
        """Close index and release resources."""
        ...

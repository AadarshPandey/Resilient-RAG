"""Vector store integration using Qdrant."""

from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.errors import BackendError
from rag.schemas import Chunk, RetrievedChunk


class VectorStore(Protocol):
    """Protocol for vector storage backends."""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update chunks and vectors."""
        ...

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Search nearest chunks for query vector."""
        ...

    def count(self) -> int:
        """Return total chunks count."""
        ...


class QdrantVectorStore:
    """Production vector store implementation backed by Qdrant (local or remote)."""

    def __init__(
        self,
        location: str = ":memory:",
        collection_name: str = "chunks",
        vector_size: int = 384,
        url: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        try:
            if url:
                self.client = QdrantClient(url=url)
            elif location == ":memory:":
                self.client = QdrantClient(location=":memory:")
            else:
                self.client = QdrantClient(path=location)

            self._ensure_collection()
        except Exception as e:
            raise BackendError(f"Failed to initialize Qdrant vector store: {e}") from e

    def _ensure_collection(self) -> None:
        """Create collection if it does not already exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
        except Exception as e:
            raise BackendError(f"Failed to ensure Qdrant collection: {e}") from e

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upsert chunks with vectors into Qdrant collection."""
        if not chunks:
            return

        try:
            points: list[PointStruct] = []
            for idx, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True)):
                payload = {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "index": chunk.index,
                    "metadata": chunk.metadata,
                }
                points.append(
                    PointStruct(
                        id=idx,  # integer or uuid ID for Qdrant point
                        vector=vec,
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        except Exception as e:
            raise BackendError(f"Failed to upsert points to Qdrant: {e}") from e

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Perform cosine similarity search in Qdrant."""
        try:
            # qdrant-client 1.x supports query_points
            query_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            hits = query_result.points

            results: list[RetrievedChunk] = []
            for hit in hits:
                payload = hit.payload or {}
                chunk = Chunk(
                    id=str(payload.get("chunk_id", str(hit.id))),
                    doc_id=str(payload.get("doc_id", "")),
                    text=str(payload.get("text", "")),
                    index=int(payload.get("index", 0)),
                    metadata=dict(payload.get("metadata", {})),
                )
                score = float(hit.score) if hit.score is not None else 0.0
                results.append(RetrievedChunk(chunk=chunk, score=score, source="vector"))

            return results
        except Exception as e:
            raise BackendError(f"Failed to search Qdrant vector store: {e}") from e

    def count(self) -> int:
        """Count total vectors in collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception as e:
            raise BackendError(f"Failed to get collection count from Qdrant: {e}") from e

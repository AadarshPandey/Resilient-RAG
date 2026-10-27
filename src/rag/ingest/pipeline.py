"""Ingestion pipeline with concurrent processing, manifest caching, and tenacity retries."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from rag.fakes import FakeVectorStore
from rag.ingest.chunking import chunk_document
from rag.ingest.loaders import load_file
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.embedder import Embedder
from rag.retrieval.vector import QdrantVectorStore, VectorStore
from rag.schemas import Chunk, Document


class IngestionResult(BaseModel):
    """Summary of an ingestion run."""

    indexed_files: int
    skipped_files: int
    total_chunks: int


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 digest of file content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class IngestionPipeline:
    """Ingestion pipeline coordinating loading, chunking, manifest caching, and indexing."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore | QdrantVectorStore | FakeVectorStore,
        bm25_index: BM25Index,
        manifest_path: str | Path = "data/manifest.json",
        max_workers: int = 4,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.manifest_path = Path(manifest_path)
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        """Load manifest from disk if present."""
        if self.manifest_path.is_file():
            try:
                with open(self.manifest_path, encoding="utf-8") as f:
                    return json.load(f)  # type: ignore[no-any-return]
            except Exception:
                return {}
        return {}

    def _save_manifest(self) -> None:
        """Persist manifest to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)

    def _load_and_chunk(self, path: Path) -> tuple[Document, list[Chunk]]:
        """Worker task to load and chunk a single file."""
        doc = load_file(path)
        chunks = chunk_document(doc, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return doc, chunks

    @retry(
        wait=wait_exponential(multiplier=0.1, min=0.1, max=1.0),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _embed_chunks_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings with exponential backoff retry."""
        return self.embedder.embed(texts)

    def run(self, input_dir: str | Path) -> IngestionResult:
        """Execute ingestion on all supported files in input_dir."""
        directory = Path(input_dir)
        if not directory.is_dir():
            return IngestionResult(indexed_files=0, skipped_files=0, total_chunks=0)

        # Collect files
        supported_exts = {".md", ".txt", ".html", ".htm", ".pdf"}
        all_files = [
            p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in supported_exts
        ]

        files_to_process: list[Path] = []
        file_hashes: dict[str, str] = {}
        skipped_count = 0

        for f_path in all_files:
            rel_key = str(f_path.resolve())
            current_hash = compute_file_sha256(f_path)
            file_hashes[rel_key] = current_hash

            if rel_key in self._manifest and self._manifest[rel_key].get("sha256") == current_hash:
                skipped_count += 1
            else:
                files_to_process.append(f_path)

        if not files_to_process:
            return IngestionResult(
                indexed_files=0,
                skipped_files=skipped_count,
                total_chunks=len(self.bm25_index.doc_ids),
            )

        # Concurrently load and chunk changed files
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            loaded_pairs = list(executor.map(self._load_and_chunk, files_to_process))

        new_chunks: list[Chunk] = []
        for path, (_doc, chunks) in zip(files_to_process, loaded_pairs, strict=True):
            new_chunks.extend(chunks)
            rel_key = str(path.resolve())
            self._manifest[rel_key] = {
                "sha256": file_hashes[rel_key],
                "chunk_ids": [c.id for c in chunks],
                "chunk_count": len(chunks),
            }

        # Embed from main thread
        if new_chunks:
            chunk_texts = [c.text for c in new_chunks]
            vectors = self._embed_chunks_with_retry(chunk_texts)
            self.vector_store.upsert(new_chunks, vectors)

            # Re-index all active chunks in BM25
            # Collect chunks across existing + new
            all_chunks = list(self.bm25_index.chunks_by_id.values()) + new_chunks
            # Deduplicate by chunk id
            unique_chunks = {c.id: c for c in all_chunks}
            self.bm25_index.index_chunks(list(unique_chunks.values()))

        self._save_manifest()

        return IngestionResult(
            indexed_files=len(files_to_process),
            skipped_files=skipped_count,
            total_chunks=len(self.bm25_index.doc_ids),
        )

"""Deterministic chunking and ID generation."""

import hashlib
from typing import Any

from rag.schemas import Chunk, Document


def compute_doc_id(path_or_identifier: str) -> str:
    """Compute deterministic document ID from file path or identifier."""
    normalized = path_or_identifier.replace("\\", "/").strip().lower()
    return "doc_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def compute_chunk_id(doc_id: str, index: int, text: str) -> str:
    """Compute content-versioned deterministic chunk ID.

    Combines document id, chunk position index, and chunk content hash.
    """
    content_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}_c{index}_{content_hash}"


def chunk_document(
    doc: Document,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split a document into chunks using recursive character boundaries with overlap.

    Args:
        doc: The document to chunk.
        chunk_size: Approximate maximum number of characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        metadata: Extra metadata to associate with chunks.

    Returns:
        A list of Chunk objects with deterministic IDs.
    """
    text = doc.content.strip()
    if not text:
        return []

    combined_metadata = dict(doc.metadata)
    if metadata:
        combined_metadata.update(metadata)

    # Recursive split separators
    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(sub_text: str, sep_idx: int) -> list[str]:
        if len(sub_text) <= chunk_size or sep_idx >= len(separators):
            return [sub_text] if sub_text.strip() else []

        sep = separators[sep_idx]
        parts = sub_text.split(sep) if sep else list(sub_text)
        splits: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part) + (len(sep) if sep else 0)
            if current_len + part_len > chunk_size and current:
                joined = sep.join(current) if sep else "".join(current)
                splits.append(joined)
                # Overlap logic
                overlap_chars = 0
                overlap_parts: list[str] = []
                for p in reversed(current):
                    overlap_parts.insert(0, p)
                    overlap_chars += len(p) + (len(sep) if sep else 0)
                    if overlap_chars >= chunk_overlap:
                        break
                current = overlap_parts
                current_len = overlap_chars

            current.append(part)
            current_len += part_len

        if current:
            joined = sep.join(current) if sep else "".join(current)
            if joined.strip():
                splits.append(joined)

        # Recurse on splits that exceed chunk_size
        final_splits: list[str] = []
        for s in splits:
            if len(s) > chunk_size and sep_idx + 1 < len(separators):
                final_splits.extend(split_text(s, sep_idx + 1))
            else:
                final_splits.append(s)

        return final_splits

    raw_chunks = split_text(text, 0)
    result_chunks: list[Chunk] = []

    for idx, chunk_text in enumerate(raw_chunks):
        c_text = chunk_text.strip()
        if not c_text:
            continue
        c_id = compute_chunk_id(doc.id, idx, c_text)
        result_chunks.append(
            Chunk(
                id=c_id,
                doc_id=doc.id,
                text=c_text,
                index=idx,
                metadata=combined_metadata,
            )
        )

    return result_chunks

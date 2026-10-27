"""BM25 sparse search index with positive IDF floor."""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rag.schemas import Chunk, Document, RetrievedChunk


def default_tokenizer(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return [token.lower() for token in re.findall(r"\b\w+\b", text)]


class BM25Index:
    """In-memory BM25 index with positive IDF floor and serialization."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        idf_floor: float = 0.1,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.idf_floor = idf_floor
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lens: list[int] = []
        self.doc_ids: list[str] = []
        self.chunks_by_id: dict[str, Chunk] = {}
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_freqs: list[dict[str, int]] = []

    def index_documents(self, documents: list[Document]) -> None:
        """Index a list of Documents."""
        chunks = [
            Chunk(
                id=doc.id,
                doc_id=doc.id,
                text=doc.content,
                index=0,
                metadata=doc.metadata,
            )
            for doc in documents
        ]
        self.index_chunks(chunks)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Index a list of Chunk objects."""
        self.chunks_by_id = {c.id: c for c in chunks}
        self.doc_ids = [c.id for c in chunks]
        self.corpus_size = len(chunks)
        self.doc_lens = []
        self.doc_freqs = []
        self.df = Counter()

        for chunk in chunks:
            tokens = default_tokenizer(chunk.text)
            self.doc_lens.append(len(tokens))
            term_freq = Counter(tokens)
            self.doc_freqs.append(dict(term_freq))
            for term in term_freq:
                self.df[term] += 1

        self.avg_doc_len = (
            sum(self.doc_lens) / float(self.corpus_size) if self.corpus_size > 0 else 0.0
        )

        # Compute IDF with floor
        self.idf = {}
        for term, freq in self.df.items():
            # BM25Okapi IDF formula: log((N - n + 0.5) / (n + 0.5) + 1)
            raw_idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = max(raw_idf, self.idf_floor)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search query against index, returning (chunk_id, score) pairs."""
        if not self.corpus_size:
            return []

        tokens = default_tokenizer(query)
        if not tokens:
            return []

        scores: list[float] = [0.0] * self.corpus_size
        for idx in range(self.corpus_size):
            doc_len = self.doc_lens[idx]
            freqs = self.doc_freqs[idx]
            len_norm = (
                1.0
                - self.b
                + self.b * (doc_len / self.avg_doc_len if self.avg_doc_len > 0 else 1.0)
            )

            for term in tokens:
                if term in freqs:
                    tf = freqs[term]
                    idf_val = self.idf.get(term, self.idf_floor)
                    term_score = idf_val * (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
                    scores[idx] += term_score

        scored_pairs = [
            (self.doc_ids[idx], scores[idx]) for idx in range(self.corpus_size) if scores[idx] > 0.0
        ]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]

    def search_chunks(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search query and return list of RetrievedChunk objects."""
        top_hits = self.search(query, top_k=top_k)
        results: list[RetrievedChunk] = []
        for chunk_id, score in top_hits:
            if chunk_id in self.chunks_by_id:
                results.append(
                    RetrievedChunk(
                        chunk=self.chunks_by_id[chunk_id],
                        score=score,
                        source="bm25",
                    )
                )
        return results

    def to_dict(self) -> dict[str, Any]:
        """Serialize index state to dictionary."""
        return {
            "k1": self.k1,
            "b": self.b,
            "idf_floor": self.idf_floor,
            "corpus_size": self.corpus_size,
            "avg_doc_len": self.avg_doc_len,
            "doc_lens": self.doc_lens,
            "doc_ids": self.doc_ids,
            "chunks": {cid: c.model_dump() for cid, c in self.chunks_by_id.items()},
            "df": dict(self.df),
            "idf": self.idf,
            "doc_freqs": self.doc_freqs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BM25Index":
        """Restore index state from dictionary."""
        instance = cls(
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
            idf_floor=data.get("idf_floor", 0.1),
        )
        instance.corpus_size = data["corpus_size"]
        instance.avg_doc_len = data["avg_doc_len"]
        instance.doc_lens = data["doc_lens"]
        instance.doc_ids = data["doc_ids"]
        instance.chunks_by_id = {cid: Chunk.model_validate(c) for cid, c in data["chunks"].items()}
        instance.df = data["df"]
        instance.idf = data["idf"]
        instance.doc_freqs = data["doc_freqs"]
        return instance

    def save(self, path: str | Path) -> None:
        """Save index to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """Load index from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

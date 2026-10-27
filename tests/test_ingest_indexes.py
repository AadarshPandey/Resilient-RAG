"""Tests for M2: Ingestion and indexes (loaders, chunking, embedder, fakes, vector store, bm25, pipeline)."""

from pathlib import Path

import pytest

from rag.fakes import FakeVectorStore, HashEmbedder
from rag.ingest.chunking import chunk_document, compute_chunk_id, compute_doc_id
from rag.ingest.loaders import load_file
from rag.ingest.pipeline import IngestionPipeline
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.vector import QdrantVectorStore
from rag.schemas import Document


def test_deterministic_doc_and_chunk_ids() -> None:
    """Verify document IDs and chunk IDs are deterministic content hashes."""
    doc_id1 = compute_doc_id("data/raw/kubelet.md")
    doc_id2 = compute_doc_id("data/raw/kubelet.md")
    assert doc_id1 == doc_id2

    chunk_id1 = compute_chunk_id(doc_id1, index=0, text="The kubelet runs on nodes.")
    chunk_id2 = compute_chunk_id(doc_id1, index=0, text="The kubelet runs on nodes.")
    chunk_id_diff = compute_chunk_id(doc_id1, index=1, text="The kubelet runs on nodes.")
    chunk_id_diff_text = compute_chunk_id(doc_id1, index=0, text="Different text.")

    assert chunk_id1 == chunk_id2
    assert chunk_id1 != chunk_id_diff
    assert chunk_id1 != chunk_id_diff_text


def test_chunking_fixed_and_recursive() -> None:
    """Verify chunking produces chunks with valid metadata and content."""
    doc = Document(
        id="doc_k8s",
        content="Section 1\n" + "Word " * 120 + "\n\nSection 2\n" + "More " * 120,
        metadata={"source": "k8s.md"},
    )
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    assert len(chunks) >= 2
    for i, ch in enumerate(chunks):
        assert ch.doc_id == "doc_k8s"
        assert ch.index == i
        assert len(ch.text) > 0
        assert ch.metadata["source"] == "k8s.md"


def test_document_loaders(tmp_path: Path) -> None:
    """Verify loaders for markdown, text, and html."""
    # Markdown
    md_file = tmp_path / "test.md"
    md_file.write_text("# Kubernetes\nKubelet is a node agent.", encoding="utf-8")
    doc_md = load_file(md_file)
    assert doc_md.id == compute_doc_id(str(md_file))
    assert "Kubelet is a node agent" in doc_md.content

    # Plain text
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Plain text documentation line.", encoding="utf-8")
    doc_txt = load_file(txt_file)
    assert "Plain text documentation" in doc_txt.content

    # HTML
    html_file = tmp_path / "test.html"
    html_file.write_text(
        "<html><body><h1>Title</h1><p>HTML paragraph.</p></body></html>", encoding="utf-8"
    )
    doc_html = load_file(html_file)
    assert "Title" in doc_html.content
    assert "HTML paragraph" in doc_html.content
    assert "<p>" not in doc_html.content  # parsed clean text


def test_bm25_idf_floor() -> None:
    """Verify BM25 positive IDF floor on a two-document corpus where terms appear in both docs."""
    # When a term appears in all docs, naive BM25 idf = log((N - n + 0.5)/(n + 0.5)) < 0.
    # Our BM25Index enforces an IDF floor of 0.1 so valid hits never score 0 or negative.
    docs = [
        Document(id="d1", content="kubernetes pod container runtime"),
        Document(id="d2", content="kubernetes worker node runtime"),
    ]
    bm25 = BM25Index()
    bm25.index_documents(docs)

    scores = bm25.search("kubernetes", top_k=2)
    assert len(scores) == 2
    for doc_id, score in scores:
        assert score > 0.0, f"Expected positive score for {doc_id}, got {score}"

    # Verify serialization
    state = bm25.to_dict()
    restored = BM25Index.from_dict(state)
    scores_restored = restored.search("kubernetes", top_k=2)
    assert len(scores_restored) == 2
    assert pytest.approx(scores[0][1]) == scores_restored[0][1]


def test_hash_embedder_and_fake_vector_store() -> None:
    """Verify HashEmbedder produces deterministic unit vectors and FakeVectorStore searches."""
    embedder = HashEmbedder(dimension=64)
    v1 = embedder.embed(["test query"])[0]
    v2 = embedder.embed(["test query"])[0]
    v3 = embedder.embed(["completely different query"])[0]

    assert len(v1) == 64
    assert v1 == v2
    assert v1 != v3

    store = FakeVectorStore()
    doc = Document(id="d1", content="Cluster networking")
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))

    query_vec = embedder.embed(["Cluster networking"])[0]
    results = store.search(query_vec, top_k=1)
    assert len(results) == 1
    assert results[0].chunk.doc_id == "d1"
    assert results[0].score > 0.9


def test_qdrant_vector_store_in_memory() -> None:
    """Verify QdrantVectorStore works in :memory: mode with HashEmbedder."""
    embedder = HashEmbedder(dimension=32)
    store = QdrantVectorStore(location=":memory:", collection_name="test_col", vector_size=32)

    doc = Document(id="d1", content="CoreDNS manages cluster internal DNS resolution.")
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
    vectors = embedder.embed([c.text for c in chunks])

    store.upsert(chunks, vectors)

    query_vec = embedder.embed(["DNS resolution"])[0]
    results = store.search(query_vec, top_k=1)
    assert len(results) == 1
    assert "DNS" in results[0].chunk.text
    assert results[0].score > 0.0


def test_ingestion_pipeline_manifest(tmp_path: Path) -> None:
    """Verify IngestionPipeline loads, chunks, indexes, and skips unchanged files via manifest."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    f1 = raw_dir / "doc1.md"
    f1.write_text("Etcd stores all cluster state securely.", encoding="utf-8")
    f2 = raw_dir / "doc2.md"
    f2.write_text("Kube-proxy maintains network rules on nodes.", encoding="utf-8")

    embedder = HashEmbedder(dimension=32)
    vector_store = FakeVectorStore()
    bm25_index = BM25Index()
    manifest_path = tmp_path / "manifest.json"

    pipeline = IngestionPipeline(
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        manifest_path=manifest_path,
    )

    # First run: indexes 2 files
    result1 = pipeline.run(input_dir=raw_dir)
    assert result1.indexed_files == 2
    assert result1.skipped_files == 0
    assert result1.total_chunks >= 2

    # Second run without changes: skips both files
    result2 = pipeline.run(input_dir=raw_dir)
    assert result2.indexed_files == 0
    assert result2.skipped_files == 2

    # Modify one file: re-indexes only that file
    f1.write_text("Etcd is a consistent and highly-available key value store.", encoding="utf-8")
    result3 = pipeline.run(input_dir=raw_dir)
    assert result3.indexed_files == 1
    assert result3.skipped_files == 1

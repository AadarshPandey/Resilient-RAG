"""Document loaders supporting markdown, text, HTML, and PDF formats."""

from pathlib import Path
from typing import Any

from rag.errors import RagError
from rag.ingest.chunking import compute_doc_id
from rag.schemas import Document


def load_file(file_path: str | Path, metadata: dict[str, Any] | None = None) -> Document:
    """Load a file from disk into a Document.

    Supports .md, .txt, .html, .htm, and .pdf.

    Args:
        file_path: Path to the target file.
        metadata: Optional dictionary of metadata.

    Returns:
        A Document instance.
    """
    path = Path(file_path)
    if not path.is_file():
        raise RagError(f"File not found: {path}")

    ext = path.suffix.lower()
    doc_id = compute_doc_id(str(path))
    meta = dict(metadata or {})
    meta.setdefault("source", str(path))
    meta.setdefault("file_name", path.name)
    meta.setdefault("extension", ext)

    if ext in (".md", ".txt"):
        content = path.read_text(encoding="utf-8")
    elif ext in (".html", ".htm"):
        from bs4 import BeautifulSoup

        raw_html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw_html, "html.parser")
        content = soup.get_text(separator="\n", strip=True)
    elif ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text_parts = [page.extract_text() or "" for page in reader.pages]
        content = "\n\n".join(text_parts).strip()
    else:
        # Fallback to plain text
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RagError(f"Unsupported binary file format: {ext}") from exc

    return Document(id=doc_id, content=content, metadata=meta)

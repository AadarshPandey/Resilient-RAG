"""Smoke test for package scaffold."""

import rag


def test_import_rag() -> None:
    """Verify that rag package imports correctly and defines version."""
    assert rag.__version__ == "0.1.0"

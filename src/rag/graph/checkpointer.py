"""Checkpointer factory supporting in-memory, SQLite, and PostgreSQL backends."""

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from rag.errors import BackendError

logger = logging.getLogger(__name__)


def create_checkpointer(dsn: str | None = None) -> BaseCheckpointSaver[Any]:
    """Construct a LangGraph checkpoint saver based on the configured DSN.

    Supports:
        - None or ":memory:" -> MemorySaver
        - sqlite:///<path> -> SqliteSaver (with tables auto-initialized)
        - postgresql://... or postgres://... -> PostgresSaver (with setup applied)
    """
    if not dsn or dsn == ":memory:":
        return MemorySaver()

    parsed = urlparse(dsn)
    scheme = parsed.scheme.lower()

    if scheme == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        # Parse file path from sqlite:///path
        path_str = dsn.replace("sqlite:///", "")
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(path), check_same_thread=False)
        sqlite_saver = SqliteSaver(conn)
        sqlite_saver.setup()
        return sqlite_saver

    elif scheme in ("postgresql", "postgres"):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            cm = PostgresSaver.from_conn_string(dsn)
            pg_saver = cm.__enter__()
            pg_saver.setup()
            return pg_saver
        except Exception as e:
            raise BackendError(f"Failed to initialize PostgreSQL checkpointer: {e}") from e

    raise ValueError(
        f"Unsupported checkpointer DSN scheme: '{scheme}'. Supported: sqlite, postgresql, :memory:"
    )


def setup_postgres_checkpointer(dsn: str) -> None:
    """Run table migrations on PostgreSQL database for LangGraph checkpoints."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(dsn) as saver:
            saver.setup()
            logger.info("PostgreSQL checkpoint tables successfully verified and initialized.")
    except Exception as e:
        raise BackendError(f"PostgreSQL migration failed for '{dsn}': {e}") from e

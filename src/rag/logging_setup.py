"""Structured single-line JSON logging configuration."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

current_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_request_id", default=None
)


def set_current_request_id(request_id: str | None) -> contextvars.Token[str | None]:
    """Set the active request ID in context."""
    return current_request_id.set(request_id)


def get_current_request_id() -> str | None:
    """Retrieve the active request ID from context."""
    return current_request_id.get()


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        req_id = getattr(record, "request_id", None) or get_current_request_id()
        if req_id is not None:
            log_obj["request_id"] = req_id

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        extra_dict = getattr(record, "extra", None)
        if isinstance(extra_dict, dict):
            log_obj.update(extra_dict)
        return json.dumps(log_obj)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with JSONFormatter to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    root_logger.addHandler(handler)

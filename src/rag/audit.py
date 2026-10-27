"""Thread-safe append-only audit logger writing to JSONL with automated PII redaction."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rag.security import redact_dict


class AuditLogger:
    """Thread-safe JSONL audit logger."""

    def __init__(self, path: str | Path = "data/audit.jsonl", redact_pii: bool = True) -> None:
        self.path = Path(path)
        self.redact_pii = redact_pii
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any] | BaseModel) -> None:
        """Append a single record as a JSON line to the audit file with PII redacted."""
        payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        if self.redact_pii:
            payload = redact_dict(payload)
        line = json.dumps(payload, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

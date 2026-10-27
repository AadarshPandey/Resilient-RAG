"""Security utilities for API key validation, rate limiting, PII redaction, sanitization, and leakage detection."""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
from typing import Any

from rag.clock import Clock, SystemClock

# --- Regex patterns for PII redaction ---
SECRET_PATTERN = re.compile(
    r"\b(?:sk-[a-zA-Z0-9_\-]{14,}|Bearer\s+[a-zA-Z0-9_\-\.]{14,})\b",
    re.IGNORECASE,
)
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")

# --- System prompt leakage markers ---
LEAKAGE_PATTERNS = [
    re.compile(r"you\s+are\s+a\s+helpful\s+assistant", re.IGNORECASE),
    re.compile(
        r"answer\s+the\s+user\s+query\s+using\s+only\s+the\s+provided\s+context", re.IGNORECASE
    ),
    re.compile(r"STRICT_MODE:\s+Return\s+JSON", re.IGNORECASE),
    re.compile(
        r"You\s+are\s+an\s+uncompromising\s+factual\s+verification\s+assistant", re.IGNORECASE
    ),
    re.compile(r"GROUNDED_GENERATION_PROMPT", re.IGNORECASE),
    re.compile(r"STRICT_GENERATION_PROMPT", re.IGNORECASE),
]

# --- Delimiters to neutralize in retrieved context ---
INSTRUCTION_DELIMITERS = [
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"\[/INST\]", re.IGNORECASE),
    re.compile(r"<<SYS>>", re.IGNORECASE),
    re.compile(r"<</SYS>>", re.IGNORECASE),
]


def hash_api_key(raw_key: str) -> str:
    """Compute SHA-256 hash of API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def verify_api_key(raw_key: str, allowed_hashes: list[str]) -> bool:
    """Verify raw API key against a list of SHA-256 hashes using constant-time comparison."""
    if not allowed_hashes:
        return True
    key_hash = hash_api_key(raw_key)
    return any(hmac.compare_digest(key_hash, allowed) for allowed in allowed_hashes)


def redact_pii(text: str) -> str:
    """Redact secrets, credit cards, emails, and phone numbers from a text string."""
    redacted = SECRET_PATTERN.sub("[REDACTED_SECRET]", text)
    redacted = CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", redacted)
    redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
    redacted = PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    return redacted


def redact_dict(data: Any) -> Any:
    """Recursively redact PII from all string values in a dictionary or list."""
    if isinstance(data, str):
        return redact_pii(data)
    elif isinstance(data, dict):
        return {k: redact_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [redact_dict(item) for item in data]
    return data


def sanitize_retrieved_text(text: str) -> str:
    """Strip dangerous control characters and neutralize role delimiter tokens."""
    # Remove control characters except newline and tab
    cleaned = "".join(ch for ch in text if ch in ("\n", "\t") or (ord(ch) >= 32 and ord(ch) != 127))
    # Neutralize injection delimiters
    for pattern in INSTRUCTION_DELIMITERS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def check_system_prompt_leakage(text: str) -> bool:
    """Detect if output text echoes system instructions or internal prompt templates."""
    return any(p.search(text) is not None for p in LEAKAGE_PATTERNS)


class RateLimiter:
    """Sliding-window in-memory rate limiter per client key."""

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
        clock: Clock | None = None,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._history: dict[str, list[float]] = {}

    def is_allowed(self, client_key: str) -> bool:
        """Check whether client request is within rate limit; record timestamp if allowed."""
        now = self.clock.now()
        threshold = now - self.window_seconds

        with self._lock:
            timestamps = self._history.setdefault(client_key, [])
            # Prune old timestamps
            self._history[client_key] = [t for t in timestamps if t > threshold]
            if len(self._history[client_key]) >= self.max_requests:
                return False
            self._history[client_key].append(now)
            return True

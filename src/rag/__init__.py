"""Resilient RAG package."""

import socket
from typing import Any

# Prefer IPv4 resolution to prevent connection hangs on environments with unrouted IPv6 WAN
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_preferred_getaddrinfo(
    host: Any,
    port: Any,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
) -> Any:
    """Prefer IPv4 resolution to prevent connection hangs on environments with unrouted IPv6."""
    if family == socket.AF_UNSPEC:
        try:
            return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
        except socket.gaierror:
            pass
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_preferred_getaddrinfo

__version__ = "0.1.0"

"""Typed exceptions for the effGen client SDK."""
from __future__ import annotations

from typing import Any


class EffGenClientError(Exception):
    """Base class for all effGen client errors."""


class EffGenConnectionError(EffGenClientError):
    """Raised when the client cannot connect to the server."""


class EffGenTimeoutError(EffGenClientError):
    """Raised when a request times out."""


class EffGenAPIError(EffGenClientError):
    """Raised when the server returns a non-2xx response."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        payload: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class EffGenAuthError(EffGenAPIError):
    """Raised on 401/403 — authentication or authorization failed."""


class EffGenRateLimitError(EffGenAPIError):
    """Raised on 429 — rate limit exceeded."""


class EffGenServerError(EffGenAPIError):
    """Raised on 5xx — server-side error."""

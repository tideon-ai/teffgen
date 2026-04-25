"""Typed exceptions for the tideon.ai client SDK."""
from __future__ import annotations

from typing import Any


class TeffgenClientError(Exception):
    """Base class for all tideon.ai client errors."""


class TeffgenConnectionError(TeffgenClientError):
    """Raised when the client cannot connect to the server."""


class TeffgenTimeoutError(TeffgenClientError):
    """Raised when a request times out."""


class TeffgenAPIError(TeffgenClientError):
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


class TeffgenAuthError(TeffgenAPIError):
    """Raised on 401/403 — authentication or authorization failed."""


class TeffgenRateLimitError(TeffgenAPIError):
    """Raised on 429 — rate limit exceeded."""


class TeffgenServerError(TeffgenAPIError):
    """Raised on 5xx — server-side error."""

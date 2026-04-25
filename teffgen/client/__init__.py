"""tideon.ai Python client SDK.

Provides sync and async clients for connecting to an tideon.ai API server.

Example:
    >>> from teffgen.client import TeffgenClient
    >>> client = TeffgenClient(base_url="http://localhost:8000", api_key="...")
    >>> result = client.chat("What is 2+2?", tools=["calculator"])
    >>> print(result.content)

Async / streaming:
    >>> import asyncio
    >>> async def main():
    ...     async for chunk in client.chat_stream("Tell me a story"):
    ...         print(chunk, end="")
    >>> asyncio.run(main())
"""
from __future__ import annotations

from teffgen.client.client import (
    ChatResponse,
    HealthStatus,
    TeffgenClient,
)
from teffgen.client.exceptions import (
    TeffgenAPIError,
    TeffgenAuthError,
    TeffgenClientError,
    TeffgenConnectionError,
    TeffgenRateLimitError,
    TeffgenServerError,
    TeffgenTimeoutError,
)

__all__ = [
    "TeffgenClient",
    "ChatResponse",
    "HealthStatus",
    "TeffgenClientError",
    "TeffgenAPIError",
    "TeffgenAuthError",
    "TeffgenConnectionError",
    "TeffgenRateLimitError",
    "TeffgenServerError",
    "TeffgenTimeoutError",
]

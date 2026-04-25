"""Production middleware for the tideon.ai API server.

Attached to an existing FastAPI app via :func:`install_production_middleware`.
Provides:

- Request ID injection (``X-Request-ID``)
- CORS configuration
- Response compression (gzip)
- Request/response validation is handled by FastAPI/Pydantic by default
- Graceful shutdown hooks (SIGTERM/SIGINT draining)
"""
from __future__ import annotations

import asyncio
import logging
import signal
import uuid
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


def install_production_middleware(
    app: Any,
    *,
    cors_origins: Iterable[str] | None = None,
    enable_gzip: bool = True,
    gzip_min_size: int = 500,
    enable_request_id: bool = True,
    shutdown_timeout: float = 10.0,
) -> None:
    """Install production-grade middleware on a FastAPI ``app``.

    Safe to call even if FastAPI/Starlette is not importable (no-op).
    """
    try:
        from fastapi import Request
        from starlette.middleware.cors import CORSMiddleware
        from starlette.middleware.gzip import GZipMiddleware
    except Exception:  # pragma: no cover
        logger.warning("fastapi/starlette not available — middleware skipped")
        return

    # 1. CORS
    origins = list(cors_origins) if cors_origins else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # 2. gzip compression
    if enable_gzip:
        app.add_middleware(GZipMiddleware, minimum_size=gzip_min_size)

    # 3. Request ID injection (ASGI-level middleware so it runs before route).
    if enable_request_id:

        @app.middleware("http")
        async def request_id_middleware(request: Request, call_next):  # type: ignore
            req_id = request.headers.get("x-request-id") or uuid.uuid4().hex
            request.state.request_id = req_id
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("request %s failed", req_id)
                raise
            response.headers["X-Request-ID"] = req_id
            return response

    # 4. Graceful shutdown
    _install_graceful_shutdown(app, shutdown_timeout)


def _install_graceful_shutdown(app: Any, timeout: float) -> None:
    """Drain in-flight requests before exiting on SIGTERM/SIGINT."""
    inflight: set = set()
    shutting_down = {"value": False}

    try:
        from fastapi import Request
    except Exception:  # pragma: no cover
        return

    @app.middleware("http")
    async def track_inflight(request: Request, call_next):  # type: ignore
        if shutting_down["value"]:
            from starlette.responses import JSONResponse

            return JSONResponse(
                {"error": "server is shutting down"}, status_code=503
            )
        task = asyncio.current_task()
        if task is not None:
            inflight.add(task)
        try:
            return await call_next(request)
        finally:
            if task is not None:
                inflight.discard(task)

    @app.on_event("shutdown")
    async def _drain() -> None:
        shutting_down["value"] = True
        deadline = asyncio.get_event_loop().time() + timeout
        while inflight and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.1)
        if inflight:
            logger.warning(
                "shutdown: %d requests still in-flight after %.1fs",
                len(inflight),
                timeout,
            )

    # Best-effort signal hookup; uvicorn handles SIGTERM itself, but when
    # embedded elsewhere this ensures flags still flip.
    try:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig, lambda: shutting_down.__setitem__("value", True)
                )
            except (NotImplementedError, RuntimeError):
                pass
    except RuntimeError:
        pass

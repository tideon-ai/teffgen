"""
Sliding-window rate-limit coordinator for tideon.ai model adapters.

In-memory implementation.  Persistence across processes is not yet supported.

Usage::

    from teffgen.models._rate_limit import RateLimitCoordinator

    coordinator = RateLimitCoordinator(
        provider="cerebras",
        model="llama3.1-8b",
        rpm=30, rph=900, rpd=14_400,
        tpm=60_000, tph=1_000_000, tpd=1_000_000,
    )

    async def call():
        await coordinator.acquire(tokens_estimate=100)
        result = await make_api_call()
        coordinator.record(actual_tokens=result.tokens_used)
        return result
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when the daily budget for a model is exhausted."""


@dataclass
class _Window:
    """Sliding-window counter for a fixed duration (seconds)."""
    duration: float                   # window length in seconds
    limit: int                        # max events allowed in the window
    _timestamps: deque[float] = field(default_factory=deque)

    def _evict(self, now: float) -> None:
        cutoff = now - self.duration
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def count(self, now: float | None = None) -> int:
        self._evict(now or time.monotonic())
        return len(self._timestamps)

    def add(self, now: float | None = None) -> None:
        self._timestamps.append(now or time.monotonic())

    def remaining(self, now: float | None = None) -> int:
        return max(0, self.limit - self.count(now))

    def wait_seconds(self, now: float | None = None) -> float:
        """Seconds to wait until one slot opens in the window."""
        now = now or time.monotonic()
        self._evict(now)
        if len(self._timestamps) < self.limit:
            return 0.0
        # Oldest event will expire at oldest_ts + duration
        return max(0.0, self._timestamps[0] + self.duration - now)


@dataclass
class _TokenWindow:
    """Sliding-window counter that tracks total tokens (not just request count)."""
    duration: float
    limit: int
    # Each entry is (timestamp, token_count)
    _entries: deque[tuple[float, int]] = field(default_factory=deque)

    def _evict(self, now: float) -> None:
        cutoff = now - self.duration
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def total(self, now: float | None = None) -> int:
        self._evict(now or time.monotonic())
        return sum(t for _, t in self._entries)

    def add(self, tokens: int, now: float | None = None) -> None:
        self._entries.append((now or time.monotonic(), tokens))

    def remaining(self, now: float | None = None) -> int:
        return max(0, self.limit - self.total(now))

    def wait_seconds(self, tokens_needed: int, now: float | None = None) -> float:
        """Seconds to wait until *tokens_needed* fit within the window."""
        now = now or time.monotonic()
        self._evict(now)
        available = self.limit - self.total(now)
        if available >= tokens_needed:
            return 0.0
        # Wait until enough old entries expire
        needed_to_free = tokens_needed - available
        freed = 0
        for ts, tok in self._entries:
            freed += tok
            if freed >= needed_to_free:
                return max(0.0, ts + self.duration - now)
        # All entries together don't cover the need (tokens_needed > limit)
        return 0.0


class RateLimitCoordinator:
    """
    Sliding-window rate-limit coordinator for a single (provider, model) pair.

    Tracks RPM / RPH / RPD and TPM / TPH / TPD simultaneously.  Any request
    that would violate a limit is delayed with ``asyncio.sleep`` until capacity
    is available.

    Args:
        provider: Provider name (e.g. ``"cerebras"``).
        model: Model ID (e.g. ``"llama3.1-8b"``).
        rpm: Max requests per minute.
        rph: Max requests per hour.
        rpd: Max requests per day.
        tpm: Max tokens per minute.
        tph: Max tokens per hour.
        tpd: Max tokens per day.

    Raises:
        RateLimitExceeded: When the *daily* budget (RPD or TPD) is fully
            consumed and the next request cannot be scheduled before the
            day window resets.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        rpm: int,
        rph: int,
        rpd: int,
        tpm: int,
        tph: int,
        tpd: int,
    ) -> None:
        self.provider = provider
        self.model = model

        # Request-count windows
        self._req_minute = _Window(duration=60.0, limit=rpm)
        self._req_hour = _Window(duration=3_600.0, limit=rph)
        self._req_day = _Window(duration=86_400.0, limit=rpd)

        # Token-count windows
        self._tok_minute = _TokenWindow(duration=60.0, limit=tpm)
        self._tok_hour = _TokenWindow(duration=3_600.0, limit=tph)
        self._tok_day = _TokenWindow(duration=86_400.0, limit=tpd)

        # Lazy-initialized so construction doesn't require a running event loop
        # (Python 3.9 binds asyncio.Lock to the event loop at creation time)
        self._lock: asyncio.Lock | None = None

        # Total counters (lifetime, not windowed) for observability
        self.total_requests: int = 0
        self.total_tokens: int = 0
        self.total_throttled: int = 0
        self.total_throttle_seconds: float = 0.0

        logger.debug(
            "RateLimitCoordinator ready: %s/%s  rpm=%d rph=%d rpd=%d  "
            "tpm=%d tph=%d tpd=%d",
            provider, model, rpm, rph, rpd, tpm, tph, tpd,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_lock(self) -> asyncio.Lock:
        """Return the lock, creating it lazily inside the current event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self, tokens_estimate: int = 0) -> None:
        """Block until a request slot (and token budget) is available.

        Must be called *before* making the API request.  Pair with
        :meth:`record` after the call completes.

        Args:
            tokens_estimate: Expected token count (prompt + completion).
                Pass ``0`` if unknown; the coordinator will only enforce
                request-count limits in that case.

        Raises:
            RateLimitExceeded: If the daily request or token budget is
                exhausted (the day window has no capacity).
        """
        async with self._get_lock():
            await self._wait_for_capacity(tokens_estimate)

    def record(self, actual_tokens: int = 0) -> None:
        """Record a completed request.

        Must be called *after* the API call returns.

        Args:
            actual_tokens: Actual tokens used (from ``usage`` in the response).
        """
        now = time.monotonic()
        self._req_minute.add(now)
        self._req_hour.add(now)
        self._req_day.add(now)

        if actual_tokens > 0:
            self._tok_minute.add(actual_tokens, now)
            self._tok_hour.add(actual_tokens, now)
            self._tok_day.add(actual_tokens, now)

        self.total_requests += 1
        self.total_tokens += actual_tokens

        logger.debug(
            "RLC record %s/%s: req=%d tokens=%d",
            self.provider, self.model, self.total_requests, self.total_tokens,
        )

    def status(self) -> dict:
        """Return a snapshot of current window usage (for debugging/logging)."""
        now = time.monotonic()
        return {
            "provider": self.provider,
            "model": self.model,
            "req_minute_used": self._req_minute.count(now),
            "req_minute_limit": self._req_minute.limit,
            "req_hour_used": self._req_hour.count(now),
            "req_hour_limit": self._req_hour.limit,
            "req_day_used": self._req_day.count(now),
            "req_day_limit": self._req_day.limit,
            "tok_minute_used": self._tok_minute.total(now),
            "tok_minute_limit": self._tok_minute.limit,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_throttled": self.total_throttled,
            "total_throttle_seconds": round(self.total_throttle_seconds, 3),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _wait_for_capacity(self, tokens_estimate: int) -> None:
        """Compute the required sleep and block.  Called inside the lock."""
        while True:
            now = time.monotonic()

            # Check daily limits first (raise immediately — daily waits are unusable)
            if self._req_day.remaining(now) == 0:
                raise RateLimitExceeded(
                    f"Daily request budget exhausted for {self.provider}/{self.model}. "
                    "Resets in 24 h."
                )

            if tokens_estimate > 0 and self._tok_day.remaining(now) < tokens_estimate:
                raise RateLimitExceeded(
                    f"Daily token budget exhausted for {self.provider}/{self.model}. "
                    "Resets in 24 h."
                )

            # Compute maximum wait needed across all sub-minute/hourly windows
            waits = [
                self._req_minute.wait_seconds(now),
                self._req_hour.wait_seconds(now),
            ]
            if tokens_estimate > 0:
                waits += [
                    self._tok_minute.wait_seconds(tokens_estimate, now),
                    self._tok_hour.wait_seconds(tokens_estimate, now),
                ]

            wait = max(waits)
            if wait <= 0:
                break

            logger.debug(
                "RLC throttling %s/%s for %.3f s (tokens_estimate=%d)",
                self.provider, self.model, wait, tokens_estimate,
            )
            self.total_throttled += 1
            self.total_throttle_seconds += wait
            # Release lock during sleep so other coroutines can check/update
            lock = self._get_lock()
            lock.release()
            try:
                await asyncio.sleep(wait)
            finally:
                await lock.acquire()

"""
In-memory cost tracker for effGen model adapters.

Accumulates prompt and completion token counts per (provider, model) pair
and converts them to USD using per-provider rate tables.  Cerebras free-tier
cost is $0.  Other providers use placeholder rates that will be refined in
later versions.

Usage::

    from effgen.models._cost import CostTracker

    tracker = CostTracker.get()
    tracker.record("cerebras", "llama3.1-8b", prompt_tokens=50, completion_tokens=20)
    print(tracker.total_cost("cerebras", "llama3.1-8b"))
    print(tracker.summary())
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-million-token rates (USD).  [input_per_M, output_per_M]
#
# Status key:
#   OFFICIAL   — rate reflects the current provider-published list price
#                (Cerebras free tier = $0 is official; verified 2026-04-24).
#   PLACEHOLDER — rate will be refined in v0.2.4 (cost-aware router phase).
#                Treat numbers here as rough guidance, not billing truth.
#
# Persistence: this tracker is IN-MEMORY only — the per-process singleton is
# cleared on restart.  A durable backend (sqlite) lands in v0.2.4.
# ---------------------------------------------------------------------------
_RATES: dict[str, dict[str, tuple[float, float]]] = {
    "cerebras": {
        # OFFICIAL: Cerebras free tier pricing = $0 for all models.
        "*": (0.0, 0.0),
    },
    "openai": {
        # PLACEHOLDER (verify against OpenAI pricing page before billing use)
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4": (30.00, 60.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "*": (1.00, 3.00),
    },
    "anthropic": {
        # PLACEHOLDER
        "claude-3-opus": (15.00, 75.00),
        "claude-3-sonnet": (3.00, 15.00),
        "claude-3-haiku": (0.25, 1.25),
        "*": (3.00, 15.00),
    },
    "gemini": {
        # PLACEHOLDER
        "gemini-pro": (0.125, 0.375),
        "gemini-1.5-pro": (3.50, 10.50),
        "*": (1.00, 3.00),
    },
}


def _rate(provider: str, model: str) -> tuple[float, float]:
    """Lookup (input_per_M, output_per_M) rate for provider/model."""
    provider_rates = _RATES.get(provider.lower(), {})
    # Exact match first, then prefix match, then wildcard
    if model in provider_rates:
        return provider_rates[model]
    for key in provider_rates:
        if key != "*" and model.startswith(key):
            return provider_rates[key]
    return provider_rates.get("*", (0.0, 0.0))


@dataclass
class _ModelStats:
    """Per-(provider, model) accumulated stats."""
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    total_cost_usd: float = 0.0


class CostTracker:
    """Thread-safe, in-memory cost tracker.

    A singleton per process (use :meth:`get`).  Supports multiple providers
    and models simultaneously.  Cerebras free-tier cost is always $0.

    Example::

        tracker = CostTracker.get()
        cost = tracker.record("cerebras", "llama3.1-8b", 50, 20)
        print(f"Cost: ${cost:.6f}")  # 0.000000 for Cerebras free tier
    """

    _instance: CostTracker | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], _ModelStats] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "CostTracker":
        """Return the process-global CostTracker instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the global tracker (useful in tests)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Record a completed API call and return the USD cost.

        Args:
            provider: Provider name, e.g. ``"cerebras"``.
            model: Model ID, e.g. ``"llama3.1-8b"``.
            prompt_tokens: Tokens in the prompt (from API usage).
            completion_tokens: Tokens in the completion (from API usage).

        Returns:
            USD cost for this call (0.0 for Cerebras free tier).
        """
        input_rate, output_rate = _rate(provider, model)
        cost = (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000

        key = (provider.lower(), model)
        with self._lock:
            if key not in self._data:
                self._data[key] = _ModelStats(provider=provider, model=model)
            stats = self._data[key]
            stats.prompt_tokens += prompt_tokens
            stats.completion_tokens += completion_tokens
            stats.requests += 1
            stats.total_cost_usd += cost

        logger.debug(
            "CostTracker.record %s/%s: prompt=%d completion=%d cost=$%.6f",
            provider, model, prompt_tokens, completion_tokens, cost,
        )
        return cost

    def total_cost(self, provider: str | None = None, model: str | None = None) -> float:
        """Return total USD cost, optionally filtered by provider and/or model.

        Args:
            provider: Filter to this provider (None = all providers).
            model: Filter to this model (None = all models).

        Returns:
            Cumulative USD cost.
        """
        with self._lock:
            total = 0.0
            for (prov, mod), stats in self._data.items():
                if provider and prov != provider.lower():
                    continue
                if model and mod != model:
                    continue
                total += stats.total_cost_usd
        return total

    def total_tokens(self, provider: str | None = None, model: str | None = None) -> dict[str, int]:
        """Return total token counts, optionally filtered.

        Returns:
            Dict with keys ``prompt``, ``completion``, ``total``.
        """
        prompt = completion = 0
        with self._lock:
            for (prov, mod), stats in self._data.items():
                if provider and prov != provider.lower():
                    continue
                if model and mod != model:
                    continue
                prompt += stats.prompt_tokens
                completion += stats.completion_tokens
        return {"prompt": prompt, "completion": completion, "total": prompt + completion}

    def summary(self) -> list[dict]:
        """Return a list of per-(provider, model) usage summaries.

        Returns:
            List of dicts with keys: ``provider``, ``model``, ``requests``,
            ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
            ``cost_usd``.
        """
        with self._lock:
            rows = []
            for stats in self._data.values():
                rows.append({
                    "provider": stats.provider,
                    "model": stats.model,
                    "requests": stats.requests,
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.completion_tokens,
                    "total_tokens": stats.prompt_tokens + stats.completion_tokens,
                    "cost_usd": round(stats.total_cost_usd, 8),
                })
        return rows

    def reset_stats(self) -> None:
        """Clear all accumulated stats (does not reset singleton)."""
        with self._lock:
            self._data.clear()

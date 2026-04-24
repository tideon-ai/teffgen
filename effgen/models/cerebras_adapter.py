"""
Cerebras Cloud SDK adapter for effGen.

Supports all free-tier Cerebras models with built-in rate-limit coordination.
Streaming and native tool-calling are not yet available.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

from effgen.models._rate_limit import RateLimitCoordinator
from effgen.models.base import (
    BaseModel,
    GenerationConfig,
    GenerationResult,
    TokenCount,
)
from effgen.models.cerebras_models import (
    CEREBRAS_DEFAULT_MODEL,
    CEREBRAS_MODELS,
    available_models,
    free_tier_models,
    model_info,
)

logger = logging.getLogger(__name__)

_CEREBRAS_MODEL_TYPE_VALUE = "cerebras"


class _CerebrasModelType:
    """Sentinel so ModelType enum doesn't need patching."""
    value = _CEREBRAS_MODEL_TYPE_VALUE


class CerebrasAdapter(BaseModel):
    """
    Adapter for Cerebras Cloud inference API.

    Wraps the ``cerebras-cloud-sdk`` with the standard effGen BaseModel
    interface.  A :class:`~effgen.models._rate_limit.RateLimitCoordinator`
    is created per adapter instance to enforce per-model free-tier limits.
    Streaming is not yet available.

    Args:
        model_name: Cerebras model ID.  Must be a key in
            :data:`~effgen.models.cerebras_models.CEREBRAS_MODELS`.
            Defaults to ``"llama3.1-8b"`` (the most accessible free-tier model).
        api_key: Cerebras API key.  If omitted, reads ``CEREBRAS_API_KEY``
            from the environment.
        max_retries: Maximum number of SDK retry attempts.
        timeout: Per-request timeout in seconds.
        enable_rate_limiting: If ``True`` (default), acquire / record calls
            via the built-in :class:`~effgen.models._rate_limit.RateLimitCoordinator`.
            Disable only for testing.
    """

    def __init__(
        self,
        model_name: str = CEREBRAS_DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
        enable_rate_limiting: bool = True,
        **kwargs: Any,
    ) -> None:
        if model_name not in CEREBRAS_MODELS:
            raise ValueError(
                f"Unknown Cerebras model '{model_name}'. "
                f"Available: {available_models()}"
            )

        info = CEREBRAS_MODELS[model_name]
        super().__init__(
            model_name=model_name,
            model_type=_CerebrasModelType(),  # type: ignore[arg-type]
            context_length=info.get("context", 128_000),
        )
        self._api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._extra_kwargs = kwargs
        self._client: Any = None

        # Rate-limit coordinator wired per-instance (in-memory)
        self._rate_limiter: RateLimitCoordinator | None = None
        if enable_rate_limiting:
            self._rate_limiter = RateLimitCoordinator(
                provider="cerebras",
                model=model_name,
                rpm=info.get("rpm", 30),
                rph=info.get("rph", 900),
                rpd=info.get("rpd", 14_400),
                tpm=info.get("tpm", 60_000),
                tph=info.get("tph", 1_000_000),
                tpd=info.get("tpd", 1_000_000),
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the Cerebras SDK client.

        Reads ``CEREBRAS_API_KEY`` from the environment if *api_key* was not
        passed to the constructor.

        Raises:
            RuntimeError: If ``cerebras-cloud-sdk`` is not installed.
            ValueError: If no API key is available.
        """
        api_key = self._api_key or os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError(
                "Cerebras API key not found. Set the CEREBRAS_API_KEY "
                "environment variable or pass api_key= to CerebrasAdapter."
            )

        try:
            from cerebras.cloud.sdk import Cerebras
        except ImportError as exc:
            raise RuntimeError(
                "cerebras-cloud-sdk is not installed. "
                "Install with: pip install 'effgen[cerebras]'"
            ) from exc

        self._client = Cerebras(api_key=api_key)
        self._is_loaded = True

        # Warn if a non-free-tier model is loaded — a free-tier key will 404.
        info = CEREBRAS_MODELS.get(self.model_name, {})
        if not info.get("free_tier", False):
            logger.warning(
                "Cerebras model '%s' is not reliably callable on the free tier "
                "(high demand / restricted access). If you have a paid-tier key "
                "this will work; otherwise consider a free-tier model: %s",
                self.model_name, free_tier_models(),
            )
        if info.get("deprecated"):
            logger.warning(
                "Cerebras model '%s' is scheduled for deprecation on %s.",
                self.model_name, info["deprecated"],
            )
        self._metadata = {
            "model_name": self.model_name,
            "context_length": self.get_context_length(),
            "provider": "cerebras",
            "free_tier": CEREBRAS_MODELS[self.model_name].get("free_tier", False),
        }
        logger.info("CerebrasAdapter loaded for model '%s'", self.model_name)

    def unload(self) -> None:
        """Release SDK client resources."""
        self._client = None
        self._is_loaded = False
        logger.info("CerebrasAdapter unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response synchronously via the Cerebras API.

        Rate-limiting is applied automatically if *enable_rate_limiting* was
        ``True`` in the constructor (the default).  The coordinator uses
        ``asyncio.get_event_loop().run_until_complete`` to acquire a slot
        when called from synchronous context; prefer the async path when
        running inside an async event loop.

        Args:
            prompt: User prompt text.
            config: Optional generation config (temperature, max_tokens, …).
            **kwargs: Forwarded to ``chat.completions.create``.

        Returns:
            GenerationResult with the generated text and token usage.

        Raises:
            RuntimeError: If ``load()`` has not been called or the API call fails.
        """
        import asyncio

        if not self._is_loaded or self._client is None:
            raise RuntimeError("CerebrasAdapter not loaded. Call load() first.")

        if config is None:
            config = GenerationConfig()

        # Estimate token cost for rate-limit pre-check
        try:
            est_tokens = self.count_tokens(prompt).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        # Acquire rate-limit slot (run sync-compatible coroutine)
        if self._rate_limiter is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're inside an async context — caller should use async_generate
                    logger.debug("Skipping blocking rate-limit acquire (nested event loop)")
                else:
                    loop.run_until_complete(self._rate_limiter.acquire(est_tokens))
            except RuntimeError:
                # No event loop — create one
                asyncio.run(self._rate_limiter.acquire(est_tokens))

        result = self._do_generate(prompt, config, **kwargs)

        # Record actual tokens after a successful call
        if self._rate_limiter is not None:
            actual = result.metadata.get("total_tokens", 0) if result.metadata else 0
            self._rate_limiter.record(actual)

        return result

    async def async_generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Async version of :meth:`generate` — preferred inside async contexts.

        Properly awaits the rate-limit coordinator without needing nested-loop
        workarounds.
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("CerebrasAdapter not loaded. Call load() first.")

        if config is None:
            config = GenerationConfig()

        try:
            est_tokens = self.count_tokens(prompt).count + (config.max_tokens or 500)
        except Exception:
            est_tokens = 500

        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(est_tokens)

        result = self._do_generate(prompt, config, **kwargs)

        if self._rate_limiter is not None:
            actual = result.metadata.get("total_tokens", 0) if result.metadata else 0
            self._rate_limiter.record(actual)

        return result

    def _do_generate(
        self,
        prompt: str,
        config: GenerationConfig,
        **kwargs: Any,
    ) -> GenerationResult:
        """Internal: make the SDK call and return a GenerationResult."""
        messages = [{"role": "user", "content": prompt}]

        request_params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        # Map GenerationConfig fields that Cerebras supports
        if config.temperature != 0.7:
            request_params["temperature"] = config.temperature
        if config.top_p != 0.9:
            request_params["top_p"] = config.top_p
        if config.max_tokens is not None:
            request_params["max_completion_tokens"] = config.max_tokens
        if config.stop_sequences:
            request_params["stop"] = config.stop_sequences
        if config.seed is not None:
            request_params["seed"] = config.seed

        request_params.update(kwargs)

        try:
            response = self._client.chat.completions.create(**request_params)
        except Exception as exc:
            msg = str(exc)
            # Translate common Cerebras API errors into actionable messages
            if "404" in msg and "model_not_found" in msg:
                info = CEREBRAS_MODELS.get(self.model_name, {})
                hint = (
                    f" Model '{self.model_name}' is currently not accessible on your tier "
                    "(Cerebras has temporarily restricted free-tier access to high-demand "
                    "models like gpt-oss-120b and zai-glm-4.7). "
                    f"Try a free-tier model: {free_tier_models()}."
                ) if not info.get("free_tier", False) else ""
                logger.error("Cerebras API 404 for model '%s': %s", self.model_name, exc)
                raise RuntimeError(f"Cerebras generation failed: {exc}.{hint}") from exc
            if "429" in msg or "rate_limit" in msg.lower():
                logger.error("Cerebras rate-limit hit: %s", exc)
                raise RuntimeError(
                    f"Cerebras rate-limit exceeded for {self.model_name}: {exc}. "
                    "Check coordinator status via adapter.rate_limit_status()."
                ) from exc
            logger.error("Cerebras API call failed: %s", exc)
            raise RuntimeError(f"Cerebras generation failed: {exc}") from exc

        choice = response.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason or "stop"

        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

        logger.info(
            "Cerebras generated %d tokens (prompt=%d, completion=%d)",
            total_tokens,
            prompt_tokens,
            completion_tokens,
        )

        return GenerationResult(
            text=text,
            tokens_used=completion_tokens,
            finish_reason=finish_reason,
            model_name=self.model_name,
            metadata={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "provider": "cerebras",
            },
        )

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Streaming is not yet implemented.

        Raises:
            NotImplementedError: Always.  Streaming is not yet implemented.
        """
        raise NotImplementedError(
            "CerebrasAdapter.generate_stream is not yet available."
        )

    # ------------------------------------------------------------------
    # Token counting & context length
    # ------------------------------------------------------------------

    # Empirical per-family multiplier applied to tiktoken counts to account for
    # chat-template overhead (BOS, role markers, system tokens). Measured by
    # comparing tiktoken(prompt) vs. usage.prompt_tokens on real API calls.
    # Empirical: multiplier + fixed overhead (BOS/system/role tokens).
    # final = raw * mult + fixed
    _CHAT_TEMPLATE_MULTIPLIER = {
        "llama":   (1.5, 25),   # chat template adds ~25 fixed tokens
        "qwen":    (1.3, 5),
        "gpt-oss": (1.3, 10),
        "zai-glm": (1.3, 10),
    }

    def count_tokens(self, text: str) -> TokenCount:
        """Estimate token count via tiktoken with a per-family chat-template adjustment.

        The Cerebras API does not expose a standalone tokeniser; gpt-4's
        cl100k_base encoding is used as an approximation and scaled by an
        empirical multiplier to account for each model's chat-template
        overhead (BOS, role markers, etc.).  This keeps the rate-limit
        coordinator's token estimate within ~15% of the actual
        ``usage.prompt_tokens`` reported by the API.
        """
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "tiktoken is not installed. Install with: pip install tiktoken"
            ) from exc

        try:
            enc = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        raw = len(enc.encode(text))
        family = CEREBRAS_MODELS.get(self.model_name, {}).get("family", "")
        mult, fixed = self._CHAT_TEMPLATE_MULTIPLIER.get(family, (1.3, 10))
        adjusted = int(raw * mult) + fixed

        return TokenCount(count=adjusted, model_name=self.model_name)

    def get_context_length(self) -> int:
        """Return context window size for the loaded model.

        Values sourced from Cerebras inference docs (verified 2026-04-23).
        """
        return CEREBRAS_MODELS.get(self.model_name, {}).get("context", 128_000)

    def get_max_output(self) -> int:
        """Return maximum completion tokens for the loaded model."""
        return CEREBRAS_MODELS.get(self.model_name, {}).get("max_output", 8_192)

    def rate_limit_status(self) -> dict:
        """Return a snapshot of the rate-limit coordinator state.

        Returns an empty dict if rate limiting is disabled.
        """
        if self._rate_limiter is None:
            return {}
        return self._rate_limiter.status()

    # ------------------------------------------------------------------
    # Class-level helpers (mirrors cerebras_models module functions)
    # ------------------------------------------------------------------

    @classmethod
    def list_models(cls) -> list[str]:
        """Return all registered Cerebras model IDs."""
        return available_models()

    @classmethod
    def list_free_tier_models(cls) -> list[str]:
        """Return model IDs callable on the free tier."""
        return free_tier_models()

    @classmethod
    def get_model_info(cls, model_id: str) -> dict:
        """Return metadata for *model_id* (context, limits, etc.)."""
        return model_info(model_id)

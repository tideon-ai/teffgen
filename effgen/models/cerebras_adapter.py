"""
Cerebras Cloud SDK adapter for effGen.

Phase 1: gpt-oss-120b only. Streaming and tool-calling arrive in Phase 3.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any

from effgen.models.base import (
    BaseModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)
from effgen.models.cerebras_models import CEREBRAS_DEFAULT_MODEL, CEREBRAS_MODELS

logger = logging.getLogger(__name__)

_CEREBRAS_MODEL_TYPE_VALUE = "cerebras"


class _CerebrasModelType:
    """Sentinel so ModelType enum doesn't need patching in Phase 1."""
    value = _CEREBRAS_MODEL_TYPE_VALUE


class CerebrasAdapter(BaseModel):
    """
    Adapter for Cerebras Cloud inference API.

    Wraps the ``cerebras-cloud-sdk`` with the standard effGen BaseModel
    interface.  Streaming arrives in v0.2.1 Phase 3.

    Attributes:
        model_name: Cerebras model ID (e.g. ``"gpt-oss-120b"``).
        api_key: Cerebras API key (reads ``CEREBRAS_API_KEY`` env var if omitted).
        max_retries: Maximum number of SDK retry attempts.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        model_name: str = CEREBRAS_DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 60,
        **kwargs: Any,
    ) -> None:
        # BaseModel expects a ModelType enum; we use a lightweight sentinel so
        # we don't break ModelType's existing enum values in Phase 1.
        super().__init__(
            model_name=model_name,
            model_type=_CerebrasModelType(),  # type: ignore[arg-type]
            context_length=CEREBRAS_MODELS.get(model_name, {}).get("context", 128_000),
        )
        self._api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._extra_kwargs = kwargs
        self._client: Any = None

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
        try:
            from cerebras.cloud.sdk import Cerebras
        except ImportError as exc:
            raise RuntimeError(
                "cerebras-cloud-sdk is not installed. "
                "Install with: pip install 'effgen[cerebras]'"
            ) from exc

        api_key = self._api_key or os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError(
                "Cerebras API key not found. Set the CEREBRAS_API_KEY "
                "environment variable or pass api_key= to CerebrasAdapter."
            )

        self._client = Cerebras(api_key=api_key)
        self._is_loaded = True
        self._metadata = {
            "model_name": self.model_name,
            "context_length": self.get_context_length(),
            "provider": "cerebras",
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

        Args:
            prompt: User prompt text.
            config: Optional generation config (temperature, max_tokens, …).
            **kwargs: Forwarded to ``chat.completions.create``.

        Returns:
            GenerationResult with the generated text and token usage.

        Raises:
            RuntimeError: If ``load()`` has not been called or the API call fails.
        """
        if not self._is_loaded or self._client is None:
            raise RuntimeError("CerebrasAdapter not loaded. Call load() first.")

        if config is None:
            config = GenerationConfig()

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
            NotImplementedError: Always.  Streaming arrives in v0.2.1 phase 3.
        """
        raise NotImplementedError(
            "CerebrasAdapter.generate_stream is not yet available. "
            "Streaming support arrives in v0.2.1 phase 3."
        )

    # ------------------------------------------------------------------
    # Token counting & context length
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> TokenCount:
        """Estimate token count via tiktoken (gpt-4 encoding as proxy).

        The Cerebras API does not expose a standalone tokeniser; gpt-4's
        cl100k_base encoding is used as an approximation.  Accuracy vs.
        Cerebras's actual count will be validated in Phase 2.

        # TODO(phase2): replace with native Cerebras token counter once available.
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

        return TokenCount(count=len(enc.encode(text)), model_name=self.model_name)

    def get_context_length(self) -> int:
        """Return context length for the loaded model.

        Returns 128 000 for gpt-oss-120b.
        Source: Cerebras inference docs (verified 2026-04-23).
        """
        return CEREBRAS_MODELS.get(self.model_name, {}).get("context", 128_000)

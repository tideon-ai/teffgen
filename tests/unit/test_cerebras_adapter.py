"""Unit tests for CerebrasAdapter — mocks allowed here per build plan."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from effgen.models.base import GenerationConfig, GenerationResult, TokenCount
from effgen.models.cerebras_adapter import CerebrasAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(text: str = "hello", prompt_tokens: int = 5, completion_tokens: int = 3):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = "stop"

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# 1. Constructor defaults
# ---------------------------------------------------------------------------

class TestCerebrasAdapterInit:
    def test_default_model(self):
        adapter = CerebrasAdapter()
        # Default is llama3.1-8b (free-tier callable); gpt-oss-120b listed but
        # requires higher tier — see TODO_p1_gpt_oss_120b_access.md
        assert adapter.model_name == "llama3.1-8b"

    def test_custom_model(self):
        adapter = CerebrasAdapter(model_name="gpt-oss-120b")
        assert adapter.model_name == "gpt-oss-120b"

    def test_not_loaded_on_init(self):
        adapter = CerebrasAdapter()
        assert not adapter.is_loaded()
        assert adapter._client is None


# ---------------------------------------------------------------------------
# 2. load() reads env key
# ---------------------------------------------------------------------------

class TestCerebrasAdapterLoad:
    def test_load_reads_env_key(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "test-key-abc")
        with patch("cerebras.cloud.sdk.Cerebras") as MockCerebras:
            MockCerebras.return_value = MagicMock()
            adapter = CerebrasAdapter()
            adapter.load()
            MockCerebras.assert_called_once_with(api_key="test-key-abc")
            assert adapter.is_loaded()

    def test_load_uses_explicit_key(self, monkeypatch):
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        with patch("cerebras.cloud.sdk.Cerebras") as MockCerebras:
            MockCerebras.return_value = MagicMock()
            adapter = CerebrasAdapter(api_key="explicit-key")
            adapter.load()
            MockCerebras.assert_called_once_with(api_key="explicit-key")

    def test_load_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        adapter = CerebrasAdapter()
        with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
            adapter.load()

    def test_load_raises_if_sdk_missing(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "k")
        with patch.dict("sys.modules", {"cerebras": None, "cerebras.cloud": None, "cerebras.cloud.sdk": None}):
            adapter = CerebrasAdapter()
            with pytest.raises((RuntimeError, ImportError)):
                adapter.load()


# ---------------------------------------------------------------------------
# 3. generate() formats messages correctly
# ---------------------------------------------------------------------------

class TestCerebrasAdapterGenerate:
    def _loaded_adapter(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
        with patch("cerebras.cloud.sdk.Cerebras") as MockCerebras:
            mock_client = MagicMock()
            MockCerebras.return_value = mock_client
            adapter = CerebrasAdapter()
            adapter.load()
        return adapter

    def test_generate_formats_messages(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
        with patch("cerebras.cloud.sdk.Cerebras") as MockCerebras:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _make_mock_response("hi")
            MockCerebras.return_value = mock_client

            adapter = CerebrasAdapter()
            adapter.load()
            result = adapter.generate("Say hello")

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["messages"] == [{"role": "user", "content": "Say hello"}]
            assert call_kwargs["model"] == adapter.model_name

    def test_generate_returns_generation_result(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
        with patch("cerebras.cloud.sdk.Cerebras") as MockCerebras:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _make_mock_response("world", 10, 4)
            MockCerebras.return_value = mock_client

            adapter = CerebrasAdapter()
            adapter.load()
            result = adapter.generate("prompt")

            assert isinstance(result, GenerationResult)
            assert result.text == "world"
            assert result.tokens_used == 4
            assert result.finish_reason == "stop"
            assert result.model_name == adapter.model_name
            assert result.metadata["prompt_tokens"] == 10

    def test_generate_propagates_api_error(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
        with patch("cerebras.cloud.sdk.Cerebras") as MockCerebras:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError("API down")
            MockCerebras.return_value = mock_client

            adapter = CerebrasAdapter()
            adapter.load()
            with pytest.raises(RuntimeError, match="Cerebras generation failed"):
                adapter.generate("prompt")

    def test_generate_raises_if_not_loaded(self):
        adapter = CerebrasAdapter()
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.generate("hello")


# ---------------------------------------------------------------------------
# 4. generate_stream() raises NotImplementedError
# ---------------------------------------------------------------------------

class TestCerebrasAdapterStream:
    def test_generate_stream_raises_not_implemented(self):
        adapter = CerebrasAdapter()
        with pytest.raises(NotImplementedError, match="phase 3"):
            # consume the iterator if it returned one
            result = adapter.generate_stream("test")
            if hasattr(result, "__next__"):
                next(result)


# ---------------------------------------------------------------------------
# 5. count_tokens and get_context_length
# ---------------------------------------------------------------------------

class TestCerebrasAdapterTokens:
    def test_count_tokens_returns_token_count(self):
        adapter = CerebrasAdapter()
        tc = adapter.count_tokens("Hello world")
        assert isinstance(tc, TokenCount)
        assert tc.count > 0
        assert tc.model_name == adapter.model_name

    def test_get_context_length(self):
        adapter = CerebrasAdapter(model_name="gpt-oss-120b")
        assert adapter.get_context_length() == 128_000

    def test_get_context_length_llama(self):
        adapter = CerebrasAdapter(model_name="llama3.1-8b")
        assert adapter.get_context_length() == 8_192

    def test_get_context_length_unknown_model(self):
        adapter = CerebrasAdapter(model_name="unknown-model")
        # Falls back to 128_000
        assert adapter.get_context_length() == 128_000


# ---------------------------------------------------------------------------
# 6. unload()
# ---------------------------------------------------------------------------

class TestCerebrasAdapterUnload:
    def test_unload_clears_client(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "k")
        with patch("cerebras.cloud.sdk.Cerebras") as MockCerebras:
            MockCerebras.return_value = MagicMock()
            adapter = CerebrasAdapter()
            adapter.load()
            assert adapter.is_loaded()
            adapter.unload()
            assert not adapter.is_loaded()
            assert adapter._client is None

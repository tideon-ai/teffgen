"""OpenAI-compatible local embeddings API.

Provides a ``/v1/embeddings`` endpoint that works out of the box without
paid APIs. Uses ``sentence-transformers`` when available, otherwise falls
back to an in-process TF-IDF hashing embedder so the endpoint remains
functional in minimal installs.

Caching:
    * In-memory LRU cache for repeated texts.
    * Optional SQLite-backed persistent cache keyed by (model, text_hash).

Usage:
    >>> from fastapi import FastAPI
    >>> from effgen.api.embeddings import create_embeddings_router
    >>> app = FastAPI()
    >>> app.include_router(create_embeddings_router())
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import threading
from collections import OrderedDict
from typing import Any

# Model aliases — OpenAI-style names mapped to local models.
MODEL_ALIASES = {
    "text-embedding-small": "sentence-transformers/all-MiniLM-L6-v2",
    "text-embedding-ada-002": "sentence-transformers/all-MiniLM-L6-v2",
    "text-embedding-3-small": "sentence-transformers/all-MiniLM-L6-v2",
    "text-embedding-3-large": "sentence-transformers/all-mpnet-base-v2",
}
DEFAULT_DIM = 384  # TF-IDF fallback dimension


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
class LRUCache:
    """Thread-safe LRU cache for embeddings keyed by (model, text)."""

    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size
        self._data: "OrderedDict[str, list[float]]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(model: str, text: str) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{model}:{h}"

    def get(self, model: str, text: str) -> list[float] | None:
        k = self.key(model, text)
        with self._lock:
            val = self._data.get(k)
            if val is not None:
                self._data.move_to_end(k)
            return val

    def put(self, model: str, text: str, vec: list[float]) -> None:
        k = self.key(model, text)
        with self._lock:
            self._data[k] = vec
            self._data.move_to_end(k)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)


class SQLiteCache:
    """Persistent embedding cache backed by SQLite."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.path.expanduser("~/.effgen/embeddings_cache.db")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  model TEXT NOT NULL,"
            "  text_hash TEXT NOT NULL,"
            "  vector BLOB NOT NULL,"
            "  dim INTEGER NOT NULL,"
            "  PRIMARY KEY (model, text_hash)"
            ")"
        )
        self._conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT vector, dim FROM embeddings WHERE model=? AND text_hash=?",
                (model, self._hash(text)),
            ).fetchone()
        if not row:
            return None
        blob, dim = row
        import struct

        return list(struct.unpack(f"{dim}f", blob))

    def put(self, model: str, text: str, vec: list[float]) -> None:
        import struct

        blob = struct.pack(f"{len(vec)}f", *vec)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (model, text_hash, vector, dim)"
                " VALUES (?, ?, ?, ?)",
                (model, self._hash(text), blob, len(vec)),
            )
            self._conn.commit()


# ---------------------------------------------------------------------------
# Embedder backends
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


class TFIDFEmbedder:
    """Hashing TF-IDF embedder — dependency-free fallback.

    Produces fixed-dimension L2-normalized sparse-to-dense vectors using a
    hashing trick. Not as good as a neural embedder, but deterministic and
    works without any additional packages.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in _WORD_RE.findall(text)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = self._tokenize(text)
            if not tokens:
                out.append(vec)
                continue
            # term frequency with hashing
            for tok in tokens:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 32) & 1 else -1.0
                vec[idx] += sign
            # IDF-ish log dampening
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


class SentenceTransformerEmbedder:
    """Wrapper around ``sentence-transformers`` when installed."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class EmbeddingEngine:
    """Main embedding engine: resolves models, caches, dispatches."""

    def __init__(
        self,
        lru_size: int = 1024,
        persistent_cache: bool = True,
        cache_path: str | None = None,
    ) -> None:
        self.lru = LRUCache(max_size=lru_size)
        self.sqlite: SQLiteCache | None
        try:
            self.sqlite = SQLiteCache(cache_path) if persistent_cache else None
        except Exception:
            self.sqlite = None
        self._backends: dict = {}
        self._backend_lock = threading.Lock()

    def _resolve_model(self, model: str) -> str:
        return MODEL_ALIASES.get(model, model)

    def _get_backend(self, model: str) -> Any:
        resolved = self._resolve_model(model)
        with self._backend_lock:
            if resolved in self._backends:
                return self._backends[resolved]
            try:
                backend: Any = SentenceTransformerEmbedder(resolved)
            except Exception:
                backend = TFIDFEmbedder()
            self._backends[resolved] = backend
            return backend

    def embed(self, texts: list[str], model: str = "text-embedding-small") -> list[list[float]]:
        if not texts:
            return []
        resolved = self._resolve_model(model)

        # Lookup caches
        results: list[list[float] | None] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_texts: list[str] = []
        for i, t in enumerate(texts):
            cached = self.lru.get(resolved, t)
            if cached is None and self.sqlite is not None:
                cached = self.sqlite.get(resolved, t)
                if cached is not None:
                    self.lru.put(resolved, t, cached)
            if cached is not None:
                results[i] = cached
            else:
                missing_idx.append(i)
                missing_texts.append(t)

        if missing_texts:
            backend = self._get_backend(model)
            new_vecs = backend.embed(missing_texts)
            for idx, text, vec in zip(missing_idx, missing_texts, new_vecs):
                results[idx] = vec
                self.lru.put(resolved, text, vec)
                if self.sqlite is not None:
                    try:
                        self.sqlite.put(resolved, text, vec)
                    except Exception:
                        pass

        return [r for r in results if r is not None]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FastAPI router (optional dependency)
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel as _PydBaseModel  # type: ignore
    from pydantic import Field as _PydField

    class EmbeddingRequest(_PydBaseModel):
        model: str = _PydField(default="text-embedding-small")
        input: str | list[str]

    class EmbeddingItem(_PydBaseModel):
        object: str = "embedding"
        index: int
        embedding: list[float]

    class EmbeddingUsage(_PydBaseModel):
        prompt_tokens: int = 0
        total_tokens: int = 0

    class EmbeddingResponse(_PydBaseModel):
        object: str = "list"
        data: list[EmbeddingItem]
        model: str
        usage: EmbeddingUsage
except Exception:  # pragma: no cover - pydantic optional
    EmbeddingRequest = None  # type: ignore
    EmbeddingItem = None  # type: ignore
    EmbeddingUsage = None  # type: ignore
    EmbeddingResponse = None  # type: ignore


def create_embeddings_router(engine: EmbeddingEngine | None = None) -> Any:
    """Create a FastAPI router exposing ``/v1/embeddings``.

    Requires ``fastapi`` and ``pydantic`` to be installed. Otherwise import
    of this function will raise.
    """
    from fastapi import APIRouter, Body, HTTPException  # type: ignore

    eng = engine or EmbeddingEngine()

    router = APIRouter()

    @router.post("/v1/embeddings", response_model=EmbeddingResponse)
    def create_embeddings(req: EmbeddingRequest = Body(...)) -> EmbeddingResponse:
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        if not texts:
            raise HTTPException(status_code=400, detail="input must not be empty")
        try:
            vecs = eng.embed(texts, model=req.model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"embedding failed: {e}")
        total_tokens = sum(len(t.split()) for t in texts)
        return EmbeddingResponse(
            data=[EmbeddingItem(index=i, embedding=v) for i, v in enumerate(vecs)],
            model=req.model,
            usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )

    return router


__all__ = [
    "MODEL_ALIASES",
    "LRUCache",
    "SQLiteCache",
    "TFIDFEmbedder",
    "SentenceTransformerEmbedder",
    "EmbeddingEngine",
    "create_embeddings_router",
]

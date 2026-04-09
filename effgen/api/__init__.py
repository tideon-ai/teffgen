"""effGen API Server v2 — Production Gateway.

Phase 12 modules:
- openai_compat: OpenAI-compatible /v1/chat/completions and /v1/completions
- queue: RequestQueue with priority, fair scheduling, backpressure
- pool: AgentPool with min/max size and auto-scaling
- tenancy: Tenant config, API key management, per-tenant limits
- middleware: Production middleware (request ID, CORS, gzip, etc.)
"""
from __future__ import annotations

from effgen.api.openai_compat import (
    MODEL_ALIASES,
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    create_openai_router,
    resolve_model_alias,
)
from effgen.api.pool import AgentPool, PooledAgent
from effgen.api.queue import QueuedRequest, RequestQueue, RequestPriority
from effgen.api.tenancy import APIKey, Tenant, TenantManager
from effgen.api.middleware import install_production_middleware
from effgen.api.embeddings import (
    EmbeddingEngine,
    LRUCache as EmbeddingLRUCache,
    SQLiteCache as EmbeddingSQLiteCache,
    TFIDFEmbedder,
    create_embeddings_router,
)

__all__ = [
    "MODEL_ALIASES",
    "ChatCompletionRequest",
    "ChatMessage",
    "CompletionRequest",
    "create_openai_router",
    "resolve_model_alias",
    "AgentPool",
    "PooledAgent",
    "RequestQueue",
    "QueuedRequest",
    "RequestPriority",
    "Tenant",
    "APIKey",
    "TenantManager",
    "install_production_middleware",
    "EmbeddingEngine",
    "EmbeddingLRUCache",
    "EmbeddingSQLiteCache",
    "TFIDFEmbedder",
    "create_embeddings_router",
]

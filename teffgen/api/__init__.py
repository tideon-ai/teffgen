"""tideon.ai API Server v2 — Production Gateway.

Phase 12 modules:
- openai_compat: OpenAI-compatible /v1/chat/completions and /v1/completions
- queue: RequestQueue with priority, fair scheduling, backpressure
- pool: AgentPool with min/max size and auto-scaling
- tenancy: Tenant config, API key management, per-tenant limits
- middleware: Production middleware (request ID, CORS, gzip, etc.)
"""
from __future__ import annotations

from teffgen.api.embeddings import (
    EmbeddingEngine,
    TFIDFEmbedder,
    create_embeddings_router,
)
from teffgen.api.embeddings import (
    LRUCache as EmbeddingLRUCache,
)
from teffgen.api.embeddings import (
    SQLiteCache as EmbeddingSQLiteCache,
)
from teffgen.api.middleware import install_production_middleware
from teffgen.api.openai_compat import (
    MODEL_ALIASES,
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    create_openai_router,
    resolve_model_alias,
)
from teffgen.api.pool import AgentPool, PooledAgent
from teffgen.api.queue import QueuedRequest, RequestPriority, RequestQueue
from teffgen.api.tenancy import APIKey, Tenant, TenantManager

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

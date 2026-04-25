"""
Agent Communication Protocol (ACP) Integration.

This package implements IBM's ACP protocol for agent-to-agent communication,
providing agent manifests, synchronous/asynchronous requests, task tracking,
and OpenTelemetry instrumentation.
"""

from .client import (
    ACPAuthHandler,
    ACPClient,
    ACPClientConfig,
    ACPDiscoveryClient,
    APIKeyAuthHandler,
    BearerAuthHandler,
    TokenAuthHandler,
    create_capability_token,
)
from .protocol import (
    ACPError,
    ACPProtocolHandler,
    ACPRequest,
    ACPResponse,
    ACPVersion,
    AgentManifest,
    CapabilityDefinition,
    CapabilityToken,
    ErrorSeverity,
    RequestType,
    SchemaDefinition,
    TaskInfo,
    TaskStatus,
)
from .server import (
    ACPCapabilityRegistry,
    ACPServer,
    ACPServerConfig,
    capability,
)

__all__ = [
    # Protocol
    "ACPProtocolHandler",
    "AgentManifest",
    "ACPRequest",
    "ACPResponse",
    "ACPError",
    "TaskInfo",
    "TaskStatus",
    "RequestType",
    "ErrorSeverity",
    "SchemaDefinition",
    "CapabilityDefinition",
    "CapabilityToken",
    "ACPVersion",
    # Client
    "ACPClient",
    "ACPClientConfig",
    "ACPAuthHandler",
    "TokenAuthHandler",
    "APIKeyAuthHandler",
    "BearerAuthHandler",
    "ACPDiscoveryClient",
    "create_capability_token",
    # Server
    "ACPServer",
    "ACPServerConfig",
    "ACPCapabilityRegistry",
    "capability",
]

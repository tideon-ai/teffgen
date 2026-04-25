"""
Agent-to-Agent (A2A) Protocol Integration.

This package implements Google's A2A protocol for agent-to-agent communication,
providing message protocol handling, task lifecycle management, and context passing.
"""

from .agent_card import (
    AgentCard,
    AuthScheme,
    Capability,
    CapabilityType,
    EndpointConfig,
)
from .client import (
    A2AClient,
    A2AClientConfig,
    APIKeyAuthHandler,
    AuthHandler,
    BearerAuthHandler,
    OAuth2AuthHandler,
    discover_agents,
)
from .protocol import (
    A2AError,
    A2AMessage,
    A2AProtocolHandler,
    A2AVersion,
    Artifact,
    ErrorCode,
    MessagePart,
    MessagePartType,
    Task,
    TaskRequest,
    TaskState,
    TaskUpdate,
)

__all__ = [
    # Protocol
    "A2AProtocolHandler",
    "Task",
    "TaskRequest",
    "TaskUpdate",
    "TaskState",
    "A2AMessage",
    "A2AError",
    "ErrorCode",
    "Artifact",
    "MessagePart",
    "MessagePartType",
    "A2AVersion",
    # Agent Card
    "AgentCard",
    "Capability",
    "CapabilityType",
    "EndpointConfig",
    "AuthScheme",
    # Client
    "A2AClient",
    "A2AClientConfig",
    "AuthHandler",
    "BearerAuthHandler",
    "OAuth2AuthHandler",
    "APIKeyAuthHandler",
    "discover_agents",
]

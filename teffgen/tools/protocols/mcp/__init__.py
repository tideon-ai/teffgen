"""
Model Context Protocol (MCP) implementation for tideon.ai.

This module provides MCP client and server implementations for tool
and resource communication.
"""

from .client import (
    ConnectionState,
    HTTPTransport,
    MCPClient,
    MCPServerConfig,
    MCPToolBridge,
    MCPTransport,
    SSETransport,
    StdioTransport,
)
from .protocol import (
    ErrorCode,
    MCPCapabilities,
    MCPError,
    MCPNotification,
    MCPProtocolHandler,
    MCPRequest,
    MCPResource,
    MCPResponse,
    MCPTool,
    MCPVersion,
    MessageType,
    TransportType,
)
from .server import (
    MCPServer,
    create_server,
    main_http,
    main_stdio,
)

__all__ = [
    # Protocol
    "MCPProtocolHandler",
    "MCPTool",
    "MCPResource",
    "MCPCapabilities",
    "MCPRequest",
    "MCPResponse",
    "MCPNotification",
    "MCPError",
    "ErrorCode",
    "TransportType",
    "MessageType",
    "MCPVersion",
    # Client
    "MCPClient",
    "MCPServerConfig",
    "MCPTransport",
    "StdioTransport",
    "HTTPTransport",
    "SSETransport",
    "MCPToolBridge",
    "ConnectionState",
    # Server
    "MCPServer",
    "create_server",
    "main_stdio",
    "main_http",
]

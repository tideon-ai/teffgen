"""
Protocol integrations for effGen.

This package provides implementations for various agent communication protocols:
- MCP (Model Context Protocol): Anthropic's protocol for context and tool sharing
  - mcp: Custom implementation (legacy)
  - mcp_official: Official MCP SDK implementation (recommended)
- A2A (Agent-to-Agent): Google's protocol for agent communication
- ACP (Agent Communication Protocol): IBM's protocol for agent interoperability
"""

from . import a2a, acp, mcp, mcp_official

__all__ = [
    "mcp",
    "mcp_official",
    "a2a",
    "acp",
]

"""Unit tests for protocol implementations."""

import pytest

from teffgen.tools.protocols.acp.protocol import (
    ACPProtocolHandler,
    AgentManifest,
    CapabilityDefinition,
    SchemaDefinition,
    TaskStatus,
)


class TestACPProtocol:
    """Tests for ACP Protocol Handler."""

    @pytest.fixture
    def manifest(self):
        return AgentManifest(
            agentId="test-agent",
            name="Test Agent",
            version="1.0.0",
            description="A test agent",
            capabilities=[
                CapabilityDefinition(
                    name="calculate",
                    description="Perform calculations",
                    inputSchema=SchemaDefinition(
                        type="object",
                        properties={"expression": {"type": "string"}},
                        required=["expression"],
                    ),
                )
            ],
        )

    @pytest.fixture
    def handler(self, manifest):
        return ACPProtocolHandler(manifest)

    def test_create_handler(self, handler):
        assert handler is not None

    def test_manifest_properties(self, manifest):
        assert manifest.agentId == "test-agent"
        assert manifest.name == "Test Agent"
        assert len(manifest.capabilities) == 1

    def test_create_request(self, handler):
        request = handler.create_request("calculate", {"expression": "2+2"})
        assert request is not None
        assert request.capability == "calculate"

    def test_validate_valid_request(self, handler):
        request = handler.create_request("calculate", {"expression": "2+2"})
        is_valid, error = handler.validate_request(request)
        assert is_valid is True

    def test_validate_missing_required_field(self, handler):
        request = handler.create_request("calculate", {})
        is_valid, error = handler.validate_request(request)
        assert is_valid is False
        assert "expression" in str(error).lower() or error is not None

    def test_task_status_values(self):
        assert TaskStatus.PENDING is not None
        assert TaskStatus.RUNNING is not None
        assert TaskStatus.COMPLETED is not None
        assert TaskStatus.FAILED is not None

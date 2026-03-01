"""
Shared test fixtures for the effGen test suite.
"""

import os
import sys
import warnings
import pytest
import tempfile
import shutil
from pathlib import Path

# Suppress ImportWarning from optional dependencies before importing effgen
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure effgen package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.fixtures.mock_models import MockModel, MockToolCallingModel, MockStreamingModel
from effgen.core.agent import Agent, AgentConfig, AgentResponse
from effgen.models.base import BaseModel, GenerationConfig, GenerationResult
from effgen.tools.builtin import Calculator, DateTimeTool, JSONTool, TextProcessingTool


# ---------------------------------------------------------------------------
# Mock Model Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """A simple mock model that returns 'Final Answer: 42'."""
    return MockModel(responses=[
        "Thought: I know the answer.\nFinal Answer: 42"
    ])


@pytest.fixture
def mock_tool_model():
    """Mock model that calls calculator then gives final answer."""
    return MockToolCallingModel(tool_sequence=[
        {
            "thought": "I need to calculate this.",
            "action": "calculator",
            "action_input": '{"expression": "2 + 2"}',
        },
        {
            "thought": "I now know the final answer.",
            "action": "Final Answer",
            "action_input": "The answer is 4.",
        },
    ])


@pytest.fixture
def mock_streaming_model():
    """Mock model for streaming tests."""
    return MockStreamingModel(responses=[
        "Thought: Let me think.\nFinal Answer: Hello, world!"
    ])


# ---------------------------------------------------------------------------
# Tool Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calculator():
    return Calculator()


@pytest.fixture
def datetime_tool():
    return DateTimeTool()


@pytest.fixture
def json_tool():
    return JSONTool()


@pytest.fixture
def text_tool():
    return TextProcessingTool()


# ---------------------------------------------------------------------------
# Agent Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_agent(mock_model):
    """Agent with no tools."""
    config = AgentConfig(
        name="test-basic",
        model=mock_model,
        tools=[],
        max_iterations=3,
        enable_memory=False,
        enable_sub_agents=False,
    )
    return Agent(config=config)


@pytest.fixture
def tool_agent(mock_tool_model, calculator):
    """Agent with calculator tool."""
    config = AgentConfig(
        name="test-tool",
        model=mock_tool_model,
        tools=[calculator],
        max_iterations=5,
        enable_memory=False,
        enable_sub_agents=False,
    )
    return Agent(config=config)


@pytest.fixture
def multi_tool_agent(mock_model, calculator, datetime_tool, json_tool):
    """Agent with multiple tools."""
    config = AgentConfig(
        name="test-multi",
        model=mock_model,
        tools=[calculator, datetime_tool, json_tool],
        max_iterations=5,
        enable_memory=False,
        enable_sub_agents=False,
    )
    return Agent(config=config)


# ---------------------------------------------------------------------------
# Temp Directory Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    d = tempfile.mkdtemp(prefix="effgen_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# GPU Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def free_gpu_id():
    """Find a free GPU with minimal memory usage."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        # Find GPU with least memory usage
        min_mem = float("inf")
        best_gpu = 0
        for i in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(i)
            if mem_used < min_mem:
                min_mem = mem_used
                best_gpu = i
        return best_gpu
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Fixtures directory path
# ---------------------------------------------------------------------------

@pytest.fixture
def fixtures_dir():
    """Path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def knowledge_base_dir(fixtures_dir):
    """Path to the knowledge base fixtures."""
    return fixtures_dir / "knowledge_base"

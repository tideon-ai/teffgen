"""Integration tests for tool fallback chains."""

import pytest
from effgen import Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator


@pytest.mark.gpu
class TestFallback:
    """Test fallback chain functionality."""

    def test_agent_with_fallback_enabled(self, real_model):
        agent = Agent(config=AgentConfig(
            name="fallback_test",
            model=real_model,
            tools=[Calculator()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
            enable_fallback=True,
        ))
        result = agent.run("What is 10 + 20?")
        assert result.success
        assert "30" in result.output

    def test_agent_with_fallback_disabled(self, real_model):
        agent = Agent(config=AgentConfig(
            name="no_fallback_test",
            model=real_model,
            tools=[Calculator()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
            enable_fallback=False,
        ))
        result = agent.run("What is 10 + 20?")
        assert result.success
        assert "30" in result.output

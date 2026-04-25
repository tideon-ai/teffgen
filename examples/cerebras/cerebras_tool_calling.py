"""
Cerebras native tool-calling example.

Prerequisites:
    pip install "teffgen[cerebras]"
    export CEREBRAS_API_KEY="your-key"   # or place in ~/.teffgen/.env

What this demonstrates:
  - Native OpenAI-compatible function calling on Cerebras models
  - Multi-step math agent: "(17 × 23) + sqrt(144)"
  - CostTracker usage reporting
  - Fallback to ReAct for unsupported models
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".teffgen" / ".env", override=False)

if not os.getenv("CEREBRAS_API_KEY"):
    raise SystemExit("Set CEREBRAS_API_KEY in ~/.teffgen/.env or the environment.")

from teffgen.core.agent import Agent, AgentConfig  # noqa: E402
from teffgen.models._cost import CostTracker  # noqa: E402
from teffgen.models.cerebras_adapter import CerebrasAdapter  # noqa: E402
from teffgen.tools.builtin.calculator import Calculator  # noqa: E402


def run_math_agent(model_id: str, task: str) -> None:
    print(f"\n{'='*60}")
    print(f"Agent: {model_id}")
    print(f"Task: {task}")
    print("="*60)

    adapter = CerebrasAdapter(model_id, enable_rate_limiting=False)
    adapter.load()

    supports_tools = adapter.supports_tool_calling()
    print(f"Supports native tools: {supports_tools}")

    config = AgentConfig(
        name=f"cerebras-{model_id}",
        model=adapter,
        tools=[Calculator()],
        system_prompt=(
            "You are a precise math assistant. Use the calculator tool for every "
            "computation — never compute mentally. Show your reasoning step by step."
        ),
        max_iterations=8,
        temperature=0.1,
        tool_calling_mode="auto",
    )
    agent = Agent(config)

    try:
        response = agent.run(task)
        print(f"\nOutput: {response.output}")
        print(f"Tool calls: {response.tool_calls}")
        print(f"Success: {response.success}")
        print(f"Tokens used: {response.tokens_used}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        adapter.unload()


def main():
    task = "Calculate (17 * 23) + sqrt(144). Use the calculator for each step."

    # llama3.1-8b supports native tools
    run_math_agent("llama3.1-8b", task)

    # qwen-3-235b also supports native tools
    run_math_agent("qwen-3-235b-a22b-instruct-2507", task)

    # Show cost (always $0 for Cerebras free tier)
    print("\n--- Cost Summary ---")
    for row in CostTracker.get().summary():
        print(
            f"  {row['provider']}/{row['model']}: "
            f"requests={row['requests']} "
            f"tokens={row['total_tokens']} "
            f"cost=${row['cost_usd']:.6f}"
        )


if __name__ == "__main__":
    main()

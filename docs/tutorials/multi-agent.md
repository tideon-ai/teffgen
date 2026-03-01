# Multi-Agent Orchestration

effGen supports task decomposition via sub-agents.

## How Sub-Agents Work

When `enable_sub_agents=True` (default), the agent can decompose complex tasks:

1. The router analyzes the task complexity
2. If sub-agents are needed, the task is split into subtasks
3. Each subtask runs with its own agent configuration
4. Results are aggregated into a final response

## Example

```python
from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import Calculator, PythonREPL, WebSearch

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

agent = Agent(AgentConfig(
    name="orchestrator",
    model=model,
    tools=[Calculator(), PythonREPL(), WebSearch()],
    enable_sub_agents=True,
    max_iterations=15,
))

# Complex task that benefits from decomposition
result = agent.run(
    "Research the GDP of the top 5 economies, then calculate "
    "the average and standard deviation"
)
```

## Execution Modes

```python
from effgen.core.agent import AgentMode

# Let the agent decide
result = agent.run(task, mode=AgentMode.AUTO)

# Force single-agent execution
result = agent.run(task, mode=AgentMode.SINGLE)

# Force sub-agent decomposition
result = agent.run(task, mode=AgentMode.SUB_AGENTS)
```

## Tips

- Sub-agents work best with larger models (3B+)
- Simple tasks run faster in `SINGLE` mode
- Use `--verbose` in CLI to see the routing decision

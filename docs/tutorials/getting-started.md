# Getting Started with effGen

Get up and running with effGen in 5 minutes.

## Installation

```bash
pip install effgen
```

For GPU support with vLLM:
```bash
pip install effgen[vllm]
```

## Your First Agent

```python
from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import Calculator

# Load a small language model (runs on a single GPU)
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# Create an agent with a calculator tool
agent = Agent(AgentConfig(
    name="my-agent",
    model=model,
    tools=[Calculator()],
))

# Run a task
result = agent.run("What is 42 * 58?")
print(result.output)  # "2436"
```

## Using Presets

Skip configuration boilerplate with presets:

```python
from effgen.presets import create_agent
from effgen import load_model

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# Math agent (Calculator + PythonREPL)
math_agent = create_agent("math", model)
result = math_agent.run("What is the square root of 144?")

# Coding agent (CodeExecutor + PythonREPL + FileOps + Bash)
code_agent = create_agent("coding", model)
result = code_agent.run("Write a Python script that lists prime numbers under 100")
```

Available presets: `math`, `research`, `coding`, `general`, `minimal`

## CLI Usage

```bash
# Run a task directly
effgen run "What is 2+2?" --model Qwen/Qwen2.5-3B-Instruct

# Use a preset
effgen run --preset math "What is the square root of 144?"

# Interactive chat
effgen chat --model Qwen/Qwen2.5-3B-Instruct

# List presets
effgen presets

# Verbose output with execution trace
effgen run "Calculate 10!" --preset math --verbose
```

## Next Steps

- [Building a Math Agent](building-math-agent.md)
- [Building a Research Agent](building-research-agent.md)
- [Custom Tools Guide](custom-tools.md)
- [API Reference](../api/reference.md)

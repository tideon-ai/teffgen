# Getting Started with tideon.ai

Get up and running with tideon.ai in 5 minutes.

## Installation

```bash
pip install teffgen
```

For GPU support with vLLM:
```bash
pip install teffgen[vllm]
```

## Your First Agent

```python
from teffgen import Agent, AgentConfig, load_model
from teffgen.tools.builtin import Calculator

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
from teffgen.presets import create_agent
from teffgen import load_model

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
teffgen run "What is 2+2?" --model Qwen/Qwen2.5-3B-Instruct

# Use a preset
teffgen run --preset math "What is the square root of 144?"

# Interactive chat
teffgen chat --model Qwen/Qwen2.5-3B-Instruct

# List presets
teffgen presets

# Verbose output with execution trace
teffgen run "Calculate 10!" --preset math --verbose
```

## Next Steps

- [Building a Math Agent](building-math-agent.md)
- [Building a Research Agent](building-research-agent.md)
- [Custom Tools Guide](custom-tools.md)
- [API Reference](../api/reference.md)

# Native Tool Calling

tideon.ai v0.2.0 supports native function calling for models that have built-in tool-use capabilities (Qwen, Llama, Mistral, and others). This bypasses text-based ReAct parsing for faster, more reliable tool execution.

## Tool Calling Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `"auto"` | Automatically selects native if model supports it, else ReAct | Default — works everywhere |
| `"native"` | Uses model's built-in function calling format | Qwen 2.5+, Llama 3.2+, Mistral |
| `"react"` | Text-based ReAct reasoning loop | Any model, maximum compatibility |
| `"hybrid"` | Tries native first, falls back to ReAct | Best accuracy with capable models |

## Basic Usage

```python
from teffgen import Agent, load_model
from teffgen.core.agent import AgentConfig
from teffgen.tools.builtin import Calculator

model = load_model("Qwen/Qwen2.5-3B-Instruct")

config = AgentConfig(
    name="native_agent",
    model=model,
    tools=[Calculator()],
    tool_calling_mode="native",  # Use native function calling
)

agent = Agent(config=config)
result = agent.run("What is 42 * 58?")
print(result.output)  # 2436
```

## Structured Output

Force the agent to return JSON matching a schema:

```python
config = AgentConfig(
    name="structured_agent",
    model=model,
    tools=[Calculator()],
    output_format="json",
    output_schema={
        "type": "object",
        "properties": {
            "answer": {"type": "number"},
            "explanation": {"type": "string"}
        },
        "required": ["answer"]
    },
)
```

### With Pydantic Models

```python
from pydantic import BaseModel

class MathResult(BaseModel):
    answer: float
    explanation: str

result = agent.run("What is 15% of 200?", output_model=MathResult)
parsed = result.metadata["parsed_output"]  # MathResult instance
print(parsed.answer)  # 30.0
```

## Checking Model Support

```python
model = load_model("Qwen/Qwen2.5-3B-Instruct")
print(model.supports_tool_calling())  # True

model = load_model("some-small-model")
print(model.supports_tool_calling())  # False — use "react" or "auto"
```

## How It Works

1. Tools are converted to JSON Schema definitions via `tools_to_definitions()`
2. Definitions are passed to the model's chat template via the `tools` parameter
3. The model produces `<tool_call>` tokens in its native format
4. `NativeFunctionCallingStrategy` parses the model-specific format (Qwen, Llama, Mistral, or generic)
5. Tool is executed, result fed back to the model

In `"hybrid"` mode, if native parsing fails, the system falls back to ReAct text parsing automatically.

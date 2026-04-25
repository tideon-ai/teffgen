# Custom Tools Guide

Learn how to create your own tools for tideon.ai agents.

## Basic Tool Structure

Every tool extends `BaseTool` and implements two things:
1. A `metadata` property describing the tool
2. An `_execute` async method that does the work

```python
from teffgen.tools.base_tool import (
    BaseTool, ToolMetadata, ToolCategory, ParameterSpec, ParameterType,
)

class UpperCaseTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="uppercase",
            description="Convert text to uppercase",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[
                ParameterSpec(
                    name="text",
                    type=ParameterType.STRING,
                    description="Text to convert",
                    required=True,
                ),
            ],
            returns={"type": "object", "properties": {"result": {"type": "string"}}},
        )

    async def _execute(self, **kwargs):
        return {"result": kwargs["text"].upper()}
```

## Using Your Tool

```python
from teffgen import Agent, AgentConfig, load_model

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = Agent(AgentConfig(
    name="my-agent",
    model=model,
    tools=[UpperCaseTool()],
))

result = agent.run("Convert 'hello world' to uppercase")
```

## Parameter Validation

`ParameterSpec` supports rich validation:

```python
ParameterSpec(
    name="count",
    type=ParameterType.INTEGER,
    description="Number of items",
    required=True,
    min_value=1,
    max_value=100,
)

ParameterSpec(
    name="format",
    type=ParameterType.STRING,
    description="Output format",
    enum=["json", "csv", "text"],
    default="json",
)
```

## Distributing as a Plugin

See the [Plugin Development Guide](../guides/plugin-development.md) to package your tools as an installable plugin.

# API Reference

## Core Classes

### Agent

```python
from effgen import Agent, AgentConfig

config = AgentConfig(
    name="my-agent",
    model=model,            # BaseModel instance or model ID string
    tools=[Calculator()],   # List of BaseTool instances
    system_prompt="...",    # Custom system prompt
    max_iterations=10,      # Max ReAct loop iterations
    temperature=0.7,        # Generation temperature
    enable_sub_agents=True, # Enable task decomposition
    enable_memory=True,     # Enable conversation memory
    enable_streaming=False, # Enable streaming output
)

agent = Agent(config)
```

#### `agent.run(task, mode=AgentMode.AUTO) -> AgentResponse`

Run a task synchronously. Returns `AgentResponse` with:
- `output: str` — Final response text
- `success: bool` — Whether execution succeeded
- `iterations: int` — ReAct iterations performed
- `tool_calls: int` — Number of tool invocations
- `tokens_used: int` — Total tokens consumed
- `execution_time: float` — Wall-clock seconds
- `execution_trace: List[Dict]` — Full ReAct trace

#### `agent.stream(task, mode=AgentMode.AUTO) -> Iterator[str]`

Stream response tokens. Yields strings as they are generated.

#### `agent.reset_memory()`

Clear conversation history.

---

### load_model

```python
from effgen import load_model

model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization="4bit",          # None, "4bit", "8bit"
    device_map="auto",            # "auto", "cpu", or GPU index
    max_memory=None,              # Dict of device -> max memory
)
```

Supported backends:
- **Transformers** (default): HuggingFace models
- **vLLM**: High-throughput serving (`pip install effgen[vllm]`)
- **OpenAI**: `load_model("openai:gpt-4o")`
- **Anthropic**: `load_model("anthropic:claude-3-haiku")`
- **Gemini**: `load_model("gemini:gemini-pro")`

---

### Presets

```python
from effgen.presets import create_agent, list_presets

# See available presets
print(list_presets())
# {'math': '...', 'research': '...', 'coding': '...', 'general': '...', 'minimal': '...'}

# Create a preset agent
agent = create_agent("math", model)
agent = create_agent("coding", model, extra_tools=[MyTool()])
```

---

## Tools

### BaseTool

All tools extend `BaseTool` from `effgen.tools.base_tool`:

```python
from effgen.tools.base_tool import BaseTool, ToolMetadata, ToolCategory, ParameterSpec, ParameterType

class MyTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="my_tool",
            description="What this tool does",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[
                ParameterSpec(name="input", type=ParameterType.STRING,
                              description="Input text", required=True),
            ],
            returns={"type": "object"},
        )

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        return {"result": kwargs["input"]}
```

### Built-in Tools

| Tool | Import | Description |
|------|--------|-------------|
| Calculator | `from effgen.tools.builtin import Calculator` | Arithmetic expressions |
| PythonREPL | `from effgen.tools.builtin import PythonREPL` | Execute Python code |
| CodeExecutor | `from effgen.tools.builtin import CodeExecutor` | Multi-language code execution |
| WebSearch | `from effgen.tools.builtin import WebSearch` | Web search (DuckDuckGo free) |
| FileOperations | `from effgen.tools.builtin import FileOperations` | Read/write/list files |
| BashTool | `from effgen.tools.builtin import BashTool` | Shell commands |
| JSONTool | `from effgen.tools.builtin import JSONTool` | JSON manipulation |
| DateTimeTool | `from effgen.tools.builtin import DateTimeTool` | Date/time operations |
| TextProcessingTool | `from effgen.tools.builtin import TextProcessingTool` | Text manipulation |
| URLFetchTool | `from effgen.tools.builtin import URLFetchTool` | Fetch URL content |
| WikipediaTool | `from effgen.tools.builtin import WikipediaTool` | Wikipedia lookup |
| AgenticSearch | `from effgen.tools.builtin import AgenticSearch` | Local file search |
| Retrieval | `from effgen.tools.builtin import Retrieval` | RAG retrieval |

---

## Plugin System

```python
from effgen.tools.plugin import ToolPlugin, PluginManager, discover_plugins

# Auto-discover all plugins
discover_plugins()

# Manual plugin loading
mgr = PluginManager()
mgr.load_plugin(my_plugin_instance)
```

See [Plugin Development Guide](../guides/plugin-development.md) for details.

---

## Configuration

```python
from effgen import ConfigLoader, Config

loader = ConfigLoader()
config = loader.load_config("config.yaml")
```

See [Configuration Reference](configuration.md) for all options.

---

## Memory

```python
from effgen import ShortTermMemory, LongTermMemory, VectorMemoryStore

# Short-term (conversation context)
stm = ShortTermMemory(max_tokens=4096)

# Long-term (persistent facts)
ltm = LongTermMemory(backend="sqlite", persist_path="~/.effgen/memory/")

# Vector store (semantic search)
vs = VectorMemoryStore()
```

# API Reference

## Core Classes

### Agent

```python
from teffgen import Agent, AgentConfig

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
from teffgen import load_model

model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization="4bit",          # None, "4bit", "8bit"
    device_map="auto",            # "auto", "cpu", or GPU index
    max_memory=None,              # Dict of device -> max memory
)
```

Supported backends:
- **Transformers** (default): HuggingFace models
- **vLLM**: High-throughput serving (`pip install teffgen[vllm]`)
- **OpenAI**: `load_model("openai:gpt-4o")`
- **Anthropic**: `load_model("anthropic:claude-3-haiku")`
- **Gemini**: `load_model("gemini:gemini-pro")`

---

### Presets

```python
from teffgen.presets import create_agent, list_presets

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

All tools extend `BaseTool` from `teffgen.tools.base_tool`:

```python
from teffgen.tools.base_tool import BaseTool, ToolMetadata, ToolCategory, ParameterSpec, ParameterType

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
| Calculator | `from teffgen.tools.builtin import Calculator` | Arithmetic expressions |
| PythonREPL | `from teffgen.tools.builtin import PythonREPL` | Execute Python code |
| CodeExecutor | `from teffgen.tools.builtin import CodeExecutor` | Multi-language code execution |
| WebSearch | `from teffgen.tools.builtin import WebSearch` | Web search (DuckDuckGo free) |
| FileOperations | `from teffgen.tools.builtin import FileOperations` | Read/write/list files |
| BashTool | `from teffgen.tools.builtin import BashTool` | Shell commands |
| JSONTool | `from teffgen.tools.builtin import JSONTool` | JSON manipulation |
| DateTimeTool | `from teffgen.tools.builtin import DateTimeTool` | Date/time operations |
| TextProcessingTool | `from teffgen.tools.builtin import TextProcessingTool` | Text manipulation |
| URLFetchTool | `from teffgen.tools.builtin import URLFetchTool` | Fetch URL content |
| WikipediaTool | `from teffgen.tools.builtin import WikipediaTool` | Wikipedia lookup |
| AgenticSearch | `from teffgen.tools.builtin import AgenticSearch` | Local file search |
| Retrieval | `from teffgen.tools.builtin import Retrieval` | RAG retrieval |
| StockPriceTool | `from teffgen.tools.builtin import StockPriceTool` | Stock prices (yfinance/Yahoo) |
| CurrencyConverterTool | `from teffgen.tools.builtin import CurrencyConverterTool` | Currency conversion (ECB) |
| CryptoTool | `from teffgen.tools.builtin import CryptoTool` | Crypto prices (CoinGecko) |
| DataFrameTool | `from teffgen.tools.builtin import DataFrameTool` | Pandas DataFrame ops |
| PlotTool | `from teffgen.tools.builtin import PlotTool` | Matplotlib charts |
| StatsTool | `from teffgen.tools.builtin import StatsTool` | Statistical analysis |
| GitTool | `from teffgen.tools.builtin import GitTool` | Git operations (read-only) |
| DockerTool | `from teffgen.tools.builtin import DockerTool` | Docker operations (read-only) |
| SystemInfoTool | `from teffgen.tools.builtin import SystemInfoTool` | System monitoring |
| HTTPTool | `from teffgen.tools.builtin import HTTPTool` | HTTP requests |
| ArxivTool | `from teffgen.tools.builtin import ArxivTool` | arXiv paper search |
| StackOverflowTool | `from teffgen.tools.builtin import StackOverflowTool` | StackOverflow search |
| GitHubTool | `from teffgen.tools.builtin import GitHubTool` | GitHub search |
| WolframAlphaTool | `from teffgen.tools.builtin import WolframAlphaTool` | Wolfram Alpha (API key) |
| EmailDraftTool | `from teffgen.tools.builtin import EmailDraftTool` | Email drafting |
| SlackDraftTool | `from teffgen.tools.builtin import SlackDraftTool` | Slack message drafting |
| NotificationTool | `from teffgen.tools.builtin import NotificationTool` | Desktop notifications |

---

## Guardrails

```python
from teffgen.guardrails import (
    GuardrailChain, PIIGuardrail, PromptInjectionGuardrail,
    ToxicityGuardrail, LengthGuardrail, TopicGuardrail,
    ToolPermissionGuardrail, get_guardrail_preset,
)

# Use a preset
chain = get_guardrail_preset("strict")

# Or build custom
chain = GuardrailChain([
    PromptInjectionGuardrail(sensitivity="high"),
    PIIGuardrail(),
    LengthGuardrail(max_length=5000),
])

# Attach to agent
config = AgentConfig(name="safe_agent", model=model, guardrails=chain)
```

---

## RAG Pipeline

```python
from teffgen.rag import DocumentIngester, HybridSearchEngine, ContextBuilder

# Ingest documents
ingester = DocumentIngester()
chunks = ingester.ingest("./docs/")

# Search
engine = HybridSearchEngine(chunks)
results = engine.search("scaling architecture", top_k=5)

# Build context with citations
builder = ContextBuilder(max_tokens=2048)
context, citations = builder.build(results)

# Or use the preset
from teffgen.presets import create_agent
agent = create_agent("rag", model, knowledge_base="./docs/")
result = agent.run("What does the architecture doc say about scaling?")
print(result.citations)
```

---

## Evaluation

```python
from teffgen.eval import AgentEvaluator, MathSuite, RegressionTracker
from teffgen.eval.evaluator import ScoringMode

evaluator = AgentEvaluator(agent, scoring=ScoringMode.CONTAINS)
results = evaluator.run_suite(MathSuite())
print(results.summary())

# Regression tracking
tracker = RegressionTracker()
tracker.save_baseline("math", results, version="0.2.0")
report = tracker.compare("math", new_results, version="0.2.1")
```

---

## Model Router

```python
from teffgen.models.router import ModelRouter
from teffgen.models.capabilities import MODEL_CAPABILITIES, estimate_complexity

# Auto-routing
config = AgentConfig(
    name="smart_agent",
    model=small_model,
    models=[small_model, large_model],  # Router auto-created
)

# Complexity estimation
level = estimate_complexity("Write a recursive merge sort in Python")
# → ComplexityLevel.COMPLEX
```

---

## Checkpointing & Sessions

```python
# Checkpointing
agent.run("Long research task...", checkpoint_interval=3)
# If interrupted:
agent.resume(checkpoint_id="latest")

# Sessions
agent = Agent(config=config, session_id="user-123")
agent.run("My name is Alice")
# Later, even in new process:
agent = Agent(config=config, session_id="user-123")
result = agent.run("What's my name?")  # Recalls "Alice"
```

---

## Human-in-the-Loop

```python
from teffgen.core.human_loop import HumanApproval

config = AgentConfig(
    name="careful_agent",
    model=model,
    tools=[BashTool()],
    approval_mode="dangerous_only",
    approval_callback=HumanApproval.cli_callback,
)
```

---

## Observability

```python
# Debug mode
result = agent.run("What is 2+2?", debug=True)
trace = result.metadata["debug_trace"]
trace.print_rich()  # Rich TUI output

# Structured logging
from teffgen.utils.structured_logging import StructuredLogger
logger = StructuredLogger("my_app")

# Prometheus metrics
from teffgen.utils.prometheus_metrics import get_metrics
metrics = get_metrics()
print(metrics.export())
```

---

## Client SDK

```python
from teffgen.client import TeffgenClient

client = TeffgenClient(base_url="http://localhost:8000", api_key="...")

# Chat
response = client.chat("What is 2+2?")

# Streaming
for chunk in client.chat_stream_sync("Tell me a story"):
    print(chunk, end="")

# Embeddings
vectors = client.embed(["hello", "world"])
```

---

## Plugin System

```python
from teffgen.tools.plugin import ToolPlugin, PluginManager, discover_plugins

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
from teffgen import ConfigLoader, Config

loader = ConfigLoader()
config = loader.load_config("config.yaml")
```

See [Configuration Reference](configuration.md) for all options.

---

## Memory

```python
from teffgen import ShortTermMemory, LongTermMemory, VectorMemoryStore

# Short-term (conversation context)
stm = ShortTermMemory(max_tokens=4096)

# Long-term (persistent facts)
ltm = LongTermMemory(backend="sqlite", persist_path="~/.teffgen/memory/")

# Vector store (semantic search)
vs = VectorMemoryStore()
```

# Architecture Guide

## System Overview

```
                    ┌─────────────────────────┐
                    │       User Input         │
                    │  (CLI / API / Python)    │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │         Agent            │
                    │    (ReAct Loop)          │
                    │                         │
                    │  Thought → Action →     │
                    │  Observation → ...      │
                    └──┬──────┬──────┬────────┘
                       │      │      │
              ┌────────▼┐  ┌──▼───┐ ┌▼────────┐
              │  Model   │  │Tools │ │ Memory  │
              │ Backend  │  │      │ │         │
              └────────┬┘  └──┬───┘ └┬────────┘
                       │      │      │
    ┌──────────────────┼──────┼──────┼──────────┐
    │ Backends:        │      │      │          │
    │ - Transformers   │  Built-in   │ Short    │
    │ - vLLM           │  + Plugins  │ + Long   │
    │ - OpenAI API     │  + MCP      │ + Vector │
    │ - Anthropic API  │  + A2A/ACP  │          │
    │ - Gemini API     │             │          │
    └──────────────────┘─────────────┘──────────┘
```

## Core Components

### Agent (`effgen/core/agent.py`)

The central class. Implements the ReAct reasoning loop:

1. Receives a task from the user
2. Generates a **Thought** (reasoning about what to do)
3. Selects an **Action** (tool to call) with parameters
4. Receives an **Observation** (tool result)
5. Repeats until a **Final Answer** is reached or max iterations hit

Key features:
- Sub-agent decomposition for complex tasks
- Streaming output support
- Memory integration (short-term, long-term, vector)
- Configurable via `AgentConfig` dataclass

### Model Backends (`effgen/models/`)

Abstraction over multiple LLM backends:

| Backend | File | Use Case |
|---------|------|----------|
| `TransformersEngine` | `transformers_engine.py` | Local GPU inference (default) |
| `VLLMEngine` | `vllm_engine.py` | High-throughput serving |
| `OpenAIAdapter` | `openai_adapter.py` | OpenAI API models |
| `AnthropicAdapter` | `anthropic_adapter.py` | Claude models |
| `GeminiAdapter` | `gemini_adapter.py` | Google Gemini models |

All implement `BaseModel` with: `generate()`, `generate_stream()`, `count_tokens()`, `get_context_length()`, `load()`, `unload()`

### Tools (`effgen/tools/`)

- `base_tool.py`: `BaseTool` abstract class with metadata and validation
- `registry.py`: `ToolRegistry` for discovery, lazy loading, dependency management
- `builtin/`: 14 built-in tools
- `plugin.py`: External plugin loading via entry points
- `protocols/`: MCP, A2A, ACP protocol implementations

### Memory (`effgen/memory/`)

Three tiers:
1. **ShortTermMemory**: Recent conversation context (token-limited)
2. **LongTermMemory**: Persistent facts across sessions (SQLite)
3. **VectorMemoryStore**: Semantic search over past interactions

### Prompts (`effgen/prompts/`)

- `TemplateManager`: Prompt template management
- `ChainManager`: Prompt chaining
- `PromptOptimizer`: SLM-specific prompt optimization
- `AgentSystemPromptBuilder`: Auto-generates system prompts from tools

### Configuration (`effgen/config/`)

YAML/JSON configuration loading with validation and defaults.

## Data Flow

```
User Query
    │
    ▼
AgentConfig (model, tools, prompts, memory)
    │
    ▼
Agent.__init__() → loads model, initializes tools & memory
    │
    ▼
Agent.run(task) → enters ReAct loop
    │
    ├──▶ Model.generate(prompt) → raw LLM output
    │        │
    │        ▼
    │    Parse: extract Thought, Action, Action Input
    │        │
    │        ▼
    │    Tool._execute(**params) → observation
    │        │
    │        ▼
    │    Append to context, check for Final Answer
    │        │
    │        └──▶ (loop back to generate)
    │
    ▼
AgentResponse (output, stats, trace)
```

## Plugin Architecture

Plugins are discovered from three sources:
1. Python entry points (`effgen.plugins` group)
2. User plugin directory (`~/.effgen/plugins/`)
3. Environment variable (`EFFGEN_PLUGINS_DIR`)

Each plugin provides a `ToolPlugin` subclass that registers `BaseTool` implementations into the global `ToolRegistry`.

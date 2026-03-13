# effGen Examples

Ready-to-run examples demonstrating effGen's agentic AI capabilities with Small Language Models.

## Requirements

- effGen installed (`pip install -e .`)
- GPU recommended (all examples use local model inference)
- Default model: `Qwen/Qwen2.5-3B-Instruct` (downloaded automatically on first run)

## Quick Start

```bash
# Set GPU (default: GPU 0)
export CUDA_VISIBLE_DEVICES=0

# Basic examples
python examples/basic/basic_agent.py            # Basic agent with calculator
python examples/basic/qa_agent.py               # Q&A agent (no tools)
python examples/basic/calculator_agent.py       # Math agent with Calculator + PythonREPL

# Tool-use examples
python examples/tools/multi_tool_agent.py       # Multi-tool agent (simple)
python examples/tools/advanced_multi_tool_agent.py  # Multi-tool with 5 tools
python examples/tools/file_operations_agent.py  # File read/write/search
python examples/tools/coding_agent.py           # Code execution + iteration

# Advanced examples
python examples/advanced/conversational_agent.py   # Multi-turn memory
python examples/advanced/advanced_streaming_agent.py   # Real-time token streaming
python examples/advanced/data_processing_agent.py  # JSON & data pipelines
python examples/advanced/multi_agent_pipeline.py   # Multi-agent orchestration
python examples/advanced/error_recovery_agent.py   # Error handling patterns

# Preset and plugin examples
python examples/plugins_presets/preset_agents.py          # Ready-to-use agent presets
python examples/plugins_presets/plugin_example.py         # Custom tool plugins

# Web and retrieval examples
python examples/web_retrieval/web_agent.py              # Web search agent
python examples/web_retrieval/weather_agent.py          # Weather via Open-Meteo (free)
python examples/web_retrieval/streaming_agent.py        # Simple streaming demo
python examples/web_retrieval/memory_agent.py           # Simple memory demo
python examples/web_retrieval/retrieval_agent.py        # RAG-based retrieval
python examples/web_retrieval/agentic_search_agent.py   # Grep-based agentic search
```

## Example Agents

### Core Examples

| Example | Description | Tools | Recommended Model |
|---------|-------------|-------|-------------------|
| `basic/qa_agent.py` | Q&A agent with no tools — direct inference | None | Qwen2.5-1.5B+ |
| `basic/calculator_agent.py` | Math agent with step-by-step calculation | Calculator, PythonREPL | Qwen2.5-1.5B+ |
| `tools/advanced_multi_tool_agent.py` | Agent with 5 tools, fallback chains, circuit breaker | Calculator, PythonREPL, DateTimeTool, BashTool, TextProcessingTool | Qwen2.5-3B+ |
| `tools/file_operations_agent.py` | File read/write/list/search with sandbox | FileOperations, BashTool, TextProcessingTool | Qwen2.5-3B+ |
| `tools/coding_agent.py` | Write, run, and iterate on code | CodeExecutor, PythonREPL, FileOperations, BashTool | Qwen2.5-3B+ |
| `advanced/conversational_agent.py` | Multi-turn conversation with memory persistence | Calculator, DateTimeTool + ShortTermMemory + LongTermMemory | Qwen2.5-3B+ |
| `advanced/error_recovery_agent.py` | Intentional failures to test framework resilience | Custom broken/slow tools | Qwen2.5-3B+ |
| `advanced/data_processing_agent.py` | JSON querying, validation, text analysis, data pipelines | JSONTool, TextProcessingTool, PythonREPL, FileOperations | Qwen2.5-1.5B+ |
| `advanced/advanced_streaming_agent.py` | Token streaming with thought/tool/answer callbacks | Calculator, DateTimeTool | Qwen2.5-1.5B+ |
| `advanced/multi_agent_pipeline.py` | Multi-agent orchestration: manual pipeline + sub-agents | Calculator, PythonREPL + SubAgentRouter | Qwen2.5-3B+ |

### Simple Examples (from v0.1.1)

| Example | Description |
|---------|-------------|
| `basic/basic_agent.py` | Basic calculator agent |
| `basic/basic_agent_vllm.py` | Basic agent with vLLM backend (5-10x faster) |
| `plugins_presets/preset_agents.py` | Create agents from built-in presets |
| `web_retrieval/streaming_agent.py` | Simple streaming demo |
| `web_retrieval/memory_agent.py` | Simple multi-turn memory demo |
| `tools/multi_tool_agent.py` | Simple multi-tool agent |
| `web_retrieval/weather_agent.py` | Weather queries (free, no API key) |
| `plugins_presets/plugin_example.py` | Custom tool plugin creation |
| `web_retrieval/web_agent.py` | Web search agent |
| `web_retrieval/retrieval_agent.py` | RAG-based retrieval |
| `web_retrieval/agentic_search_agent.py` | Grep-based search |

### Utilities

| File | Description |
|------|-------------|
| `utils/sweep_model.py` | Cross-model compatibility sweep runner |
| `utils/compatibility_matrix.md` | Full model compatibility results (11 models x 10 agents) |

## Model Recommendations

Based on testing across 11 models (see [utils/compatibility_matrix.md](utils/compatibility_matrix.md)):

| Model | Size | Score | Best For |
|-------|------|-------|----------|
| **Qwen2.5-1.5B-Instruct** | 1.5B | 10/10 | Best value — perfect score at small size |
| **Qwen2.5-3B-Instruct** | 3B | 10/10 | Recommended default — fast, reliable |
| **Phi-4-mini-instruct** | 3.8B | 10/10 | Best cross-family validation |
| Qwen3-1.7B | 1.7B | 9.5/10 | Strong alternative at 1.7B |
| Qwen2.5-7B-Instruct | 7B | 9.0/10 | When 7B quality is needed |
| Llama-3.2-3B-Instruct | 3B | 8.5/10 | Good for single-turn tasks |

## Configuration

All examples support:
- `CUDA_VISIBLE_DEVICES` environment variable for GPU selection
- `--model` argument to specify the model
- Most support `--interactive` for chat mode

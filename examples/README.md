# effGen Examples

Ready-to-run examples to get started with effGen.

## Requirements

- effGen installed (`pip install -e .`)
- GPU recommended (all examples use 4-bit quantization)
- Model: `Qwen/Qwen2.5-3B-Instruct` (downloaded automatically on first run)
- For retrieval examples: `pip install datasets` (to download ARC dataset)

## Examples

| Example | Description |
|---------|-------------|
| `basic_agent.py` | Calculator agent with code execution tools |
| `preset_agents.py` | Create agents from built-in presets (math, research, coding) |
| `streaming_agent.py` | Real-time token-by-token streaming output |
| `memory_agent.py` | Multi-turn conversation with context recall |
| `multi_tool_agent.py` | Agent using Calculator + BashTool + DateTimeTool together |
| `weather_agent.py` | Weather queries via free Open-Meteo API (no API key needed) |
| `plugin_example.py` | Create and register a custom tool plugin |
| `web_agent.py` | Web search agent using DuckDuckGo |
| `retrieval_agent.py` | Knowledge-base Q&A using embedding-based RAG |
| `agentic_search_agent.py` | Knowledge-base Q&A using grep-based exact matching |

## Quick Start

```bash
# Set GPU (default: GPU 0)
export CUDA_VISIBLE_DEVICES=0

# Basic calculator agent
python examples/basic_agent.py

# Preset agents (math, research, coding)
python examples/preset_agents.py

# Streaming output
python examples/streaming_agent.py

# Multi-turn memory
python examples/memory_agent.py

# Multi-tool agent
python examples/multi_tool_agent.py

# Weather (no API key required)
python examples/weather_agent.py

# Custom plugin
python examples/plugin_example.py

# Retrieval examples (requires ARC dataset)
python examples/data/download_arc.py --output-dir examples/data
python examples/retrieval_agent.py
python examples/agentic_search_agent.py
```

## Configuration

- Each example uses `CUDA_VISIBLE_DEVICES` to select the GPU (default: `"0"`)
- Set `DETAILED_LOGGING = True` in older examples for verbose debug output
- All examples use `Qwen/Qwen2.5-3B-Instruct` with 4-bit quantization by default

# effGen v0.1.0 Release Notes

**Release Date:** March 1, 2026

effGen v0.1.0 is the first feature-complete release, upgrading the framework from Alpha to Beta status. This release transforms effGen into a full-featured agentic AI framework optimized for Small Language Models (1B-7B parameters).

## Highlights

- **14 Built-in Tools** — 7 new tools added: BashTool, WeatherTool, JSONTool, DateTimeTool, TextProcessingTool, URLFetchTool, and WikipediaTool
- **Protocol Support** — Complete MCP, A2A, and ACP protocol implementations for tool and agent interoperability
- **Real Token Streaming** — True streaming via `generate_stream()` with callbacks for thoughts, tool calls, observations, and answers
- **Memory System** — ShortTermMemory, LongTermMemory, and VectorMemoryStore integrated into the Agent lifecycle
- **Agent Presets** — One-line agent creation with `create_agent("math", model)` for math, research, coding, general, and minimal configurations
- **Plugin System** — Extend effGen with custom tools via entry points or directory-based discovery
- **CLI Enhancements** — Rich progress display, `--preset`, `--explain`, `--verbose` flags, tab completion for bash/zsh/fish, and persistent chat history
- **API Server** — WebSocket streaming, API key authentication, rate limiting, and OpenAPI documentation
- **CI/CD & Testing** — 6 GitHub Actions workflows, 67 unit tests, health monitoring, OpenTelemetry tracing, and Prometheus metrics

## What's Changed

- Structured tool descriptions with parameter types and usage examples
- `stream()` now uses real token streaming (previously character-by-character)
- `run_async()` is natively async (previously wrapped sync in executor)
- Memory uses proper ShortTermMemory/LongTermMemory classes
- Development status upgraded from Alpha to Beta

## What's Fixed

- All `NotImplementedError` paths in retrieval tool
- ACP JSON Schema validation (was checking required fields only)
- Streaming placeholder removed (`time.sleep(0.01)`)
- Direct inference now retains multi-turn conversation context

## Upgrading from v0.0.2

No breaking API changes. Existing `Agent(config=AgentConfig(...))` and `load_model()` calls work without modification. New features are opt-in.

```bash
pip install --upgrade effgen
```

# effGen Release Notes

## v0.1.2 — March 12, 2026

v0.1.2 is a test-driven hardening release. Every feature was built by creating a real agent, testing it across multiple models (0.5B to 8B), watching what breaks, and fixing the framework.

### Highlights

- **10 comprehensive example agents** — Q&A, calculator, multi-tool, file operations, code execution, conversational memory, error recovery, data processing, streaming, and multi-agent pipeline orchestration
- **19 framework bugs fixed** — discovered through real inference testing, not unit tests. Fixes cover tool parsing, answer extraction, memory management, and model-specific edge cases
- **Cross-model compatibility matrix** — 11 models tested across all 10 agents. 73% pass rate (80 PASS, 23 PARTIAL, 7 FAIL out of 110 combinations)
- **Top models (10/10 PASS):** Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Phi-4-mini-instruct

### What's New

- 10 example agents in `examples/` with full documentation, model recommendations, and interactive modes
- Compatibility matrix at `examples/compatibility_matrix.md` with per-agent model recommendations
- User-explicit sub-agent trigger detection (e.g., "use 3 agents to parallelize this")
- Sweep runner (`examples/sweep_model.py`) for automated cross-model testing

### What's Improved

- ReAct loop is more robust — better loop detection, answer extraction, and error recovery
- Tool input parsing handles single-quoted JSON, non-JSON inputs, and markdown fences
- Conversation history is better managed — configurable turn limits, auto-summarization, response truncation
- Tool results are properly formatted for the model (no more raw dicts)

### What's Fixed

- 4-bit quantization now works correctly with TransformersEngine
- gemma-3 context length detection fixed for nested config
- DateTimeTool `now` operation respects date parameter
- PythonREPL no longer double-prints output
- Absolute file paths no longer get their leading slash stripped
- Many more — see [CHANGELOG.md](CHANGELOG.md) for the full list

### Model Recommendations

| Use Case | Minimum | Recommended |
|----------|---------|-------------|
| Q&A (no tools) | 0.5B | 1.5B+ |
| Tool calling | 1.5B | 3B |
| Multi-turn conversation | 1.5B | 3B |
| Multi-agent pipeline | 1.5B | 3B |

---

## v0.1.1 — March 6, 2026

v0.1.1 is a stabilization release that fixes metadata inconsistencies, improves error handling, adds 6 new examples, and expands the test suite.

### What's Fixed
- License references now consistently say Apache-2.0 everywhere (was MIT in some files)
- `setup.py` entry points, Development Status, and dependency versions now match `pyproject.toml`
- 5 bare `except:` handlers in GPU monitoring replaced with specific exception types
- 15+ stray `print()` calls converted to structured logging

### What's New
- 6 example scripts: presets, streaming, memory, multi-tool, weather, and plugin usage
- 50+ new tests covering CLI, API server, plugins, presets, fallback chains, and circuit breakers
- Top-level convenience imports for `ToolFallbackChain`, `CircuitBreaker`, `ToolPromptGenerator`, `AgentSystemPromptBuilder`
- `NEWS.md` for user-friendly release summaries

### What's Changed
- Error handlers across execution modules now log exceptions instead of silently swallowing them
- Comprehensive lint cleanup via ruff (2200+ auto-fixes)

---

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

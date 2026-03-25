# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.3] - 2026-03-25

### Added
- **Sub-agent depth limiting** — `max_sub_agent_depth` config option (default 3) prevents unbounded sub-agent recursion (ISSUE-005)
- **"No tool needed" guidance** in ReAct prompt — explicit instruction and example for direct answers, reducing unnecessary tool calls by SLMs (ISSUE-016)
- **Model-aware token counting** — `ShortTermMemory` now accepts an optional `model` parameter for accurate tokenization instead of the `len(text)//4` heuristic (ISSUE-009)
- **Circuit breaker persistence** — optional JSON file persistence for circuit breaker state via `persist_path` parameter (ISSUE-012)
- **Streaming timeout safety** — all streaming examples now use `signal.SIGALRM` timeouts to prevent indefinite hangs (ISSUE-013)
- **`pytest-timeout`** added to dev dependencies with 120s default timeout (ISSUE-001)
- **`bitsandbytes`** added to dev dependencies for 4-bit quantization testing (ISSUE-002)

### Improved
- **Loop detection** — exact loop now allows 1 retry before triggering (was zero-tolerance); fuzzy loop threshold raised to 7 for `DATA_PROCESSING` category tools; action inputs normalized (JSON key sorting, whitespace stripping) before comparison (ISSUE-004, ISSUE-019)
- **Partial answer extraction** — observations now scanned for day names and numeric results; multiple valid observations combined for multi-tool tasks (ISSUE-017)
- **"Answer now" nudge** — when iterations are running low and a tool returned successfully, the scratchpad hints the model to emit `Final Answer:` (ISSUE-017)
- **Model-family prompt formatters** — Qwen format uses `<|tools|>` section markers; Llama format uses `<|begin_of_text|>` header/EOT tags (ISSUE-010)
- **Stop sequences** — removed overly aggressive `\n\n\n` stop sequence that could truncate legitimate multi-paragraph output (ISSUE-015)
- **System prompt** — added "Do NOT use tools for greetings, jokes, opinions, or recalling information" to mistakes section (ISSUE-016)
- **Model loading warning** — logs a clear warning when `require_model=False` and loading fails, instead of silently setting `self.model = None` (ISSUE-011)
- **Integration test robustness** — `real_model` fixture falls back to fp16 if bitsandbytes is not installed (ISSUE-002)

### Fixed
- **NotImplementedError messages** — MCP transport stubs and Retrieval tool stubs now include descriptive messages instead of bare `raise NotImplementedError` (ISSUE-006, ISSUE-008)

### Internal
- 19 issues from v0.1.2 verification addressed across 12 files
- Streaming examples hardened with timeout handling
- Conversational agent example tuned for memory summarization

## [0.1.2] - 2026-03-12

### Added
- **10 comprehensive example agents** covering Q&A, calculator, multi-tool, file operations, code execution, conversational memory, error recovery, data processing, streaming, and multi-agent pipeline orchestration
- **Cross-model compatibility matrix** — 11 models tested across all 10 agents (110 combinations), 73% pass rate ([compatibility_matrix.md](examples/compatibility_matrix.md))
- **User-explicit sub-agent trigger detection** in `SubAgentRouter` — regex-based fuzzy matching for phrases like "use sub-agents", "launch 3 agents", "spawn agents" (router.py)
- **Compatibility sweep runner** (`examples/sweep_model.py`) for automated cross-model testing

### Improved
- **ReAct loop robustness** — loop detection breaks repeated identical actions (BUG-003), fuzzy loop detection for 5+ calls with different inputs (BUG-017)
- **Tool input parsing** — single-quoted JSON via `ast.literal_eval` fallback (BUG-016), non-JSON input mapping, markdown fence stripping for code params
- **Conversation history** — `max_turns` increased from 5 to 25, summary inclusion, assistant response truncation (300 chars), configurable `keep_recent_messages` (BUG-014, BUG-015)
- **Answer extraction** — line-start anchor for "Answer:" regex with `re.MULTILINE` (BUG-004), trailing text trimming (BUG-005), newline boundary fix for `Action: Final Answer` (BUG-008)
- **Tool result formatting** — proper extraction of `data`/`message` keys from FileOperations dict results (BUG-010), stderr extraction for CodeExecutor errors (BUG-013), stdout preference over None for PythonREPL (BUG-012)
- **Default max_tokens** increased from 512 to 1024 for long tool data (BUG-018)

### Fixed
- **BUG-001:** `quantization="4bit"` silently ignored by TransformersEngine — now properly passed through (model_loader.py)
- **BUG-002:** gemma-3 context length detection fails when config uses nested `text_config` (transformers_engine.py)
- **BUG-006:** DateTimeTool `now` operation ignores `date` parameter (datetime_tool.py)
- **BUG-007:** `validate_parameters` rejects unknown parameters hallucinated by SLMs — now warns instead of failing (base_tool.py)
- **BUG-009:** `_map_input_to_parameters` strips leading slash from absolute paths via `lstrip('/')` (agent.py)
- **BUG-011:** PythonREPL `_execute()` re-evaluates last `ast.Call` expression causing double `print()` output (python_repl.py)

### Internal
- Test-driven development across 12 phases with real GPU inference
- 19 framework bugs discovered and fixed through systematic agent testing
- Compatibility testing across 11 model families (0.5B to 8B parameters)
- Verification sweep: 116 unit tests pass, all integration tests pass

## [0.1.1] - 2026-03-06

### Fixed
- Fixed license inconsistency: all files now correctly reference Apache-2.0 (was MIT in some files)
- Fixed `setup.py` entry point mismatch: `effgen-agent` now correctly points to `agent_main` (was `main`)
- Fixed `setup.py` Development Status: now correctly says Beta (was Alpha)
- Fixed `setup.py` dependency version mismatches with `pyproject.toml` (duckduckgo-search, cloud-secrets, monitoring groups)
- Fixed missing `fastapi` and `uvicorn` in `pyproject.toml` dependencies (`effgen serve` now works out of the box with `pip install effgen`)
- Replaced 5 bare `except:` in `gpu/monitor.py` with specific exception handlers
- Replaced 15+ `print()` calls with proper logger calls in `docker_sandbox`, `decomposition_engine`, `router`, `complexity_analyzer`, `gpu/utils`
- Added logging to silent `except Exception:` handlers in execution modules (`docker_sandbox.py`, `sandbox.py`, `code_executor.py`)

### Added
- `NEWS.md` with user-friendly release summaries
- 6 new example scripts: `preset_agents`, `streaming_agent`, `memory_agent`, `multi_tool_agent`, `weather_agent`, `plugin_example`
- Updated `examples/README.md` with descriptions for all examples
- Top-level imports for `ToolFallbackChain`, `CircuitBreaker`, `ToolPromptGenerator`, `AgentSystemPromptBuilder`
- CLI smoke tests (`tests/integration/test_cli.py`)
- API server tests (`tests/integration/test_api_server.py`)
- Plugin system tests (`tests/unit/test_plugin.py`)
- Preset tests (`tests/unit/test_presets.py`)
- Fallback chain tests (`tests/unit/test_fallback.py`)
- Circuit breaker tests (`tests/unit/test_circuit_breaker.py`)
- Benchmark baseline (`tests/benchmarks/baseline.json`)

### Changed
- All error handlers in `gpu/monitor.py` now catch specific exceptions instead of bare `except:`
- Diagnostic output in `docker_sandbox`, `decomposition_engine`, `router`, `complexity_analyzer`, and `gpu/utils` now uses structured logging

### Internal
- Lint cleanup via ruff (2200+ auto-fixes)
- mypy fixes on modified files
- Validated all GitHub Actions YAML files

---

## [0.1.0] - 2026-03-01

### Added

#### Foundation Hardening
- **ToolPromptGenerator**: Dynamic system prompts with exact tool usage examples for SLMs
- **Model-Specific Prompts**: Optimized prompt formatting for Qwen, Llama, and Phi model families
- **Tool Fallback Chains**: Automatic fallback when tools fail (e.g., calculator → python_repl → code_executor)
- **CircuitBreaker**: Tracks tool failure rates and temporarily disables failing tools
- **Enhanced Tool Descriptions**: Structured format with parameter types, defaults, and usage examples
- **Retry Logic**: Exponential backoff for empty model responses with temperature adjustment
- **Partial Answer Extraction**: Extracts best answer from scratchpad when max iterations reached
- **Input Sanitization**: Validates and sanitizes all tool inputs before execution
- **Async Context Manager**: `async with Agent(config) as agent:` support
- **True Async**: `run_async()` is now natively async (not executor-wrapped)

#### Tool Ecosystem (7 New Tools — 14 Total)
- **BashTool**: Shell command execution with security controls (command whitelist/blacklist)
- **WeatherTool**: Weather data via Open-Meteo API (free, no API key required)
- **JSONTool**: Parse, query (JSONPath), transform, and validate JSON
- **DateTimeTool**: Current time, timezone conversion, date arithmetic
- **TextProcessingTool**: Word count, regex operations, text comparison
- **URLFetchTool**: Fetch and extract text from web pages
- **WikipediaTool**: Search and retrieve Wikipedia articles (free API)
- **Enhanced Retrieval**: Document loaders (txt, md, pdf, csv, json), chunking strategies, hybrid search (vector + BM25)
- **Enhanced AgenticSearch**: ripgrep backend, multi-query, file-type awareness, summarization
- **AgentSystemPromptBuilder**: Auto-generates tool-aware system prompts per agent configuration

#### Protocols & Streaming
- **ACP Protocol Complete**: Full JSON Schema validation, server/client modes, async task polling
- **MCP Client Enhanced**: Auto-reconnection, MCP→effGen tool bridge, resource→context bridge, health monitoring
- **Real Streaming**: True token streaming via generate_stream() (replaces placeholder)
- **Streaming Callbacks**: on_thought, on_tool_call, on_observation, on_answer
- **SSE Streaming**: Server-Sent Events endpoint for real-time API streaming
- **Memory Integration**: ShortTermMemory, LongTermMemory, VectorMemoryStore connected to Agent
- **Memory Configuration**: Configurable backends, persistence paths, auto-summarization

#### Infrastructure
- **CI/CD Pipelines**: GitHub Actions for CI, releases, docs, nightly tests, health checks, PR gates
- **Health Monitoring**: Website, DNS, SSL checks for effgen.org and docs.effgen.org
- **Test Suite**: 67 unit tests, 8 benchmarks, integration and e2e tests with MockModel and fixtures
- **Observability**: OpenTelemetry tracing (no-op fallback), Prometheus metrics
- **`effgen health` Command**: CLI health checker for all infrastructure
- **Code Quality**: Pre-commit hooks (black, isort, flake8, mypy, bandit), CONTRIBUTING.md

#### Developer Experience
- **Plugin System**: ToolPlugin base class with entry point and directory discovery
- **Agent Presets**: Ready-to-use configs — math, research, coding, general, minimal
- **`create_agent()` Factory**: One-line agent creation from presets
- **CLI Enhancements**: Rich progress, verbose/explain modes, tab completion (bash/zsh/fish), session persistence
- **API Server**: WebSocket streaming, API key authentication, rate limiting, OpenAPI docs, /health, /metrics
- **Documentation**: API reference, 6 tutorials, architecture guide, configuration reference, FAQ, migration guide
- **Packaging**: py.typed (PEP 561), Dockerfile, conda-forge recipe, optional dependency groups

### Changed
- `_get_tools_description()` now outputs structured format with parameter details
- `stream()` now uses real token streaming (previously character-by-character placeholder)
- `run_async()` is now truly async (previously wrapped sync in executor)
- Memory system uses proper ShortTermMemory/LongTermMemory classes (previously plain list)
- ACP `validate_request()` now does full JSON Schema validation (previously only checked required fields)
- User-Agent strings now use dynamic version from `effgen.__version__`
- Development status upgraded from Alpha to Beta

### Fixed
- All `NotImplementedError` paths in retrieval tool
- ACP TODO for JSON schema validation
- Streaming placeholder (`time.sleep(0.01)`)
- Memory as plain list (`self.short_term_memory = []`)
- Direct inference path now includes conversation history for multi-turn context retention

---

## [0.0.2] - 2026-02-03

### Added
- **Retrieval Tool**: RAG-based semantic search tool for knowledge base Q&A
- **Agentic Search Tool**: Grep-based exact match search with async support

### Fixed
- **vLLM Backend**: Fixed automatic chat template support for instruction-tuned models
- **GPU Memory Control**: Improved `gpu_memory_utilization` parameter handling
- **OOM Error Handling**: Better error messages and suggestions for CUDA out-of-memory errors
- **Tensor Parallel Auto-Selection**: Fixed auto-detection of tensor parallel size for small models (1.7B, 4B, etc.)
- **vLLM Cache Directory**: Resolved issues with vLLM cache directory handling

### Changed
- **Model Loader**: Improved small model detection for tensor parallel size selection
- **Version Management**: Consolidated `__version__` to single source in main `effgen/__init__.py`

### Compatibility
- Tested with multiple model families:
  - Qwen (Qwen3-1.7B, Qwen2.5-3B-Instruct)
  - Meta Llama (Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct)
  - Microsoft Phi (Phi-4-mini-instruct)
  - HuggingFace SmolLM (SmolLM2-1.7B-Instruct, SmolLM3-3B)
  - Google Gemma (Gemma-3-4b-it)

---

## [0.0.1] - 2026-01-31

### Added

#### Core Framework
- **Agent System**: Complete agentic framework optimized for Small Language Models (1B-7B parameters)
- **Task Management**: Task and SubTask classes with priority levels and status tracking
- **Agent State**: Comprehensive state management for agent execution
- **ReAct Pattern**: Reasoning and Acting pattern implementation for structured problem-solving

#### Model Support
- **Multi-Backend Support**:
  - HuggingFace Transformers (local models)
  - vLLM (fast inference with 5-10x speedup)
  - OpenAI API adapter
  - Anthropic API adapter
  - Google Gemini API adapter
- **Model Loader**: Automatic model detection and loading with intelligent fallback
- **Generation Configuration**: Flexible configuration for temperature, tokens, sampling, etc.

#### Tool System
- **Built-in Tools**:
  - Calculator (basic math, conversions, financial calculations)
  - Web Search (DuckDuckGo integration with caching)
  - Code Executor (Python, JavaScript, Bash in sandboxed environment)
  - File Operations (read, write, list, search)
  - Python REPL (interactive Python execution)
- **Tool Registry**: Dynamic tool registration and discovery
- **Protocol Support**:
  - MCP (Model Context Protocol) - Official Anthropic SDK integration
  - A2A (Agent-to-Agent) protocol
  - ACP (Agent Communication Protocol)

#### Prompt Engineering
- **Template Manager**: Jinja2-based template system with versioning
- **Chain Manager**: Multi-step prompt chaining with conditional execution
- **Prompt Optimizer**: SLM-specific optimization techniques
- **Few-Shot Learning**: Dynamic example selection for improved performance

#### Memory Systems
- **Short-Term Memory**: Conversation history and context management
- **Long-Term Memory**: Persistent storage with importance-based retrieval
- **Vector Store**: Semantic search with FAISS, ChromaDB, and Qdrant support
- **Storage Backends**: JSON and SQLite storage options

#### Task Decomposition
- **Complexity Analysis**: Automatic task complexity assessment
- **Decomposition Engine**: Break complex tasks into manageable subtasks
- **Sub-Agent Manager**: Specialized sub-agents for different task types
- **Orchestrator**: Coordinate multi-agent execution with parallel/sequential strategies

#### GPU Management
- **GPU Allocator**: Intelligent GPU allocation with memory requirements
- **GPU Monitor**: Real-time monitoring of utilization, temperature, and power
- **Multi-GPU Support**: Automatic distribution across available GPUs

#### Code Execution
- **Sandboxed Execution**: Safe code execution with Docker containers
- **Code Validator**: Static analysis and security checks
- **Multiple Languages**: Support for Python, JavaScript, Bash, and more
- **Resource Limits**: Configurable CPU, memory, and timeout limits

#### Configuration
- **YAML Configuration**: Hierarchical configuration with validation
- **JSON Schema Validation**: Type-safe configuration with comprehensive schemas
- **Environment Variables**: Secure secret management with .env support
- **Cloud Secrets**: AWS Secrets Manager, HashiCorp Vault, Azure Key Vault integration

#### CLI Interface
- **Interactive Chat**: Real-time chat interface with rich formatting
- **One-Shot Execution**: Direct task execution from command line
- **API Server**: FastAPI-based REST API server
- **Web Agent**: Autonomous web browsing and interaction
- **Tool Management**: List, inspect, and test tools

#### Utilities
- **Logging System**: Rich, structured logging with multiple levels and formats
- **Metrics Tracking**: Performance metrics, token usage, and cost tracking
- **Error Handling**: Comprehensive error handling with retry logic
- **Async Support**: Full async/await support for concurrent operations

#### Examples & Documentation
- **Basic Agent Example**: Simple agent with calculator and web search
- **Web Agent Example**: Agent that can browse and extract information
- **Installation Script**: Interactive installer with animations
- **Security Policy**: Comprehensive security guidelines and vulnerability reporting

### Configuration Files
- `pyproject.toml`: Modern Python packaging with build system configuration
- `setup.py`: Traditional setuptools configuration for compatibility
- `.gitignore`: Comprehensive ignore patterns for Python, IDEs, and system files
- `requirements.txt`: Core dependencies with version specifications

### Package Metadata
- **License**: MIT License
- **Python Support**: 3.10, 3.11, 3.12, 3.13
- **Development Status**: Alpha
- **Keywords**: ai, agents, llm, slm, language-models, tool-use, multi-agent

### Optional Dependencies
- `dev`: Development tools (pytest, black, isort, flake8, mypy)
- `vllm`: Fast inference engine
- `flash-attn`: Flash Attention for faster transformer inference
- `vector-db`: Vector database backends (FAISS, ChromaDB, Qdrant)
- `search`: Advanced search engines (Google, DuckDuckGo)
- `cloud-secrets`: Cloud secret management (AWS, Azure, Vault)
- `monitoring`: Experiment tracking (Weights & Biases, TensorBoard)
- `all`: All optional dependencies combined

### Entry Points
- `effgen`: Main CLI entry point
- `effgen-agent`: Agent-specific commands
- `effgen-web`: Web agent interface

---

## Version History

### Version Naming Convention
- **Major.Minor.Patch** (Semantic Versioning)
- **Major**: Breaking changes, major new features
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, minor improvements

### Release Schedule
- **Patch releases**: As needed for critical bugs
- **Minor releases**: Monthly feature updates
- **Major releases**: Quarterly for significant changes

---

## Links

- **GitHub**: https://github.com/ctrl-gaurav/effGen
- **PyPI**: https://pypi.org/project/effgen/
- **Documentation**: https://effgen.org/docs/
- **Issues**: https://github.com/ctrl-gaurav/effGen/issues

---

## Contributors

Thank you to all contributors who helped make effGen possible!

- Gaurav Srivastava (@ctrl-gaurav) - Creator and maintainer

---

[Unreleased]: https://github.com/ctrl-gaurav/effGen/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/ctrl-gaurav/effGen/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/ctrl-gaurav/effGen/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ctrl-gaurav/effGen/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ctrl-gaurav/effGen/compare/v0.0.2...v0.1.0
[0.0.2]: https://github.com/ctrl-gaurav/effGen/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/ctrl-gaurav/effGen/releases/tag/v0.0.1

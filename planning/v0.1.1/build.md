# effGen v0.1.1 Build Plan

> **Version:** 0.1.0 → 0.1.1
> **Status:** Planning
> **Approach:** 5 Phases, incremental, each phase testable independently
> **Repository:** [github.com/ctrl-gaurav/effGen](https://github.com/ctrl-gaurav/effGen) (Open Source)

---

## Summary

### What v0.1.0 Accomplished
v0.1.0 was the first major feature release, transforming effGen from a v0.0.2 alpha into a Beta-quality framework. All 101 checklist items were completed across 5 phases:
- **Phase 1:** Foundation hardening — ToolPromptGenerator, fallback chains, CircuitBreaker, retry logic, async cleanup
- **Phase 2:** 7 new tools (14 total) — BashTool, WeatherTool, JSONTool, DateTimeTool, TextProcessingTool, URLFetchTool, WikipediaTool; enhanced retrieval with hybrid search; AgentSystemPromptBuilder
- **Phase 3:** ACP/MCP protocol completion, real token streaming, memory integration (ShortTermMemory, LongTermMemory, VectorMemoryStore)
- **Phase 4:** CI/CD (6 GitHub Actions workflows), 67 unit tests, health monitoring, OpenTelemetry tracing, Prometheus metrics, pre-commit hooks
- **Phase 5:** Plugin system, 5 agent presets, CLI enhancements (--preset, --explain, --verbose, tab completion, chat history), API server (WebSocket, auth, rate limiting), documentation (12 docs), packaging (Dockerfile, conda recipe)

### What v0.1.1 Will Accomplish
**Theme: Robustness, Consistency, and Quality-of-Life**

v0.1.1 is a stabilization release focused on fixing inconsistencies, improving error handling, cleaning up code quality issues, adding missing examples, and improving the developer experience. No major new features — instead, making everything that exists work better and more reliably.

---

## Audit Findings

### Bugs / Inconsistencies
1. **License mismatch:** `effgen/__init__.py:3` says `__license__ = "MIT"`, `pyproject.toml:12` says `license = "MIT"`, but actual LICENSE file is Apache 2.0, `setup.py:63` says Apache, README says Apache 2.0. GPU module headers (`effgen/gpu/utils.py:8`, `allocator.py:9`, `monitor.py:9`, `__init__.py:15`) also say MIT.
2. **setup.py vs pyproject.toml entry point mismatch:** `setup.py:132` has `effgen-agent=effgen.cli:main` but `pyproject.toml:105` has `effgen-agent = "effgen.cli:agent_main"` — these point to different functions.
3. **setup.py Development Status says Alpha** (`setup.py:59` — `"Development Status :: 3 - Alpha"`) but `pyproject.toml:24` correctly says Beta (`"Development Status :: 4 - Beta"`).
4. **setup.py duckduckgo-search version mismatch:** `setup.py:97` has `>=3.9.0` but `pyproject.toml:63` has `>=8.1.0`. The newer version has breaking API changes.

### Error Handling Issues (High Priority)
5. **5 bare `except:` in `effgen/gpu/monitor.py`** — lines 338, 345, 352, 364, 695. Catches SystemExit, KeyboardInterrupt, etc. Should catch specific NVML exceptions.
6. **25+ overly broad `except Exception:` handlers** across: `effgen/execution/docker_sandbox.py` (4), `effgen/execution/sandbox.py` (2), `effgen/execution/validators.py` (1), `effgen/utils/metrics.py` (2), `effgen/memory/vector_store.py` (2), `effgen/tools/protocols/mcp/client.py` (1), `effgen/tools/plugin.py` (1), `effgen/cli.py` (7), `effgen/tools/builtin/code_executor.py` (1).

### Code Quality Issues (Medium Priority)
7. **15 `print()` statements that should be `logger` calls:**
   - `effgen/execution/docker_sandbox.py`: lines 104, 107, 386, 455, 457, 472, 485, 510, 531
   - `effgen/core/decomposition_engine.py`: line 297
   - `effgen/core/router.py`: line 225
   - `effgen/core/complexity_analyzer.py`: lines 580, 586, 591
   - `effgen/gpu/utils.py`: lines 655, 659, 661, 663
8. **3 files over 1000 lines:**
   - `effgen/cli.py` — 1,945 lines (should extract command handlers into submodules)
   - `effgen/core/agent.py` — 1,676 lines (acceptable but could extract helpers)
   - `effgen/tools/builtin/retrieval.py` — 1,009 lines (acceptable)

### Missing Content
9. **No NEWS.md file** — referenced in v0.1.0 verification docs but never created.
10. **Missing examples for new v0.1.0 features:** No examples for presets, streaming, memory, weather tool, plugin system, multi-tool agents, API server usage.
11. **Missing `__init__.py` exports** for some new modules: `effgen/tools/fallback.py` exports (ToolFallbackChain), `effgen/utils/circuit_breaker.py` exports (CircuitBreaker), `effgen/prompts/tool_prompt_generator.py` exports (ToolPromptGenerator), `effgen/prompts/agent_system_prompt.py` exports (AgentSystemPromptBuilder) — not all are importable from top-level `effgen`.
12. **No example for direct API model usage** (OpenAI/Anthropic/Gemini adapters).

### Testing Gaps
13. **No test for CLI commands** — only manual testing.
14. **No test for API server endpoints** — integration test file exists in planning/v0.1.0/build.md but not in tests/.
15. **No test for plugin loading from entry points** — only basic class test.
16. **No test for ACP/MCP protocol interop** — only unit-level validation tests.
17. **Benchmark tests exist but no regression baseline** — no stored baseline to compare against.

---

## Guiding Principles

- **Open Source First:** Every feature must work without paid external APIs. If a feature *can* use a paid API, it must have a free fallback and display a clear message.
- **SLM-Optimized:** All prompts, tools, and system designs must work well with 1B-7B parameter models.
- **Test with Real Agents:** Every feature gets a real agent-based integration test (GPU inference), not just mocks.
- **Backwards Compatible:** Don't break existing `Agent`, `AgentConfig`, `load_model` APIs.
- **Minimal Changes:** This is a patch release — fix what's broken, don't add new features.

---

## Phase 1: Consistency & Correctness

### Goals
Fix all metadata inconsistencies (license, version, entry points) and ensure setup.py and pyproject.toml are perfectly aligned. Fix the development status. Ensure clean installs work without surprises.

### Tasks
- [x] **1.1** Fix license to Apache 2.0 everywhere:
  - `effgen/__init__.py:3` — change `__license__ = "MIT"` to `__license__ = "Apache-2.0"`
  - `pyproject.toml:12` — change `license = "MIT"` to `license = "Apache-2.0"`
  - `effgen/gpu/utils.py:8` — change `License: MIT` to `License: Apache-2.0`
  - `effgen/gpu/__init__.py:15` — change `License: MIT` to `License: Apache-2.0`
  - `effgen/gpu/allocator.py:9` — change `License: MIT` to `License: Apache-2.0`
  - `effgen/gpu/monitor.py:9` — change `License: MIT` to `License: Apache-2.0`
- [x] **1.2** Fix setup.py entry point: change `effgen-agent=effgen.cli:main` (line 132) to `effgen-agent=effgen.cli:agent_main` to match pyproject.toml
- [x] **1.3** Fix setup.py Development Status: change `"Development Status :: 3 - Alpha"` (line 59) to `"Development Status :: 4 - Beta"` to match pyproject.toml
- [x] **1.4** Fix setup.py duckduckgo-search version: change `>=3.9.0` (line 97) to `>=8.1.0` to match pyproject.toml
- [x] **1.5** Audit all other version differences between setup.py `extras_require` and pyproject.toml `[project.optional-dependencies]` — fix any mismatches
- [x] **1.6** Create NEWS.md with a user-friendly summary of v0.1.0 changes (referenced but never created)

### Tests
- [x] Test: Run `pip install -e .` in a fresh conda env and verify no warnings about conflicting metadata
- [x] Test: Run `python -c "import effgen; print(effgen.__license__)"` — verify prints "Apache-2.0"
- [x] Test: Run `python -c "from effgen import Agent, load_model; print('OK')"` — verify clean import
- [x] Test: Verify `effgen --version` outputs `0.1.0` (until we bump version later)

### Success Criteria
All metadata files are consistent. `pip install -e .` works cleanly. No license mismatches anywhere in the codebase.

---

## Phase 2: Error Handling & Logging Cleanup

### Goals
Replace all bare `except:` with specific exceptions. Convert all inappropriate `print()` calls to proper `logger` calls. Make error handling robust and informative rather than silently swallowing failures.

### Tasks
- [x] **2.1** Fix 5 bare `except:` in `effgen/gpu/monitor.py` (lines 338, 345, 352, 364, 695) — replace with `except (AttributeError, RuntimeError, Exception)` or the specific pynvml exception types
- [x] **2.2** Convert `print()` to `logger` calls in `effgen/execution/docker_sandbox.py` (9 instances at lines 104, 107, 386, 455, 457, 472, 485, 510, 531) — use `logger.info()` for progress, `logger.warning()` for warnings
- [x] **2.3** Convert `print()` to `logger` calls in `effgen/core/decomposition_engine.py` (line 297) and `effgen/core/router.py` (line 225) — use `logger.warning()`
- [x] **2.4** Convert `print()` to `logger` calls in `effgen/core/complexity_analyzer.py` (lines 580, 586, 591) — use `logger.info()` and `logger.warning()`
- [x] **2.5** Convert `print()` to `logger` calls in `effgen/gpu/utils.py` (lines 655, 659, 661, 663) — use `logger.info()` for device info display
- [x] **2.6** Review and tighten the most critical `except Exception:` handlers — focus on `effgen/execution/docker_sandbox.py` (lines 215, 251, 379, 506), `effgen/execution/sandbox.py` (lines 320, 383), and `effgen/tools/builtin/code_executor.py` (line 278). Add specific exception types where possible, or at minimum add `logger.debug()` calls so failures aren't completely silent.
- [x] **2.7** Ensure every file that uses `print()` for logging has a proper `logger = logging.getLogger(__name__)` or `from effgen.utils.logging import get_logger; logger = get_logger(__name__)` at the top

### Tests
- [x] Test: Create agent with Calculator tool on GPU, run "What is 99 * 101?" — verify answer contains 9999 and no print() output leaks to stdout (capture stdout and check)
- [ ] Test: Trigger a GPU monitoring call when no GPU is available (mock pynvml to raise) — verify it handles gracefully with a log message instead of crashing
- [x] Test: Import all modules that were changed and verify no import errors: `from effgen.gpu import monitor; from effgen.execution import docker_sandbox; from effgen.core import decomposition_engine, router, complexity_analyzer`

### Success Criteria
Zero bare `except:` in codebase. Zero `print()` calls used for logging/diagnostics (CLI output `print()` is acceptable). All error handlers either catch specific exceptions or log the exception at debug level.

---

## Phase 3: Examples & Developer Experience

### Goals
Add practical, runnable example scripts for all major v0.1.0 features. Users should be able to `python examples/<feature>.py` and see each feature in action. Also add missing top-level exports.

### Tasks
- [x] **3.1** Create `examples/preset_agents.py` — demonstrates `create_agent("math", model)`, `create_agent("research", model)`, `create_agent("coding", model)` with actual queries
- [x] **3.2** Create `examples/streaming_agent.py` — demonstrates `agent.stream("What is 2+2?")` with real-time token output
- [x] **3.3** Create `examples/memory_agent.py` — demonstrates multi-turn memory: agent remembers context across calls
- [x] **3.4** Create `examples/multi_tool_agent.py` — demonstrates agent with Calculator + BashTool + DateTimeTool solving a multi-step task
- [x] **3.5** Create `examples/weather_agent.py` — demonstrates WeatherTool with Open-Meteo (free API, no key)
- [x] **3.6** Create `examples/plugin_example.py` — demonstrates creating and registering a simple custom tool plugin
- [x] **3.7** Update `examples/README.md` with descriptions and usage instructions for all examples (old + new)
- [x] **3.8** Add missing convenience imports to `effgen/__init__.py`:
  - `from effgen.tools.fallback import ToolFallbackChain`
  - `from effgen.utils.circuit_breaker import CircuitBreaker`
  - `from effgen.prompts.tool_prompt_generator import ToolPromptGenerator`
  - `from effgen.prompts.agent_system_prompt import AgentSystemPromptBuilder`
  - Add these to `__all__` as well

### Tests
- [x] Test: Run `examples/preset_agents.py` on a free GPU with Qwen2.5-3B-Instruct — verify math preset produces correct answer
- [x] Test: Run `examples/streaming_agent.py` — verify tokens stream (not all at once)
- [x] Test: Run `examples/memory_agent.py` — verify agent recalls earlier context
- [x] Test: Verify new imports work: `from effgen import ToolFallbackChain, CircuitBreaker, ToolPromptGenerator, AgentSystemPromptBuilder`

### Success Criteria
Every major v0.1.0 feature has a working example. All examples run successfully with a real model. Top-level imports cover all public API classes.

---

## Phase 4: Test Suite Hardening

### Goals
Fill gaps in the test suite. Add tests for CLI commands, API server endpoints, and protocol interop. Ensure the test suite catches regressions reliably. Store benchmark baselines.

### Tasks
- [ ] **4.1** Add CLI smoke tests in `tests/integration/test_cli.py`:
  - Test `effgen --version` outputs correct version
  - Test `effgen --help` shows help without error
  - Test `effgen presets` lists available presets
  - Test `effgen health` runs health checks (may skip DNS in CI)
- [ ] **4.2** Add API server tests in `tests/integration/test_api_server.py`:
  - Test `/health` returns 200
  - Test `/metrics` returns metrics data
  - Test authentication with `EFFGEN_API_KEY`
  - Test rate limiting returns 429 when exceeded
  (Use `TestClient` from FastAPI, not a real running server)
- [ ] **4.3** Add plugin system tests in `tests/unit/test_plugin.py`:
  - Test `ToolPlugin` base class instantiation
  - Test `PluginManager.discover_plugins()` from entry points
  - Test `PluginManager.load_from_directory()` with a temp directory
- [ ] **4.4** Add preset tests in `tests/unit/test_presets.py`:
  - Test `list_presets()` returns all 5 presets
  - Test `create_agent("math", model)` returns agent with Calculator + PythonREPL
  - Test `create_agent("invalid", model)` raises appropriate error
- [ ] **4.5** Store benchmark baseline in `tests/benchmarks/baseline.json` — record current performance numbers for agent init time, tool latency, e2e response time
- [ ] **4.6** Add `tests/unit/test_fallback.py` — test ToolFallbackChain with mock tools that succeed and fail
- [ ] **4.7** Add `tests/unit/test_circuit_breaker.py` — test CircuitBreaker open/closed/half-open states

### Tests
- [ ] Test: Run `pytest tests/unit/ --no-cov -v` — all unit tests pass
- [ ] Test: Run `pytest tests/integration/test_cli.py --no-cov -v` — CLI tests pass
- [ ] Test: Create a real agent with Calculator on GPU, run "What is 7 * 8?" — verify output contains 56
- [ ] Test: Run `pytest tests/unit/test_presets.py --no-cov -v` — preset tests pass

### Success Criteria
Test count increases by 15+. CLI, API server, plugins, presets, fallback, and circuit breaker all have dedicated test files. All tests pass.

---

## Phase 5: Documentation & Polish

### Goals
Update README with v0.1.1 changes. Ensure CHANGELOG is complete. Clean up any remaining rough edges. Prepare for release.

### Tasks
- [ ] **5.1** Update CHANGELOG.md with v0.1.1 section (will be finalized at release time, but structure it now with [Unreleased] section)
- [ ] **5.2** Review and update README.md:
  - Verify all code examples still work
  - Add mention of `effgen health` command
  - Add mention of `effgen presets` command
  - Ensure example code matches current API
- [ ] **5.3** Update `docs/tutorials/getting-started.md` if any API changes were made
- [ ] **5.4** Ensure `effgen/cli.py` help text is accurate for all commands (--preset, --explain, --verbose, health, presets, create-plugin)
- [ ] **5.5** Run full lint pass: `ruff check effgen/` and fix any issues found (excluding line-length which is handled by black)
- [ ] **5.6** Verify all `.github/workflows/*.yml` files are valid YAML and reference correct Python versions
- [ ] **5.7** Run `mypy effgen/ --ignore-missing-imports` and fix any type errors in files modified during v0.1.1

### Tests
- [ ] Test: Full integration test — create agent with Calculator + PythonREPL + DateTimeTool, ask a multi-step question, verify correct answer with real GPU inference
- [ ] Test: `pip install -e .` in fresh conda env, run `python -c "import effgen; print(effgen.__version__)"` — verify clean install
- [ ] Test: `pytest tests/ --no-cov -q` — all tests pass
- [ ] Test: Verify `python examples/basic_agent.py` still works (regression)

### Success Criteria
README is accurate. CHANGELOG has v0.1.1 section. Linting passes. All tests pass. Clean install works.

---

## Master Checklist

### Phase 1: Consistency & Correctness
- [x] P1-1.1: Fix license to Apache-2.0 everywhere (6 files)
- [x] P1-1.2: Fix setup.py effgen-agent entry point
- [x] P1-1.3: Fix setup.py Development Status to Beta
- [x] P1-1.4: Fix setup.py duckduckgo-search version
- [x] P1-1.5: Audit and fix all setup.py vs pyproject.toml version mismatches
- [x] P1-1.6: Create NEWS.md
- [x] P1-TEST: pip install -e . clean install test
- [x] P1-TEST: License string verification
- [x] P1-TEST: Import verification
- [x] P1-TEST: Version output verification

### Phase 2: Error Handling & Logging
- [x] P2-2.1: Fix bare except: in gpu/monitor.py (5 instances)
- [x] P2-2.2: Convert print() to logger in docker_sandbox.py (9 instances)
- [x] P2-2.3: Convert print() to logger in decomposition_engine.py and router.py
- [x] P2-2.4: Convert print() to logger in complexity_analyzer.py
- [x] P2-2.5: Convert print() to logger in gpu/utils.py
- [x] P2-2.6: Tighten critical except Exception: handlers
- [x] P2-2.7: Ensure logger setup in all modified files
- [x] P2-TEST: Calculator agent produces correct output
- [ ] P2-TEST: GPU monitoring graceful failure
- [x] P2-TEST: All modified modules import cleanly

### Phase 3: Examples & Developer Experience
- [x] P3-3.1: Create examples/preset_agents.py
- [x] P3-3.2: Create examples/streaming_agent.py
- [x] P3-3.3: Create examples/memory_agent.py
- [x] P3-3.4: Create examples/multi_tool_agent.py
- [x] P3-3.5: Create examples/weather_agent.py
- [x] P3-3.6: Create examples/plugin_example.py
- [x] P3-3.7: Update examples/README.md
- [x] P3-3.8: Add missing convenience imports to effgen/__init__.py
- [x] P3-TEST: preset_agents.py runs correctly on GPU
- [x] P3-TEST: streaming_agent.py streams tokens
- [x] P3-TEST: memory_agent.py recalls context
- [x] P3-TEST: New top-level imports work

### Phase 4: Test Suite Hardening
- [ ] P4-4.1: Add CLI smoke tests
- [ ] P4-4.2: Add API server tests
- [ ] P4-4.3: Add plugin system tests
- [ ] P4-4.4: Add preset tests
- [ ] P4-4.5: Store benchmark baseline
- [ ] P4-4.6: Add fallback chain tests
- [ ] P4-4.7: Add circuit breaker tests
- [ ] P4-TEST: All unit tests pass
- [ ] P4-TEST: CLI tests pass
- [ ] P4-TEST: Real agent calculator test on GPU
- [ ] P4-TEST: Preset tests pass

### Phase 5: Documentation & Polish
- [ ] P5-5.1: Update CHANGELOG.md with v0.1.1 section
- [ ] P5-5.2: Review and update README.md
- [ ] P5-5.3: Update getting-started tutorial if needed
- [ ] P5-5.4: Verify CLI help text accuracy
- [ ] P5-5.5: Run ruff lint pass and fix issues
- [ ] P5-5.6: Validate GitHub Actions YAML files
- [ ] P5-5.7: Run mypy on modified files and fix errors
- [ ] P5-TEST: Full integration test on GPU
- [ ] P5-TEST: Clean pip install test
- [ ] P5-TEST: Full test suite passes
- [ ] P5-TEST: Regression test with basic_agent.py

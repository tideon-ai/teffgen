# Migration Guide

## 0.0.2 → 1.0.0

### New Features
- **Presets**: Use `create_agent("math", model)` for instant agent setup
- **Plugin system**: Distribute tools as installable packages
- **CLI**: `--preset`, `--explain`, `--completion`, `create-plugin` commands
- **API server**: WebSocket streaming, API key auth, rate limiting, metrics
- **Tab completion**: `eval "$(effgen --completion bash)"`

### Breaking Changes
None. All existing `Agent`, `AgentConfig`, and `load_model` APIs remain unchanged.

### New Imports
```python
# Presets (new)
from effgen.presets import create_agent, list_presets

# Plugin system (new)
from effgen.tools.plugin import ToolPlugin, PluginManager, discover_plugins
```

### CLI Changes
```bash
# New commands
effgen presets                              # List available presets
effgen run --preset math "What is 2+2?"     # Use preset
effgen run --explain "..."                  # Show tool reasoning
effgen create-plugin my_tools               # Generate plugin scaffold
effgen --completion bash                    # Print completion script
```

### API Server Changes
- New endpoints: `WS /ws`, `GET /metrics`
- Auth: Set `EFFGEN_API_KEY` environment variable
- Rate limiting: Set `EFFGEN_RATE_LIMIT` (default: 60 req/min)
- `POST /run` now accepts `preset` field

# Plugin Development Guide

effGen supports external tool plugins that can be distributed as Python packages and auto-discovered at runtime.

## Quick Start

Generate a plugin scaffold:

```bash
effgen create-plugin my_tools
cd effgen-plugin-my_tools
pip install -e .
```

Your plugin's tools will be auto-discovered by effGen via Python entry points.

## Writing a Tool

Create a class that extends `BaseTool`:

```python
from effgen.tools.base_tool import (
    BaseTool, ToolMetadata, ToolCategory, ParameterSpec, ParameterType,
)

class MyTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="my_tool",
            description="Does something useful.",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[
                ParameterSpec(
                    name="input",
                    type=ParameterType.STRING,
                    description="Input text",
                    required=True,
                ),
            ],
            returns={"type": "object", "properties": {"result": {"type": "string"}}},
        )

    async def _execute(self, **kwargs):
        return {"result": kwargs["input"].upper()}
```

## Registering Tools via Plugin Class

Wrap your tools in a `ToolPlugin`:

```python
from effgen.tools.plugin import ToolPlugin
from my_tools.tools import MyTool

class MyPlugin(ToolPlugin):
    name = "my_tools"
    version = "1.0.0"
    tools = [MyTool]
```

## Entry Point (Recommended)

Add this to your `pyproject.toml` so effGen finds your plugin automatically:

```toml
[project.entry-points."effgen.plugins"]
my_tools = "my_tools.plugin:MyPlugin"
```

## Directory-Based Loading

Alternatively, drop `.py` files into `~/.effgen/plugins/` or set `EFFGEN_PLUGINS_DIR`:

```bash
export EFFGEN_PLUGINS_DIR=/path/to/my/plugins
```

Any file containing a `ToolPlugin` subclass will be loaded.

## Manual Loading

```python
from effgen.tools.plugin import PluginManager

mgr = PluginManager()
mgr.discover_all()            # Load from all sources
print(mgr.loaded_plugins)     # See what was loaded
```

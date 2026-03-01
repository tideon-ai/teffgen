"""
Built-in tools for the effGen framework.

This module contains the standard set of tools that ship with effGen.
"""

# Import built-in tools (lazy loading handled by registry)
__all__ = [
    "CodeExecutor",
    "PythonREPL",
    "WebSearch",
    "FileOperations",
    "Calculator",
    "Retrieval",
    "AgenticSearch",
    "BashTool",
    "WeatherTool",
    "JSONTool",
    "DateTimeTool",
    "TextProcessingTool",
    "URLFetchTool",
    "WikipediaTool",
]


def __getattr__(name):
    """Lazy import of tools."""
    if name == "CodeExecutor":
        from .code_executor import CodeExecutor
        return CodeExecutor
    elif name == "PythonREPL":
        from .python_repl import PythonREPL
        return PythonREPL
    elif name == "WebSearch":
        from .web_search import WebSearch
        return WebSearch
    elif name == "FileOperations":
        from .file_ops import FileOperations
        return FileOperations
    elif name == "Calculator":
        from .calculator import Calculator
        return Calculator
    elif name == "Retrieval":
        from .retrieval import Retrieval
        return Retrieval
    elif name == "AgenticSearch":
        from .agentic_search import AgenticSearch
        return AgenticSearch
    elif name == "BashTool":
        from .bash_tool import BashTool
        return BashTool
    elif name == "WeatherTool":
        from .weather import WeatherTool
        return WeatherTool
    elif name == "JSONTool":
        from .json_tool import JSONTool
        return JSONTool
    elif name == "DateTimeTool":
        from .datetime_tool import DateTimeTool
        return DateTimeTool
    elif name == "TextProcessingTool":
        from .text_processing import TextProcessingTool
        return TextProcessingTool
    elif name == "URLFetchTool":
        from .url_fetch import URLFetchTool
        return URLFetchTool
    elif name == "WikipediaTool":
        from .wikipedia_tool import WikipediaTool
        return WikipediaTool
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

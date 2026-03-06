"""
OpenTelemetry tracing integration for effGen.

Provides span-level tracing for agent execution, tool calls,
and ReAct iterations. Uses no-op tracer if OpenTelemetry is not installed.

Usage:
    from effgen.utils.tracing import get_tracer
    tracer = get_tracer()
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("key", "value")
        ...
"""

import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class NoOpSpan:
    """No-op span when OpenTelemetry is not installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class NoOpTracer:
    """No-op tracer when OpenTelemetry is not installed."""

    def start_as_current_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()

    @contextmanager
    def start_span(self, name: str, **kwargs):
        yield NoOpSpan()


def get_tracer(name: str = "effgen") -> Any:
    """
    Get an OpenTelemetry tracer, or a no-op tracer if OTEL is not installed.

    Args:
        name: Tracer name (default: "effgen")

    Returns:
        Tracer instance
    """
    if OTEL_AVAILABLE:
        return trace.get_tracer(name)
    return NoOpTracer()


def setup_tracing(
    service_name: str = "effgen",
    export_to_console: bool = False,
) -> None:
    """
    Set up OpenTelemetry tracing for effGen.

    Args:
        service_name: Service name for traces
        export_to_console: If True, export spans to console (useful for debugging)
    """
    if not OTEL_AVAILABLE:
        logger.info(
            "OpenTelemetry not installed. Install with: pip install opentelemetry-api opentelemetry-sdk"
        )
        return

    provider = TracerProvider()
    if export_to_console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    logger.info(f"OpenTelemetry tracing initialized for {service_name}")


def trace_agent_run(agent_name: str, task: str):
    """Create a span for an agent run."""
    tracer = get_tracer()
    span = tracer.start_as_current_span(
        "agent.run",
        attributes={
            "agent.name": agent_name,
            "agent.task": task[:200],
        },
    )
    return span


def trace_tool_call(tool_name: str, tool_input: str):
    """Create a span for a tool call."""
    tracer = get_tracer()
    span = tracer.start_as_current_span(
        f"tool.{tool_name}",
        attributes={
            "tool.name": tool_name,
            "tool.input": tool_input[:200],
        },
    )
    return span


def trace_react_iteration(iteration: int, agent_name: str):
    """Create a span for a ReAct iteration."""
    tracer = get_tracer()
    span = tracer.start_as_current_span(
        f"react.iteration.{iteration}",
        attributes={
            "agent.name": agent_name,
            "react.iteration": iteration,
        },
    )
    return span

"""
Prometheus-compatible metrics collection for effGen.

Provides counters, histograms, and gauges for agent monitoring.
Uses a simple in-memory implementation that can export to Prometheus
text format. No external dependencies required.

Usage:
    from effgen.utils.prometheus_metrics import metrics
    metrics.agent_runs.inc()
    metrics.tool_call_duration.observe(0.5, labels={"tool": "calculator"})
    print(metrics.export())
"""

import time
import threading
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Counter:
    """A monotonically increasing counter."""
    name: str
    help: str
    _value: float = 0.0
    _labels: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            if labels:
                key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                self._labels[key] += amount
            else:
                self._value += amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        if labels:
            key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return self._labels.get(key, 0.0)
        return self._value

    def export(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        if self._value > 0:
            lines.append(f"{self.name} {self._value}")
        for key, val in self._labels.items():
            lines.append(f"{self.name}{{{key}}} {val}")
        return "\n".join(lines)


@dataclass
class Histogram:
    """A histogram for measuring distributions."""
    name: str
    help: str
    _sum: float = 0.0
    _count: int = 0
    _labels_sum: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _labels_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            if labels:
                key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                self._labels_sum[key] += value
                self._labels_count[key] += 1
            else:
                self._sum += value
                self._count += 1

    def get_avg(self, labels: Optional[Dict[str, str]] = None) -> float:
        if labels:
            key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            count = self._labels_count.get(key, 0)
            return self._labels_sum.get(key, 0) / count if count else 0.0
        return self._sum / self._count if self._count else 0.0

    def export(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        if self._count > 0:
            lines.append(f"{self.name}_sum {self._sum}")
            lines.append(f"{self.name}_count {self._count}")
        for key in self._labels_sum:
            lines.append(f"{self.name}_sum{{{key}}} {self._labels_sum[key]}")
            lines.append(f"{self.name}_count{{{key}}} {self._labels_count[key]}")
        return "\n".join(lines)


@dataclass
class Gauge:
    """A gauge that can go up and down."""
    name: str
    help: str
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    def get(self) -> float:
        return self._value

    def export(self) -> str:
        return "\n".join([
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} gauge",
            f"{self.name} {self._value}",
        ])


class EffGenMetrics:
    """Collection of all effGen metrics."""

    def __init__(self):
        self.agent_runs = Counter(
            "effgen_agent_runs_total",
            "Total number of agent runs",
        )
        self.tool_calls = Counter(
            "effgen_tool_calls_total",
            "Total number of tool calls",
        )
        self.tool_failures = Counter(
            "effgen_tool_failures_total",
            "Total number of failed tool calls",
        )
        self.agent_duration = Histogram(
            "effgen_agent_duration_seconds",
            "Agent execution duration in seconds",
        )
        self.tool_duration = Histogram(
            "effgen_tool_duration_seconds",
            "Tool execution duration in seconds",
        )
        self.tokens_used = Counter(
            "effgen_tokens_used_total",
            "Total tokens consumed",
        )
        self.active_agents = Gauge(
            "effgen_active_agents",
            "Number of currently active agents",
        )

    def export(self) -> str:
        """Export all metrics in Prometheus text format."""
        sections = [
            self.agent_runs.export(),
            self.tool_calls.export(),
            self.tool_failures.export(),
            self.agent_duration.export(),
            self.tool_duration.export(),
            self.tokens_used.export(),
            self.active_agents.export(),
        ]
        return "\n\n".join(sections) + "\n"


# Global metrics instance
metrics = EffGenMetrics()

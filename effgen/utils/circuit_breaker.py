"""
Circuit breaker pattern for tool execution.

Tracks tool failure rates and temporarily disables tools that
fail too frequently, preventing repeated wasteful calls.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation — tool is available
    OPEN = "open"          # Tool is disabled due to repeated failures
    HALF_OPEN = "half_open"  # Cooldown expired, allowing a test call


@dataclass
class _ToolCircuit:
    """Internal state for a single tool's circuit."""
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    state: CircuitState = CircuitState.CLOSED


class CircuitBreaker:
    """
    Track tool failure rates and temporarily disable tools that fail too often.

    After `failure_threshold` consecutive failures, the tool is marked OPEN
    (skipped). After `cooldown_seconds`, it transitions to HALF_OPEN and
    allows one test call. A success resets the circuit to CLOSED.

    Args:
        failure_threshold: Number of consecutive failures before opening.
        cooldown_seconds: Seconds before an open circuit allows a retry.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0, persist_path: str | None = None):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._circuits: dict[str, _ToolCircuit] = {}
        self._persist_path = persist_path
        if persist_path:
            self._load_state()

    def _get_circuit(self, tool_name: str) -> _ToolCircuit:
        if tool_name not in self._circuits:
            self._circuits[tool_name] = _ToolCircuit()
        return self._circuits[tool_name]

    def is_available(self, tool_name: str) -> bool:
        """
        Check if a tool is available (circuit not open).

        Returns True for CLOSED or HALF_OPEN (test call allowed).
        """
        circuit = self._get_circuit(tool_name)
        if circuit.state == CircuitState.CLOSED:
            return True
        if circuit.state == CircuitState.OPEN:
            # Check if cooldown has elapsed
            if time.time() - circuit.last_failure_time >= self.cooldown_seconds:
                circuit.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit for '{tool_name}' moved to HALF_OPEN (cooldown elapsed)")
                return True
            return False
        # HALF_OPEN — allow one test call
        return True

    def record_success(self, tool_name: str) -> None:
        """Record a successful tool execution, resetting the circuit."""
        circuit = self._get_circuit(tool_name)
        if circuit.state != CircuitState.CLOSED:
            logger.info(f"Circuit for '{tool_name}' reset to CLOSED after success")
        circuit.consecutive_failures = 0
        circuit.state = CircuitState.CLOSED
        self._save_state()

    def record_failure(self, tool_name: str) -> None:
        """Record a tool failure. May open the circuit."""
        circuit = self._get_circuit(tool_name)
        circuit.consecutive_failures += 1
        circuit.last_failure_time = time.time()

        if circuit.consecutive_failures >= self.failure_threshold:
            if circuit.state != CircuitState.OPEN:
                logger.warning(
                    f"Circuit for '{tool_name}' OPENED after "
                    f"{circuit.consecutive_failures} consecutive failures"
                )
            circuit.state = CircuitState.OPEN
        self._save_state()

    def get_state(self, tool_name: str) -> CircuitState:
        """Get the current circuit state for a tool."""
        return self._get_circuit(tool_name).state

    def _save_state(self) -> None:
        """Save circuit breaker state to JSON file."""
        if not self._persist_path:
            return
        import json
        data = {}
        for name, circuit in self._circuits.items():
            data[name] = {
                "consecutive_failures": circuit.consecutive_failures,
                "last_failure_time": circuit.last_failure_time,
                "state": circuit.state.value,
            }
        try:
            with open(self._persist_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_state(self) -> None:
        """Load circuit breaker state from JSON file."""
        if not self._persist_path:
            return
        import json
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            for name, state in data.items():
                self._circuits[name] = _ToolCircuit(
                    consecutive_failures=state["consecutive_failures"],
                    last_failure_time=state["last_failure_time"],
                    state=CircuitState(state["state"]),
                )
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

    def reset(self, tool_name: str) -> None:
        """Manually reset a tool's circuit to CLOSED."""
        if tool_name in self._circuits:
            self._circuits[tool_name] = _ToolCircuit()
            logger.debug(f"Circuit for '{tool_name}' manually reset")

    def reset_all(self) -> None:
        """Reset all circuits."""
        self._circuits.clear()

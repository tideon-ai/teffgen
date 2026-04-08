"""
Model Router for effGen framework.

Routes queries to the optimal model based on task complexity, available GPU
memory, model capabilities, latency budget, and cost budget.

Complexity estimation is heuristic-based and designed to be fast (< 1ms).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from effgen.models.base import BaseModel
from effgen.models.capabilities import (
    ModelCapability,
    estimate_capability,
    get_model_capability,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task complexity estimation
# ---------------------------------------------------------------------------

class ComplexityLevel(Enum):
    """Discrete complexity levels for routing decisions."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ComplexityEstimate:
    """Result of complexity estimation."""
    level: ComplexityLevel
    score: float  # 0.0 (trivial) to 1.0 (very complex)
    required_capabilities: list[str] = field(default_factory=list)
    reasoning: str = ""


# Keyword sets for heuristic complexity detection.
# Kept as module-level frozensets for speed (no allocation per call).
_SIMPLE_KEYWORDS = frozenset([
    "what", "who", "when", "where", "define", "meaning", "name",
    "capital", "color", "colour", "list", "true", "false", "yes", "no",
    "hello", "hi", "hey", "thanks", "thank",
])

_CODE_KEYWORDS = frozenset([
    "code", "program", "function", "class", "implement", "debug", "fix",
    "refactor", "algorithm", "api", "endpoint", "sql", "query", "script",
    "python", "javascript", "typescript", "java", "rust", "golang", "cpp",
    "html", "css", "regex", "compile", "runtime", "error", "exception",
    "test", "unittest", "pytest", "recursive", "recursion", "sort",
    "sorting", "binary", "tree", "stack", "queue", "hash", "database",
    "write", "build", "create", "deploy", "docker", "kubernetes",
])

_MATH_KEYWORDS = frozenset([
    "calculate", "compute", "solve", "equation", "integral", "derivative",
    "matrix", "vector", "probability", "statistics", "sum", "product",
    "factorial", "fibonacci", "prime", "graph", "optimization", "linear",
    "algebra", "geometry", "trigonometry", "calculus",
])

_REASONING_KEYWORDS = frozenset([
    "explain", "why", "how", "compare", "contrast", "analyse", "analyze",
    "evaluate", "argue", "reason", "logic", "proof", "infer", "deduce",
    "hypothesis", "theory", "because", "therefore", "strategy", "plan",
    "design", "architecture", "trade-off", "tradeoff",
])

_MULTILINGUAL_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff"
    r"\u0400-\u04ff\u0600-\u06ff\u0900-\u097f"
    r"\uac00-\ud7af]"
)

_COMPLEX_STRUCTURE_PATTERNS = [
    re.compile(r"step[- ]?by[- ]?step", re.IGNORECASE),
    re.compile(r"multi[- ]?step", re.IGNORECASE),
    re.compile(r"with error handling", re.IGNORECASE),
    re.compile(r"edge cases?", re.IGNORECASE),
    re.compile(r"comprehensive", re.IGNORECASE),
    re.compile(r"full[- ]?(stack|implementation)", re.IGNORECASE),
]


def estimate_complexity(
    query: str,
    tools: list[Any] | None = None,
) -> ComplexityEstimate:
    """Estimate task complexity from the query text.

    This is a pure-heuristic function optimised for speed (< 1ms).
    It analyses keyword density, query length, and structural patterns.

    Args:
        query: The user query / task description.
        tools: Optional list of tools available (more tools = potentially
               more complex orchestration).

    Returns:
        ComplexityEstimate with level, score, and detected capabilities.
    """
    if not query:
        return ComplexityEstimate(
            level=ComplexityLevel.SIMPLE, score=0.0, reasoning="empty query"
        )

    words = set(query.lower().split())
    query_lower = query.lower()
    score = 0.0
    capabilities: list[str] = []

    # --- Length signal ---
    n_chars = len(query)
    n_words = len(words)
    if n_chars > 500:
        score += 0.2
    elif n_chars > 200:
        score += 0.1
    elif n_chars < 30:
        score -= 0.1

    # --- Simple-query signal ---
    simple_hits = words & _SIMPLE_KEYWORDS
    if simple_hits and n_words < 12:
        score -= 0.15

    # --- Domain keyword signals ---
    code_hits = words & _CODE_KEYWORDS
    if code_hits:
        score += min(len(code_hits) * 0.08, 0.3)
        capabilities.append("code")

    math_hits = words & _MATH_KEYWORDS
    if math_hits:
        score += min(len(math_hits) * 0.08, 0.3)
        capabilities.append("math")

    reasoning_hits = words & _REASONING_KEYWORDS
    if reasoning_hits:
        score += min(len(reasoning_hits) * 0.06, 0.25)
        capabilities.append("reasoning")

    # --- Multilingual signal ---
    if _MULTILINGUAL_PATTERN.search(query):
        score += 0.1
        capabilities.append("multilingual")

    # --- Structural complexity ---
    for pat in _COMPLEX_STRUCTURE_PATTERNS:
        if pat.search(query):
            score += 0.1

    # --- Tool signal ---
    if tools:
        n_tools = len(tools)
        if n_tools > 5:
            score += 0.15
            capabilities.append("tool_calling")
        elif n_tools > 2:
            score += 0.08
            capabilities.append("tool_calling")

    # Clamp
    score = max(0.0, min(1.0, score))

    # Map to discrete level
    if score < 0.25:
        level = ComplexityLevel.SIMPLE
    elif score < 0.50:
        level = ComplexityLevel.MODERATE
    else:
        level = ComplexityLevel.COMPLEX

    if not capabilities:
        capabilities.append("instruction_following")

    return ComplexityEstimate(
        level=level,
        score=round(score, 3),
        required_capabilities=capabilities,
        reasoning=f"len={n_chars}, words={n_words}, "
                  f"code={len(code_hits)}, math={len(math_hits)}, "
                  f"reasoning={len(reasoning_hits)}",
    )


# ---------------------------------------------------------------------------
# Routing configuration
# ---------------------------------------------------------------------------

@dataclass
class RoutingConfig:
    """Configuration for the model router.

    Attributes:
        complexity_threshold_simple: Max score to still use the smallest model.
        complexity_threshold_moderate: Max score for a mid-sized model.
        gpu_memory_headroom_gb: GB of GPU memory to keep free as headroom.
        prefer_loaded: If True, prefer models already loaded in memory.
        latency_budget_ms: Optional max latency in ms (unused for now).
        cost_budget: Optional cost ceiling (unused for now, reserved for API models).
    """
    complexity_threshold_simple: float = 0.25
    complexity_threshold_moderate: float = 0.50
    gpu_memory_headroom_gb: float = 1.0
    prefer_loaded: bool = True
    latency_budget_ms: float | None = None
    cost_budget: float | None = None


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    model: BaseModel
    model_name: str
    complexity: ComplexityEstimate
    reason: str
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------

class ModelRouter:
    """Routes queries to the optimal model from a pool of available models.

    The router scores each candidate model against the estimated task
    complexity and selects the best fit.  It prefers smaller models for
    simple tasks and larger/more-capable models for complex ones.

    Args:
        models: List of BaseModel instances or model-name strings.
        config: Optional RoutingConfig.
    """

    def __init__(
        self,
        models: list[BaseModel] | None = None,
        config: RoutingConfig | None = None,
    ):
        self.config = config or RoutingConfig()
        self._models: list[BaseModel] = list(models) if models else []
        self._capabilities: dict[str, ModelCapability] = {}

        # Pre-fetch capability profiles
        for m in self._models:
            name = getattr(m, "model_name", str(m))
            self._capabilities[name] = estimate_capability(name)

    # -- public API --

    def add_model(self, model: BaseModel) -> None:
        """Add a model to the router's candidate pool."""
        self._models.append(model)
        name = getattr(model, "model_name", str(model))
        self._capabilities[name] = estimate_capability(name)
        logger.info("Router: added model '%s'", name)

    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the router by name."""
        for i, m in enumerate(self._models):
            if getattr(m, "model_name", "") == model_name:
                self._models.pop(i)
                self._capabilities.pop(model_name, None)
                logger.info("Router: removed model '%s'", model_name)
                return True
        return False

    @property
    def models(self) -> list[BaseModel]:
        """Return the list of candidate models."""
        return list(self._models)

    def select(
        self,
        query: str,
        tools: list[Any] | None = None,
    ) -> RoutingDecision:
        """Select the best model for a query.

        Args:
            query: The user query / task description.
            tools: Optional list of tools the agent has.

        Returns:
            RoutingDecision with the chosen model and reasoning.

        Raises:
            ValueError: If no models are available.
        """
        if not self._models:
            raise ValueError("ModelRouter has no candidate models")

        t0 = time.monotonic()
        complexity = estimate_complexity(query, tools)

        best_model = None
        best_score = -1.0
        best_reason = ""

        for model in self._models:
            name = getattr(model, "model_name", str(model))
            cap = self._capabilities.get(name)
            if cap is None:
                cap = estimate_capability(name)
                self._capabilities[name] = cap

            score = self._score_model(model, cap, complexity)

            if score > best_score:
                best_score = score
                best_model = model
                best_reason = (
                    f"score={score:.3f}, size={cap.size_billions:.1f}B, "
                    f"complexity={complexity.level.value}"
                )

        elapsed = (time.monotonic() - t0) * 1000

        decision = RoutingDecision(
            model=best_model,
            model_name=getattr(best_model, "model_name", str(best_model)),
            complexity=complexity,
            reason=best_reason,
            elapsed_ms=round(elapsed, 3),
        )
        logger.info(
            "Router selected '%s' for query (complexity=%s, %.2fms)",
            decision.model_name, complexity.level.value, elapsed,
        )
        return decision

    # -- scoring --

    def _score_model(
        self,
        model: BaseModel,
        cap: ModelCapability,
        complexity: ComplexityEstimate,
    ) -> float:
        """Score a model for a given complexity estimate.

        Higher is better.  The scoring balances:
        - Capability match (does the model have the required skills?)
        - Size efficiency (prefer smaller models for simple tasks)
        - Loaded-preference (avoid load latency)
        """
        score = 0.0

        # 1. Capability match — average score on required dimensions
        if complexity.required_capabilities:
            cap_scores = [
                cap.get_score(dim) for dim in complexity.required_capabilities
            ]
            avg_cap = sum(cap_scores) / len(cap_scores)
        else:
            avg_cap = cap.overall_score()

        # Weight capability match by complexity
        score += avg_cap * (0.5 + complexity.score * 0.5)

        # 2. Size efficiency — penalise oversized models for simple tasks
        if complexity.level == ComplexityLevel.SIMPLE:
            # Strongly prefer the smallest available model to save resources.
            # Inverse-size bonus: 0.5B -> +0.45, 1.5B -> +0.35, 3B -> +0.2, 7B -> -0.2
            if cap.size_billions > 0:
                size_bonus = max(-0.2, 0.5 - cap.size_billions * 0.1)
            else:
                size_bonus = 0.3
            score += size_bonus
        elif complexity.level == ComplexityLevel.COMPLEX:
            # Prefer larger models
            if cap.size_billions >= 7.0:
                score += 0.25
            elif cap.size_billions >= 3.0:
                score += 0.15
            else:
                score -= 0.05

        # 3. Prefer loaded models
        if self.config.prefer_loaded and hasattr(model, "is_loaded"):
            if model.is_loaded():
                score += 0.15

        return round(score, 4)

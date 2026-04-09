"""
effGen Evaluation, Benchmarking & Regression Testing.

Provides tools to measure agent quality, detect regressions, and compare
models/configurations.
"""

from __future__ import annotations

from .evaluator import AgentEvaluator, EvalResult, SuiteResults, TestCase
from .suites import (
    ConversationSuite,
    MathSuite,
    ReasoningSuite,
    SafetySuite,
    TestSuite,
    ToolUseSuite,
    get_suite,
    list_suites,
)
from .regression import RegressionTracker
from .comparison import ModelComparison

__all__ = [
    "AgentEvaluator",
    "EvalResult",
    "SuiteResults",
    "TestCase",
    "TestSuite",
    "MathSuite",
    "ToolUseSuite",
    "ReasoningSuite",
    "SafetySuite",
    "ConversationSuite",
    "get_suite",
    "list_suites",
    "RegressionTracker",
    "ModelComparison",
]

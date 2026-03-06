"""
EffGen Core Module

This module contains the core agent system components:
- Agent: Main agent class with ReAct loop and sub-agent support
- Router: Intelligent routing for sub-agent decisions
- SubAgentManager: Sub-agent lifecycle management
- ExecutionTracker: Transparent execution tracking
- Orchestrator: Multi-agent coordination
- Task and State management
"""

# Agent
from .agent import Agent, AgentConfig, AgentMode, AgentResponse

# Complexity Analyzer
from .complexity_analyzer import ComplexityAnalyzer, ComplexityScore

# Decomposition Engine
from .decomposition_engine import DecompositionEngine, TaskStructure

# Execution Tracker
from .execution_tracker import (
    EventType,
    ExecutionEvent,
    ExecutionNode,
    ExecutionStatus,
    ExecutionTracker,
)

# Orchestrator
from .orchestrator import MultiAgentOrchestrator, OrchestrationPattern, TeamConfig, TeamResponse

# Router
from .router import RoutingDecision, RoutingStrategy, SubAgentRouter

# State
from .state import AgentState

# Sub-Agent Manager
from .sub_agent_manager import (
    SubAgentConfig,
    SubAgentManager,
    SubAgentResult,
    SubAgentSpecialization,
)

# Task
from .task import SubTask, Task, TaskPriority, TaskStatus

__all__ = [
    # Agent
    "Agent",
    "AgentConfig",
    "AgentResponse",
    "AgentMode",

    # Router
    "SubAgentRouter",
    "RoutingDecision",
    "RoutingStrategy",

    # Sub-Agent Manager
    "SubAgentManager",
    "SubAgentConfig",
    "SubAgentResult",
    "SubAgentSpecialization",

    # Execution Tracker
    "ExecutionTracker",
    "ExecutionEvent",
    "ExecutionStatus",
    "ExecutionNode",
    "EventType",

    # Orchestrator
    "MultiAgentOrchestrator",
    "TeamConfig",
    "TeamResponse",
    "OrchestrationPattern",

    # Complexity Analyzer
    "ComplexityAnalyzer",
    "ComplexityScore",

    # Decomposition Engine
    "DecompositionEngine",
    "TaskStructure",

    # Task
    "Task",
    "SubTask",
    "TaskStatus",
    "TaskPriority",

    # State
    "AgentState",
]

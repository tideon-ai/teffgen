"""
Prompt Engineering System for Small Language Models

Provides comprehensive prompt management, chaining, and optimization
specifically designed for SLMs (1B-7B parameter models).
"""

from .template_manager import (
    PromptTemplate,
    FewShotExample,
    TemplateManager,
    create_default_template_manager,
)

from .chain_manager import (
    ChainType,
    StepStatus,
    ChainStep,
    ChainState,
    PromptChain,
    ChainManager,
    create_default_chain_manager,
)

from .optimizer import (
    ModelSize,
    OptimizationConfig,
    OptimizationResult,
    PromptOptimizer,
    create_optimizer_for_model,
)

from .tool_prompt_generator import ToolPromptGenerator
from .agent_system_prompt import AgentSystemPromptBuilder

__all__ = [
    # Template Manager
    'PromptTemplate',
    'FewShotExample',
    'TemplateManager',
    'create_default_template_manager',

    # Chain Manager
    'ChainType',
    'StepStatus',
    'ChainStep',
    'ChainState',
    'PromptChain',
    'ChainManager',
    'create_default_chain_manager',

    # Tool Prompt Generator
    'ToolPromptGenerator',

    # Agent System Prompt Builder
    'AgentSystemPromptBuilder',

    # Optimizer
    'ModelSize',
    'OptimizationConfig',
    'OptimizationResult',
    'PromptOptimizer',
    'create_optimizer_for_model',
]

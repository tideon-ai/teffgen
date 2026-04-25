"""
tideon.ai: A comprehensive framework for building agents with Small Language Models.

This framework enables SLMs to function as powerful agentic systems through:
- Tool integration (built-in tools + MCP/A2A/ACP protocols)
- Advanced prompt engineering optimized for SLMs
- Smart sub-agent decomposition for complex tasks
- Multi-GPU support with vLLM and Transformers
- Comprehensive configuration management
"""

__version__ = "0.2.0"
__author__ = "tideon.ai Team"
__license__ = "Apache-2.0"

# Core imports
# Configuration imports
from teffgen.config import Config, ConfigLoader, ConfigValidator
from teffgen.core.agent import Agent, AgentConfig
from teffgen.core.aggregation import ResultAggregator

# Batch & Aggregation imports
from teffgen.core.batch import BatchConfig, BatchResult, BatchRunner
from teffgen.core.state import AgentState
from teffgen.core.task import SubTask, Task, TaskPriority, TaskStatus

# Domain imports
from teffgen.domains import (
    Domain,
    FinanceDomain,
    HealthDomain,
    KeywordExpander,
    LegalDomain,
    ScienceDomain,
    TechDomain,
)

# GPU imports
from teffgen.gpu import GPUAllocator, GPUMonitor, gpu_utils

# Guardrails imports
from teffgen.guardrails import (
    Guardrail,
    GuardrailChain,
    GuardrailPosition,
    GuardrailResult,
    LengthGuardrail,
    PIIGuardrail,
    PromptInjectionGuardrail,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    ToolPermissionGuardrail,
    TopicGuardrail,
    ToxicityGuardrail,
    get_guardrail_preset,
)

# Memory imports
from teffgen.memory import (
    ImportanceLevel,
    JSONStorageBackend,
    LongTermMemory,
    MemoryEntry,
    MemoryType,
    Message,
    MessageRole,
    ShortTermMemory,
    SQLiteStorageBackend,
    VectorMemoryStore,
)

# Model imports
from teffgen.models import (
    AnthropicAdapter,
    BaseModel,
    CerebrasAdapter,
    GeminiAdapter,
    GenerationConfig,
    GenerationResult,
    ModelLoader,
    OpenAIAdapter,
    TransformersEngine,
    VLLMEngine,
    load_model,
)
from teffgen.models._rate_limit import RateLimitCoordinator, RateLimitExceeded  # noqa: I001
from teffgen.models.cerebras_models import available_models as cerebras_available_models
from teffgen.models.cerebras_models import free_tier_models as cerebras_free_tier_models
from teffgen.models.cerebras_models import model_info as cerebras_model_info
from teffgen.models.errors import ModelRefusalError
from teffgen.models.openai_models import available_models as openai_available_models
from teffgen.models.openai_models import chat_models as openai_chat_models
from teffgen.models.openai_models import model_info as openai_model_info
from teffgen.models.openai_models import reasoning_models as openai_reasoning_models  # noqa: I001
from teffgen.models.openai_schema import to_openai_schema

# Preset imports
from teffgen.presets import create_agent, list_presets

# Prompt imports
from teffgen.prompts import ChainManager, PromptOptimizer, TemplateManager

# Tool imports
from teffgen.tools import BaseTool, ToolRegistry
from teffgen.tools import get_registry as get_tool_registry

# MLX engine imports (Apple Silicon only)
try:
    from teffgen.models.mlx_engine import MLXEngine
    from teffgen.models.mlx_vlm_engine import MLXVLMEngine
except ImportError:
    pass

# Additional convenience imports
try:
    from teffgen.tools.fallback import ToolFallbackChain
except ImportError:
    pass

try:
    from teffgen.utils.circuit_breaker import CircuitBreaker
except ImportError:
    pass

try:
    from teffgen.prompts.tool_prompt_generator import ToolPromptGenerator
except ImportError:
    pass

try:
    from teffgen.prompts.agent_system_prompt import AgentSystemPromptBuilder
except ImportError:
    pass

# Eval imports
try:
    from teffgen.eval import (
        AgentEvaluator,
        EvalResult,
        ModelComparison,
        RegressionTracker,
        SuiteResults,
        TestCase,
        TestSuite,
    )
except ImportError:
    pass

# Execution imports
from teffgen.execution import (
    CodeExecutor,
    CodeValidator,
    ExecutionResult,
    ExecutionStatus,
    SandboxConfig,
    ValidationResult,
    ValidationSeverity,
)

__all__ = [
    # Core
    "Agent",
    "AgentConfig",
    "Task",
    "SubTask",
    "TaskStatus",
    "TaskPriority",
    "AgentState",

    # Models
    "load_model",
    "BaseModel",
    "VLLMEngine",
    "TransformersEngine",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "CerebrasAdapter",
    "ModelLoader",
    "GenerationConfig",
    "GenerationResult",
    "RateLimitCoordinator",
    "RateLimitExceeded",
    "ModelRefusalError",
    "to_openai_schema",
    # Cerebras helpers
    "cerebras_available_models",
    "cerebras_free_tier_models",
    "cerebras_model_info",
    # OpenAI helpers
    "openai_available_models",
    "openai_chat_models",
    "openai_reasoning_models",
    "openai_model_info",

    # Tools
    "BaseTool",
    "ToolRegistry",
    "get_tool_registry",

    # Configuration
    "ConfigLoader",
    "Config",
    "ConfigValidator",

    # Prompts
    "TemplateManager",
    "ChainManager",
    "PromptOptimizer",

    # GPU
    "GPUAllocator",
    "GPUMonitor",
    "gpu_utils",

    # Memory
    "ShortTermMemory",
    "LongTermMemory",
    "VectorMemoryStore",
    "Message",
    "MessageRole",
    "MemoryEntry",
    "MemoryType",
    "ImportanceLevel",
    "JSONStorageBackend",
    "SQLiteStorageBackend",

    # Batch & Aggregation
    "BatchRunner",
    "BatchConfig",
    "BatchResult",
    "ResultAggregator",

    # Domains
    "Domain",
    "KeywordExpander",
    "TechDomain",
    "ScienceDomain",
    "FinanceDomain",
    "HealthDomain",
    "LegalDomain",

    # Presets
    "create_agent",
    "list_presets",

    # Additional convenience exports
    "ToolFallbackChain",
    "CircuitBreaker",
    "ToolPromptGenerator",
    "AgentSystemPromptBuilder",

    # Eval
    "AgentEvaluator",
    "EvalResult",
    "SuiteResults",
    "TestCase",
    "TestSuite",
    "ModelComparison",
    "RegressionTracker",

    # Execution
    "CodeExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    "CodeValidator",
    "ValidationResult",
    "ValidationSeverity",
    "SandboxConfig",
]

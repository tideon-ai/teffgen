"""
Model infrastructure for tideon.ai framework.

This package provides a unified interface for various model backends including:
- vLLM for fast local inference
- HuggingFace Transformers as fallback
- OpenAI API adapter
- Anthropic Claude API adapter
- Google Gemini API adapter

Example:
    >>> from teffgen.models import load_model
    >>>
    >>> # Load a local HuggingFace model (tries vLLM, falls back to Transformers)
    >>> model = load_model("meta-llama/Llama-2-7b-hf")
    >>>
    >>> # Load an API model
    >>> gpt4 = load_model("gpt-4")
    >>> claude = load_model("claude-3-opus-20240229")
    >>>
    >>> # Generate text
    >>> result = model.generate("What is the capital of France?")
    >>> print(result.text)
"""

from teffgen.models.anthropic_adapter import AnthropicAdapter
from teffgen.models.base import (
    BaseModel,
    BatchModel,
    FunctionCallingModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)
from teffgen.models.batching import ContinuousBatcher
from teffgen.models.capabilities import (
    MODEL_CAPABILITIES,
    ModelCapability,
    estimate_capability,
    get_model_capability,
    list_registered_models,
    register_model_capability,
)
from teffgen.models.cerebras_adapter import CerebrasAdapter
from teffgen.models.errors import ModelRefusalError
from teffgen.models.gemini_adapter import GeminiAdapter
from teffgen.models.lazy import LazyModel
from teffgen.models.model_loader import ModelLoader, load_model
from teffgen.models.openai_adapter import OpenAIAdapter
from teffgen.models.openai_schema import to_openai_schema
from teffgen.models.pool import ModelPool, PoolConfig
from teffgen.models.router import (
    ComplexityEstimate,
    ComplexityLevel,
    ModelRouter,
    RoutingConfig,
    RoutingDecision,
    estimate_complexity,
)
from teffgen.models.transformers_engine import TransformersEngine
from teffgen.models.vllm_engine import VLLMEngine

# MLX engines (Apple Silicon only, lazy import)
try:
    from teffgen.models.mlx_engine import MLXEngine
except ImportError:
    pass

try:
    from teffgen.models.mlx_vlm_engine import MLXVLMEngine
except ImportError:
    pass

__all__ = [
    # Base classes
    "BaseModel",
    "BatchModel",
    "FunctionCallingModel",
    "ModelType",

    # Data classes
    "GenerationConfig",
    "GenerationResult",
    "TokenCount",

    # Engine implementations
    "VLLMEngine",
    "TransformersEngine",
    "MLXEngine",
    "MLXVLMEngine",

    # API adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "CerebrasAdapter",

    # Loader
    "ModelLoader",
    "load_model",

    # Router
    "ModelRouter",
    "RoutingConfig",
    "RoutingDecision",
    "ComplexityEstimate",
    "ComplexityLevel",
    "estimate_complexity",

    # Capabilities
    "ModelCapability",
    "MODEL_CAPABILITIES",
    "register_model_capability",
    "get_model_capability",
    "estimate_capability",
    "list_registered_models",

    # Pool
    "ModelPool",
    "PoolConfig",

    "LazyModel",
    "ContinuousBatcher",

    # Errors
    "ModelRefusalError",

    # Schema helpers
    "to_openai_schema",
]

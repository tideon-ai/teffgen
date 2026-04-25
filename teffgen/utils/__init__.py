"""
Utility modules for tideon.ai framework.

This package provides comprehensive utility functions including logging,
metrics collection, validation, and other helper functions used throughout
the tideon.ai framework.

Modules:
    - logging: Advanced logging with rich formatting and rotation
    - metrics: Performance metrics and cost tracking
    - validators: Input validation and schema checking

Example:
    Basic usage:
        >>> from teffgen.utils import setup_logger, get_logger
        >>> from teffgen.utils import MetricsCollector
        >>> from teffgen.utils import validate_model_name

    Setup logging:
        >>> setup_logger(level="INFO", log_dir="./logs")
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")

    Track metrics:
        >>> metrics = MetricsCollector()
        >>> with metrics.track_execution("task"):
        ...     perform_task()
        >>> print(metrics.get_summary())

    Validate inputs:
        >>> validate_model_name("microsoft/phi-2")
        >>> validate_path("/path/to/file", must_exist=True)
"""

# Logging utilities
from teffgen.utils.logging import (
    ColoredFormatter,
    LogContext,
    LoggerManager,
    LogLevel,
    PerformanceLogger,
    StructuredFormatter,
    get_logger,
    log_agent_start,
    log_agent_stop,
    log_exception,
    log_model_inference,
    log_task_complete,
    log_task_error,
    log_task_start,
    log_tool_call,
    set_log_level,
    setup_logger,
)

# Metrics utilities
from teffgen.utils.metrics import (
    MODEL_PRICING,
    ExecutionMetric,
    MetricsCollector,
    ResourceSnapshot,
    TokenMetric,
    get_global_metrics,
)

# Prometheus metrics
from teffgen.utils.prometheus_metrics import (
    metrics as prometheus_metrics,
)

# Structured logging
from teffgen.utils.structured_logging import (
    LogRunContext,
    StructuredLogger,
    TeffgenJSONFormatter,
    generate_run_id,
    generate_workflow_id,
    get_structured_logger,
    setup_json_logging,
)

# Tracing
from teffgen.utils.tracing import (
    get_tracer,
    setup_tracing,
    shutdown_tracing,
    trace_agent_run,
    trace_tool_execute,
)

# Validation utilities
from teffgen.utils.validators import (
    ValidationError,
    load_and_validate_json,
    require_type,
    sanitize_filename,
    sanitize_string,
    validate_config_dict,
    validate_dict,
    validate_email,
    validate_file_extension,
    validate_input,
    validate_json_schema,
    validate_list,
    validate_model_name,
    validate_number,
    validate_output,
    validate_path,
    validate_pydantic_model,
    validate_string,
    validate_type,
    validate_url,
)

__all__ = [
    # Logging
    'LogLevel',
    'StructuredFormatter',
    'ColoredFormatter',
    'LoggerManager',
    'setup_logger',
    'get_logger',
    'set_log_level',
    'LogContext',
    'log_exception',
    'PerformanceLogger',
    'log_agent_start',
    'log_agent_stop',
    'log_task_start',
    'log_task_complete',
    'log_task_error',
    'log_tool_call',
    'log_model_inference',

    # Metrics
    'MODEL_PRICING',
    'ExecutionMetric',
    'TokenMetric',
    'ResourceSnapshot',
    'MetricsCollector',
    'get_global_metrics',

    # Structured logging
    'TeffgenJSONFormatter',
    'LogRunContext',
    'StructuredLogger',
    'generate_run_id',
    'generate_workflow_id',
    'get_structured_logger',
    'setup_json_logging',

    # Tracing
    'get_tracer',
    'setup_tracing',
    'shutdown_tracing',
    'trace_agent_run',
    'trace_tool_execute',

    # Prometheus metrics
    'prometheus_metrics',

    # Validators
    'ValidationError',
    'validate_type',
    'validate_string',
    'validate_number',
    'validate_list',
    'validate_dict',
    'validate_path',
    'validate_file_extension',
    'validate_url',
    'validate_email',
    'validate_model_name',
    'validate_config_dict',
    'validate_json_schema',
    'load_and_validate_json',
    'validate_pydantic_model',
    'sanitize_string',
    'sanitize_filename',
    'require_type',
    'validate_input',
    'validate_output',
]

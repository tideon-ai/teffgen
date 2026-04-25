"""
GPU Management Module for tideon.ai Framework

This module provides comprehensive GPU management capabilities including:
- Intelligent GPU allocation with multiple strategies
- Real-time GPU monitoring with alerts
- Utility functions for GPU operations and memory estimation

Components:
    - GPUAllocator: Smart GPU allocation and resource management
    - GPUMonitor: Real-time monitoring with threshold alerts
    - Utility functions: Memory estimation, device selection, etc.

Author: tideon.ai Team
License: Apache-2.0
"""

from teffgen.gpu import utils as gpu_utils
from teffgen.gpu.allocator import (
    Allocation,
    AllocationRequest,
    AllocationStrategy,
    GPUAllocator,
    GPUInfo,
    ParallelismType,
)
from teffgen.gpu.monitor import (
    Alert,
    AlertLevel,
    GPUMetrics,
    GPUMonitor,
    MetricType,
    MonitorConfig,
)

__all__ = [
    # Allocator classes
    "GPUAllocator",
    "AllocationStrategy",
    "ParallelismType",
    "GPUInfo",
    "AllocationRequest",
    "Allocation",

    # Monitor classes
    "GPUMonitor",
    "MonitorConfig",
    "GPUMetrics",
    "Alert",
    "AlertLevel",
    "MetricType",

    # Utilities module
    "gpu_utils",
]

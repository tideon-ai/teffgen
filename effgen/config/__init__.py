"""
Configuration management for effGen.

This module provides configuration loading, validation, and management capabilities
with support for environment variable substitution, config merging, and hot reloading.
"""

from .loader import Config, ConfigLoader
from .validator import ConfigValidator, ValidationError

__all__ = [
    "ConfigLoader",
    "Config",
    "ConfigValidator",
    "ValidationError",
]

"""E2E test fixtures."""

import os
import warnings
import pytest

warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(scope="session")
def real_model():
    """Session-scoped real model for e2e tests."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No GPU available for e2e tests")
    except ImportError:
        pytest.skip("PyTorch not available")

    from effgen import load_model
    model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
    yield model
    try:
        model.unload()
    except Exception:
        pass

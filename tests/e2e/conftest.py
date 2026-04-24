"""E2E test fixtures."""

import gc
import warnings

import pytest

warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _cuda_cleanup():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


@pytest.fixture(scope="module")
def real_model():
    """Module-scoped real model for e2e tests.

    Module scope (not session) prevents bitsandbytes 4-bit CUDA state from
    leaking across test modules, which historically caused downstream
    integration streaming tests to deadlock in the same pytest session.
    """
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
    del model
    gc.collect()
    _cuda_cleanup()

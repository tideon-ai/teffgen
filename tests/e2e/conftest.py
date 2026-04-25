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


@pytest.fixture(scope="class")
def real_model():
    """Class-scoped real model for e2e tests.

    Class scope (not module / session) isolates CUDA state between test
    classes so bitsandbytes / accelerate dispatch hooks cannot leak RMSNorm
    corruption into the next class's forward passes — a pre-existing
    upstream issue that showed up as sporadic full-suite failures.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("No GPU available for e2e tests")
    except ImportError:
        pytest.skip("PyTorch not available")

    from teffgen import load_model

    model = load_model("Qwen/Qwen2.5-3B-Instruct")
    yield model
    try:
        model.unload()
    except Exception:
        pass
    del model
    gc.collect()
    _cuda_cleanup()

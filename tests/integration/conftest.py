"""
Integration test fixtures requiring real GPU models.
"""

import os
import warnings

import pytest

warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _find_free_gpu():
    """Find a free GPU (minimal memory usage)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        min_mem = float("inf")
        best = 0
        for i in range(torch.cuda.device_count()):
            torch.cuda.get_device_properties(i)
            mem_used = torch.cuda.memory_allocated(i)
            # Also check via nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = info.used
            except Exception:
                pass
            if mem_used < min_mem:
                min_mem = mem_used
                best = i
        return best
    except ImportError:
        return None


@pytest.fixture(scope="session")
def gpu_id():
    """Session-scoped fixture for GPU ID."""
    gid = _find_free_gpu()
    if gid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
    return gid


def _load_and_yield(gpu_id, quantization=None):
    """Load a model and clean up CUDA state on teardown."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from teffgen import load_model
    if quantization:
        try:
            model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization=quantization)
        except Exception:
            model = load_model("Qwen/Qwen2.5-3B-Instruct")
    else:
        model = load_model("Qwen/Qwen2.5-3B-Instruct")
    yield model
    try:
        model.unload()
    except Exception:
        pass
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


@pytest.fixture(scope="class")
def real_model(gpu_id):
    """Class-scoped fixture that loads a real model once per test class.

    Class scope isolates accelerate's device-dispatch hook state between test
    classes so a prior test's forward pass cannot corrupt Qwen2 RMSNorm on a
    subsequent class — an upstream torch/accelerate issue that otherwise
    manifests as a C-level abort. Loaded in fp16 (no bitsandbytes 4-bit).
    """
    if gpu_id is None:
        pytest.skip("No GPU available")
    yield from _load_and_yield(gpu_id, quantization=None)


@pytest.fixture(scope="class")
def streaming_model(gpu_id):
    """Class-scoped fixture for streaming tests — loaded without 4-bit quant.

    bitsandbytes 4-bit kernels leave CUDA stream state that causes
    TextIteratorStreamer.text_queue.get() to block indefinitely.  Loading
    the model in fp16 / bf16 avoids those kernels entirely. Class scope
    (not module / session) prevents cross-class state bleed.
    """
    if gpu_id is None:
        pytest.skip("No GPU available")
    yield from _load_and_yield(gpu_id, quantization=None)

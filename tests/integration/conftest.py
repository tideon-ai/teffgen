"""
Integration test fixtures requiring real GPU models.
"""

import os
import sys
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
            props = torch.cuda.get_device_properties(i)
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
    from effgen import load_model
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


@pytest.fixture(scope="module")
def real_model(gpu_id):
    """Module-scoped fixture that loads a real model once per test module.

    Module scope (not session) prevents bitsandbytes 4-bit CUDA state from
    accumulating across test modules, which causes TextIteratorStreamer to
    deadlock in the streaming tests.
    """
    if gpu_id is None:
        pytest.skip("No GPU available")
    yield from _load_and_yield(gpu_id, quantization="4bit")


@pytest.fixture(scope="module")
def streaming_model(gpu_id):
    """Module-scoped fixture for streaming tests — loads WITHOUT 4-bit quantization.

    bitsandbytes 4-bit kernels leave CUDA stream state that causes
    TextIteratorStreamer.text_queue.get() to block indefinitely.  Loading
    the model in fp16 / bf16 avoids those kernels entirely.
    """
    if gpu_id is None:
        pytest.skip("No GPU available")
    yield from _load_and_yield(gpu_id, quantization=None)

# Installation Guide

## Standard install

```bash
pip install teffgen                    # base install (all core features)
pip install teffgen[all]                # everything except flash-attn
pip install teffgen[dev]                # dev tools (pytest, ruff, mypy, pytest-forked)
pip install teffgen[rag]                # sentence-transformers + faiss-cpu
pip install teffgen[vector-db]          # faiss + chromadb + qdrant
pip install teffgen[vllm]               # vLLM backend (NVIDIA GPUs)
pip install teffgen[mlx]                # MLX backend (Apple Silicon)
```

> **Why isn't `flash-attn` in `[all]`?**
> `flash-attn`'s own `setup.py` imports `torch` during wheel-metadata generation,
> but pip's isolated build environment does not have `torch` available at that
> moment. This is a well-known upstream bug in `flash-attn` — any package that
> lists `flash-attn` as a dependency will cause `pip install` to fail. To keep
> `pip install teffgen[all]` working for everyone, `flash-attn` is kept out of
> `[all]` and installed separately (see below).

## Installing flash-attn (optional, NVIDIA GPUs only)

**Step 1 — install teffgen first (gets torch and everything else):**

```bash
pip install teffgen[all]
```

**Step 2 — install flash-attn with build isolation disabled:**

```bash
pip install flash-attn --no-build-isolation
```

`--no-build-isolation` lets flash-attn's setup.py reuse the torch already
installed in your environment, bypassing the bug.

### Requirements for flash-attn

- NVIDIA GPU with compute capability ≥ 7.5 (Turing or newer)
- CUDA toolkit (`nvcc`) matching your torch CUDA version
- GCC 9+ and a few GB of RAM for compilation (can take 10–30 minutes)

If the build fails, prefer the official pre-built wheel from
<https://github.com/Dao-AILab/flash-attention/releases> matching your
exact `(python, torch, cuda)` triple.

## Installing vLLM (optional)

vLLM ships pre-built wheels on PyPI for common
`(python, torch, cuda)` combinations:

```bash
pip install teffgen[vllm]
```

If the resolver cannot find a matching wheel, use vLLM's extra index:

```bash
pip install teffgen
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124
```

## Supported Python versions

`teffgen` officially supports Python **3.10, 3.11, 3.12, 3.13**. Python 3.14
is best-effort — several upstream packages (torch, bitsandbytes) do not yet
ship cp314 wheels.

## Verifying your install

```bash
python -c "import teffgen; print(teffgen.__version__)"
python -c "from teffgen import Agent; print(Agent)"
```

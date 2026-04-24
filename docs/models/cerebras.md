# Cerebras Backend

effGen supports [Cerebras Cloud](https://inference.cerebras.ai/) as a hosted
inference backend via `CerebrasAdapter`.

## Setup

```bash
pip install "effgen[cerebras]"
```

Set your API key (get one at <https://cloud.cerebras.ai/>):

```bash
export CEREBRAS_API_KEY="your-key-here"
# or place it in ~/.effgen/.env
```

## Quick start

```python
from effgen import CerebrasAdapter

adapter = CerebrasAdapter(model_name="llama3.1-8b")
adapter.load()

result = adapter.generate("What is 7 * 8?")
print(result.text)
print("tokens:", result.metadata["prompt_tokens"], "->", result.metadata["completion_tokens"])

adapter.unload()
```

### Via `load_model`

```python
from effgen.models import load_model

model = load_model("llama3.1-8b", provider="cerebras")
print(model.generate("Hello!").text)
model.unload()
```

## Supported models (Phase 1)

| Model | Context | Max output | Notes |
|-------|---------|-----------|-------|
| `llama3.1-8b` | 8 192 | 8 192 | **Default** — free-tier callable |
| `qwen-3-235b-a22b-instruct-2507` | 128 000 | 16 000 | Free-tier callable |
| `gpt-oss-120b` | 128 000 | 8 192 | Listed by the API but currently returns 404 on the free tier; upgrade account to access |
| `zai-glm-4.7` | 128 000 | 8 192 | Listed; availability depends on tier |

Phase 2 will add full rate-limit metadata and a rate-limit coordinator for
all four models.

## Streaming

Streaming is not yet available — it arrives in v0.2.1 Phase 3.
Calling `generate_stream()` raises `NotImplementedError` with a clear message.

## Environment variables

| Variable | Description |
|----------|-------------|
| `CEREBRAS_API_KEY` | Cerebras Cloud API key |

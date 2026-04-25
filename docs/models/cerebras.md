# Cerebras Backend

tideon.ai supports [Cerebras Cloud](https://inference.cerebras.ai/) as a hosted
inference backend via `CerebrasAdapter`.  All four free-tier models are
registered, with automatic rate-limit enforcement built in.

## Setup

```bash
pip install "teffgen[cerebras]"
```

Set your API key (get one at <https://cloud.cerebras.ai/>):

```bash
export CEREBRAS_API_KEY="your-key-here"
# or place it in ~/.teffgen/.env
```

## Quick start

```python
from teffgen import CerebrasAdapter

adapter = CerebrasAdapter(model_name="llama3.1-8b")
adapter.load()

result = adapter.generate("What is 7 * 8?")
print(result.text)
print("tokens:", result.metadata["prompt_tokens"], "->", result.metadata["completion_tokens"])

adapter.unload()
```

### Via `load_model`

```python
from teffgen.models import load_model

model = load_model("llama3.1-8b", provider="cerebras")
print(model.generate("Hello!").text)
model.unload()
```

### Async generation (recommended in async contexts)

```python
import asyncio
from teffgen import CerebrasAdapter

async def main():
    adapter = CerebrasAdapter(model_name="llama3.1-8b")
    adapter.load()
    result = await adapter.async_generate("Explain quantum entanglement briefly.")
    print(result.text)
    adapter.unload()

asyncio.run(main())
```

## Supported models

Values fetched from Cerebras docs on 2026-04-24.  Free-tier context /
max-output shown first; paid-tier values in parentheses.

| Model | Context (free / paid) | Max output (free / paid) | RPM | RPH | RPD | TPM | Free-tier callable |
|-------|-----------------------|--------------------------|-----|-----|-----|-----|--------------------|
| `llama3.1-8b` | 8k / 32k | 8k / 8k | 30 | 900 | 14 400 | 60 000 | ✓ |
| `qwen-3-235b-a22b-instruct-2507` | 65k / 131k | 32k / 40k | 30 | 900 | 14 400 | 60 000 | ✓ |
| `gpt-oss-120b` | 65k / 131k | 32k / 40k | 30 | 900 | 14 400 | 64 000 | ✗ (restricted) |
| `zai-glm-4.7` | 64k / 131k | 40k / 40k | 10 | 100 | 100 | 60 000 | ✗ (restricted) |

`gpt-oss-120b` and `zai-glm-4.7` are listed by the API (and work on paid
tier) but Cerebras has "temporarily reduced free-tier rate limits" on them
due to high demand — free-tier keys typically receive a `404 model_not_found`
response. `llama3.1-8b` and `qwen-3-235b-a22b-instruct-2507` are reliably
callable on the free tier today.

**Deprecation notice:** `llama3.1-8b` and `qwen-3-235b-a22b-instruct-2507`
are scheduled for deprecation on **2026-05-27** per Cerebras docs.

### Paid-tier limits

The Pay-as-You-Go tier has no hourly/daily caps and significantly higher
per-minute limits (e.g. `llama3.1-8b` is 2M TPM / 2K RPM on paid tier).
The tideon.ai `RateLimitCoordinator` is initialised with free-tier limits by
default — construct `CerebrasAdapter(model_name=..., enable_rate_limiting=False)`
or pass a custom `RateLimitCoordinator` if you're on the paid tier.

### Inspect the registry in code

```python
from teffgen.models.cerebras_models import available_models, free_tier_models, model_info

print(available_models())          # all 4 models
print(free_tier_models())          # models accessible on the free tier

info = model_info("llama3.1-8b")
print(info["rpm"], info["tpm"])    # 30, 60000
```

## Rate-limit coordinator

Each `CerebrasAdapter` instance comes with a built-in `RateLimitCoordinator`
that tracks sliding-window RPM / RPH / RPD and TPM / TPH / TPD limits.

When you call `generate()` or `async_generate()`, the coordinator:
1. Estimates the token cost from the prompt.
2. Blocks with `asyncio.sleep` if a per-minute / per-hour limit would be exceeded.
3. Records the actual tokens used after each call.

```python
adapter = CerebrasAdapter(model_name="llama3.1-8b")
adapter.load()

result = adapter.generate("Hello!")

# Inspect coordinator state
status = adapter.rate_limit_status()
print(status["req_minute_used"], "/", status["req_minute_limit"])  # e.g. 1 / 30
print(status["total_throttled"])   # number of requests that were delayed
```

To disable rate limiting (e.g., in tests or when managing limits externally):

```python
adapter = CerebrasAdapter(model_name="llama3.1-8b", enable_rate_limiting=False)
```

### RateLimitExceeded

If the **daily** budget is fully consumed, `acquire()` raises
`teffgen.models._rate_limit.RateLimitExceeded` instead of sleeping indefinitely.

```python
from teffgen import RateLimitExceeded

try:
    result = adapter.generate("hello")
except RateLimitExceeded as exc:
    print(f"Daily budget exhausted: {exc}")
```

## Agent integration

```python
from teffgen.core.agent import Agent, AgentConfig
from teffgen.models.cerebras_adapter import CerebrasAdapter
from teffgen.tools.builtin import Calculator, DateTimeTool

adapter = CerebrasAdapter(model_name="llama3.1-8b")
adapter.load()

config = AgentConfig(
    name="cerebras-assistant",
    model=adapter,
    tools=[Calculator(), DateTimeTool()],
    max_iterations=5,
)

with Agent(config) as agent:
    response = agent.run(
        "If I invest $5000 at 7% annual interest for 10 years, "
        "what is the final amount? Use compound interest: A = P*(1+r)^t"
    )
    print(response.output)
```

## Streaming

Streaming (`generate_stream`) is not yet available.
Calling it raises `NotImplementedError`.

## Environment variables

| Variable | Description |
|----------|-------------|
| `CEREBRAS_API_KEY` | Cerebras Cloud API key (required) |

## Examples

See `examples/cerebras/` for runnable examples:

| File | Description |
|------|-------------|
| `basic_cerebras.py` | Simple generation, token counting |
| `cerebras_agent.py` | Agent with Calculator and DateTimeTool |
| `cerebras_all_models.py` | Compare all models in parallel |
| `cerebras_rate_limits.py` | Rate-limit coordinator demo |
| `cerebras_hard_agent.py` | Hard multi-step agentic tasks |
| `cerebras_multi_turn.py` | Multi-turn conversation |

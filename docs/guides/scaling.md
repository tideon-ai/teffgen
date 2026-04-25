# Scaling Guide

This guide covers scaling tideon.ai for production workloads — from request throughput to multi-GPU model management.

## Request Queue Tuning

The production API server uses a priority queue with backpressure:

```python
from teffgen.api.queue import RequestQueue, RequestPriority

queue = RequestQueue(
    max_size=1000,          # Max pending requests before backpressure (503)
    default_priority=RequestPriority.NORMAL,
)
```

Priority levels: `HIGH` > `NORMAL` > `LOW`. High-priority requests (e.g., from premium tenants) are always dequeued first.

When the queue is full, new requests receive a `QueueFullError` (HTTP 503). Clients should implement exponential backoff.

## Agent Pool Sizing

Pre-warm agents to avoid cold-start latency:

```python
from teffgen.api.pool import AgentPool

pool = AgentPool(
    factory=lambda: create_agent("general", model),
    min_size=2,     # Always keep 2 agents warm
    max_size=10,    # Scale up to 10 under load
    idle_ttl=300,   # Reclaim idle agents after 5 minutes
)
```

**Sizing guidelines:**
- `min_size` = expected baseline concurrency
- `max_size` = peak concurrency (limited by GPU memory)
- `idle_ttl` = balance between memory and cold-start latency

## Model Pool & LRU Eviction

Manage multiple models across GPUs:

```python
from teffgen.models.pool import ModelPool, PoolConfig

pool = ModelPool(config=PoolConfig(
    max_loaded_models=4,        # Keep at most 4 models in GPU memory
    gpu_memory_limit_gb=40,     # Total GPU memory budget
))

# Pre-warm critical models
pool.prewarm("Qwen/Qwen2.5-3B-Instruct")

# Models are loaded on demand and evicted LRU when limits are reached
model = pool.get_or_load("Qwen/Qwen2.5-7B-Instruct")
```

## Lazy Model Loading

Defer model loading until first use:

```python
from teffgen.models import LazyModel

model = LazyModel(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    idle_timeout=600,   # Unload after 10 minutes of inactivity
)
# Model is NOT loaded yet
result = model.generate(prompt)  # NOW it loads, then generates
```

## Continuous Batching

Coalesce concurrent requests for higher GPU throughput:

```python
from teffgen.models import ContinuousBatcher

batcher = ContinuousBatcher(
    model=model,
    max_batch_size=8,     # Flush after 8 requests
    max_wait_ms=50,       # Or after 50ms, whichever comes first
)

with batcher:
    # Multiple concurrent submit() calls are batched automatically
    future = batcher.submit(prompt, generation_config)
    result = future.result()
```

## Multi-GPU Distribution

tideon.ai uses `CUDA_VISIBLE_DEVICES` for GPU targeting:

```bash
# Run on specific GPUs
CUDA_VISIBLE_DEVICES=0,1 teffgen serve --port 8000

# Model loader respects device_map
model = load_model("Qwen/Qwen2.5-7B-Instruct", device_map="auto")
```

For vLLM, tensor parallelism is auto-detected based on model size.

## Batch Execution

Process large query sets efficiently:

```python
results = agent.run_batch(
    queries=["query1", "query2", ...],
    concurrency=5,
    timeout=60,
    retries=2,
)
```

Or via CLI:

```bash
teffgen batch --input queries.jsonl --output results.jsonl \
  --concurrency 10 --batch-size 100 --timeout 60 --retries 2
```

## Domain Keyword Expansion

Scale keyword coverage per domain:

```python
from teffgen.domains import KeywordExpander

expander = KeywordExpander()

# Template-based: 2 seeds → ~18 terms
expanded = expander.expand_template(["machine learning", "data science"])

# WordNet-based: 2 seeds → ~300 terms
expanded = expander.expand_wordnet(["machine learning", "data science"])

# LLM-based: 2 seeds → ~44 terms (uses the agent's own model)
expanded = expander.expand_llm(["machine learning", "data science"], model=model)
```

## Caching

### Prompt Cache

Avoid re-computing prompts for identical or similar queries:

```python
from teffgen.cache import PromptCache

cache = PromptCache(max_size=10000, ttl=3600)
cache.put(prompt, result)
cached = cache.get(prompt)  # O(1) lookup via sha256 fingerprint
print(cache.stats)  # {"hits": 42, "misses": 10, "hit_rate": 0.81}
```

### Result Cache

Cache tool results with per-tool TTL:

```python
from teffgen.cache import ResultCache

cache = ResultCache(max_size=5000)
cache.set_tool_ttl("web_search", 300)   # 5 minutes for web search
cache.set_tool_ttl("calculator", 86400)  # 24 hours for math (deterministic)
```

## Token Budget Management

Optimize context window usage:

```python
from teffgen.memory.token_budget import TokenBudget

budget = TokenBudget(
    total_tokens=4096,
    system_share=0.20,   # 20% for system prompt
    tools_share=0.30,    # 30% for tool descriptions
    history_share=0.40,  # 40% for conversation history
    response_share=0.10, # 10% reserved for response
)

truncated = budget.fit_to_budget(system=system_prompt, tools=tools_text, history=history)
```

## Monitoring at Scale

- **Prometheus metrics** — `GET /metrics` exposes latency histograms (p50/p95/p99), throughput, error rates, GPU memory
- **Grafana dashboard** — import `configs/grafana/teffgen-dashboard.json`
- **OpenTelemetry** — trace propagation across agents, exporters for OTLP/Jaeger/Zipkin
- **Structured logging** — JSON format with run_id correlation for distributed tracing

# tideon.ai (`teffgen`)

Agentic AI framework for small and large language models.
Multi-agent workflows, tools, RAG, guardrails — usable as a Python SDK or as a self-hosted HTTP API.

## Install

```bash
pip install -e .
```

For Apple Silicon MLX (faster local inference):
```bash
pip install -e '.[mlx]'
```

## Use as SDK

```python
from teffgen import Agent, AgentConfig, load_model
from teffgen.tools.builtin import Calculator

model = load_model("Qwen/Qwen2.5-1.5B-Instruct")  # or any HF model id / API adapter
agent = Agent(config=AgentConfig(name="x", model=model, tools=[Calculator()]))
print(agent.run("What is 17 * 23?").output)
```

End-to-end demo of 9 subsystems (agents, memory, guardrails, RAG, workflow DAG, structured output):
```bash
python examples/demo_all.py --engine mlx --model LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit
```

## Use as API

```bash
docker run -p 8000:8000 -e TEFFGEN_API_KEY=secret ghcr.io/tideon-ai/teffgen-api:0.2.0
curl -H "Authorization: Bearer secret" http://localhost:8000/health
```

Endpoints: `/run`, `/health`, `/tools`, `/metrics`, `/ws` (WebSocket streaming).

## Use as base image

```dockerfile
FROM ghcr.io/tideon-ai/teffgen-sdk:0.2.0
COPY my_app/ /app/
ENV WORKER_CMD="python /app/main.py"
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

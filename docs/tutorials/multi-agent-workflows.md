# Multi-Agent Workflows

tideon.ai v0.2.0 provides advanced multi-agent orchestration with a message bus, DAG-based workflows, shared state, and agent lifecycle management.

## MessageBus — Agent Communication

```python
from teffgen.core.message_bus import MessageBus, AgentMessage, MessageType

bus = MessageBus()

# Subscribe to topics
bus.subscribe("results.*", callback=lambda msg: print(f"Got result: {msg.payload}"))

# Publish messages
bus.publish("results.math", AgentMessage(
    type=MessageType.RESULT,
    sender="math_agent",
    payload={"answer": 42},
))

# Mailbox-based (direct agent-to-agent)
bus.send("research_agent", AgentMessage(
    type=MessageType.TASK_ASSIGNMENT,
    sender="coordinator",
    payload={"task": "Search for quantum computing papers"},
))
messages = bus.receive("research_agent")
```

## DAG-Based Workflows

Define complex workflows as directed acyclic graphs:

### Python API

```python
from teffgen.core.workflow import WorkflowDAG, WorkflowNode

dag = WorkflowDAG()
dag.add_node(WorkflowNode(id="research", agent=research_agent, task="Find papers on topic"))
dag.add_node(WorkflowNode(id="summarize", agent=summary_agent, task="Summarize findings"))
dag.add_node(WorkflowNode(id="format", agent=format_agent, task="Format as report"))

dag.add_edge("research", "summarize")
dag.add_edge("summarize", "format")

results = dag.execute()  # Runs in topological order, parallelizes independent nodes
```

### YAML Workflow Definitions

```yaml
# workflow.yaml
nodes:
  - id: fetch_data
    preset: research
    task: "Fetch latest data on {topic}"
  - id: analyze
    preset: coding
    task: "Analyze the data and compute statistics"
    depends_on: [fetch_data]
  - id: visualize
    preset: coding
    task: "Create charts from the analysis"
    depends_on: [analyze]
  - id: report
    preset: general
    task: "Write a summary report"
    depends_on: [analyze, visualize]
```

```bash
# CLI
teffgen workflow validate workflow.yaml
teffgen workflow run workflow.yaml
```

### Conditional Branching

Edges can have conditions based on previous node outputs:

```python
dag.add_edge("classify", "handle_urgent", condition=lambda result: "urgent" in result.output)
dag.add_edge("classify", "handle_normal", condition=lambda result: "urgent" not in result.output)
```

## Shared State

Thread-safe key-value store shared across agents in a workflow:

```python
from teffgen.core.shared_state import SharedState

state = SharedState()

# Namespaced access
state.set("research", "papers_found", 42)
state.set("analysis", "top_topic", "quantum computing")

count = state.get("research", "papers_found")  # 42

# Snapshots for rollback
snapshot = state.snapshot()
# ... do work ...
state.restore(snapshot)  # Rollback
```

## Agent Lifecycle Management

```python
from teffgen.core.lifecycle import AgentRegistry, AgentPool

# Registry — track all agents
registry = AgentRegistry()
registry.register(agent, timeout=300)  # 5-minute timeout

# Pool — pre-warmed agents for fast allocation
pool = AgentPool(factory=lambda: create_agent("general", model), min_size=2, max_size=10)
agent = pool.acquire()
try:
    result = agent.run("task")
finally:
    pool.release(agent)

# Timeout and cancellation
registry.check_timeouts()    # Cancel agents that exceeded timeout
registry.cancel("agent-id")  # Cancel specific agent
```

## CLI Commands

```bash
# Workflow operations
teffgen workflow run pipeline.yaml --model "Qwen/Qwen2.5-3B-Instruct"
teffgen workflow validate pipeline.yaml

# Batch execution
teffgen batch --input queries.jsonl --output results.jsonl --concurrency 5 --preset research
```

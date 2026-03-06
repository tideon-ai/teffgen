"""Streaming agent example — real-time token-by-token output.

Demonstrates agent.stream() which yields tokens as the model generates them.

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/streaming_agent.py
"""

import logging
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import Calculator

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

print("Loading model:", MODEL_ID)
model = load_model(MODEL_ID, quantization="4bit")

config = AgentConfig(
    name="streaming_agent",
    model=model,
    tools=[Calculator()],
    system_prompt="You are a helpful assistant. Use the calculator for math.",
    enable_streaming=True,
    max_iterations=5,
)

agent = Agent(config=config)

print("\n" + "=" * 60)
print("Streaming response for: 'What is 2+2?'")
print("=" * 60 + "\n")

# Stream tokens as they arrive
for token in agent.stream("What is 2+2?"):
    print(token, end="", flush=True)

print("\n\n" + "=" * 60)
print("Streaming complete!")
print("=" * 60)

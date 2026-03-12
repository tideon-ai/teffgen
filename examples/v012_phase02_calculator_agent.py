#!/usr/bin/env python3
"""
effGen v0.1.2 — Phase 2: Calculator Agent (Single Tool)

A math agent using the Calculator and PythonREPL tools via the ReAct loop.
Tests the core tool-calling pipeline: model produces Thought/Action/Action Input,
framework parses them, tool executes, observation returned.

Tested models (post-fix results):
  - Qwen/Qwen2.5-0.5B-Instruct (0.5B)  — 5/7 PASS, 2 PARTIAL (reasoning limits)
  - Qwen/Qwen2.5-1.5B-Instruct (1.5B)  — 6/7 PASS, 1 PARTIAL
  - Qwen/Qwen2.5-3B-Instruct (3B)      — 7/7 PASS, primary dev model
  - Qwen/Qwen2.5-7B-Instruct (7B)      — 7/7 PASS, best quality
  - meta-llama/Llama-3.2-3B-Instruct    — 5/7 PASS, 2 PARTIAL (multi-step reasoning)
  - microsoft/Phi-4-mini-instruct       — 7/7 PASS (after trailing-text fix)
  - google/gemma-3-4b-it                — 7/7 PASS (after loop detection fix)

Framework bugs fixed in this phase:
  - BUG-003: Loop detection — models that repeat the same action+input are now
    detected and short-circuited with the last observation as the answer
  - BUG-004: "Answer:" pattern tightened — now requires line-start anchor to
    prevent greedy mid-text matching
  - BUG-005: Trailing unrelated text stripped from Final Answer — fixes Phi-4-mini
    generating follow-up questions after the answer

Tools used: Calculator (primary), PythonREPL (fallback)
Path: Agent.run() → _run_single_agent() → ReAct loop → _execute_tool() → Calculator._execute()

Usage:
  # Run calculator demo (default model: Qwen2.5-3B-Instruct)
  CUDA_VISIBLE_DEVICES=0 python examples/v012_phase02_calculator_agent.py

  # Specify a model
  CUDA_VISIBLE_DEVICES=0 python examples/v012_phase02_calculator_agent.py --model Qwen/Qwen2.5-7B-Instruct

  # Interactive mode
  CUDA_VISIBLE_DEVICES=0 python examples/v012_phase02_calculator_agent.py --interactive
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure effgen is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effgen import load_model
from effgen.presets import create_agent


def demo_calculator(agent):
    """Demonstrate calculator tool usage with various math tasks."""
    tasks = [
        ("Simple arithmetic", "What is 247 * 83?", "20501"),
        ("Math function", "What is the square root of 144?", "12"),
        ("Multi-step expression", "What is (15 + 27) * 3 - 10?", "116"),
        ("Natural language math", "Add 5 and 3, then multiply the result by 2.", "16"),
        ("Direct computation", "Use the calculator to compute 2+2.", "4"),
    ]

    results = []
    for desc, question, expected in tasks:
        print(f"\n{'='*50}")
        print(f"Test: {desc}")
        print(f"Q: {question}")
        print(f"Expected: {expected}")

        t0 = time.time()
        resp = agent.run(question)
        dt = time.time() - t0

        output = resp.output if resp.output else ""
        passed = expected.lower() in output.lower()

        print(f"A: {output}")
        print(f"Tool calls: {resp.tool_calls}, Iterations: {resp.iterations}, Time: {dt:.1f}s")
        print(f"Result: {'PASS' if passed else 'CHECK'}")
        results.append((desc, passed, dt))

    print(f"\n{'='*50}")
    print("Summary:")
    for desc, passed, dt in results:
        print(f"  {'PASS' if passed else 'CHECK':5s} — {desc} ({dt:.1f}s)")
    pass_count = sum(1 for _, p, _ in results if p)
    print(f"\n{pass_count}/{len(results)} passed")


def interactive_mode(agent):
    """Interactive math calculator chat."""
    print("\n--- Interactive Calculator (type 'quit' to exit) ---")
    print("Ask any math question. The agent will use Calculator or PythonREPL.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            t0 = time.time()
            resp = agent.run(user_input)
            dt = time.time() - t0
            print(f"Agent: {resp.output}")
            print(f"   (tools={resp.tool_calls}, iters={resp.iterations}, {dt:.1f}s)")
        except (KeyboardInterrupt, EOFError):
            break
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="effGen Phase 2: Calculator Agent")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to use (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive math chat")
    args = parser.parse_args()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(f"effGen v0.1.2 — Phase 2: Calculator Agent")
    print(f"Model: {args.model}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")

    # Load model
    print(f"\nLoading model...")
    t0 = time.time()
    model = load_model(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Create math agent (Calculator + PythonREPL)
    agent = create_agent("math", model)
    print(f"Math agent created with tools: {list(agent.tools.keys())}")

    if args.interactive:
        interactive_mode(agent)
    else:
        demo_calculator(agent)

    # Cleanup
    model.unload()
    print("\nDone.")


if __name__ == "__main__":
    main()

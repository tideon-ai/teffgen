"""
Basic Cerebras example — direct adapter usage.

Prerequisites:
    pip install "teffgen[cerebras]"
    export CEREBRAS_API_KEY="your-key"   # or place in ~/.teffgen/.env

What this demonstrates:
  - Loading CerebrasAdapter
  - Simple text generation
  - Token counting
  - Context length query
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".teffgen" / ".env", override=False)

if not os.getenv("CEREBRAS_API_KEY"):
    raise SystemExit("Set CEREBRAS_API_KEY in ~/.teffgen/.env or the environment.")

from teffgen.models.base import GenerationConfig  # noqa: E402
from teffgen.models.cerebras_adapter import CerebrasAdapter  # noqa: E402

# -----------------------------------------------------------
# 1. Load the adapter
# -----------------------------------------------------------
print("Loading Cerebras adapter (llama3.1-8b) ...")
adapter = CerebrasAdapter(model_name="llama3.1-8b")
adapter.load()

print(f"Context length : {adapter.get_context_length():,} tokens")

# -----------------------------------------------------------
# 2. Simple generation
# -----------------------------------------------------------
result = adapter.generate("What is the speed of light in km/s? Give a one-line answer.")
print(f"\nResponse       : {result.text}")
print(f"Tokens used    : {result.tokens_used}  (prompt={result.metadata['prompt_tokens']})")
print(f"Finish reason  : {result.finish_reason}")

# -----------------------------------------------------------
# 3. Custom generation config
# -----------------------------------------------------------
config = GenerationConfig(temperature=0.2, max_tokens=64)
result2 = adapter.generate(
    "List three famous physicists. Be brief.",
    config=config,
)
print(f"\nWith custom config:\n{result2.text}")

# -----------------------------------------------------------
# 4. Token counting
# -----------------------------------------------------------
sample = "The quick brown fox jumps over the lazy dog."
tc = adapter.count_tokens(sample)
print(f"\nToken count for '{sample}': {tc.count}")

# -----------------------------------------------------------
# 5. Cleanup
# -----------------------------------------------------------
adapter.unload()
print("\nDone.")

"""
Cerebras model registry for Phase 1 (gpt-oss-120b only).
Phase 2 will extend this to all 4 free-tier models.
"""

# Source: Cerebras inference docs + live models.list() call (verified 2026-04-23)
# NOTE: gpt-oss-120b is listed by the API but inference returns 404 on the free
# tier — see build_plan/v0.2.1/followups/TODO_p1_gpt_oss_120b_access.md.
# llama3.1-8b is the callable free-tier model used for Phase 1 live validation.
CEREBRAS_MODELS: dict[str, dict] = {
    "gpt-oss-120b": {
        "context": 128_000,
        "max_output": 8_192,
    },
    "llama3.1-8b": {
        "context": 8_192,
        "max_output": 8_192,
    },
    # Free-tier callable (tested 2026-04-23); may hit 429 under high traffic
    "qwen-3-235b-a22b-instruct-2507": {
        "context": 128_000,
        "max_output": 16_000,
    },
    # Listed in models.list() but inference returns 404 on current tier
    # See TODO_p1_gpt_oss_120b_access.md
    "zai-glm-4.7": {
        "context": 128_000,
        "max_output": 8_192,
    },
}

# Default model for live testing (free-tier callable)
CEREBRAS_DEFAULT_MODEL = "llama3.1-8b"

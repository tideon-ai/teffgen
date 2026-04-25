"""
Prompt injection detection guardrail for tideon.ai.

Detects common prompt injection patterns including instruction override,
role-play injection, system prompt extraction, and delimiter injection.
All detection is regex/keyword-based — no external APIs or ML models.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Guardrail, GuardrailPosition, GuardrailResult


class PromptInjectionGuardrail(Guardrail):
    """Detect prompt injection attempts at configurable sensitivity levels.

    Sensitivity levels:
        - ``low``: Only obvious, high-confidence injection patterns.
        - ``medium``: Adds system prompt extraction and role-play patterns.
        - ``high``: Adds delimiter injection and more aggressive matching.

    Designed to have ZERO false positives on normal questions like
    "What is a system prompt?" or "Ignore the noise and focus on the data."
    """

    # ---- HIGH-CONFIDENCE patterns (all levels) ----

    # Direct instruction override: "Ignore previous instructions", etc.
    # Requires imperative form + "instructions/prompts" to avoid false positives.
    _INSTRUCTION_OVERRIDE: list[re.Pattern[str]] = [
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass|skip|do\s+not\s+follow)"
            r"\s+(?:all\s+)?(?:previous|prior|above|earlier|original|initial|your|the|my)?"
            r"\s*(?:instructions?|prompts?|rules?|guidelines?|directives?|constraints?|context)",
            re.I,
        ),
        re.compile(
            r"\b(?:ignore|disregard|forget)\s+(?:everything|anything)\s+"
            r"(?:above|before|previously|i\s+(?:said|told|wrote))",
            re.I,
        ),
    ]

    # Explicit new identity assignment: "You are now ...", "Act as ..."
    # Must be at start of content or after a delimiter to avoid matching
    # normal text like "If you are now ready..."
    _IDENTITY_OVERRIDE: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are|"
            r"pretend\s+(?:to\s+be|you\s+are)|"
            r"you\s+(?:must|should|will)\s+now\s+(?:act|behave|respond)\s+as)\b",
            re.I,
        ),
    ]

    # ---- MEDIUM-CONFIDENCE patterns (medium + high) ----

    # System prompt extraction attempts
    _SYSTEM_PROMPT_EXTRACTION: list[re.Pattern[str]] = [
        re.compile(
            r"\b(?:(?:show|reveal|print|display|output|repeat|echo|tell|give|what\s+(?:is|are))"
            r"(?:\s+me)?\s+(?:your|the)\s+(?:system\s+(?:prompt|message|instructions?)|"
            r"initial\s+(?:prompt|instructions?)|hidden\s+(?:prompt|instructions?)|"
            r"(?:original|full|complete|entire)\s+(?:prompt|instructions?)))",
            re.I,
        ),
    ]

    # Role-play injection: "You are DAN", jailbreak patterns
    _ROLEPLAY_INJECTION: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:you\s+are\s+(?:DAN|STAN|DUDE|AIM|KEVIN|BARD|Sydney)\b|"
            r"enable\s+(?:developer|DAN|jailbreak)\s+mode|"
            r"(?:enter|switch\s+to|activate)\s+(?:unrestricted|unfiltered|uncensored|DAN)\s+mode)",
            re.I,
        ),
    ]

    # ---- HIGH-SENSITIVITY patterns (high only) ----

    # Delimiter injection: attempting to create fake system/user boundaries
    _DELIMITER_INJECTION: list[re.Pattern[str]] = [
        # Fake message boundaries
        re.compile(
            r"(?:^|\n)\s*(?:<\|?(?:system|im_start|endoftext)\|?>|"
            r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|"
            r"###\s*(?:System|Human|Assistant|User)\s*(?::|###))",
            re.I,
        ),
        # Markdown-style system prompt injection
        re.compile(
            r"(?:^|\n)\s*(?:---+\s*\n\s*(?:System|New\s+System)\s*(?:Prompt|Instructions?)\s*:)",
            re.I,
        ),
    ]

    # Direct "System:" / "System prompt:" header injection
    _SYSTEM_HEADER_INJECTION: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:System\s*(?:Prompt|Message|Instructions?)\s*:\s*.{10,})",
            re.I,
        ),
    ]

    def __init__(
        self,
        sensitivity: str = "medium",
        positions: list[GuardrailPosition] | None = None,
        enabled: bool = True,
    ):
        super().__init__(
            name="PromptInjectionGuardrail",
            positions=positions or [GuardrailPosition.INPUT],
            enabled=enabled,
        )
        if sensitivity not in ("low", "medium", "high"):
            raise ValueError(f"Invalid sensitivity: {sensitivity!r}. Must be 'low', 'medium', or 'high'.")
        self.sensitivity = sensitivity

    def _get_patterns(self) -> list[tuple[str, re.Pattern[str]]]:
        """Return (label, pattern) pairs for the current sensitivity level."""
        patterns: list[tuple[str, re.Pattern[str]]] = []

        # Low: always included
        for p in self._INSTRUCTION_OVERRIDE:
            patterns.append(("instruction_override", p))
        for p in self._IDENTITY_OVERRIDE:
            patterns.append(("identity_override", p))

        if self.sensitivity in ("medium", "high"):
            for p in self._SYSTEM_PROMPT_EXTRACTION:
                patterns.append(("system_prompt_extraction", p))
            for p in self._ROLEPLAY_INJECTION:
                patterns.append(("roleplay_injection", p))

        if self.sensitivity == "high":
            for p in self._DELIMITER_INJECTION:
                patterns.append(("delimiter_injection", p))
            for p in self._SYSTEM_HEADER_INJECTION:
                patterns.append(("system_header_injection", p))

        return patterns

    def check(self, content: str, **kwargs: Any) -> GuardrailResult:
        for label, pattern in self._get_patterns():
            match = pattern.search(content)
            if match:
                return GuardrailResult(
                    passed=False,
                    reason=f"Prompt injection guardrail: {label} detected",
                    metadata={
                        "pattern_type": label,
                        "matched_text": match.group()[:100],
                        "sensitivity": self.sensitivity,
                    },
                )

        return GuardrailResult(passed=True)

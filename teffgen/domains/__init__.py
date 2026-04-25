"""
tideon.ai Domains — Knowledge domain definitions with keyword expansion.

Provides domain-specific configurations (keywords, prompts, tools, guardrails)
and a KeywordExpander that grows N seed keywords to 10N+ using WordNet,
templates, and optional LLM-based expansion.

Usage:
    from teffgen.domains import TechDomain, KeywordExpander, Domain

    domain = TechDomain(keywords=["Python", "machine learning"])
    expanded = domain.expand_keywords(factor=10)
"""

from teffgen.domains.base import Domain
from teffgen.domains.expander import KeywordExpander
from teffgen.domains.presets import (
    FinanceDomain,
    HealthDomain,
    LegalDomain,
    ScienceDomain,
    TechDomain,
)

__all__ = [
    "Domain",
    "KeywordExpander",
    "TechDomain",
    "ScienceDomain",
    "FinanceDomain",
    "HealthDomain",
    "LegalDomain",
]

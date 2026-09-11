"""Complexity tier definitions shared by the dataset, classifier, and router.

Tier 1 (simple): reformatting, extraction, basic Q&A from provided context —
mechanical, near-zero judgment, a short deterministic answer.
Tier 2 (moderate): summarization, classification, structured analysis — some
judgment, single-pass reasoning over given material.
Tier 3 (complex): multi-step reasoning, creative generation, nuanced
judgment calls — the kind of task where a weak model visibly struggles.
"""
from __future__ import annotations

TIER_SIMPLE = 1
TIER_MODERATE = 2
TIER_COMPLEX = 3

TIERS = (TIER_SIMPLE, TIER_MODERATE, TIER_COMPLEX)

TIER_NAMES = {
    TIER_SIMPLE: "simple",
    TIER_MODERATE: "moderate",
    TIER_COMPLEX: "complex",
}

TIER_DESCRIPTIONS = {
    TIER_SIMPLE: "Reformatting, extraction, basic Q&A from provided context.",
    TIER_MODERATE: "Summarization, classification, structured analysis.",
    TIER_COMPLEX: "Multi-step reasoning, creative generation, nuanced judgment calls.",
}


def tier_name(tier: int) -> str:
    """Return the human-readable name for ``tier`` or raise ``ValueError``."""
    try:
        return TIER_NAMES[tier]
    except KeyError as exc:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}") from exc

"""Risk-based verification sampling: check the answers that look like they
might be wrong, instead of a flat random slice of everything.

Flat random sampling (the previous approach, still available via
FORCE_RANDOM_SAMPLING) has one real virtue: it gives an unbiased read on the
true error rate, since it doesn't cherry-pick which answers to check. But it
wastes most of its budget checking answers that were never in doubt. This
module scores each response for how likely a check is to find something,
using signals that are free (already-computed classifier confidence) or
near-free (regex over the routed answer's own text) - no extra LLM call
needed to decide whether an LLM call is worth making.

A small random baseline is kept underneath the risk-triggered checks for
two reasons: it catches whatever the risk signals don't anticipate (their
own blind spots), and it keeps a fraction of the escalation-rate number
statistically honest rather than describing "how often our risk-guesser
was right to be suspicious" instead of "how often the system is wrong."
"""
from __future__ import annotations

import re

# Below this confidence, the classifier itself wasn't sure which tier this
# prompt belonged in - a wrong tier guess is more likely near that boundary.
LOW_CONFIDENCE_THRESHOLD = 0.6

# A small unconditional sample, independent of any risk signal, so the
# escalation-rate metric stays an honest estimate of the true error rate
# and so problems the heuristics below don't anticipate still get caught.
BASELINE_RANDOM_RATE = 0.05

_HEDGE_PATTERNS = [
    r"\bi(?:'?m| am) not sure\b", r"\bi don'?t know\b", r"\bcannot determine\b",
    r"\bunable to\b", r"\bas an ai\b", r"\bi apologize\b", r"\bi cannot\b",
    r"\bit depends\b", r"\bunclear\b", r"\bnot certain\b", r"\bi(?:'?m| am) unable\b",
    r"\bcan'?t help with that\b", r"\bi don'?t have (?:enough|access)\b",
]
_HEDGE_RE = re.compile("|".join(_HEDGE_PATTERNS), re.IGNORECASE)


def looks_risky(output_text: str) -> bool:
    """Cheap, no-API-call heuristics on the routed model's own answer: is it
    empty, or does it hedge/refuse instead of actually answering?"""
    text = output_text.strip()
    if not text:
        return True
    return bool(_HEDGE_RE.search(text))


def should_verify(output_text: str, confidence: float, *, rng) -> tuple[bool, str]:
    """Decide whether to verify this response; returns ``(decision, reason)``
    so the reason can be logged rather than thrown away.

    No flat sample-rate knob on top of the signals below - that would just
    be re-adding the "check a fixed fraction regardless of risk" behavior
    this replaces. Total verification volume floats with how often the
    signals actually fire, which is the point: spend the checking budget
    where it's likely to find something.
    """
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return True, "low_classifier_confidence"
    if looks_risky(output_text):
        return True, "risky_answer"
    if rng.random() < BASELINE_RANDOM_RATE:
        return True, "random_baseline"
    return False, "not_sampled"

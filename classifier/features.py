"""Turn a raw prompt string into a fixed row of numeric features.

Phase 2 is about the routing skeleton, not a perfect model, so the features are
cheap, interpretable heuristics rather than embeddings:

* how long the prompt is,
* which kinds of instruction verbs it uses (mechanical vs. moderate vs. reasoning),
* how many hard constraints it imposes ("must", "only", word limits, ...),
* whether it carries a block of context to work on,
* whether it asks for structured output or step-by-step reasoning.

``FEATURE_NAMES`` is the canonical column order. ``extract_features`` returns a
dict keyed by those names; ``feature_vector`` flattens it to a list in order.
"""
from __future__ import annotations

import re

# Instruction verbs grouped by how much reasoning they usually demand. The count
# of each group is a strong signal for the three tiers.
MECHANICAL_VERBS = {
    "extract", "reformat", "format", "convert", "translate", "list", "rename",
    "capitalize", "lowercase", "uppercase", "replace", "count", "sort", "copy",
    "repeat", "label", "tag", "parse", "strip", "join", "split", "round",
}
MODERATE_VERBS = {
    "summarize", "summarise", "classify", "categorize", "categorise",
    "paraphrase", "describe", "explain", "outline", "group", "rank", "compare",
    "identify", "define", "match",
}
REASONING_VERBS = {
    "analyze", "analyse", "evaluate", "assess", "justify", "reason", "prove",
    "derive", "critique", "synthesize", "synthesise", "design", "plan",
    "recommend", "brainstorm", "imagine", "compose", "invent", "speculate",
    "weigh", "debate", "hypothesize", "strategize", "draft", "forecast",
}

_CONSTRAINT_PATTERNS = [
    r"\bmust\b", r"\bonly\b", r"\bdo not\b", r"\bdon't\b", r"\bwithout\b",
    r"\bnever\b", r"\bexactly\b", r"\bat least\b", r"\bat most\b",
    r"\bno more than\b", r"\bfewer than\b",
    r"\bin \d+ (?:words|sentences|lines|bullets|steps)\b",
    r"\b\d+[- ]word\b", r"\b\d+[- ]line\b", r"\b\d+[- ]sentence\b",
]
_STRUCTURED_OUTPUT = [
    r"\bjson\b", r"\bcsv\b", r"\btable\b", r"\bmarkdown\b", r"\bbullet",
    r"\bnumbered list\b", r"\bschema\b", r"\bxml\b", r"\byaml\b",
    r"\bkey[- ]value\b", r"\barray\b",
]
_SHORT_OUTPUT = [
    r"\bone word\b", r"\bone sentence\b", r"\bsingle sentence\b",
    r"\bjust the\b", r"\bonly the\b", r"\bin one line\b", r"\byes or no\b",
    r"\banswer with (?:just|only)\b",
]
_MULTISTEP = [
    r"\bstep[- ]by[- ]step\b", r"\bshow your (?:work|reasoning|working)\b",
    r"\bwalk through\b", r"\bexplain why\b", r"\breason through\b",
    r"\bthink through\b", r"\bfirst\b.*\bthen\b",
]
_CONTEXT_MARKERS = [
    r"\bfollowing\b", r"\bbelow\b", r"\babove\b", r"\bgiven the\b",
    r"\bthis (?:text|passage|paragraph|article|review|email|document|snippet|comment|synopsis|excerpt)\b",
]

FEATURE_NAMES = [
    "n_words",
    "n_chars",
    "n_sentences",
    "n_questions",
    "avg_word_len",
    "n_mechanical_verbs",
    "n_moderate_verbs",
    "n_reasoning_verbs",
    "n_constraints",
    "has_provided_context",
    "context_len_words",
    "wants_structured_output",
    "wants_short_output",
    "has_multistep_request",
]

_WORD_RE = re.compile(r"[A-Za-z']+")
_QUOTED_RE = re.compile(r"[\"'“”‘’](.+?)[\"'“”‘’]", re.DOTALL)


def _count_any(patterns: list[str], text: str) -> int:
    """Total number of matches across every pattern in ``patterns``."""
    return sum(len(re.findall(p, text)) for p in patterns)


def _longest_quoted_span_words(text: str) -> int:
    """Word count of the longest quoted run - a proxy for 'context was pasted in'."""
    spans = _QUOTED_RE.findall(text)
    return max((len(_WORD_RE.findall(s)) for s in spans), default=0)


def extract_features(prompt: str) -> dict[str, float]:
    """Return the feature dict for a single prompt, keyed by ``FEATURE_NAMES``."""
    text = prompt.strip()
    lower = text.lower()
    words = _WORD_RE.findall(text)
    n_words = len(words)

    quoted_words = _longest_quoted_span_words(text)
    has_context = int(
        quoted_words >= 12
        or (_count_any(_CONTEXT_MARKERS, lower) > 0 and n_words >= 25)
    )

    return {
        "n_words": float(n_words),
        "n_chars": float(len(text)),
        "n_sentences": float(max(1, len(re.findall(r"[.!?]+", text)))),
        "n_questions": float(text.count("?")),
        "avg_word_len": (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        "n_mechanical_verbs": float(sum(w.lower() in MECHANICAL_VERBS for w in words)),
        "n_moderate_verbs": float(sum(w.lower() in MODERATE_VERBS for w in words)),
        "n_reasoning_verbs": float(sum(w.lower() in REASONING_VERBS for w in words)),
        "n_constraints": float(_count_any(_CONSTRAINT_PATTERNS, lower)),
        "has_provided_context": float(has_context),
        "context_len_words": float(quoted_words),
        "wants_structured_output": float(_count_any(_STRUCTURED_OUTPUT, lower) > 0),
        "wants_short_output": float(_count_any(_SHORT_OUTPUT, lower) > 0),
        "has_multistep_request": float(_count_any(_MULTISTEP, lower) > 0),
    }


def feature_vector(prompt: str) -> list[float]:
    """Return the feature values as a plain list, in ``FEATURE_NAMES`` order."""
    feats = extract_features(prompt)
    return [feats[name] for name in FEATURE_NAMES]


if __name__ == "__main__":  # quick manual check: python -m classifier.features "prompt"
    import json
    import sys

    sample = sys.argv[1] if len(sys.argv) > 1 else (
        "You have $10k to cut costs. Give a prioritized three-step plan and "
        "justify the ordering. Show your reasoning."
    )
    print(sample, "\n")
    print(json.dumps(extract_features(sample), indent=2))

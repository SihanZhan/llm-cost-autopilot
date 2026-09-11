"""Per-use-case quality scoring for the verification loop.

Three request types get the purpose-built check the project brief calls for
(extraction: field coverage, classification: label match, summarization:
LLM-as-judge); everything else falls back to a lightweight token-overlap
heuristic. Every scorer returns ``(score, extra_cost)`` — a float in [0, 1]
plus whatever it spent doing the check (only the judge call spends anything),
so a single QUALITY_THRESHOLDS table can gate all of them and the verifier
can still account for every dollar.
"""
from __future__ import annotations

import re

from llm_clients import LLMRequestError, send_request
from models import get_model

EXTRACTION_KEYWORDS = ("extract", "parse the", "pull out")
CLASSIFICATION_KEYWORDS = ("classify", "sentiment", "categorize", "categorise", "label this")
SUMMARIZATION_KEYWORDS = ("summarize", "summarise", "tl;dr", "in one sentence", "key takeaway")

# extraction/classification demand an exact match against the top-tier model;
# summarization accepts "4/5 or better" from the judge; general is a looser
# similarity bar since it has no purpose-built check.
QUALITY_THRESHOLDS = {
    "extraction": 1.0,
    "classification": 1.0,
    "summarization": 0.8,
    "general": 0.6,
}

DEFAULT_JUDGE_MODEL_ID = "gpt-4o-mini"

_WORD_RE = re.compile(r"[a-z0-9']+")


def use_case_for_prompt(prompt: str) -> str:
    """Bucket a prompt into extraction / classification / summarization / general."""
    lower = prompt.lower()
    if any(k in lower for k in EXTRACTION_KEYWORDS):
        return "extraction"
    if any(k in lower for k in CLASSIFICATION_KEYWORDS):
        return "classification"
    if any(k in lower for k in SUMMARIZATION_KEYWORDS):
        return "summarization"
    return "general"


def _normalize(text: str) -> str:
    return text.strip().strip(".").lower()


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity between the token sets of ``a`` and ``b``."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def score_extraction(routed_output: str, reference_output: str) -> tuple[float, float]:
    """Did the routed model's extracted value show up in the top-tier
    model's answer too (or vice versa)? Falls back to token overlap for
    partial credit rather than a hard 0."""
    a, b = _normalize(routed_output), _normalize(reference_output)
    score = 1.0 if (a == b or a in b or b in a) else _token_overlap(a, b)
    return score, 0.0


def score_classification(routed_output: str, reference_output: str) -> tuple[float, float]:
    """Exact label match after normalizing case/punctuation."""
    score = 1.0 if _normalize(routed_output) == _normalize(reference_output) else 0.0
    return score, 0.0


def score_summarization(
    routed_output: str,
    reference_output: str,
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID,
) -> tuple[float, float]:
    """LLM-as-judge: ask a cheap model to rate 1-5 how well the routed
    summary captures the same key information as the reference summary.
    Falls back to token overlap (at zero extra cost) if the judge call fails."""
    rubric = (
        "You are grading a summary. Score how well SUMMARY A captures the same "
        "key information as REFERENCE SUMMARY B, on a scale of 1 (misses the "
        "point or contradicts B) to 5 (fully equivalent in substance).\n\n"
        f"SUMMARY A:\n{routed_output}\n\nREFERENCE SUMMARY B:\n{reference_output}\n\n"
        "Respond with only the integer score, nothing else."
    )
    try:
        judge = get_model(judge_model_id)
        resp = send_request(rubric, judge, max_tokens=5)
        match = re.search(r"[1-5]", resp.output_text)
        if match:
            return int(match.group()) / 5.0, resp.cost
    except (LLMRequestError, KeyError):
        pass
    return _token_overlap(routed_output, reference_output), 0.0


def score_general(routed_output: str, reference_output: str) -> tuple[float, float]:
    return _token_overlap(routed_output, reference_output), 0.0


_SCORERS = {
    "extraction": score_extraction,
    "classification": score_classification,
    "summarization": score_summarization,
    "general": score_general,
}


def score_agreement(use_case: str, routed_output: str, reference_output: str) -> tuple[float, float]:
    """Dispatch to the scorer for ``use_case``; returns ``(score, extra_cost)``."""
    scorer = _SCORERS.get(use_case, score_general)
    return scorer(routed_output, reference_output)

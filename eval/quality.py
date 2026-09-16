"""Per-use-case quality scoring for the verification loop.

Three request types get the purpose-built check the project brief calls for:
- extraction: field coverage — detect typed fields (email, date, dollar
  amount, phone, ...) in both outputs and check whether the fields the
  top-tier model found are also present in the routed model's answer.
- classification: exact label match against the top-tier model.
- summarization: LLM-as-judge, scored 1-5.
Everything else falls back to a lightweight token-overlap heuristic. Every
scorer returns ``(score, extra_cost)`` — a float in [0, 1] plus whatever it
spent doing the check (only the judge call spends anything), so a single
QUALITY_THRESHOLDS table can gate all of them and the verifier can still
account for every dollar.
"""
from __future__ import annotations

import re
from datetime import date, datetime

from llm_clients import LLMRequestError, send_request
from models import get_model

EXTRACTION_KEYWORDS = ("extract", "parse the", "pull out")
CLASSIFICATION_KEYWORDS = ("classify", "sentiment", "categorize", "categorise", "label this")
SUMMARIZATION_KEYWORDS = ("summarize", "summarise", "tl;dr", "in one sentence", "key takeaway")

# extraction/classification demand an exact match against the top-tier model;
# summarization and general both accept "4/5 or better" from their LLM-as-
# judge (same judge mechanism, different rubric - see _SUMMARIZATION_RUBRIC
# / _GENERAL_RUBRIC below).
QUALITY_THRESHOLDS = {
    "extraction": 1.0,
    "classification": 1.0,
    "summarization": 0.8,
    "general": 0.8,
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


# --- extraction: typed-field detection -----------------------------------
#
# "Did it get all the key fields?" only means something if you can say what
# a "field" is. Without a separate labeled ground-truth dataset (none was
# built for this project), the top-tier model's output stands in for
# "expected" - but the comparison happens at the level of actual typed
# fields (an email, a date, a dollar amount, ...), not raw string/token
# overlap, so two answers that both found "the date" but formatted it
# differently ("2026-09-22" vs "September 22, 2026") still count as a match.

_DATE_FORMATS = (
    "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y",
    "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y",
)


def _normalize_date(text: str) -> str:
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return _normalize(text)


def _normalize_phone(text: str) -> str:
    return re.sub(r"\D", "", text)


def _normalize_money(text: str) -> str:
    digits = re.sub(r"[^\d.]", "", text)
    try:
        return f"{float(digits):.2f}"
    except ValueError:
        return digits


def _normalize_percent(text: str) -> str:
    digits = re.sub(r"[^\d.]", "", text)
    try:
        return f"{float(digits):g}"
    except ValueError:
        return digits


def _normalize_time(text: str) -> str:
    cleaned = text.strip().upper().replace(" ", "")
    for fmt in ("%I:%M%p", "%H:%M"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%H:%M")
        except ValueError:
            continue
    return _normalize(text)


# (field_type, pattern, normalizer), in priority order. Order matters here:
# extract_fields masks each match out of the text before the next (lower-
# priority) pattern runs, so a phone number's digits can't also get picked
# up piecemeal by the generic alnum_code catch-all, and so the SAME real
# value found via different patterns (an ISO date vs. a month-name date)
# still lands under one shared type label ("date") instead of two labels
# that would never compare equal even after normalizing to the same value.
_FIELD_PATTERNS: tuple[tuple[str, re.Pattern, "callable"], ...] = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), str.lower),
    ("url", re.compile(r"https?://[^\s,'\")]+"), lambda s: s.lower().rstrip("/")),
    ("phone", re.compile(r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), _normalize_phone),
    ("amount", re.compile(r"\$\s?\d[\d,]*\.?\d*"), _normalize_money),
    ("amount", re.compile(r"\b\d+\.\d{2}\b"), _normalize_money),  # plain "482.19", no $
    ("percentage", re.compile(r"\b\d+(?:\.\d+)?\s?%"), _normalize_percent),
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), _normalize_date),
    ("date", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), _normalize_date),
    (
        "date",
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z]*\.?\s+"
            r"\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
            re.IGNORECASE,
        ),
        _normalize_date,
    ),
    ("time", re.compile(r"\b\d{1,2}:\d{2}\s?(?:[APap][Mm])?\b"), _normalize_time),
    ("alnum_code", re.compile(r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{2,}\b"), str.upper),
)


def extract_fields(text: str) -> set[tuple[str, str]]:
    """Return every typed field found in ``text`` as ``(type, normalized_value)``.

    Applies patterns in priority order, masking each match out of the
    working text before the next pattern runs — see the comment above
    _FIELD_PATTERNS for why that matters.
    """
    fields: set[tuple[str, str]] = set()
    remaining = text
    for field_type, pattern, normalizer in _FIELD_PATTERNS:
        for match in pattern.finditer(remaining):
            fields.add((field_type, normalizer(match.group())))
        remaining = pattern.sub(lambda m: " " * len(m.group()), remaining)
    return fields


def score_extraction(routed_output: str, reference_output: str) -> tuple[float, float]:
    """Field coverage: of the typed fields found in the top-tier model's
    (reference) answer, what fraction also appear in the routed model's
    answer? Falls back to whole-string containment/token-overlap only when
    neither answer contains a field this extractor recognizes (e.g. a
    plain-word answer like a name), so free-text extractions still get
    scored rather than defaulting to 0."""
    routed_fields = extract_fields(routed_output)
    reference_fields = extract_fields(reference_output)

    if reference_fields:
        return len(routed_fields & reference_fields) / len(reference_fields), 0.0
    if routed_fields:
        # Reference found nothing typed but routed did - can't confirm
        # coverage of "expected" fields; fall through to text comparison.
        pass

    a, b = _normalize(routed_output), _normalize(reference_output)
    score = 1.0 if (a == b or a in b or b in a) else _token_overlap(a, b)
    return score, 0.0


def score_classification(routed_output: str, reference_output: str) -> tuple[float, float]:
    """Exact label match after normalizing case/punctuation."""
    score = 1.0 if _normalize(routed_output) == _normalize(reference_output) else 0.0
    return score, 0.0


def _llm_judge_score(
    rubric: str, routed_output: str, reference_output: str, judge_model_id: str
) -> tuple[float, float]:
    """Ask ``judge_model_id`` to rate agreement 1-5 per ``rubric`` (which must
    reference {routed} and {reference}); returns (score/5, judge call cost).
    Falls back to token overlap (at zero extra cost) if the judge call fails."""
    try:
        judge = get_model(judge_model_id)
        resp = send_request(
            rubric.format(routed=routed_output, reference=reference_output),
            judge, max_tokens=5,
        )
        match = re.search(r"[1-5]", resp.output_text)
        if match:
            return int(match.group()) / 5.0, resp.cost
    except (LLMRequestError, KeyError):
        pass
    return _token_overlap(routed_output, reference_output), 0.0


_SUMMARIZATION_RUBRIC = (
    "You are grading a summary. Score how well SUMMARY A captures the same "
    "key information as REFERENCE SUMMARY B, on a scale of 1 (misses the "
    "point or contradicts B) to 5 (fully equivalent in substance).\n\n"
    "SUMMARY A:\n{routed}\n\nREFERENCE SUMMARY B:\n{reference}\n\n"
    "Respond with only the integer score, nothing else."
)

# The catch-all bucket (no keyword matched extraction/classification/
# summarization) used to be scored by raw token overlap - counting shared
# words. That's blind to two answers that are both correct but phrased
# differently, which is exactly the common case between two different
# models: measured at 500-request scale (docs/phase6_notes.md), it
# wrongly rejected good answers 56% of the time, the single biggest driver
# of both the cost problem (Phase 4/6) and the classifier-accuracy problem
# (Phase 3/5) this project tracked. An LLM-as-judge - the same mechanism
# summarization already used successfully (0% false-rejection rate) - asks
# whether the two answers actually mean the same thing instead of whether
# they're phrased the same way.
_GENERAL_RUBRIC = (
    "You are comparing two AI answers to the same request. Score how "
    "equivalent ANSWER A is to REFERENCE ANSWER B in substance and "
    "correctness - different wording, formatting, or length is fine as "
    "long as the actual content/meaning matches, on a scale of 1 (wrong or "
    "substantively different) to 5 (equivalent).\n\n"
    "ANSWER A:\n{routed}\n\nREFERENCE ANSWER B:\n{reference}\n\n"
    "Respond with only the integer score, nothing else."
)


def score_summarization(
    routed_output: str,
    reference_output: str,
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID,
) -> tuple[float, float]:
    """LLM-as-judge: rate 1-5 how well the routed summary captures the same
    key information as the reference summary."""
    return _llm_judge_score(_SUMMARIZATION_RUBRIC, routed_output, reference_output, judge_model_id)


def score_general(
    routed_output: str,
    reference_output: str,
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID,
) -> tuple[float, float]:
    """LLM-as-judge: rate 1-5 whether the two answers are substantively
    equivalent, tolerating different wording/formatting."""
    return _llm_judge_score(_GENERAL_RUBRIC, routed_output, reference_output, judge_model_id)


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

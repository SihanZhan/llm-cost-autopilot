"""Quality verification: re-run a routed request against the top-tier model,
score agreement, and auto-escalate on failure.

``verify_and_maybe_escalate`` is the scoring/escalation/logging core; it's
invoked by eval.verification_worker, a genuinely separate process from the
API (see eval.pipeline.route_and_verify, which enqueues a job in db.py
instead of calling this in-process) — matching the brief's "API service +
a background worker for async verification" architecture. Every check —
pass, fail, escalate, or skipped — gets one row in ``logs/verification_log.jsonl``
(full detail, incl. prompt text, for the Phase-3 feedback loop) *and* one row
in the ``requests`` table in ``db.py`` (prompt hashed, not full text — the
Phase 4 dashboard's data source).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import db
from classifier.routing import model_for_tier
from classifier.tiers import TIER_COMPLEX
from eval.quality import QUALITY_THRESHOLDS, score_agreement, use_case_for_prompt
from llm_clients import LLMRequestError, Response, send_request

LOG_FILE = Path(__file__).parent / "logs" / "verification_log.jsonl"


@dataclass
class VerificationResult:
    timestamp: str
    prompt: str
    use_case: str
    tier: int
    routed_model_id: str
    routed_cost: float
    skipped: bool                    # True when the routed model already is the top tier
    verifier_model_id: str | None
    verifier_cost: float
    quality_score: float | None
    threshold: float
    passed: bool
    escalated: bool
    final_model_id: str
    final_output: str
    cost_delta: float                # extra $ spent because of escalation (0 if not escalated)
    verification_cost: float         # $ spent just to check quality (verifier + judge calls)
    quality_gap: float               # max(0, threshold - score); 0 when passed
    error: str | None = None


def _log(
    request_id: int,
    result: VerificationResult,
    routed_response: Response,
    verifier_response: Response | None = None,
) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(result)) + "\n")

    # baseline_cost should be "what BASELINE_MODEL_ID actually costs for this
    # prompt." When verification ran, we already have a real call to the
    # top-tier model (verifier_response) - use ITS token counts, not an
    # estimate from the routed (possibly very different) model's tokenizer.
    # Only fall back to the routed response's tokens when there's no other
    # data: the skipped case (routed model already IS the top tier, so its
    # own tokens are exact) and the error case (verification failed, no
    # verifier tokens exist).
    baseline_tokens_source = verifier_response if verifier_response is not None else routed_response

    db.update_verification_result(
        request_id,
        {
            "baseline_cost": db.baseline_cost_for(
                baseline_tokens_source.input_tokens, baseline_tokens_source.output_tokens
            ),
            "verified": not result.skipped and result.error is None,
            "quality_score": result.quality_score,
            "passed": result.passed,
            "escalated": result.escalated,
            "final_model": result.final_model_id,
            "cost_delta": result.cost_delta,
            "verification_cost": result.verification_cost,
        },
    )


def verify_and_maybe_escalate(
    request_id: int,
    prompt: str,
    tier: int,
    routed_response: Response,
    *,
    escalate_on_failure: bool = True,
) -> VerificationResult:
    """Score ``routed_response`` against the top-tier model and update the
    request's row (``request_id``, from ``db.log_routed_request``) with the
    outcome.

    Runs synchronously in whatever thread/process calls it —
    eval.verification_worker is what actually makes this async, by running
    it in a separate process from the API.
    """
    use_case = use_case_for_prompt(prompt)
    threshold = QUALITY_THRESHOLDS[use_case]
    timestamp = datetime.now(timezone.utc).isoformat()
    top_tier_model = model_for_tier(TIER_COMPLEX)

    if routed_response.model_id == top_tier_model.model_id:
        # Nothing to verify against - the routed model already is the top tier.
        # (eval.pipeline.route_and_verify already filters these out before
        # enqueueing; this stays as a defensive fallback for direct callers.)
        result = VerificationResult(
            timestamp=timestamp, prompt=prompt, use_case=use_case, tier=tier,
            routed_model_id=routed_response.model_id, routed_cost=routed_response.cost,
            skipped=True, verifier_model_id=None, verifier_cost=0.0,
            quality_score=None, threshold=threshold, passed=True, escalated=False,
            final_model_id=routed_response.model_id, final_output=routed_response.output_text,
            cost_delta=0.0, verification_cost=0.0, quality_gap=0.0,
        )
        _log(request_id, result, routed_response)
        return result

    try:
        verifier_response = send_request(prompt, top_tier_model)
    except LLMRequestError as exc:
        # Can't verify right now (provider hiccup) - don't punish the routed
        # response for it; log the gap so it's visible, not silent.
        result = VerificationResult(
            timestamp=timestamp, prompt=prompt, use_case=use_case, tier=tier,
            routed_model_id=routed_response.model_id, routed_cost=routed_response.cost,
            skipped=False, verifier_model_id=top_tier_model.model_id, verifier_cost=0.0,
            quality_score=None, threshold=threshold, passed=True, escalated=False,
            final_model_id=routed_response.model_id, final_output=routed_response.output_text,
            cost_delta=0.0, verification_cost=0.0, quality_gap=0.0, error=str(exc),
        )
        _log(request_id, result, routed_response)
        return result

    score, judge_cost = score_agreement(
        use_case, routed_response.output_text, verifier_response.output_text
    )
    passed = score >= threshold
    escalate = (not passed) and escalate_on_failure
    verification_cost = verifier_response.cost + judge_cost

    if escalate:
        final_model_id = verifier_response.model_id
        final_output = verifier_response.output_text
        cost_delta = verifier_response.cost - routed_response.cost
    else:
        final_model_id = routed_response.model_id
        final_output = routed_response.output_text
        cost_delta = 0.0

    result = VerificationResult(
        timestamp=timestamp, prompt=prompt, use_case=use_case, tier=tier,
        routed_model_id=routed_response.model_id, routed_cost=routed_response.cost,
        skipped=False, verifier_model_id=top_tier_model.model_id,
        verifier_cost=verifier_response.cost, quality_score=score, threshold=threshold,
        passed=passed, escalated=escalate, final_model_id=final_model_id,
        final_output=final_output, cost_delta=cost_delta,
        verification_cost=verification_cost,
        quality_gap=0.0 if passed else round(threshold - score, 4),
    )
    _log(request_id, result, routed_response, verifier_response)
    return result

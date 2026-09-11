"""Async quality verification: re-run a routed request against the top-tier
model, score agreement, and auto-escalate on failure.

The "async" part is real, not simulated: ``submit_verification`` runs
``verify_and_maybe_escalate`` on a background thread via a module-level
``ThreadPoolExecutor``, so the caller gets its primary (cheap-model) response
immediately and verification happens after, without blocking. Every check —
pass, fail, escalate, or skipped — gets one row in ``logs/verification_log.jsonl``,
which is both the audit trail for the Phase 4 dashboard and the raw material
for the Phase-3-mandated feedback loop (see ``classifier.feedback``).
"""
from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from classifier.routing import model_for_tier
from classifier.tiers import TIER_COMPLEX
from eval.quality import QUALITY_THRESHOLDS, score_agreement, use_case_for_prompt
from llm_clients import LLMRequestError, Response, send_request

LOG_FILE = Path(__file__).parent / "logs" / "verification_log.jsonl"
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="verifier")


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


def _log(result: VerificationResult) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(result)) + "\n")


def verify_and_maybe_escalate(
    prompt: str,
    tier: int,
    routed_response: Response,
    *,
    escalate_on_failure: bool = True,
) -> VerificationResult:
    """Score ``routed_response`` against the top-tier model and log the outcome.

    Runs synchronously in whatever thread calls it — ``submit_verification``
    is what actually makes this async by handing it to a background thread.
    """
    use_case = use_case_for_prompt(prompt)
    threshold = QUALITY_THRESHOLDS[use_case]
    timestamp = datetime.now(timezone.utc).isoformat()
    top_tier_model = model_for_tier(TIER_COMPLEX)

    if routed_response.model_id == top_tier_model.model_id:
        # Nothing to verify against - the routed model already is the top tier.
        result = VerificationResult(
            timestamp=timestamp, prompt=prompt, use_case=use_case, tier=tier,
            routed_model_id=routed_response.model_id, routed_cost=routed_response.cost,
            skipped=True, verifier_model_id=None, verifier_cost=0.0,
            quality_score=None, threshold=threshold, passed=True, escalated=False,
            final_model_id=routed_response.model_id, final_output=routed_response.output_text,
            cost_delta=0.0, verification_cost=0.0, quality_gap=0.0,
        )
        _log(result)
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
        _log(result)
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
    _log(result)
    return result


def submit_verification(prompt: str, tier: int, routed_response: Response) -> "Future[VerificationResult]":
    """Fire off verification on a background thread; returns immediately."""
    return _EXECUTOR.submit(verify_and_maybe_escalate, prompt, tier, routed_response)

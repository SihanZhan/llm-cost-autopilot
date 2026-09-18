"""Ties the classifier, router, and verifier together into one call.

``route_and_verify`` is the "handle one request" entry point the API calls:
classify -> route -> call the cheap model -> log it -> return immediately,
while quality verification (for a subset of requests, chosen by risk, not
a coin flip) is handed off to a genuinely separate process (see
eval.verification_worker) by enqueueing a job in db.py.

Verifying every single request is what the system originally did, and what
made checking cost more than routing saved. Verifying a flat random
fraction of them (VERIFICATION_SAMPLE_RATE, the first fix) cut that cost
without being smart about which ones to spend it on. This version
(eval.risk.should_verify) checks the ones actually likely to be wrong -
low classifier confidence, or an answer that looks empty/hedging - plus a
small random baseline to catch blind spots and keep the escalation-rate
number an honest estimate. Set FORCE_VERIFY_ALL=1 to check every non-top-
tier request regardless of risk signals (eval.demo uses this so its
10-prompt showcase still verifies everything). See docs/cost_fix_results.md
and docs/risk_based_verification.md.
"""
from __future__ import annotations

import os
import random

import db
from classifier.predict import predict_tier_with_confidence
from classifier.routing import model_for_tier
from classifier.tiers import TIER_COMPLEX
from eval.risk import should_verify
from llm_clients import Response, send_request


def route_and_verify(
    prompt: str, *, max_tokens: int = 512, callback_url: str | None = None
) -> tuple[int, Response, int, int | None]:
    """Classify + route + call the cheap model, log it, and - for requests a
    risk check flags as worth checking - enqueue async verification.

    Returns ``(tier, routed_response, request_id, verification_job_id)``.
    The caller can hand ``routed_response`` back to the user right away, and
    give them ``request_id`` to poll ``GET /v1/completions/{request_id}``
    later for the verified/possibly-escalated final answer - or pass
    ``callback_url`` to have eval.verification_worker POST the outcome
    there instead of making the caller ask again.
    ``verification_job_id`` is ``None`` when this request wasn't flagged for
    verification (or was already routed to the top tier, where there's
    nothing to check against) - ``request_id``'s row is logged either way,
    just without a verification outcome filled in (yet, if ever).
    """
    tier, confidence = predict_tier_with_confidence(prompt)
    model = model_for_tier(tier)
    response = send_request(prompt, model, max_tokens=max_tokens)

    request_id = db.log_routed_request(
        {
            "timestamp": _now_iso(),
            "prompt_hash": db.prompt_hash(prompt),
            "use_case": _use_case_hint(prompt),
            "tier": tier,
            "routed_model": response.model_id,
            "routed_cost": response.cost,
            "routed_latency": response.latency,
            "baseline_cost": db.baseline_cost_for(response.input_tokens, response.output_tokens),
            "routed_output": response.output_text,
        }
    )

    top_tier_model = model_for_tier(TIER_COMPLEX)
    already_top_tier = response.model_id == top_tier_model.model_id

    reason = "already_top_tier"
    sampled = False
    if not already_top_tier:
        if os.getenv("FORCE_VERIFY_ALL", "").lower() in ("1", "true", "yes"):
            sampled, reason = True, "force_verify_all"
        else:
            sampled, reason = should_verify(response.output_text, confidence, rng=random)

    job_id = None
    if sampled:
        job_id = db.enqueue_verification_job(
            prompt, tier, response, request_id, reason, callback_url
        )

    return tier, response, request_id, job_id


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _use_case_hint(prompt: str) -> str:
    from eval.quality import use_case_for_prompt

    return use_case_for_prompt(prompt)

"""Ties the classifier, router, and verifier together into one call.

``route_and_verify`` is the "handle one request" entry point the API calls:
classify -> route -> call the cheap model -> log it -> return immediately,
while quality verification (for a sampled subset of requests) is handed off
to a genuinely separate process (see eval.verification_worker) by
enqueueing a job in db.py.

VERIFICATION_SAMPLE_RATE controls what fraction of non-top-tier requests
get checked at all. Verifying every single one (rate 1.0) is what the
system originally did - and what made checking cost more than routing
saved: it means calling the top-tier model on ~two-thirds of all traffic
just to grade the cheap model's homework, whether or not it needed
grading. Sampling keeps enough checks to catch systematic problems while
spending a fraction of the cost. See docs/cost_fix_results.md.
"""
from __future__ import annotations

import os
import random

import db
from classifier.predict import predict_tier
from classifier.routing import model_for_tier
from classifier.tiers import TIER_COMPLEX
from llm_clients import Response, send_request

DEFAULT_VERIFICATION_SAMPLE_RATE = 0.2


def _sample_rate() -> float:
    """Read the sample rate fresh on every call (not just at import time) so
    scripts can override it via ``os.environ`` after importing this module."""
    return float(os.getenv("VERIFICATION_SAMPLE_RATE", str(DEFAULT_VERIFICATION_SAMPLE_RATE)))


def route_and_verify(
    prompt: str, *, max_tokens: int = 512
) -> tuple[int, Response, int, int | None]:
    """Classify + route + call the cheap model, log it, and - for a sampled
    subset of non-top-tier requests - enqueue async verification.

    Returns ``(tier, routed_response, request_id, verification_job_id)``.
    The caller can hand ``routed_response`` back to the user right away, and
    give them ``request_id`` to poll ``GET /v1/completions/{request_id}``
    later for the verified/possibly-escalated final answer.
    ``verification_job_id`` is ``None`` when this request wasn't sampled for
    verification (or was already routed to the top tier, where there's
    nothing to check against) - ``request_id``'s row is logged either way,
    just without a verification outcome filled in (yet, if ever).
    """
    tier = predict_tier(prompt)
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
    sampled = (not already_top_tier) and random.random() < _sample_rate()

    job_id = None
    if sampled:
        job_id = db.enqueue_verification_job(prompt, tier, response, request_id)

    return tier, response, request_id, job_id


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _use_case_hint(prompt: str) -> str:
    from eval.quality import use_case_for_prompt

    return use_case_for_prompt(prompt)

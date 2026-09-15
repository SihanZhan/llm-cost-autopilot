"""Ties the classifier, router, and verifier together into one call.

``route_and_verify`` is the "handle one request" entry point the API calls:
classify -> route -> call the cheap model -> return immediately, while
quality verification is handed off to a genuinely separate process (see
eval.verification_worker) by enqueueing a job in db.py — matching the
brief's "API service + a background worker for async verification"
architecture, rather than running verification in-process.
"""
from __future__ import annotations

import db
from classifier.predict import predict_tier
from classifier.routing import model_for_tier
from llm_clients import Response, send_request


def route_and_verify(prompt: str, *, max_tokens: int = 512) -> tuple[int, Response, int]:
    """Classify + route + call the cheap model, then enqueue async verification.

    Returns ``(tier, routed_response, verification_job_id)``. The caller can
    hand ``routed_response`` back to the user right away; a separately
    running ``eval.verification_worker`` process picks up the queued job and
    logs the verified / possibly-escalated outcome independently.
    """
    tier = predict_tier(prompt)
    model = model_for_tier(tier)
    response = send_request(prompt, model, max_tokens=max_tokens)
    job_id = db.enqueue_verification_job(prompt, tier, response)
    return tier, response, job_id

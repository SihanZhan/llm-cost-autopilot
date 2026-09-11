"""Ties the classifier, router, and verifier together into one call.

``route_and_verify`` is the "handle one request" entry point Phase 5's API
will eventually call: classify -> route -> call the cheap model -> return
immediately, while quality verification (and possible escalation) runs on a
background thread and gets logged for the dashboard and the feedback loop.
"""
from __future__ import annotations

from concurrent.futures import Future

from classifier.predict import predict_tier
from classifier.routing import model_for_tier
from eval.verifier import VerificationResult, submit_verification
from llm_clients import Response, send_request


def route_and_verify(
    prompt: str, *, max_tokens: int = 512
) -> tuple[int, Response, "Future[VerificationResult]"]:
    """Classify + route + call the cheap model, then kick off async verification.

    Returns ``(tier, routed_response, verification_future)``. The caller can
    hand ``routed_response`` back to the user right away; call
    ``verification_future.result()`` only when it actually needs the
    verified / possibly-escalated outcome — verification logs regardless.
    """
    tier = predict_tier(prompt)
    model = model_for_tier(tier)
    response = send_request(prompt, model, max_tokens=max_tokens)
    future = submit_verification(prompt, tier, response)
    return tier, response, future

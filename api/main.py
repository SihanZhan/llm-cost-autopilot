"""Phase 5: FastAPI service exposing the router built in Phases 1-4.

Endpoints:
    POST /v1/completions      classify -> route -> call the cheap model,
                               return immediately; verification (Phase 3)
                               keeps running in the background and shows up
                               later in /v1/stats and the dashboard.
    GET  /v1/models           the model registry: providers, per-token
                               pricing, measured latency, quality tier.
    GET  /v1/stats            the same cost-savings summary the dashboard
                               shows (stats.compute_stats), as JSON.
    PUT  /v1/routing-config   change tier -> model mappings live; takes
                               effect on the very next request, no redeploy.
    GET  /health               liveness check.

Run:
    uvicorn api.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import stats
from classifier.routing import load_routing_config, update_routing_config
from classifier.tiers import tier_name
from eval.pipeline import route_and_verify
from llm_clients import LLMRequestError
from models import MODEL_REGISTRY

app = FastAPI(
    title="LLM Cost Autopilot",
    description="Routes each request to the cheapest model that can handle it, "
    "verifies quality asynchronously, and tracks the savings.",
    version="0.5.0",
)


# --- schemas -------------------------------------------------------------


class CompletionRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=4096)


class CompletionResponse(BaseModel):
    output_text: str
    model_id: str
    provider: str
    tier: int
    tier_name: str
    why: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency: float
    verification_job_id: int | None
    note: str = (
        "If verification_job_id is set, quality verification is queued for "
        "the verification-worker process and is not reflected in this "
        "response — check /v1/stats or the dashboard once it's processed. "
        "Only a sample of requests are verified (VERIFICATION_SAMPLE_RATE); "
        "a null verification_job_id means this one wasn't sampled, or was "
        "already routed to the top-tier model."
    )


class ModelInfo(BaseModel):
    provider: str
    model_id: str
    cost_per_input_token: float
    cost_per_output_token: float
    avg_latency: float
    quality_tier: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class StatsResponse(BaseModel):
    n_requests: int
    n_escalated: int
    escalation_rate: float
    total_routed_cost: float
    total_baseline_cost: float
    escalation_cost_delta: float
    verification_cost: float
    served_answer_cost: float
    true_total_cost: float
    pct_saved_routing_only: float
    pct_saved_served_answer: float
    pct_saved_true: float
    routing_distribution: dict[str, int]
    tier_distribution: dict[int, int]


class RoutingConfigUpdate(BaseModel):
    routing: dict[int, str]


class RoutingConfigResponse(BaseModel):
    routing: dict[int, str]


# --- endpoints -------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(request: CompletionRequest) -> CompletionResponse:
    """Classify the prompt, route it to a model, call it, and kick off
    async verification. Returns as soon as the routed model responds."""
    try:
        tier, response, verification_job_id = route_and_verify(
            request.prompt, max_tokens=request.max_tokens
        )
    except LLMRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        # bad/missing routing.yaml entry, unknown model_id, etc.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    routing_config = load_routing_config()
    provider = next(
        (m.provider for m in MODEL_REGISTRY if m.model_id == response.model_id), "unknown"
    )
    why = (
        f"Classified as tier {tier} ({tier_name(tier)}); routing.yaml maps tier {tier} "
        f"to '{routing_config[tier]}'."
    )

    return CompletionResponse(
        output_text=response.output_text,
        model_id=response.model_id,
        provider=provider,
        tier=tier,
        tier_name=tier_name(tier),
        why=why,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost=response.cost,
        latency=response.latency,
        verification_job_id=verification_job_id,
    )


@app.get("/v1/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(models=[ModelInfo(**vars(m)) for m in MODEL_REGISTRY])


@app.get("/v1/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    return StatsResponse(**stats.compute_stats())


@app.put("/v1/routing-config", response_model=RoutingConfigResponse)
def put_routing_config(update: RoutingConfigUpdate) -> RoutingConfigResponse:
    try:
        new_config = update_routing_config(update.routing)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RoutingConfigResponse(routing=new_config)


@app.get("/v1/routing-config", response_model=RoutingConfigResponse)
def get_routing_config() -> RoutingConfigResponse:
    """Not in the roadmap's endpoint list, but PUT-without-GET is an odd API
    to ship - this just reads back what's currently in routing.yaml."""
    return RoutingConfigResponse(routing=load_routing_config())

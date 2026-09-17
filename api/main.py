"""Phase 5: FastAPI service exposing the router built in Phases 1-4.

Endpoints:
    POST /v1/completions           classify -> route -> call the cheap
                                    model, return immediately; verification
                                    (Phase 3) keeps running separately and
                                    shows up later in /v1/stats, the
                                    dashboard, and GET /v1/completions/{id}.
    GET  /v1/completions/{id}      poll for the verified / possibly-
                                    escalated final answer to a request
                                    already made via POST /v1/completions.
    GET  /v1/models                the model registry: providers, per-token
                                    pricing, measured latency, quality tier.
    GET  /v1/stats                 the same cost-savings summary the
                                    dashboard shows (stats.compute_stats).
    PUT  /v1/routing-config        change tier -> model mappings live;
                                    takes effect next request, no redeploy.
    GET  /health                   liveness check.

Run:
    uvicorn api.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import db
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
    request_id: int
    verification_job_id: int | None
    note: str = (
        "This response is always the routed (cheap-model) answer, even if "
        "verification later escalates it - GET /v1/completions/{request_id} "
        "returns the corrected answer once verification (if sampled) "
        "completes. Only a sample of requests are verified "
        "(VERIFICATION_SAMPLE_RATE); a null verification_job_id means this "
        "one wasn't sampled, or was already routed to the top-tier model."
    )


class CompletionResultResponse(BaseModel):
    request_id: int
    status: str  # "not_checked" | "pending" | "check_failed" | "checked"
    output_text: str
    model_id: str
    escalated: bool
    quality_score: float | None
    passed: bool | None


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
        tier, response, request_id, verification_job_id = route_and_verify(
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
        request_id=request_id,
        verification_job_id=verification_job_id,
    )


@app.get("/v1/completions/{request_id}", response_model=CompletionResultResponse)
def get_completion_result(request_id: int) -> CompletionResultResponse:
    """Poll for the verified / possibly-escalated final answer to a request
    made via POST /v1/completions. This is what actually closes the loop the
    project brief's "auto-escalation... return the better result" implies -
    POST /v1/completions itself always returns the routed answer immediately
    and never waits on verification, so without this endpoint an escalation
    only ever updated internal stats and the caller had no way to get the
    corrected answer at all."""
    row = db.get_request(request_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no request with id {request_id}")

    if row["verified"]:
        status = "checked"
    else:
        job = db.get_verification_job_for_request(request_id)
        if job is None:
            status = "not_checked"
        elif job["status"] in ("pending", "claimed"):
            status = "pending"
        elif job["status"] == "error":
            status = "check_failed"
        else:
            status = "not_checked"

    return CompletionResultResponse(
        request_id=request_id,
        status=status,
        output_text=row["final_output"],
        model_id=row["final_model"],
        escalated=bool(row["escalated"]),
        quality_score=row["quality_score"],
        passed=None if row["passed"] is None else bool(row["passed"]),
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

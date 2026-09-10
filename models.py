"""Model registry: static metadata for every model the router can pick from.

Costs are USD **per token** — take the provider's public "$ per 1M tokens"
figure and divide by 1_000_000. Verified against OpenAI and Anthropic
first-party pricing, September 2026. `avg_latency` is a rough seconds-per-call
hint used only for routing; the Phase 1 baseline replaces these with measured
numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelConfig:
    provider: str            # "openai" | "anthropic" | "ollama"
    model_id: str            # provider-native model identifier
    cost_per_input_token: float
    cost_per_output_token: float
    avg_latency: float       # seconds per request (rough hint)
    quality_tier: str        # "high" | "medium" | "low"


MODEL_REGISTRY = [
    ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        cost_per_input_token=2.50 / _PER_MILLION,
        cost_per_output_token=10.00 / _PER_MILLION,
        avg_latency=1.5,
        quality_tier="high",
    ),
    ModelConfig(
        provider="openai",
        model_id="gpt-4o-mini",
        cost_per_input_token=0.15 / _PER_MILLION,
        cost_per_output_token=0.60 / _PER_MILLION,
        avg_latency=0.5,
        quality_tier="medium",
    ),
    ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-5",
        cost_per_input_token=2.00 / _PER_MILLION,
        cost_per_output_token=10.00 / _PER_MILLION,
        avg_latency=1.2,
        quality_tier="high",
    ),
    ModelConfig(
        provider="anthropic",
        model_id="claude-haiku-4-5",
        cost_per_input_token=1.00 / _PER_MILLION,
        cost_per_output_token=5.00 / _PER_MILLION,
        avg_latency=0.6,
        quality_tier="medium",
    ),
    ModelConfig(
        provider="ollama",
        model_id="llama3",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        avg_latency=0.8,
        quality_tier="low",
    ),
]


def get_model(model_id: str) -> ModelConfig:
    """Return the registry entry for ``model_id`` or raise ``KeyError``."""
    for model in MODEL_REGISTRY:
        if model.model_id == model_id:
            return model
    raise KeyError(f"no model registered with id {model_id!r}")


def models_by_tier(tier: str) -> list[ModelConfig]:
    """Return every registered model whose ``quality_tier`` matches ``tier``."""
    return [m for m in MODEL_REGISTRY if m.quality_tier == tier]

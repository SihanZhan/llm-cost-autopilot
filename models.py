from dataclasses import dataclass

@dataclass
class ModelConfig:
    provider: str
    model_id: str
    cost_per_input_token: float
    cost_per_output_token: float
    avg_latency: float
    quality_tier: str
MODEL_REGISTRY = [
    ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        cost_per_input_token=0.0000025,
        cost_per_output_token=0.00001,
        avg_latency=1.5,
        quality_tier="high"
    ),

    ModelConfig(
        provider = "openai",
        model_id = "gpt-4o-mini",
        cost_per_input_token = 0.00000015,
        cost_per_output_token = 0.0000006,
        avg_latency = 0.5,
        quality_tier = "medium"
    ),

    ModelConfig(
        provider = "anthropic",
        model_id = "claude-sonnet-5",
        cost_per_input_token = 0.000002,
        cost_per_output_token = 0.00001,
        avg_latency = 0.5,
        quality_tier = "medium"
    ),


    ModelConfig(
        provider = "anthropic",
        model_id = "claude-haiku-4-5-20251001",
        cost_per_input_token = 0.000001,
        cost_per_output_token = 0.000005,
        avg_latency = 0.5,
        quality_tier = "low"
    ),


    ModelConfig(
        provider = "ollama",
        model_id = "llama3",
        cost_per_input_token = 0.00,
        cost_per_output_token = 0.00,
        avg_latency = 0.5,
        quality_tier = "low"
    )
]
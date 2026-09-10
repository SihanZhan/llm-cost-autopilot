# LLM Cost Autopilot

Route LLM requests to the cheapest model that meets a quality bar, and track cost/latency per call.

## Overview

- **`models.py`** – `ModelConfig` dataclass and `MODEL_REGISTRY`, a list of supported
  models (OpenAI, Anthropic, Ollama) with per-token cost, average latency, and a
  quality tier.
- **`llm_clients.py`** – `Response` dataclass and `send_request(prompt, model_config)`,
  which dispatches a prompt to the right provider SDK and returns output text, token
  counts, latency, and computed cost.
```


## Status

Early work in progress. `send_request` still needs latency/cost wiring and error
handling, and Ollama support is stubbed.

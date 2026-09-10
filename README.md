# LLM Cost Autopilot

Route LLM requests to the cheapest model that meets a quality bar, and track cost/latency per call.

## Overview

- **`models.py`** – `ModelConfig` dataclass and `MODEL_REGISTRY`, a list of supported
  models (OpenAI, Anthropic, Ollama) with per-token cost, average latency, and a
  quality tier.
- **`llm_clients.py`** – `Response` dataclass and `send_request(prompt, model_config)`,
  which dispatches a prompt to the right provider SDK and returns output text, token
  counts, latency, and computed cost.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Set provider API keys as environment variables (do not commit them):

```bash
setx OPENAI_API_KEY "..."
setx ANTHROPIC_API_KEY "..."
```

## Status

Early work in progress. `send_request` still needs latency/cost wiring and error
handling, and Ollama support is stubbed.

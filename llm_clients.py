"""Unified interface to every provider in the model registry.

``send_request(prompt, model_config)`` dispatches to the correct provider SDK
and returns a standardized :class:`Response` (text, token counts, measured
latency, and computed dollar cost). Credentials come from the environment:

    OPENAI_API_KEY       required for provider="openai"
    ANTHROPIC_API_KEY    required for provider="anthropic"
    OLLAMA_HOST          optional, defaults to http://localhost:11434

Provider SDKs are imported lazily, so you only need the packages for the
providers you actually call.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from models import ModelConfig

try:  # load a local .env when python-dotenv is available; harmless if not
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

DEFAULT_MAX_TOKENS = 1024
DEFAULT_RETRIES = 2


class LLMRequestError(RuntimeError):
    """Raised when a provider call fails, or fails to succeed after retries."""


@dataclass
class Response:
    output_text: str
    input_tokens: int
    output_tokens: int
    latency: float          # wall-clock seconds for the provider call
    cost: float             # USD, from the registry's per-token rates
    model_id: str


def send_request(
    prompt: str,
    model_config: ModelConfig,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = DEFAULT_RETRIES,
) -> Response:
    """Send ``prompt`` to the model described by ``model_config``.

    Times the call, computes cost from the registry's per-token rates, and
    returns a :class:`Response`. Raises :class:`LLMRequestError` on an unknown
    provider, missing credentials, or a provider error that survives retries.
    """
    providers = {
        "openai": _call_openai,
        "anthropic": _call_anthropic,
        "ollama": _call_ollama,
    }
    try:
        provider_call = providers[model_config.provider]
    except KeyError:
        raise LLMRequestError(f"unknown provider: {model_config.provider!r}") from None

    start = time.perf_counter()
    text, input_tokens, output_tokens = _with_retry(
        lambda: provider_call(prompt, model_config, max_tokens), retries
    )
    latency = time.perf_counter() - start

    cost = (
        input_tokens * model_config.cost_per_input_token
        + output_tokens * model_config.cost_per_output_token
    )
    return Response(
        output_text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency=latency,
        cost=cost,
        model_id=model_config.model_id,
    )


def _with_retry(call, retries: int):
    """Call ``call`` up to ``retries`` extra times with exponential backoff.

    Provider SDKs raise disparate exception types, so this retries on any
    exception and re-wraps the last one as :class:`LLMRequestError`.
    """
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            return call()
        except LLMRequestError:
            raise  # non-transient (bad provider, missing credentials) - fail fast
        except Exception as exc:  # noqa: BLE001 - deliberately provider-agnostic
            if attempt == retries:
                raise LLMRequestError(str(exc)) from exc
            time.sleep(delay)
            delay *= 2


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise LLMRequestError(f"{name} is not set in the environment")
    return value


def _get(obj, key):
    """Read ``key`` from a dict or an attribute of an object (SDK-version safe)."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _call_openai(prompt: str, model_config: ModelConfig, max_tokens: int):
    from openai import OpenAI

    client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model_config.model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    return text, resp.usage.prompt_tokens, resp.usage.completion_tokens


def _call_anthropic(prompt: str, model_config: ModelConfig, max_tokens: int):
    from anthropic import Anthropic

    client = Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model_config.model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _call_ollama(prompt: str, model_config: ModelConfig, max_tokens: int):
    import ollama

    client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    resp = client.chat(
        model=model_config.model_id,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens},
    )
    message = _get(resp, "message")
    text = _get(message, "content") or ""
    # Ollama returns token counts when it can; fall back to a rough estimate.
    input_tokens = _get(resp, "prompt_eval_count") or _estimate_tokens(prompt)
    output_tokens = _get(resp, "eval_count") or _estimate_tokens(text)
    return text, input_tokens, output_tokens


def _estimate_tokens(text: str) -> int:
    """Very rough token estimate (~4 chars/token) for providers that omit counts."""
    return max(1, len(text) // 4)

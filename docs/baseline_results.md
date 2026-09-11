# Phase 1 baseline results

Run: 2026-09-10, `python baseline.py` (10 prompts x registry). Full per-row
data (prompt id, tier, tokens, latency, cost, output preview) is in
`baseline_results.csv`, generated locally and gitignored — regenerate with
the command above.

## Coverage

3 of 5 registry models ran live. `gpt-4o` and `gpt-4o-mini` were skipped —
the OpenAI account had no API credits at run time (`429 insufficient_quota`).
Ollama (`llama3`) and both Anthropic models ran cleanly.

## Summary

| model | calls | avg latency | total cost |
|---|---:|---:|---:|
| claude-sonnet-5 | 10 | 2.71s | $0.01537 |
| claude-haiku-4-5 | 10 | 1.72s | $0.00592 |
| llama3 (local) | 10 | 13.68s | $0.00000 |
| gpt-4o | — | not measured | — |
| gpt-4o-mini | — | not measured | — |
| **total** | **30** | | **$0.02129** |

## Notes

- Haiku is ~2.6x cheaper than Sonnet on this set and noticeably faster — the
  gap should widen further once GPT-4o-mini is in the mix as a Tier 2 option.
- `llama3` latency is highly variable: sub-4s on short Tier 1 prompts, up to
  60s on the Tier 3 planning/justification prompt (first call also pays a
  one-time model-load cost). Fine as a free Tier 1 option; too slow as-is for
  anything latency-sensitive at Tier 2+.
- `models.py` `avg_latency` values for the three measured models now reflect
  these numbers. `gpt-4o` / `gpt-4o-mini` still carry the original estimates
  pending a funded OpenAI account — re-run the baseline and update those two
  once credits are added.

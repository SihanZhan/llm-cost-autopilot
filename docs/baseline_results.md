# Phase 1 baseline results

Run: 2026-09-10, `python baseline.py` (10 prompts x all 5 registry models).
Full per-row data (prompt id, tier, tokens, latency, cost, output preview) is
in `baseline_results.csv`, generated locally and gitignored — regenerate with
the command above.

## Summary

| model | calls | avg latency | total cost |
|---|---:|---:|---:|
| gpt-4o | 10 | 16.19s | $0.01197 |
| gpt-4o-mini | 10 | 1.60s | $0.00077 |
| claude-sonnet-5 | 10 | 2.67s | $0.01601 |
| claude-haiku-4-5 | 10 | 1.72s | $0.00580 |
| llama3 (local) | 10 | 15.84s | $0.00000 |
| **total** | **50** | | **$0.03456** |

## Notes

- **Cost:** the "mini" tier is dramatically cheaper than its frontier
  sibling — `gpt-4o-mini` is ~15x cheaper than `gpt-4o` on this set, and
  `claude-haiku-4-5` ~2.8x cheaper than `claude-sonnet-5`. `gpt-4o-mini` is
  the single cheapest paid model measured.
- **Latency:** `gpt-4o` was surprisingly slow this run (16.19s avg, several
  calls 20-25s) — well above `claude-sonnet-5` (2.67s) despite similar
  pricing. Worth re-checking on a future run before leaning on it for
  latency-sensitive routing; could be transient API load rather than
  representative.
- `llama3` latency is highly variable: sub-4s on short Tier 1 prompts, up to
  ~60s on the hardest Tier 3 prompt. Free, but too slow as-is for anything
  beyond Tier 1 given the alternatives above.
- `models.py` `avg_latency` for all five models now reflects these measured
  numbers.

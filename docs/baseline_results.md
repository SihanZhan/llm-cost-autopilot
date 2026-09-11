# Phase 1 baseline results

Run: 2026-09-10, `python baseline.py` (10 prompts x all 5 registry models).
Full per-row data (prompt id, tier, tokens, latency, cost, output preview) is
in `baseline_results.csv`, generated locally and gitignored — regenerate with
the command above. Prompts were rewritten for realism partway through Phase
1 (see `prompts/baseline_prompts.jsonl`); this run is against the current set.

## Summary

| model | calls | avg latency | total cost |
|---|---:|---:|---:|
| gpt-4o | 10 | 1.87s | $0.01145 |
| gpt-4o-mini | 10 | 1.64s | $0.00072 |
| claude-sonnet-5 | 10 | 2.96s | $0.01646 |
| claude-haiku-4-5 | 10 | 1.87s | $0.00573 |
| llama3 (local) | 10 | 16.08s | $0.00000 |
| **total** | **50** | | **$0.03436** |

## Notes

- **Cost:** the "mini" tier is dramatically cheaper than its frontier
  sibling — `gpt-4o-mini` is ~16x cheaper than `gpt-4o` on this set, and
  `claude-haiku-4-5` ~2.9x cheaper than `claude-sonnet-5`. `gpt-4o-mini` is
  the single cheapest paid model measured.
- **Latency:** an earlier run clocked `gpt-4o` at 16.19s avg, well above
  `claude-sonnet-5`. This re-run puts it back at 1.87s — in line with
  `claude-haiku-4-5` and faster than `claude-sonnet-5` (2.96s). The first
  number was transient API load, not representative; treat this run's
  figures as the reliable ones.
- `llama3` latency is highly variable: sub-4s on short Tier 1 prompts, up to
  ~60s on the hardest Tier 3 prompt. Free, but too slow as-is for anything
  beyond Tier 1 given the alternatives above.
- `models.py` `avg_latency` for all five models reflects this run.

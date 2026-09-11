"""Phase 3 demo: run the full classify -> route -> verify pipeline over a
fixed prompt set and report what happened.

Reuses prompts/baseline_prompts.jsonl (10 prompts spanning all three tiers
and the extraction/classification/summarization use cases) so this doesn't
need its own dataset. Each prompt is routed by the trained classifier, the
cheap-model response comes back immediately, and verification runs in the
background against the top-tier model — this script just waits for every
future before printing the summary so the report is complete.

Usage:
    python -m eval.demo
"""
from __future__ import annotations

import json
from pathlib import Path

from eval.pipeline import route_and_verify

ROOT = Path(__file__).parent.parent
PROMPTS_FILE = ROOT / "prompts" / "baseline_prompts.jsonl"


def load_prompts() -> list[dict]:
    rows = []
    with PROMPTS_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    prompts = load_prompts()
    print(f"routing {len(prompts)} prompts through classify -> route -> verify\n")

    pending = []
    for row in prompts:
        tier, response, future = route_and_verify(row["prompt"])
        pending.append((row["id"], tier, response, future))
        print(f"  {row['id']:<20} tier {tier}  ->  {response.model_id:<16} ${response.cost:.5f}  (verifying...)")

    print("\nwaiting on verification...\n")
    results = [(pid, tier, resp, future.result()) for pid, tier, resp, future in pending]

    header = f"{'prompt_id':<20} {'use_case':<15} {'passed':<7} {'escalated':<10} {'score':<7} {'cost_delta':>11}"
    print(header)
    print("-" * len(header))
    total_verify_cost = 0.0
    total_cost_delta = 0.0
    n_escalated = 0
    for pid, tier, resp, v in results:
        total_verify_cost += v.verification_cost
        total_cost_delta += v.cost_delta
        n_escalated += int(v.escalated)
        score_str = "skip" if v.skipped else (f"{v.quality_score:.2f}" if v.quality_score is not None else "err")
        print(
            f"{pid:<20} {v.use_case:<15} {str(v.passed):<7} {str(v.escalated):<10} "
            f"{score_str:<7} {v.cost_delta:>11.5f}"
        )

    print("-" * len(header))
    print(f"\n{n_escalated}/{len(results)} escalated")
    print(f"verification overhead: ${total_verify_cost:.5f}")
    print(f"escalation cost delta: ${total_cost_delta:.5f}")
    print(f"\nfull log: eval/logs/verification_log.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 4: seed the dashboard's SQLite DB with a live sample run.

eval.demo already exercises the pipeline over the 10 baseline prompts,
but that's a thin dataset for a dashboard's charts (routing distribution,
quality-score histogram, escalation rate). This samples a larger, tier-balanced
slice of the 224-prompt labeled set instead and routes every one of them
through eval.pipeline.route_and_verify for real, so the dashboard reads
actual measured cost/latency/quality data, not synthetic numbers.

Usage:
    python -m eval.seed_dashboard              # 45 prompts (15/tier)
    python -m eval.seed_dashboard --per-tier 10 # 30 prompts (10/tier)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from eval.pipeline import route_and_verify

ROOT = Path(__file__).parent.parent
LABELED_FILE = ROOT / "classifier" / "data" / "labeled_prompts.jsonl"
SEED = 42


def load_labeled() -> list[dict]:
    rows = []
    with LABELED_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_by_tier(rows: list[dict], per_tier: int) -> list[dict]:
    by_tier: dict[int, list[dict]] = {}
    for row in rows:
        by_tier.setdefault(row["tier"], []).append(row)

    rng = random.Random(SEED)
    sampled = []
    for tier, tier_rows in sorted(by_tier.items()):
        rng.shuffle(tier_rows)
        sampled.extend(tier_rows[:per_tier])
    return sampled


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the dashboard DB with a live sample run.")
    parser.add_argument("--per-tier", type=int, default=15, help="prompts sampled from each tier")
    args = parser.parse_args()

    prompts = sample_by_tier(load_labeled(), args.per_tier)
    print(f"routing {len(prompts)} prompts ({args.per_tier} per tier) through the live pipeline\n")

    pending = []
    for i, row in enumerate(prompts, start=1):
        tier, response, future = route_and_verify(row["prompt"])
        pending.append(future)
        print(f"  [{i:>3}/{len(prompts)}] {row['id']:<14} tier {tier}  ->  {response.model_id:<16} ${response.cost:.5f}")

    print("\nwaiting on verification...")
    n_escalated = 0
    total_cost_delta = 0.0
    for future in pending:
        result = future.result()
        n_escalated += int(result.escalated)
        total_cost_delta += result.cost_delta

    print(f"\ndone. {n_escalated}/{len(prompts)} escalated, ${total_cost_delta:.5f} escalation cost delta")
    print("data written to autopilot.db and eval/logs/verification_log.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

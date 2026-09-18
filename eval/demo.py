"""Phase 3 demo: run the full classify -> route -> verify pipeline over a
fixed prompt set and report what happened.

Reuses prompts/baseline_prompts.jsonl (10 prompts spanning all three tiers
and the extraction/classification/summarization use cases) so this doesn't
need its own dataset. Each prompt is routed by the trained classifier, the
cheap-model response comes back immediately, and verification is queued for
eval.verification_worker — a genuinely separate process from this script,
matching the brief's architecture. This script drains that queue itself
afterward (same worker code, run in-process for convenience) so it can
still print an immediate, complete summary.

Forces FORCE_VERIFY_ALL=1 (verify every prompt) rather than the production
default (eval.risk.should_verify's risk-based sampling) - with only 10
prompts, letting the risk signals decide would leave most of this demo's
output empty. eval.seed_dashboard and eval.load_test use the real default
instead, since demonstrating the sampled system's actual cost profile is
the point of those.

Usage:
    python -m eval.demo
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("FORCE_VERIFY_ALL", "1")

from eval.pipeline import route_and_verify  # noqa: E402
from eval.verification_worker import drain  # noqa: E402

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
        tier, response, request_id, job_id = route_and_verify(row["prompt"])
        pending.append((row["id"], tier, response, row["prompt"], job_id))
        note = f"queued for verification-worker, job {job_id}" if job_id else "already top-tier, nothing to verify against"
        print(f"  {row['id']:<20} tier {tier}  ->  {response.model_id:<16} ${response.cost:.5f}  ({note})")

    print("\ndraining the verification queue (eval.verification_worker)...\n")
    verified_by_prompt = {v.prompt: v for v in drain()}

    header = f"{'prompt_id':<20} {'use_case':<15} {'passed':<7} {'escalated':<10} {'score':<7} {'cost_delta':>11}"
    print(header)
    print("-" * len(header))
    total_verify_cost = 0.0
    total_cost_delta = 0.0
    n_escalated = 0
    n_verified = 0
    for pid, tier, resp, prompt, job_id in pending:
        v = verified_by_prompt.get(prompt)
        if v is None:
            print(f"{pid:<20} {'-':<15} {'-':<7} {'-':<10} {'skip':<7} {0.0:>11.5f}")
            continue
        n_verified += 1
        total_verify_cost += v.verification_cost
        total_cost_delta += v.cost_delta
        n_escalated += int(v.escalated)
        score_str = f"{v.quality_score:.2f}" if v.quality_score is not None else "err"
        print(
            f"{pid:<20} {v.use_case:<15} {str(v.passed):<7} {str(v.escalated):<10} "
            f"{score_str:<7} {v.cost_delta:>11.5f}"
        )

    print("-" * len(header))
    print(f"\n{n_escalated}/{n_verified} verified requests escalated ({len(pending) - n_verified} not verified - already top-tier)")
    print(f"verification overhead: ${total_verify_cost:.5f}")
    print(f"escalation cost delta: ${total_cost_delta:.5f}")
    print(f"\nfull log: eval/logs/verification_log.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

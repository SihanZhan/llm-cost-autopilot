"""Phase 3 feedback loop: turn logged routing failures into new classifier
training examples.

Each escalation in eval/logs/verification_log.jsonl means the routed tier
guessed wrong — the prompt actually needed the top tier. Re-labeling those
prompts at TIER_COMPLEX and folding them into training data is the flywheel
described in the roadmap: run this, then re-run `classifier.train`. The
project brief suggests a weekly cadence — that's an ops decision (cron /
scheduled task), not something this script does on its own.

Usage:
    python -m classifier.feedback
"""
from __future__ import annotations

import json
from pathlib import Path

from classifier.tiers import TIER_COMPLEX
from eval.verifier import LOG_FILE

OUT_FILE = Path(__file__).parent / "data" / "failure_feedback.jsonl"


def _load_existing_prompts() -> set[str]:
    if not OUT_FILE.exists():
        return set()
    with OUT_FILE.open(encoding="utf-8") as fh:
        return {json.loads(line)["prompt"] for line in fh if line.strip()}


def harvest() -> int:
    """Append new escalation-derived training rows; return how many were added."""
    if not LOG_FILE.exists():
        print(f"no verification log at {LOG_FILE} — nothing to harvest")
        return 0

    seen = _load_existing_prompts()
    new_rows = []
    with LOG_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("escalated"):
                continue
            prompt = row["prompt"]
            if prompt in seen:
                continue
            seen.add(prompt)
            new_rows.append(
                {
                    "id": f"failure-{len(seen):04d}",
                    "tier": TIER_COMPLEX,  # the top tier is what the prompt actually needed
                    "prompt": prompt,
                    "source": "routing_failure",
                    "original_tier": row["tier"],
                    "original_model": row["routed_model_id"],
                }
            )

    if new_rows:
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with OUT_FILE.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row) + "\n")

    print(f"harvested {len(new_rows)} new training example(s) from routing failures -> {OUT_FILE.name}")
    return len(new_rows)


if __name__ == "__main__":
    harvest()

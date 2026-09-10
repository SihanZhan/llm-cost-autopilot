"""Phase 1 baseline: run a fixed prompt set through every model in the registry.

Writes one CSV row per (prompt, model) and prints a per-model summary of calls,
average latency, and total cost. A model whose credentials or local runtime are
unavailable is skipped with a note after its first failure, so a partial run
still produces usable data.

Usage:
    python baseline.py                 # all prompts, all models
    python baseline.py --limit 3       # first 3 prompts only
    python baseline.py --out runs/first.csv
    python baseline.py --max-tokens 256
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from llm_clients import LLMRequestError, send_request
from models import MODEL_REGISTRY

ROOT = Path(__file__).parent
PROMPTS_FILE = ROOT / "prompts" / "baseline_prompts.jsonl"
DEFAULT_OUT = ROOT / "baseline_results.csv"


def load_prompts(limit: int | None = None) -> list[dict]:
    rows = []
    with PROMPTS_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def run(prompts: list[dict], out_path: Path, max_tokens: int) -> None:
    total_calls = len(prompts) * len(MODEL_REGISTRY)
    print(f"{len(prompts)} prompts x {len(MODEL_REGISTRY)} models = {total_calls} calls\n")

    results: list[tuple] = []
    skipped: dict[str, str] = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "prompt_id", "tier", "provider", "model_id",
                "input_tokens", "output_tokens", "latency_s", "cost_usd",
                "output_preview",
            ]
        )
        for model in MODEL_REGISTRY:
            for prompt in prompts:
                try:
                    resp = send_request(prompt["prompt"], model, max_tokens=max_tokens)
                except LLMRequestError as exc:
                    skipped[model.model_id] = str(exc)
                    print(f"  skip {model.model_id}: {exc}")
                    break
                results.append((model, prompt, resp))
                writer.writerow(
                    [
                        prompt["id"], prompt["tier"], model.provider, model.model_id,
                        resp.input_tokens, resp.output_tokens,
                        f"{resp.latency:.3f}", f"{resp.cost:.6f}",
                        resp.output_text[:80].replace("\n", " "),
                    ]
                )
                print(
                    f"  {model.model_id:<22} {prompt['id']:<20} "
                    f"{resp.latency:5.2f}s  ${resp.cost:.5f}"
                )

    _print_summary(results, skipped, out_path)


def _print_summary(results: list[tuple], skipped: dict[str, str], out_path: Path) -> None:
    print("\n" + "=" * 64)
    print(f"{'model':<22} {'calls':>6} {'avg latency':>13} {'total cost':>14}")
    print("-" * 64)

    by_model: dict[str, list] = {}
    for model, _prompt, resp in results:
        by_model.setdefault(model.model_id, []).append(resp)

    grand_total = 0.0
    for model_id, responses in by_model.items():
        avg_latency = statistics.mean(r.latency for r in responses)
        total_cost = sum(r.cost for r in responses)
        grand_total += total_cost
        print(f"{model_id:<22} {len(responses):>6} {avg_latency:>12.2f}s {total_cost:>13.5f}$")

    print("-" * 64)
    print(f"{'TOTAL':<22} {len(results):>6} {'':>13} {grand_total:>13.5f}$")

    if skipped:
        print("\nskipped models:")
        for model_id, reason in skipped.items():
            print(f"  {model_id}: {reason}")

    if results:
        print(f"\n{len(results)} rows written to {out_path}")
    else:
        print(f"\nno successful calls - {out_path} has only its header row")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 1 model baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV output path")
    parser.add_argument("--max-tokens", type=int, default=512, help="cap output tokens per call")
    args = parser.parse_args(argv)

    run(load_prompts(args.limit), args.out, args.max_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())

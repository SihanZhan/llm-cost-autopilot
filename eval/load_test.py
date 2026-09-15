"""Phase 6: load test - 500+ prompts through the full live pipeline.

Corpus is every unique prompt this project has (the 224-prompt labeled set
+ the 10 personalized baseline prompts = 234), cycled to reach the target
count. A load test's job is to stress volume and concurrency, not to prove
lexical novelty per request - that's what the 224-prompt hand-labeled set
was already built and diversity-checked for in Phase 2.

Routed calls run concurrently (ThreadPoolExecutor - llm_clients.send_request
is a blocking call, so this is real parallelism, not cosmetic); each
request also kicks off Phase 3's async verification exactly like production
use would. A single provider hiccup shouldn't sink 500 requests, so
failures are caught per-prompt and tallied, not raised.

Usage:
    python -m eval.load_test                    # 500 requests, 8 concurrent
    python -m eval.load_test --n 750 --workers 12
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from eval.pipeline import route_and_verify
from eval.verification_worker import drain
from llm_clients import LLMRequestError

ROOT = Path(__file__).parent.parent
LABELED_FILE = ROOT / "classifier" / "data" / "labeled_prompts.jsonl"
BASELINE_FILE = ROOT / "prompts" / "baseline_prompts.jsonl"


def load_corpus() -> list[str]:
    prompts: list[str] = []
    for path in (LABELED_FILE, BASELINE_FILE):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line)["prompt"])
    # de-dup while preserving order, in case a prompt appears in both files
    seen = set()
    return [p for p in prompts if not (p in seen or seen.add(p))]


def build_run_list(n: int) -> list[str]:
    corpus = load_corpus()
    cycler = itertools.cycle(corpus)
    return [next(cycler) for _ in range(n)]


def _run_one(prompt: str) -> tuple[bool, str | None, float]:
    """Route one prompt; return (ok, error, routed_cost)."""
    try:
        _tier, response, _job_id = route_and_verify(prompt)
        return True, None, response.cost
    except LLMRequestError as exc:
        return False, str(exc), 0.0
    except Exception as exc:  # noqa: BLE001 - one bad prompt shouldn't kill 500 others
        return False, f"{type(exc).__name__}: {exc}", 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 load test.")
    parser.add_argument("--n", type=int, default=500, help="total requests")
    parser.add_argument("--workers", type=int, default=8, help="concurrent routed calls")
    parser.add_argument("--verify-workers", type=int, default=4, help="concurrent verification-worker draining")
    args = parser.parse_args()

    run_list = build_run_list(args.n)
    corpus_size = len(load_corpus())
    start_ts = datetime.now(timezone.utc).isoformat()
    print(
        f"load test starting: {args.n} requests ({corpus_size} unique prompts, "
        f"cycled {args.n / corpus_size:.1f}x), {args.workers} concurrent workers\n"
        f"start: {start_ts}\n"
    )

    t0 = time.perf_counter()
    n_ok = n_fail = 0
    errors: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, p): i for i, p in enumerate(run_list)}
        for i, future in enumerate(as_completed(futures), start=1):
            ok, error, _cost = future.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                errors[error] = errors.get(error, 0) + 1
            if i % 50 == 0 or i == len(run_list):
                elapsed = time.perf_counter() - t0
                print(f"  {i}/{len(run_list)} routed  ({n_ok} ok, {n_fail} failed)  {elapsed:.0f}s elapsed")

    routed_elapsed = time.perf_counter() - t0
    print(f"\nall routed calls done in {routed_elapsed:.0f}s. draining the verification queue "
          f"(eval.verification_worker, {args.verify_workers} concurrent)...")

    verify_results = drain(workers=args.verify_workers)
    total_elapsed = time.perf_counter() - t0
    print(f"verification drained: {len(verify_results)} job(s) processed")

    if errors:
        print("\nerrors:")
        for msg, count in sorted(errors.items(), key=lambda kv: -kv[1]):
            print(f"  {count}x  {msg}")

    print(f"\ndone. {n_ok} ok, {n_fail} failed, {total_elapsed:.0f}s total")
    print(f"start_timestamp for report filtering: {start_ts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

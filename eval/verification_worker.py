"""The async verification worker, as a genuinely separate process from the
API — matching the brief's architecture: "API service + a background worker
for async verification + SQLite" as two docker-compose services sharing a
database, not one process with an in-process thread pool.

The API (api/main.py, via eval.pipeline.route_and_verify) enqueues a job to
db.py's verification_jobs table and returns immediately. This worker polls
that table, and for each claimed job reconstructs the routed response and
runs eval.verifier.verify_and_maybe_escalate — the same scoring/escalation/
logging logic as before, just invoked from a different process.

Usage:
    python -m eval.verification_worker            # persistent polling loop
    python -m eval.verification_worker --once      # process what's queued, then exit
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import db
from eval.verifier import VerificationResult, verify_and_maybe_escalate
from llm_clients import Response

POLL_INTERVAL_SECONDS = 2.0
DEFAULT_DRAIN_WORKERS = 4


def _job_to_response(job: dict) -> Response:
    return Response(
        output_text=job["routed_output_text"],
        input_tokens=job["routed_input_tokens"],
        output_tokens=job["routed_output_tokens"],
        latency=job["routed_latency"],
        cost=job["routed_cost"],
        model_id=job["routed_model_id"],
    )


def process_one(job: dict) -> VerificationResult:
    """Run verification for a single claimed job, mark it done, return the result."""
    routed_response = _job_to_response(job)
    try:
        result = verify_and_maybe_escalate(job["prompt"], job["tier"], routed_response)
        db.mark_job_done(job["id"])
        return result
    except Exception as exc:  # noqa: BLE001 - a bad job shouldn't kill the worker
        db.mark_job_error(job["id"], str(exc))
        raise


def drain(workers: int = DEFAULT_DRAIN_WORKERS, verbose: bool = False) -> list[VerificationResult]:
    """Process every currently-queued job (concurrently, ``workers`` at a
    time) and return the results.

    Used by eval.demo / eval.seed_dashboard / eval.load_test so those
    scripts can still print an immediate summary during local development,
    without needing a separately-running worker process. The deployed
    system (docker-compose) uses main() below instead - a persistent
    worker process, exactly as the brief specifies. Claiming happens in a
    tight sequential loop first (cheap - just a DB update), so this only
    drains what's queued at the moment it's called.
    """
    jobs = []
    while True:
        job = db.claim_next_verification_job()
        if job is None:
            break
        jobs.append(job)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
                if verbose:
                    print(f"  verified job {job['id']} (tier {job['tier']}, {job['routed_model_id']})")
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  job {job['id']} failed: {exc}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Async verification worker.")
    parser.add_argument(
        "--once", action="store_true",
        help="drain whatever is queued right now, then exit (default: poll forever)",
    )
    args = parser.parse_args()

    if args.once:
        results = drain(verbose=True)
        print(f"processed {len(results)} job(s)")
        return 0

    print(f"verification worker starting, polling every {POLL_INTERVAL_SECONDS}s")
    while True:
        job = db.claim_next_verification_job()
        if job is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        try:
            process_one(job)
            print(f"verified job {job['id']} (tier {job['tier']}, {job['routed_model_id']})")
        except Exception as exc:  # noqa: BLE001 - keep polling after a bad job
            print(f"job {job['id']} failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())

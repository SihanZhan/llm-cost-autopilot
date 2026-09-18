"""The async verification worker, as a genuinely separate process from the
API — matching the brief's architecture: "API service + a background worker
for async verification + SQLite" as two docker-compose services sharing a
database, not one process with an in-process thread pool.

The API (api/main.py, via eval.pipeline.route_and_verify) enqueues a job to
db.py's verification_jobs table and returns immediately. This worker polls
that table, and for each claimed job reconstructs the routed response and
runs eval.verifier.verify_and_maybe_escalate — the same scoring/escalation/
logging logic as before, just invoked from a different process. If the job
carries a callback_url, the outcome is also POSTed there - see
docs/completion_polling.md for why polling alone doesn't close the brief's
"return the better result" loop, and _send_callback below for the push
alternative.

Usage:
    python -m eval.verification_worker            # persistent polling loop
    python -m eval.verification_worker --once      # process what's queued, then exit
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import db
from eval.verifier import VerificationResult, verify_and_maybe_escalate
from llm_clients import Response

POLL_INTERVAL_SECONDS = 2.0
DEFAULT_DRAIN_WORKERS = 4
CALLBACK_TIMEOUT_SECONDS = 5.0


def _job_to_response(job: dict) -> Response:
    return Response(
        output_text=job["routed_output_text"],
        input_tokens=job["routed_input_tokens"],
        output_tokens=job["routed_output_tokens"],
        latency=job["routed_latency"],
        cost=job["routed_cost"],
        model_id=job["routed_model_id"],
    )


def _send_callback(callback_url: str, request_id: int, result: VerificationResult) -> None:
    """Best-effort POST of the verification outcome to ``callback_url`` -
    the push alternative to making the caller poll
    GET /v1/completions/{request_id}. Never raises: a caller's unreachable
    or misbehaving endpoint must not be allowed to break verification
    itself, which is why this is logged and swallowed, not retried or
    propagated. No SSRF protection (allowlisting destination hosts, blocking
    internal/link-local addresses) - fine for a demo callback URL you
    control yourself, a real deployment accepting third-party callback URLs
    would need it.
    """
    parsed = urlparse(callback_url)
    if parsed.scheme not in ("http", "https"):
        print(f"callback for request {request_id} skipped: bad scheme in {callback_url!r}")
        return

    payload = json.dumps(
        {
            "request_id": request_id,
            "status": "checked",
            "output_text": result.final_output,
            "model_id": result.final_model_id,
            "escalated": result.escalated,
            "quality_score": result.quality_score,
            "passed": result.passed,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        callback_url, data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=CALLBACK_TIMEOUT_SECONDS) as resp:
            print(f"callback for request {request_id} -> {callback_url} ({resp.status})")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"callback for request {request_id} -> {callback_url} failed: {exc}")


def process_one(job: dict) -> VerificationResult:
    """Run verification for a single claimed job, mark it done, fire the
    callback if one was requested, and return the result."""
    routed_response = _job_to_response(job)
    try:
        result = verify_and_maybe_escalate(
            job["request_id"], job["prompt"], job["tier"], routed_response
        )
        db.mark_job_done(job["id"])
        callback_url = job.get("callback_url")
        if callback_url:
            _send_callback(callback_url, job["request_id"], result)
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

"""Phase 5: the second docker-compose service - automates the feedback loop
that Phase 3 built but left as a manual two-command step.

Per-request verification is already async and non-blocking inside the API
process itself (a ThreadPoolExecutor in eval.verifier - see
docs/phase3_notes.md); that's not what this is. This is the "each routing
failure becomes a training example; weekly retrain on accumulated failures"
half of Phase 3's roadmap item, which was explicitly left as an ops decision
back then. Now it's a container: harvest escalations, retrain if anything
new came in, sleep, repeat.

RETRAIN_INTERVAL_SECONDS controls the cadence - default is 1 hour so a demo
deployment actually shows activity; the project brief's suggested
production cadence is weekly (set RETRAIN_INTERVAL_SECONDS=604800).

Known limitation: classifier.predict caches the loaded model in-process
(``@lru_cache``), so a retrain from this worker doesn't reach the API
process until it restarts. Fine for a demo; a real deployment would want
the API to poll model.joblib's mtime or the worker to signal it.

Usage:
    python -m classifier.retrain_worker
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from classifier import feedback, train

DEFAULT_INTERVAL_SECONDS = 3600  # 1 hour; brief's suggested prod cadence is weekly (604800)


def _log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def run_once() -> None:
    _log("harvesting routing failures from the verification log...")
    n_new = feedback.harvest()
    if n_new == 0:
        _log("nothing new - skipping retrain")
        return
    _log(f"{n_new} new example(s) harvested - retraining")
    train.main()


def main() -> int:
    interval = int(os.getenv("RETRAIN_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS))
    _log(f"retrain worker starting, interval={interval}s")
    while True:
        try:
            run_once()
        except Exception as exc:  # noqa: BLE001 - one bad cycle shouldn't kill the worker
            _log(f"retrain cycle failed: {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())

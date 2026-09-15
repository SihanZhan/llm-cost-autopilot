# Phase 5 notes: the API

Run: 2026-09-15, `uvicorn api.main:app` started locally and every endpoint
exercised live (real provider calls, not mocked).

> **Updated 2026-09-15, later the same day:** at the time this doc was
> written, verification still ran in-process (a `ThreadPoolExecutor`
> inside the API), and the second docker-compose service below was
> `retrain-worker`. A closer read of the brief ("API service + a
> background worker for async verification") led to adding a real third
> service, `verification-worker`, decoupling verification into its own
> process. See [docs/brief_conformance_fixes.md](brief_conformance_fixes.md).
> The endpoint-by-endpoint results below are otherwise unaffected.

## Endpoints

| Endpoint | Verified |
|---|---|
| `GET /health` | 200, `{"status": "ok"}` |
| `POST /v1/completions` | Two live requests — a tier-1 extraction (routed to `llama3`, $0.0000) and a tier-3 constrained poem (routed to `gpt-4o`, $0.00041) — both returned real output with a `why` field explaining the tier + routing.yaml mapping |
| `GET /v1/models` | Returns all 5 registry entries with real per-token pricing and measured `avg_latency` |
| `GET /v1/stats` | Returns the exact same numbers as the Phase 4 dashboard (routing-only + true net) — they share `stats.compute_stats()`, so a correction made there (see docs/phase6_notes.md) reaches both automatically |
| `PUT /v1/routing-config` | Changed tier 1 from `llama3` to `claude-haiku-4-5`, confirmed via `GET`, confirmed an invalid `model_id` is rejected with `400`, reverted back to `llama3` |
| `GET /v1/routing-config` | Not in the roadmap's list, but added since PUT-without-GET is an awkward API to ship |

Both `/v1/completions` requests logged through the full Phase 3/4 pipeline —
verification ran async and both rows showed up in `autopilot.db` a few
seconds later (one `passed`, one `skipped` since it was already routed to
the top-tier model).

## Shared stats module

`GET /v1/stats` and the dashboard's headline metric used to compute the same
numbers two different ways. Extracted `stats.compute_stats()` so both read
from one implementation — the dashboard was refactored to call it too
(non-breaking; the chart-specific pandas grouping stays in `dashboard/app.py`,
only the headline sums moved).

## The retrain worker, and the story it's telling

`classifier/retrain_worker.py` is a docker-compose service (an addition
beyond the brief's minimum, alongside the now-separate `verification-worker`
that IS what the brief specifies): harvest escalations, retrain if anything
new came in, sleep, repeat. Ran it once by hand instead of just reading the code:

| checkpoint | training rows | held-out accuracy |
|---|---:|---:|
| Phase 2 (hand-labeled only) | 224 | 97.8% |
| Phase 3 (+2 feedback rows, one demo run) | 226 | 95.7% |
| **Phase 5 (+12 more, from Phase 4's 45-request seed run)** | **238** | **85.4%** |

Still above the 80% target, but the trend across three phases is now
unambiguous, not a one-off: `complex` recall specifically has fallen from
93% → 88% → 67-61% as more `general`-bucket escalations (the same weak
token-overlap check flagged in [docs/phase3_notes.md](phase3_notes.md) and
shown to be costing real money in [docs/phase4_notes.md](phase4_notes.md))
get folded in as tier-3 training examples. The mechanism works exactly as
designed — that's precisely the problem. An automated feedback loop with an
unreliable judge doesn't fail loudly; it quietly drifts the model in one
consistent direction, run after run. This isn't a Phase 5 bug to fix here;
it's the same root cause three phases in a row now, and the fix (a real
check for the `general` bucket, or a human/second-opinion gate before an
escalation is trusted as a training label) belongs with `eval.quality`, not
the worker that faithfully executes what it's told to.

**Known limitations:**
- `classifier.predict` caches the loaded model in-process (`@lru_cache`), so
  a retrain from the worker doesn't reach a running API process until it
  restarts. Acceptable for a demo; a real deployment would want the API to
  poll `model.joblib`'s mtime or have the worker signal it.
- `classifier.routing.update_routing_config` (used by `PUT /v1/routing-config`)
  rewrites `routing.yaml` from scratch, so a programmatic update strips the
  hand-written per-tier comments (why each model was picked). Caught this
  while testing the endpoint live — the comments are restored in this repo,
  but the next `PUT` will drop them again. Worth a smarter writer later
  (ruamel.yaml round-trips comments); not worth blocking Phase 5 on.

## Docker

`Dockerfile` + `docker-compose.yml` define three services (`api`,
`verification-worker`, `retrain-worker`) sharing state via a bind mount of
the whole repo — SQLite isn't a client-server database, so there's no
separate "SQLite service"; all three containers just see the same
`autopilot.db` file. `docker compose config`
validated the full resolved config (build context, volumes, env injection)
with no errors. **Not run end-to-end**: this environment has the `docker`
and `docker compose` CLIs but no running Docker Desktop daemon, so
`docker compose up --build` couldn't be exercised live. Everything short of
an actual container boot has been checked; treat the Docker path as
reviewed, not proven, until someone runs it on a machine with a working
daemon.

`OLLAMA_HOST` is set to `host.docker.internal`, which resolves to the host
from a container on Docker Desktop (Mac/Windows) — on Linux you'd need
`extra_hosts: ["host.docker.internal:host-gateway"]` or the host's real IP.

## Reproducing

```bash
uvicorn api.main:app --reload          # local dev
docker compose up --build              # containerized (needs a running Docker daemon)
```

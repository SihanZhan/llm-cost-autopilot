# Closing two gaps between the project brief and what was built

Prompted by a detailed line-by-line restatement of the brief that made two
gaps visible that earlier phase notes had glossed over. Both are now closed
exactly as specified, not just functionally approximated.

## 1. The docker-compose "background worker for async verification"

**Before:** verification ran asynchronously, but as a `ThreadPoolExecutor`
inside the API process (`eval/verifier.py`'s `submit_verification`). The
second docker-compose service (`retrain-worker`) automated the classifier
feedback loop instead of verification — real value, but not what the brief
asked for in that slot.

**Now:** `eval.pipeline.route_and_verify` enqueues a row in a new
`verification_jobs` table (`db.py`) and returns immediately.
`eval/verification_worker.py` is a genuinely separate process — its own
`python -m eval.verification_worker` entrypoint, its own docker-compose
service — that polls that table and does the actual re-run-against-top-
tier + score + escalate + log work via the same `eval.verifier.
verify_and_maybe_escalate` as before.

Verified live, not just by reading the code:

1. Enqueued a job from one Python process, confirmed nothing processed it
   until a *separate* `python -m eval.verification_worker --once` process
   claimed and completed it.
2. Ran the worker in persistent polling mode in the background, enqueued a
   job from a different process, confirmed it got picked up automatically.
3. Ran the full API (`uvicorn api.main:app`) alongside a separately-running
   worker process, hit `POST /v1/completions` with curl, confirmed the
   response includes a `verification_job_id` and that the separate worker
   process (not the API) is what logged the verified outcome to
   `autopilot.db`.

`docker-compose.yml` now has three services: `api`, `verification-worker`
(the brief's required second service), and `retrain-worker` (the feedback-
automation addition from Phase 5, kept as a bonus beyond the brief's
minimum, not a replacement for it).

`eval.demo` / `eval.seed_dashboard` / `eval.load_test` still print an
immediate summary after routing — they call `eval.verification_worker.
drain()`, which runs the exact same worker code, just invoked in-process
for local-script convenience rather than requiring a second terminal. The
deployed system (`docker-compose up`) uses the real separate process.

**Bug caught during this change, before it ever ran live:** `db.init_db`
called `conn.execute(_SCHEMA)`, but `_SCHEMA` now has two `CREATE TABLE`
statements — `sqlite3.execute()` only accepts one. Fixed to
`executescript()`. Caught immediately by actually running `eval.demo`
rather than trusting that the code compiled cleanly.

## 2. Extraction verification: real field detection, not text comparison

**Before:** `score_extraction` checked whether the routed model's raw
output string appeared inside (or overlapped with) the top-tier model's
raw output string — the same mechanism as the generic `general` fallback,
just with a different threshold. Not what "check whether the expected
fields were found" means.

**Now:** `eval/quality.py::extract_fields` detects typed fields — email,
phone, date (three formats, normalized to ISO so "September 22, 2026" and
"2026-09-22" compare equal), dollar amount (with or without the `$`),
percentage, time, and a generic alphanumeric-code catch-all for things like
order numbers and zip codes — in both the routed and reference outputs.
`score_extraction` is now genuine field coverage: of the fields the
top-tier model's answer contains, what fraction does the routed model's
answer also contain? Falls back to the old text-comparison approach only
when neither answer contains a field this extractor recognizes (e.g. a
plain word like a name, or hashtags with no digits in them).

**Two real bugs found and fixed before this ever ran live** (caught by
testing the scorer directly against hand-picked cases, not by trusting the
first draft):

- Fields were typed by *which regex found them* (`date_iso` vs.
  `date_month_name`), so normalizing "September 22, 2026" and "2026-09-22"
  to the same value didn't help — they still carried different type tags
  and never compared equal. Fixed by giving every date pattern the same
  `"date"` type label.
- The generic alphanumeric catch-all re-tagged pieces of values a more
  specific pattern had already matched (e.g. a phone number's digits also
  partially matching the catch-all), inconsistently depending on
  formatting — `"555-330-2200"` and `"(555) 330-2200"` scored 0.33 instead
  of a perfect match. Fixed by masking each pattern's matches out of the
  text before the next, lower-priority pattern runs.

**Verified live against real extraction prompts and real model output**,
not just synthetic test cases — e.g. one escalation this surfaced was
`llama3` answering "The order number is: 29184" for a prompt whose correct
answer was "A29184" (it dropped the leading letter). That's a genuine
extraction error, correctly caught — not a scoring artifact.

**Known limitation, left as-is:** field coverage is scored against the
top-tier model's answer as a stand-in for "expected," since no separate
ground-truth-labeled extraction dataset exists. A verbose reference answer
that restates extra context (e.g. echoing the full input string alongside
the extracted value) can inflate the "expected field count" and dock an
otherwise-correct terse answer. Observed once in testing (a username-
extraction prompt scored 0.5 because the reference answer happened to
restate the full email address alongside the username). This is an honest
constraint of using model output as ground truth, not a coding bug — a
fixed answer-key dataset would remove it, at the cost of building one.

## What wasn't re-measured

The published 500-request load-test numbers (25.7% routing-only, -2.6% true
net, 26.6% escalation rate) were measured under the *old* extraction scorer
and the *old* in-process verification architecture. Neither change alters
the routing-cost math (that's driven by which model answered each request,
untouched here) or the dominant `general`-bucket finding (44% of traffic,
56% false-escalation rate, still the same code). Extraction was 38/500
requests (7.6%) at a 24% escalation rate under the old scorer; the new
scorer will produce a different (likely lower, per the spot-checks above)
rate for that slice specifically, but re-running the full 500-request load
test to get an exact refreshed number wasn't judged worth another ~$0.80 in
API spend for a change scoped to under 8% of traffic. Flagged here rather
than left unstated.

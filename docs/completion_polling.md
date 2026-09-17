# Closing the "return the better result" gap

The project brief's Phase 3, step 3 ("Implement auto-escalation"):

> "If the verifier catches a failure, automatically re-run the request with
> the higher-tier model and **return the better result** (if latency
> permits)."

Before this change, "return" didn't happen. `POST /v1/completions` returns
the routed (cheap) model's answer immediately, before verification even
starts — that's the whole point of making verification async. When
verification later escalated a bad answer, the correction only ever updated
`autopilot.db` for `/v1/stats` and the dashboard. The caller who asked the
original question had no way to get the corrected answer — escalation was
pure bookkeeping from their point of view, not a correction they could ever
receive.

## What's built now

- `requests` gained a `final_output` column: the routed model's answer at
  first, overwritten with the escalated answer if/when verification swaps
  it in. Previously the table tracked *that* an answer changed (`escalated`,
  `final_model`) but not *what it changed to* — the actual corrected text
  only ever lived in the gitignored `verification_log.jsonl`, keyed by
  prompt text, not designed for lookup.
- `POST /v1/completions` now returns `request_id` alongside the routed
  answer.
- New `GET /v1/completions/{request_id}` — poll this after the fact to get
  the current state: `status` (`not_checked` / `pending` / `check_failed` /
  `checked`), and `output_text` — the routed answer until/unless
  verification escalates it, at which point this becomes the corrected one.

## Verified live

1. `POST /v1/completions` with a prompt not sampled for verification ->
   `verification_job_id: null`. `GET /v1/completions/{id}` ->
   `status: "not_checked"`, `output_text` unchanged from the original
   response (correct — nothing to correct).
2. Forced `VERIFICATION_SAMPLE_RATE=1.0` and sent an order-number extraction
   prompt. `llama3` answered "The order number is: 29184" (dropped the
   leading letter, a real error - same failure mode documented in
   `docs/brief_conformance_fixes.md`). The verifier escalated it.
   `GET /v1/completions/{id}` returned `status: "checked"`,
   `output_text: "A29184"`, `model_id: "gpt-4o"`, `escalated: true` — **the
   actual corrected answer, retrievable for the first time.**

## What this still doesn't do

This is polling, not push. The brief's "return the better result" phrasing
reads like the caller should be handed the fix without asking again — that
would need a webhook, a websocket, or some other channel the caller
registers up front, none of which exists here or is specified in the brief.
Polling is the smallest change that makes the corrected answer retrievable
at all; a production system would likely want a real push mechanism instead
of leaving it to the caller to know to check back.

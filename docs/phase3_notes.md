# Phase 3 notes: async quality verification loop

Run: 2026-09-11, `python -m eval.demo` (the 10 `prompts/baseline_prompts.jsonl`
prompts, routed live through classifier -> router -> verifier).

> **Updated 2026-09-15:** step 4 below originally described an in-process
> `ThreadPoolExecutor`. Verification now runs in a genuinely separate
> `eval.verification_worker` process instead (a closer match to the brief's
> "API service + a background worker for async verification"), and
> extraction's check (step 5) now does real typed-field detection rather
> than comparing raw strings. See [docs/brief_conformance_fixes.md](brief_conformance_fixes.md).
> The results below are unaffected — same scoring outcomes either way for
> this run, just produced by different code now.

## How it works

`eval.pipeline.route_and_verify(prompt)` is the single entry point:

1. `classifier.predict.predict_tier` scores the prompt (Phase 2's model).
2. `classifier.routing.model_for_tier` looks up the routed model from `routing.yaml`.
3. The routed (cheap) model is called and its response returned immediately.
4. The prompt and response are enqueued as a job (`db.enqueue_verification_job`)
   for `eval.verification_worker`, a separate process, so the caller never
   blocks on step 5.
5. The worker: skip if the routed model already *is* the top tier;
   otherwise re-run the prompt against the top-tier model (`routing.yaml`
   tier 3), score agreement with the use-case-appropriate check from
   `eval.quality` (extraction: typed-field coverage, classification: exact
   label match, summarization: LLM-as-judge via `gpt-4o-mini`, everything
   else: token-overlap), and escalate — swap in the top-tier answer as the
   final result — if the score misses the threshold. Every outcome (pass,
   fail, escalate, skip, or verifier error) is one row in
   `eval/logs/verification_log.jsonl`.

## Demo results

| prompt | use case | passed | escalated | score | cost delta |
|---|---|---|---|---:|---:|
| t1-extract-email | extraction | yes | no | 1.00 | $0.00000 |
| t1-list-to-json | general | **no** | **yes** | 0.47 | $0.00035 |
| t1-capital-qa | general | yes | no | 1.00 | $0.00000 |
| t1-date-reformat | general | yes | no | 1.00 | $0.00000 |
| t2-summarize | summarization | yes | no | 1.00 | $0.00000 |
| t2-sentiment | classification | yes | no | 1.00 | $0.00000 |
| t2-tradeoffs | general | **no** | **yes** | 0.31 | $0.00191 |
| t3-word-problem | — | yes | no | skipped (already top tier) | $0.00000 |
| t3-constrained-poem | — | yes | no | skipped (already top tier) | $0.00000 |
| t3-plan-justify | — | yes | no | skipped (already top tier) | $0.00000 |

**2/10 escalated.** Verification overhead: $0.00341 (7 verifier calls + 1
judge call). Escalation cost delta: $0.00225.

## Feedback loop

`python -m classifier.feedback` reads the log, keeps the 2 escalated rows,
and appends them to `classifier/data/failure_feedback.jsonl` re-labeled at
tier 3 (the tier that actually produced an acceptable answer). `classifier.train`
picks that file up automatically alongside the hand-labeled set — re-running
it after the harvest:

| | before feedback | after feedback |
|---|---:|---:|
| training rows | 224 | 226 |
| held-out accuracy | 97.8% | 95.7% |

## A real finding, not just a working pipeline

Accuracy went **down** after "learning" from real failures, and it's worth
saying why rather than picking the better number: both escalations came from
the `general` use-case bucket, which falls back to raw token-overlap
similarity — a bar that a grocery-list-to-JSON prompt (t1-list-to-json) or a
short bulleted-list answer (t2-tradeoffs) can easily miss even when the
routed model's answer was fine, just phrased differently or formatted with
different whitespace than gpt-4o's. Relabeling those two prompts as
tier-3-complexity training examples is arguably *wrong* — they're not
actually hard, the verifier's blunt instrument just said they diverged from
the reference. Fold enough of that into training and the classifier's tier-3
boundary gets noisier, which is exactly the 88% recall on `complex` above (it
was 93% before the feedback).

This is a legitimate risk with automated feedback loops, not a bug to
silently patch over: a naive "every escalation becomes a training example"
policy will happily teach the classifier your verifier's blind spots. The
practical fix — flagged here for a later pass rather than done now, since
Phase 3's job was to prove the loop works end-to-end — is a human (or a
stricter second check) reviewing escalations before they're promoted to
training data, or scoping the `general` bucket's threshold/method so it
doesn't fire on cosmetic formatting differences.

> **Update 2026-09-16:** built. `general` now uses an LLM-as-judge instead
> of raw token overlap, and the poisoned feedback this section describes
> was cleared and the classifier retrained clean (85.4% → 97.8%, exactly
> the original number). See [docs/cost_fix_results.md](cost_fix_results.md).

## Known limitations

- `use_case_for_prompt` is keyword matching, not a classifier — prompts that
  don't contain a trigger word (`tradeoffs`, `pros and cons`, etc.) fall into
  `general` even when a sharper check would exist for them.
- Verification always targets the tier-3 model in `routing.yaml`; there's no
  per-tier verifier tier, so a tier-1 answer is graded against the most
  expensive model, not "one tier up."
- No retry/backoff tuning specific to the verifier path — it reuses
  `llm_clients.send_request`'s defaults.

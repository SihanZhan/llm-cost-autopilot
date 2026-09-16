# The fix, validated: from a 2.6% loss to a 20.3% real saving

Run: 2026-09-16, `python -m eval.load_test --n 500 --workers 8 --verify-workers 4`
— same corpus, same scale, same method as the run that discovered the
problem, run again after implementing the two fixes discussed in
[CASE_STUDY.md](../CASE_STUDY.md): sampled verification instead of checking
every request, and a real LLM-as-judge for the `general` bucket instead of
raw token overlap.

## The headline number, before and after

| | before | after |
|---|---:|---:|
| routing-only cost reduction | 25.7% | 26.4% |
| **true net cost reduction (all verification counted)** | **-2.6% (a loss)** | **+20.3%** |
| escalation rate | 26.6% (133/500) | 1.2% (6/500) |
| verification cost | $0.2166 | $0.0466 |
| requests actually verified | 336/500 (100% of non-top-tier) | 66/500 (13.2% of all, ~19% of non-top-tier — matches the 20% sample rate) |

Same 500 requests, same corpus, same providers, same classifier — nothing
about the routing logic changed. The entire swing, from losing money to
saving over a fifth of the baseline, came from fixing how verification
checks quality and how often it bothers to.

## What each fix contributed

**Sampling (verify ~20% of non-top-tier requests instead of 100%)** cut
verification spend from $0.217 to a mix of sampling reduction and fewer
false escalations — most of the swing. Checking fewer requests means paying
the top-tier model's rate far less often just to confirm an answer that was
already fine.

**The `general`-bucket LLM-as-judge** cut that bucket's escalation rate from
56% to 9% (measured on the sampled subset: 4 escalations out of 46 checked
`general`-bucket requests, vs. the old checker's 122/218). Fewer false
escalations means fewer "pay for both models" events, and it means the
escalations that do still happen are far more likely to be real routing
failures, not phrasing differences the old raw-word-overlap check couldn't
tell apart from a real error.

| use case (this run, verified subset) | n | escalated | rate |
|---|---:|---:|---:|
| classification (exact match) | 3 | 0 | 0% |
| summarization (LLM-as-judge) | 12 | 0 | 0% |
| general (LLM-as-judge, new) | 46 | 4 | 9% |
| extraction (typed-field coverage) | 5 | 2 | 40% |

Extraction's rate looks high here, but n=5 at a 20% sample rate is too
small a sample to read much into — it was 24% at n=38 under the old checker
(see [docs/phase6_notes.md](phase6_notes.md)) and hasn't been re-measured
at comparable scale under the reworked field-based checker from
[docs/brief_conformance_fixes.md](brief_conformance_fixes.md).

## The classifier accuracy recovery

The old `general`-bucket checker didn't just cost money — every false
escalation got relabeled as "this needed the top tier" and fed back into
classifier training, dragging held-out accuracy from 97.8% down to 85.4%
across three retrains (documented in [docs/phase3_notes.md](phase3_notes.md),
[phase5_notes.md](phase5_notes.md)). That poisoned feedback
(`classifier/data/failure_feedback.jsonl`, 14 mislabeled examples like a
grocery-list-to-JSON prompt tagged "complex") was cleared, and the
classifier retrained from the clean 224-prompt hand-labeled set alone:

| | before cleanup | after |
|---|---:|---:|
| training rows | 238 (224 clean + 14 poisoned) | 224 (clean only) |
| held-out accuracy | 85.4% | **97.8%** |

Exactly back to the original Phase 2 number - not a coincidence, the same
clean data produces the same result. Going forward, the feedback loop
(`classifier/feedback.py`) will only harvest escalations from the now-fixed
verifier, so it shouldn't re-poison itself the same way - but that's a
claim worth re-checking after enough real usage accumulates, not something
proven by this one retrain.

## What changed in the code

- `eval/pipeline.py`: `VERIFICATION_SAMPLE_RATE` (env var, default 0.2)
  gates whether a non-top-tier request's verification job gets enqueued at
  all. Every routed request is still logged immediately
  (`db.log_routed_request`) regardless of sampling, so cost/routing
  tracking stays complete even though verification coverage dropped -
  cost accounting must not depend on whether a request happened to get
  checked.
- `db.py`: split the old single-shot `log_request` into
  `log_routed_request` (insert immediately) and `update_verification_result`
  (update later, only for the sampled subset that actually gets checked).
- `eval/quality.py`: `score_general` now calls the same LLM-as-judge
  mechanism `score_summarization` already used, via a shared
  `_llm_judge_score` helper, instead of `_token_overlap`. Threshold raised
  from 0.6 (calibrated for word-overlap) to 0.8 (matching summarization's
  judge threshold, "4/5 or better").

## Reproducing

```bash
python -m classifier.train                              # confirm 97.8% on the clean dataset
python -m eval.load_test --n 500 --workers 8 --verify-workers 4
python -m eval.report_charts --since <start timestamp load_test prints>
```

`VERIFICATION_SAMPLE_RATE` defaults to 0.2; override via environment
variable to try other rates (`eval.demo` forces 1.0 so its 10-prompt
showcase still verifies every prompt).

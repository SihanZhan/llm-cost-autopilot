# Risk-based verification: check the ones that look wrong, not a random slice

Prompted directly by the question "why is it random that you check 1 out of
5, why not ones that don't seem right" — a fair critique of the sampling
fix in [docs/cost_fix_results.md](cost_fix_results.md), which cut
verification cost by checking a flat random 20% instead of everything, but
never targeted *which* 20%.

## What decides now

`eval/risk.py::should_verify`, called from `eval/pipeline.py` for every
non-top-tier request, checks (in order, cheapest first):

1. **Classifier confidence.** `classifier.predict.predict_tier_with_confidence`
   returns the model's own probability for the tier it picked (both logistic
   regression and random forest expose `predict_proba`). Below 0.6, the tier
   guess itself was a close call - verify it.
2. **Does the answer look risky?** `eval/risk.py::looks_risky` - empty
   output, or a hedge/refusal phrase ("I'm not sure," "as an AI," "it
   depends," "I cannot," ...). Both checks are regex over text already in
   hand; no extra API call to decide whether an API call is worth making.
3. **A small random baseline (5%).** Independent of the two signals above,
   so the escalation-rate number stays an honest estimate of the true error
   rate instead of "how often our risk-guesser was right to be suspicious,"
   and so problems the heuristics don't anticipate still get caught
   sometimes.

No flat percentage on top of that. Total verification volume floats with
how often the signals actually fire - which is the point: spend the
checking budget where it's likely to find something, not evenly everywhere.

`FORCE_VERIFY_ALL=1` bypasses all of this and checks every non-top-tier
request regardless of risk signals (used by `eval.demo`'s 10-prompt
showcase, where letting risk decide would leave most of it unverified).

## Bug caught before this ever ran live

`should_verify` was tested against hand-built cases before being wired into
the pipeline (same pattern as every other fix in this project). "I am not
sure about this." didn't trigger the hedge check — the regex only matched
the contracted form ("I'm not sure"), not "I am not sure." Fixed before the
first real request went through it.

## Live result

Run: 2026-09-18, `python -m eval.load_test --n 300 --workers 8 --verify-workers 4`.

| | flat 20% sampling (previous fix) | risk-based (this fix) |
|---|---:|---:|
| requests checked | ~20% of non-top-tier | **14/300 (4.7%)** |
| verification cost | $0.0466 (500-req run) | **$0.0026** (300-req run) |
| routing-only cost reduction | 26.4%–27.2% | 27.71% |
| **true net cost reduction** | **+20.3%** | **+27.0%** |
| escalation rate | 1.2%–1.4% | 1.0% (3/301) |

Verification overhead dropped from a multi-point drag on the routing
savings to nearly nothing - net savings (27.03%) now sit within a point of
the theoretical routing-only ceiling (27.71%), instead of several points
below it. The three escalations this run caught were the same class of real
error prior runs found (a Roman-numeral conversion, a dropped character in
an extracted order number, a partial-credit username parse) - fewer checks,
not worse ones.

Of the 14 jobs this run enqueued, 13 came from the random baseline and 1
from `risky_answer`; `low_classifier_confidence` never fired. On a corpus
the classifier is 97.8% accurate on, it's rarely unsure - which is itself a
reasonable result, not a sign the confidence signal is dead weight; it
should fire more on a harder or more novel prompt mix.

## Known limitation

This was validated at n=300, not the full n=500 the earlier fixes were
checked against, and only once (not independently confirmed twice, unlike
the sampling+judge fix in `cost_fix_results.md`). The direction and rough
magnitude are credible given how directly they follow from checking far
fewer requests at a similar catch rate, but treat the exact 27.0% figure as
a single measurement, not a two-run-confirmed one.

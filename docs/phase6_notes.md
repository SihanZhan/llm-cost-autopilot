# Phase 6 notes: load test and final numbers

Run: 2026-09-15, `python -m eval.load_test --n 500 --workers 8`. Corpus:
all 234 unique prompts this project has (224 hand-labeled + 10 personalized
baseline prompts), cycled 2.1x to reach 500, routed concurrently (8 workers)
through the full live pipeline built in Phases 1-5, each request also
triggering Phase 3's async verification exactly like production traffic
would.

**500/500 succeeded, 0 failures**, in 448s wall time (≈1.1 req/s at 8
concurrent workers — bottlenecked by `llama3`'s 12.8s average local
inference latency, see below).

## Headline numbers

> **Correction (2026-09-15, same day, found a few hours after this doc was
> first written):** the "net cost reduction" row below was computed by
> `stats.py` as routed cost + the extra cost of the 133 *escalated*
> requests. It never added back the verification cost spent checking the
> other 203 requests that passed — money that was spent regardless of the
> outcome. That's $0.217, more than the entire savings the "net" row
> claimed. Corrected numbers:

| | value |
|---|---:|
| requests | 500 |
| routing-only cost reduction (verification-free) | **25.7%** |
| ~~net cost reduction (incl. escalation cost)~~ *(incomplete, see above)* | ~~3.7%~~ |
| **true net cost reduction (all verification counted)** | **-2.6% (a net loss)** |
| escalation rate | 26.6% (133/500) |
| quality parity (pass rate among verified requests) | 60.4% (203/336) |
| total routed cost | $0.5698 |
| all-GPT-4o baseline cost | $0.7668 |
| verification cost (all 336 checks, pass or fail) | $0.2166 |
| true total cost | $0.7864 |

**The corrected true-net figure is not a fluke of the earlier 45-request
sample either** — recomputed on that run's data it's -3.9%, the same sign
and the same rough magnitude as this 500-request run's -2.6%. Both runs
independently say the same thing: as built, this system spends more than
the baseline it was meant to beat. Routing-only savings did move between
samples (30.4% → 25.7%), which makes sense — a larger, more varied sample
pulls the average toward the true mix of tier costs. The sign of the *true*
net number didn't move: negative both times.

**How this was found:** by asking, in plain terms, "if verification calls
the expensive model on two-thirds of all traffic, how could that possibly
be cheaper than just using the expensive model?" — a question about the
mechanism, not the metric. Working through the actual dollar figures by
hand (not just trusting `stats.py`'s output) surfaced the gap. See
[CASE_STUDY.md](../CASE_STUDY.md) for the full corrected writeup.

## Where the escalations come from, confirmed at scale

| use case | n | escalated | rate |
|---|---:|---:|---:|
| summarization (LLM-as-judge) | 48 | 0 | 0% |
| classification (exact match) | 32 | 2 | 6% |
| extraction (field-coverage proxy)* | 38 | 9 | 24% |
| **general (token-overlap fallback)** | **218** | **122** | **56%** |

\* *extraction's checker was reworked afterward to detect real typed fields
(dates, emails, amounts, phone numbers, ...) instead of comparing raw
output strings — this row is from the older text-comparison version. See
[docs/brief_conformance_fixes.md](brief_conformance_fixes.md); not
re-measured at full 500-request scale since extraction is 7.6% of traffic.*

This is the same pattern Phases 3-5 found on smaller samples, now confirmed
at n=500 rather than n=10 or n=45: the three purpose-built checks
(summarization, classification, extraction) have low-to-zero false-escalation
rates. The `general` fallback — which catches the majority of traffic
(218/500, 44%) because most prompts don't contain one of the three
use-case trigger keywords — escalates *more than half the time*. That's not
"occasionally too strict," that's the dominant driver of both the accuracy
regression (Phases 3/5) and the cost problem (Phase 4/6) this project has
been tracking — severe enough, once counted fully, to flip the whole
system into a net loss rather than merely eroding its savings. At this
point across four phases of consistent evidence, this
isn't a finding anymore, it's a known, unfixed defect: `eval.quality`'s
`general` bucket needs a real check (an LLM-as-judge like `summarization`
already has, or research into what threshold token-overlap should actually
use) before this system should be trusted with real escalation decisions.

**Escalation rate by tier** is also worth a second look: tier 2 escalates
*more* than tier 1 (45% vs. 35%), which is counter to the intuition that
"simpler tasks are safer to route cheap." It's really a use-case-mix
artifact — tier 1 skews toward extraction/classification prompts (the
better-behaved checks), tier 2 skews toward `general`-bucket prompts
(comparisons, "list pros and cons," open-ended asks) that the loose check
handles worst.

## Latency at scale

| model | avg latency | n |
|---|---:|---:|
| gpt-4o-mini | 1.43s | 154 |
| gpt-4o | 5.87s | 164 |
| llama3 (local) | 12.82s | 182 |

`llama3` is the free option and also the slow one, consistent with every
earlier baseline in this project — at concurrency 8, it's the load test's
throughput bottleneck. A production deployment routing real user-facing
traffic to tier 1 would want either a faster local model or more concurrent
Ollama capacity; this project treats it as an acceptable tradeoff for a
zero-cost tier, not a hidden cost.

## Charts

Generated from this run's data via `eval/report_charts.py` (matplotlib, not
a browser screenshot — this environment has no browser to screenshot the
live Streamlit dashboard with, so these are a direct rendering of the same
`stats.compute_stats()` numbers the dashboard and `/v1/stats` show):

- `docs/images/cost_comparison.png`
- `docs/images/routing_distribution.png`
- `docs/images/quality_distribution.png`
- `docs/images/escalation_by_tier.png`

## Reproducing

```bash
python -m eval.load_test --n 500 --workers 8
python -m eval.report_charts --since <start timestamp printed by load_test>
```

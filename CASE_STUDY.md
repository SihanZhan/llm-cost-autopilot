# Case study: LLM Cost Autopilot

**Routing 500 live requests through a complexity-based router cut LLM API
spend 25.7% against an all-GPT-4o baseline — but once every dollar spent
verifying those decisions was counted honestly, the system actually cost
2.6% *more* than the baseline. Finding that, tracing it to its exact cause,
building the fix, and validating it on another live 500-request run —
**20.3% real net savings, escalation rate down from 26.6% to 1.2%, classifier
accuracy recovered from 85.4% to 97.8%** — is the actual arc of this
project: not a single flattering number, but catching a system that looked
fine, proving it wasn't, and proving the fix.**

## The idea

Most teams running LLMs in production send every request — a one-line
extraction and a multi-step planning task alike — to the same frontier
model. That's simple, but it's also paying frontier prices for work a
$0.15-per-million-token model could do just as well. LLM Cost Autopilot
treats model choice as a routing decision: classify each request's
complexity, send it to the cheapest model that can actually handle it, and
continuously check that the cheap model's answer was good enough — with a
mechanism to catch and fix it when it wasn't.

## The system

```mermaid
flowchart LR
    user[Caller] --> api[api container<br/>POST /v1/completions]
    api --> clf[Classifier]
    clf -->|simple| cheap[Llama 3 - free, local]
    clf -->|moderate| mid[GPT-4o-mini]
    clf -->|complex| top[GPT-4o]
    cheap & mid & top --> resp[Response]
    resp --> api --> user
    api -.enqueue job.-> queue[(SQLite<br/>verification_jobs)]
    queue -.poll.-> worker[verification-worker container<br/>separate process]
    worker --> verify[Verify vs. top tier]
    verify -->|fail| esc[Escalate + log]
    esc -.feedback.-> retrain[retrain-worker container]
    retrain -.updates.-> clf
    verify --> db[(SQLite<br/>requests)]
    db --> dash[Dashboard / API stats]
```

- **Classifier** (scikit-learn, trained on 224 hand-labeled prompts across
  three complexity tiers) scores every request in milliseconds using cheap
  lexical features — no LLM call needed to decide which LLM to call.
- **Router** maps tier → model via a YAML file, live-editable through
  `PUT /v1/routing-config` — no redeploy to change the policy.
- **Verifier** re-runs the same prompt against the top-tier model
  *after* the cheap response has already gone back to the caller, scores
  agreement with a check suited to the request type, and escalates
  (substitutes the top-tier answer) on failure — all in a separate
  `verification-worker` process from the API, so verification never adds
  latency to the response and can't compete with the API for resources.
- **Feedback loop** turns every escalation into a new training example and
  retrains the classifier on accumulated failures.
- **Dashboard and API** (`GET /v1/stats`, Streamlit) read the same logged
  data through one shared aggregation, so they can't quote different numbers
  for the same thing.

Full technical detail, live-tested phase by phase, is in
[docs/](docs) (`baseline_results.md` through `phase6_notes.md`) and
[ROADMAP.md](ROADMAP.md).

## The headline number, and the number under it

500 requests, sampled across all three complexity tiers, routed live
through the whole pipeline above (not simulated — real OpenAI, Anthropic,
and local Ollama calls):

![Cost vs. all-GPT-4o baseline](docs/images/cost_comparison.png)

Routing alone looks great: **25.7% cheaper than sending everything to
GPT-4o.** That number is real, but it's answering a narrower question than
it looks like it is — "what did the cheap/expensive model calls cost,
ignoring verification entirely?" Here's what verification actually costs:

| | what's counted | total spent | vs. $0.7668 baseline |
|---|---|---:|---:|
| Routing only | whichever model answered | $0.5698 | 25.7% saved |
| + cost of fixing wrong answers | + the 133 escalation replacements | $0.7381 | 3.7% saved |
| **+ cost of checking right answers** | **+ verification on all 336 checked requests, pass or fail** | **$0.7864** | **2.6% MORE than baseline** |

The middle row is the trap. 336 of the 500 requests got double-checked
against the top-tier model — not just the ones that turned out wrong. Only
133 of those 336 actually needed the top-tier model's answer; the other 203
passed, and the system paid the top-tier model's rate to confirm that
anyway. That's money spent with nothing to show for it: you already had a
fine answer, and paid again to be told so.

An earlier version of this report stopped at the middle row and called
**3.7% saved** the honest number — an improvement over the flattering 25.7%,
but still incomplete, because it only added back the cost of the 133
failures and never added back the verification spend on the 203 passes.
That gap is $0.217 — bigger than the entire "savings" the middle row
claimed. The corrected bottom row is what this system actually spent:
**more than the naive baseline it was built to beat.** This was caught by
asking the sharper question — "if you're calling both models on two-thirds
of traffic, how is that ever cheaper than calling just one?" — rather than
stopping once a number looked responsible enough to publish.

## Where the money actually goes

![Escalation rate by tier](docs/images/escalation_by_tier.png)

Escalations aren't evenly spread. Breaking them down by *why* the verifier
checks a request the way it does tells the real story:

| verification method | requests | escalation rate |
|---|---:|---:|
| LLM-as-judge (summarization) | 48 | 0% |
| exact match (classification) | 32 | 6% |
| field-coverage proxy* (extraction) | 38 | 24% |
| **raw token-overlap (everything else)** | **218** | **56%** |

\* *Extraction's check was reworked after this run to detect real typed
fields (dates, emails, amounts, ...) instead of comparing raw output
strings — see [docs/brief_conformance_fixes.md](docs/brief_conformance_fixes.md).
The 24% figure above is from the old text-comparison version; it wasn't
re-measured under the new one (extraction is 7.6% of traffic, not enough
to justify another full paid load test on its own).*

Three purpose-built checks work well. The fallback used for anything that
doesn't match a keyword trigger — a blunt "do the two answers share enough
words" heuristic — escalates over half the time, and it's the single
biggest use case by volume (44% of all traffic). That one weak check is
responsible for most of both numbers above: the cost gap, and — traced
through the feedback loop — a real accuracy cost too.

## The feedback loop, and what happens when you trust it blindly

Every escalation gets relabeled as a "this needed the top tier" training
example and folded back into the classifier's next retrain. That mechanism
works exactly as designed. Which is the problem: across three retrains, each
one triggered by real escalations from real runs, held-out accuracy went
**97.8% → 95.7% → 85.4%** — still above the 80% target, but a clean,
repeating trend, not noise. (Separately, a handful of near-duplicate
hand-labeled prompts — e.g. "convert 5 miles to km" / "convert 3 kg to
lbs," same skeleton, different numbers — were rewritten for more genuine
variety, and accuracy on the current dataset sits at 87.5%; that's a data-
quality fix, not evidence the feedback-loop problem went away.) The same
over-eager `general`-bucket check that
erodes cost savings is also quietly teaching the classifier the wrong
lesson every time it fires: a grocery-list-to-JSON request or a "list three
pros and cons" prompt isn't actually complex, but the verifier says it
diverged from GPT-4o's phrasing, and the feedback loop takes that as gospel.

An automated feedback loop is only as trustworthy as its judge. This one's
judge has an identified, measured blind spot, and the system faithfully
amplifies it every cycle. That's the finding worth remembering here more
than any single percentage: **a system with real automated feedback is not
automatically a system that's getting better** — you have to watch what it's
actually learning from.

## The fix, validated on another live 500-request run

Two changes, targeting the two root causes above:

1. **Stop checking every answer.** Verification now samples ~20% of
   non-top-tier requests (`VERIFICATION_SAMPLE_RATE`) instead of all of
   them — still enough to catch systematic problems, at a fraction of the
   cost of grading every answer whether or not it needed grading.
2. **Replace the bad `general`-bucket check.** It now uses the same
   LLM-as-judge mechanism `summarization` already had (proven at 0% false-
   escalation rate) instead of raw token overlap.

Nothing about the routing logic changed — same classifier, same models,
same tiers. Run again at the identical scale (500 requests, same corpus):

| | before the fix | after the fix |
|---|---:|---:|
| routing-only cost reduction | 25.7% | 26.4% |
| **true net cost reduction** | **-2.6% (a loss)** | **+20.3%** |
| escalation rate | 26.6% | 1.2% |
| `general`-bucket false-escalation rate | 56% | 9% |
| classifier held-out accuracy | 85.4% (poisoned by bad feedback) | **97.8%** (recovered) |

The classifier number needed one more step beyond the two code changes:
the training examples the old broken checker had already fed back
(`classifier/data/failure_feedback.jsonl` — real cases, like a grocery-list-
to-JSON prompt mislabeled "complex") were cleared, and the classifier
retrained from the clean hand-labeled set alone. It landed at exactly the
original Phase 2 number, which is the whole point: the same clean data
produces the same result, confirming the earlier drop really was the bad
feedback and nothing else.

Full breakdown, including what each of the two fixes contributed
separately: [docs/cost_fix_results.md](docs/cost_fix_results.md).

## What's real vs. what would come next

Everything above is measured, not projected: live-tested against real
provider APIs on two separate full-scale (500-request) runs — one that
found the problem, one that confirmed the fix — every claim traceable to a
script anyone can re-run (`python -m eval.load_test`,
`python -m eval.report_charts`). What this project doesn't claim: it isn't
running in production, and the Docker deployment is config-validated but
not container-tested end-to-end (no daemon in this build environment) —
though the underlying multi-process architecture it depends on (a separate
verification-worker process) was verified live outside Docker.

Two earlier gaps between the brief and the build were also found and
closed along the way — verification runs as a genuinely separate worker
process (not an in-process thread pool), and extraction verification checks
real typed fields instead of comparing raw output strings. Both were caught
the same way everything else in this project was: by testing the actual
behavior, not trusting that the code looked right. See
[docs/brief_conformance_fixes.md](docs/brief_conformance_fixes.md).

## Repo

Six phases end to end: unified provider interface → complexity classifier →
async verification with auto-escalation → cost dashboard → FastAPI service
→ this load test. See [README.md](README.md) for setup and
[ROADMAP.md](ROADMAP.md) for the full phase-by-phase build log.

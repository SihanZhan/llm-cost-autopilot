# Phase 4 notes: logging and cost dashboard

Run: 2026-09-15, `python -m eval.seed_dashboard` (45 prompts, 15 sampled per
tier from the 224-prompt labeled set, routed live through the full
classify → route → verify pipeline built in Phases 2–3).

## What's logged

Every request gets one row in `autopilot.db` (SQLite, gitignored —
regenerate with the commands below) via `db.py`: timestamp, a **hashed**
prompt (not full text — full text stays in the separate, also-gitignored
`eval/logs/verification_log.jsonl` that Phase 3's feedback loop needs), use
case, tier, routed model, routed cost, routed latency, a computed baseline
cost (what the same token counts would have cost on `gpt-4o`), quality
score, pass/escalate flags, final model, and cost delta.

`dashboard/app.py` (`streamlit run dashboard/app.py`) reads that table and
renders the headline metric, daily cost vs. baseline, routing distribution,
quality-score distribution, and escalation rate over time. I ran the server
locally and confirmed it boots with no exceptions and the underlying
pandas aggregations are correct (verified standalone before wiring them into
Streamlit) — I don't have a way to visually screenshot a running browser
session from here, so "renders correctly" is confirmed structurally, not
visually.

## Seed run results (45 requests, 15/tier)

| tier | model | n | routed cost | baseline cost | routing-only savings |
|---|---|---:|---:|---:|---:|
| 1 | llama3 | 15 | $0.00000 | $0.00308 | 100.0% |
| 2 | gpt-4o-mini | 15 | $0.00116 | $0.01932 | 94.0% |
| 3 | gpt-4o | 15 | $0.04740 | $0.04740 | 0.0% (already the baseline model) |
| **total** | | **45** | **$0.04856** | **$0.06980** | **30.4%** |

## The headline number needs a second look

30.4% is the number you'd put on a slide, and it's real — but it only counts
the routed (cheap-model) cost, not what verification actually spent to get
there. **12 of 45 requests (27%) escalated** — 6/15 at tier 1, 6/15 at tier
2 — at a combined cost delta of **$0.01863**. Fold that into "what this
system actually spent" and the picture changes a lot:

| | routing-only | net (incl. escalation cost) |
|---|---:|---:|
| actual spend | $0.04856 | $0.06719 |
| vs. $0.06980 baseline | **30.4% saved** | **3.7% saved** |

The dashboard shows both numbers side by side rather than leading with only
the flattering one, and surfaces a warning banner when escalation cost eats
more than half the routing savings (it does here).

## Why so many escalations

All 12 escalations came from the `general` use-case bucket — [eval/quality.py](../eval/quality.py)'s
token-overlap fallback, threshold 0.6. This is the same weak spot
[docs/phase3_notes.md](phase3_notes.md) already flagged after just 10 prompts;
seeding with 45 confirms it's not noise. Two free-form answers to the same
prompt from two different models routinely score well below 0.6 on raw
token overlap even when both are reasonable answers — different phrasing,
different list formatting, different length — so `general` fires escalation
far more often than the `extraction` / `classification` / `summarization`
buckets do (which had zero escalations here).

**This is the real lesson of Phases 3–4 together:** an auto-escalation loop
is only as trustworthy as its quality check. A cheap, blunt check (token
overlap) doesn't just mislabel training data (Phase 3's finding) — it also
directly erodes the cost savings the whole system exists to produce (Phase
4's finding), because every false-positive escalation pays for a full
top-tier call it didn't need. The fix isn't more escalation, it's a better
`general`-bucket check — e.g. a real LLM-as-judge for it too (currently only
`summarization` gets one), or a looser threshold backed by evidence instead
of a guessed 0.6.

## Reproducing

```bash
python -m eval.seed_dashboard --per-tier 15   # or eval.demo for a smaller run
streamlit run dashboard/app.py
```

"""Cost-savings summary shared by dashboard/app.py and api/main.py's
GET /v1/stats — one place computes the headline numbers so the API and the
dashboard can't drift apart on what "cost reduction" means.

CORRECTED 2026-09-15: the previous version of ``net_actual_cost`` added
back only ``escalation_cost_delta`` (the extra cost of the requests that
failed verification and got escalated). It never added back
``verification_cost`` spent checking the requests that *passed* — that
money was real, spent regardless of outcome, and simply missing from the
"net" figure. The old field represented "cost of whichever answer was
actually served," which is a real but narrower number than "what did this
system actually spend." ``true_total_cost`` below is the latter, and is the
one that should be treated as the honest bottom line. See docs/phase6_notes.md
for how this was found and CASE_STUDY.md for the corrected writeup.
"""
from __future__ import annotations

from collections import Counter

import db


def compute_stats(rows: list[dict] | None = None) -> dict:
    """Aggregate every logged request into the headline cost/quality numbers.

    Pass ``rows`` (e.g. a filtered subset) to avoid a second DB read; omit it
    to read everything from ``db.fetch_requests()``.
    """
    if rows is None:
        rows = db.fetch_requests()

    total_routed = sum(r["routed_cost"] for r in rows)
    total_baseline = sum(r["baseline_cost"] for r in rows)
    escalated_rows = [r for r in rows if r["escalated"]]
    escalation_cost_delta = sum(r["cost_delta"] for r in escalated_rows)
    verification_cost = sum(r["verification_cost"] for r in rows)

    # "cost of whichever answer was actually served" - real, but doesn't
    # count verification spend on requests that passed the check.
    served_answer_cost = total_routed + escalation_cost_delta
    # "what this system actually spent" - every dollar, checks included.
    true_total_cost = total_routed + verification_cost

    pct_saved_routing = (
        (total_baseline - total_routed) / total_baseline * 100 if total_baseline else 0.0
    )
    pct_saved_served_answer = (
        (total_baseline - served_answer_cost) / total_baseline * 100 if total_baseline else 0.0
    )
    pct_saved_true = (
        (total_baseline - true_total_cost) / total_baseline * 100 if total_baseline else 0.0
    )

    return {
        "n_requests": len(rows),
        "n_escalated": len(escalated_rows),
        "escalation_rate": len(escalated_rows) / len(rows) if rows else 0.0,
        "total_routed_cost": round(total_routed, 6),
        "total_baseline_cost": round(total_baseline, 6),
        "escalation_cost_delta": round(escalation_cost_delta, 6),
        "verification_cost": round(verification_cost, 6),
        "served_answer_cost": round(served_answer_cost, 6),
        "true_total_cost": round(true_total_cost, 6),
        "pct_saved_routing_only": round(pct_saved_routing, 2),
        "pct_saved_served_answer": round(pct_saved_served_answer, 2),
        "pct_saved_true": round(pct_saved_true, 2),
        "routing_distribution": dict(Counter(r["routed_model"] for r in rows)),
        "tier_distribution": dict(Counter(r["tier"] for r in rows)),
    }

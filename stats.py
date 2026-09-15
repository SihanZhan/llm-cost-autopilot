"""Cost-savings summary shared by dashboard/app.py and api/main.py's
GET /v1/stats — one place computes the headline numbers so the API and the
dashboard can't drift apart on what "cost reduction" means.
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
    net_actual = total_routed + escalation_cost_delta

    pct_saved_routing = (
        (total_baseline - total_routed) / total_baseline * 100 if total_baseline else 0.0
    )
    pct_saved_net = (
        (total_baseline - net_actual) / total_baseline * 100 if total_baseline else 0.0
    )

    return {
        "n_requests": len(rows),
        "n_escalated": len(escalated_rows),
        "escalation_rate": len(escalated_rows) / len(rows) if rows else 0.0,
        "total_routed_cost": round(total_routed, 6),
        "total_baseline_cost": round(total_baseline, 6),
        "escalation_cost_delta": round(escalation_cost_delta, 6),
        "verification_cost": round(verification_cost, 6),
        "net_actual_cost": round(net_actual, 6),
        "pct_saved_routing_only": round(pct_saved_routing, 2),
        "pct_saved_net": round(pct_saved_net, 2),
        "routing_distribution": dict(Counter(r["routed_model"] for r in rows)),
        "tier_distribution": dict(Counter(r["tier"] for r in rows)),
    }

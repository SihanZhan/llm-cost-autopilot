"""Phase 6: render the dashboard's key charts to static PNGs for the case
study / README, since this environment has no browser to screenshot the
live Streamlit dashboard with.

Reads db.py directly (same data source as dashboard/app.py and
GET /v1/stats) so these charts are never out of sync with the live numbers -
they're a rendering of real logged data, not illustrations.

Usage:
    python -m eval.report_charts                  # all logged data
    python -m eval.report_charts --since <ISO ts>  # scope to one run
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import db
import stats

OUT_DIR = Path(__file__).parent.parent / "docs" / "images"

# A brand-neutral, colorblind-safe categorical palette.
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def _rows(since: str | None) -> list[dict]:
    rows = db.fetch_requests()
    if since:
        rows = [r for r in rows if r["timestamp"] >= since]
    return rows


def chart_cost_comparison(rows: list[dict], out: Path) -> None:
    s = stats.compute_stats(rows)
    labels = ["All-GPT-4o\nbaseline", "Routed cost\n(before escalation)", "Net cost\n(incl. escalation)"]
    values = [s["total_baseline_cost"], s["total_routed_cost"], s["net_actual_cost"]]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, values, color=[COLORS[3], COLORS[2], COLORS[0]])
    ax.set_ylabel("Total cost (USD)")
    ax.set_title(f"Cost vs. all-GPT-4o baseline ({s['n_requests']} requests)")
    for bar, val in zip(bars, values):
        ax.annotate(f"${val:.4f}", (bar.get_x() + bar.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.annotate(
        f"routing-only: {s['pct_saved_routing_only']:.1f}% saved\n"
        f"net of escalation: {s['pct_saved_net']:.1f}% saved",
        xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#cccccc"),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_routing_distribution(rows: list[dict], out: Path) -> None:
    dist = Counter(r["routed_model"] for r in rows)
    labels, values = zip(*sorted(dist.items(), key=lambda kv: -kv[1]))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, values, color=COLORS[: len(labels)])
    ax.set_ylabel("Requests")
    ax.set_title(f"Routing distribution ({sum(values)} requests)")
    for i, v in enumerate(values):
        ax.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_quality_distribution(rows: list[dict], out: Path) -> None:
    scored = [r["quality_score"] for r in rows if r["quality_score"] is not None]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if scored:
        ax.hist(scored, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0001], color=COLORS[0], edgecolor="white")
    ax.set_xlabel("Quality score (agreement with top-tier model)")
    ax.set_ylabel("Requests")
    unscored = len(rows) - len(scored)
    ax.set_title(f"Quality-score distribution ({len(scored)} scored, {unscored} skipped/unverified)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_escalation_by_tier(rows: list[dict], out: Path) -> None:
    by_tier: dict[int, list[dict]] = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)

    tiers = sorted(by_tier)
    rates = [sum(1 for r in by_tier[t] if r["escalated"]) / len(by_tier[t]) * 100 for t in tiers]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar([f"tier {t}" for t in tiers], rates, color=COLORS[: len(tiers)])
    ax.set_ylabel("Escalation rate (%)")
    ax.set_title("Escalation rate by complexity tier")
    for i, v in enumerate(rates):
        ax.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render static charts from logged request data.")
    parser.add_argument("--since", default=None, help="ISO timestamp - only include rows at/after this")
    args = parser.parse_args()

    rows = _rows(args.since)
    if not rows:
        print("no rows to chart - run eval.demo / eval.seed_dashboard / eval.load_test first")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_cost_comparison(rows, OUT_DIR / "cost_comparison.png")
    chart_routing_distribution(rows, OUT_DIR / "routing_distribution.png")
    chart_quality_distribution(rows, OUT_DIR / "quality_distribution.png")
    chart_escalation_by_tier(rows, OUT_DIR / "escalation_by_tier.png")

    print(f"wrote 4 charts to {OUT_DIR} from {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

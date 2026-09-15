"""Phase 4: Streamlit cost dashboard.

Reads every logged request from db.py's SQLite table (no live API calls —
this only visualizes what eval.pipeline / eval.demo / eval.seed_dashboard
already ran and logged) and renders:

  - the headline metric: cost saved vs. sending every request to GPT-4o
  - daily cost, actual vs. that baseline
  - routing distribution (which model handled what share of requests)
  - quality-score distribution from the verifier
  - escalation rate over time

Usage:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

import db  # noqa: E402  (needs the sys.path insert above)
from stats import compute_stats  # noqa: E402

st.set_page_config(page_title="LLM Cost Autopilot", page_icon="💸", layout="wide")


@st.cache_data(ttl=30)
def load_data() -> pd.DataFrame:
    rows = db.fetch_requests()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    for col in ("verified", "passed", "escalated"):
        df[col] = df[col].astype("boolean")
    return df


def main() -> None:
    st.title("💸 LLM Cost Autopilot")
    st.caption("Cost and quality dashboard — routing decisions vs. an all-GPT-4o baseline")

    df = load_data()
    if df.empty:
        st.warning(
            "No requests logged yet. Run `python -m eval.demo` or "
            "`python -m eval.seed_dashboard` to generate data, then reload."
        )
        return

    # --- Headline metric (same computation the API's /v1/stats returns) ---
    s = compute_stats(df.to_dict("records"))

    st.markdown("### Cost saved vs. sending everything to GPT-4o")
    m1, m2, m3 = st.columns(3)
    m1.metric("Routing-only reduction", f"{s['pct_saved_routing_only']:.1f}%", help="Routed cost vs. baseline. Ignores verification entirely - pretends checking is free.")
    m2.metric("True net reduction", f"{s['pct_saved_true']:.1f}%", help="Every dollar spent - routed cost + ALL verification calls (pass or fail) - vs. baseline. The honest bottom line.")
    m3.metric("All-GPT-4o baseline", f"${s['total_baseline_cost']:.4f}")

    if s["pct_saved_true"] < 0:
        st.error(
            f"This system currently costs MORE than the baseline once every dollar is counted: "
            f"${s['true_total_cost']:.4f} spent vs. ${s['total_baseline_cost']:.4f} baseline "
            f"({s['pct_saved_true']:.1f}%). Verification checks every non-top-tier request "
            f"(${s['verification_cost']:.4f} total), not just the ones that fail - that cost is "
            f"paid whether or not the check catches anything. See docs/phase6_notes.md and "
            f"CASE_STUDY.md for the full breakdown."
        )
    elif s["pct_saved_true"] < s["pct_saved_routing_only"] * 0.5:
        st.warning(
            f"Verification overhead is eating most of the routing savings on this data: "
            f"a {s['pct_saved_routing_only']:.1f}% routing-only reduction becomes only "
            f"{s['pct_saved_true']:.1f}% once every verification dollar is counted. "
            f"See docs/phase4_notes.md — this traces back to the same overly strict "
            f"'general' use-case check flagged in docs/phase3_notes.md."
        )

    st.caption(
        f"Routed cost: ${s['total_routed_cost']:.4f}  •  verification cost (all checks, pass or fail): ${s['verification_cost']:.4f}  •  "
        f"escalation cost delta: ${s['escalation_cost_delta']:.4f}  •  true total spent: ${s['true_total_cost']:.4f}  •  "
        f"{s['n_requests']} requests across {df['date'].nunique()} day(s) of logged data"
    )

    st.divider()

    col1, col2 = st.columns(2)

    # --- Daily cost: actual vs. baseline ----------------------------------
    with col1:
        st.subheader("Daily cost: actual vs. baseline")
        daily = df.groupby("date")[["routed_cost", "baseline_cost"]].sum()
        daily = daily.rename(columns={"routed_cost": "actual", "baseline_cost": "all-GPT-4o baseline"})
        st.bar_chart(daily)
        if df["date"].nunique() == 1:
            st.caption("Only one day of logged data so far — this fills in as more runs accumulate.")

    # --- Routing distribution ----------------------------------------------
    with col2:
        st.subheader("Routing distribution")
        dist = df["routed_model"].value_counts()
        st.bar_chart(dist)

    col3, col4 = st.columns(2)

    # --- Quality-score distribution ------------------------------------
    with col3:
        st.subheader("Quality-score distribution")
        scored = df["quality_score"].dropna()
        if scored.empty:
            st.info("No scored requests yet (all skipped or unverified).")
        else:
            bins = pd.cut(scored, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0001], right=False)
            counts = bins.value_counts().sort_index()
            counts.index = [str(i) for i in counts.index]
            st.bar_chart(counts)
            unverified = int((~df["verified"]).sum())
            st.caption(f"{len(scored)} verified requests scored, {unverified} skipped or unverified")

    # --- Escalation rate over time ------------------------------------
    with col4:
        st.subheader("Escalation rate over time")
        verified = df[df["verified"]]
        daily_esc = verified.groupby("date")["escalated"].mean() * 100
        if daily_esc.empty:
            st.info("No verified requests yet.")
        else:
            st.line_chart(daily_esc.rename("escalation rate (%)"))

    st.divider()
    st.subheader("Raw requests")
    st.dataframe(
        df[
            [
                "timestamp", "prompt_hash", "use_case", "tier", "routed_model",
                "routed_cost", "routed_latency", "baseline_cost", "quality_score",
                "passed", "escalated", "final_model", "cost_delta",
            ]
        ].sort_values("timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()

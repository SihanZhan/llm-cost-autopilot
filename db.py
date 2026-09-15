"""Phase 4: the per-request audit log the dashboard reads from.

One row per verified request. Prompts are stored as a hash, not full text —
this table is the privacy-conscious "what happened and what did it cost"
record; the full prompt text (needed for classifier retraining) already
lives in eval/logs/verification_log.jsonl, which is a separate, gitignored,
internal log. ``baseline_cost`` is what the request would have cost had it
gone straight to BASELINE_MODEL_ID (gpt-4o) instead of the routed model,
computed from the routed call's own token counts — an approximation (token
counts aren't perfectly comparable across tokenizers) but a real, defensible
one, and it's what makes the headline "cost saved vs. all-GPT-4o" number
possible.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from models import get_model

DB_FILE = Path(__file__).parent / "autopilot.db"
BASELINE_MODEL_ID = "gpt-4o"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    use_case TEXT NOT NULL,
    tier INTEGER NOT NULL,
    routed_model TEXT NOT NULL,
    routed_cost REAL NOT NULL,
    routed_latency REAL NOT NULL,
    baseline_cost REAL NOT NULL,
    verified INTEGER NOT NULL,
    quality_score REAL,
    passed INTEGER,
    escalated INTEGER NOT NULL,
    final_model TEXT NOT NULL,
    cost_delta REAL NOT NULL,
    verification_cost REAL NOT NULL
);
"""


def prompt_hash(prompt: str) -> str:
    """Short, stable hash of a prompt for dedup/audit without storing raw text."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def init_db(path: Path = DB_FILE) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(_SCHEMA)


def baseline_cost_for(input_tokens: int, output_tokens: int) -> float:
    """What this request would have cost on BASELINE_MODEL_ID's rates."""
    baseline = get_model(BASELINE_MODEL_ID)
    return (
        input_tokens * baseline.cost_per_input_token
        + output_tokens * baseline.cost_per_output_token
    )


def log_request(row: dict, path: Path = DB_FILE) -> None:
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO requests (
                timestamp, prompt_hash, use_case, tier, routed_model,
                routed_cost, routed_latency, baseline_cost, verified,
                quality_score, passed, escalated, final_model, cost_delta,
                verification_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["timestamp"], row["prompt_hash"], row["use_case"], row["tier"],
                row["routed_model"], row["routed_cost"], row["routed_latency"],
                row["baseline_cost"], int(row["verified"]), row["quality_score"],
                None if row["passed"] is None else int(row["passed"]),
                int(row["escalated"]), row["final_model"], row["cost_delta"],
                row["verification_cost"],
            ),
        )


def fetch_requests(path: Path = DB_FILE) -> list[dict]:
    """Return every logged request as a list of plain dicts, oldest first."""
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM requests ORDER BY timestamp").fetchall()
    return [dict(r) for r in rows]

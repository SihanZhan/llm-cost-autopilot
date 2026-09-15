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

CREATE TABLE IF NOT EXISTS verification_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    prompt TEXT NOT NULL,
    tier INTEGER NOT NULL,
    routed_model_id TEXT NOT NULL,
    routed_output_text TEXT NOT NULL,
    routed_input_tokens INTEGER NOT NULL,
    routed_output_tokens INTEGER NOT NULL,
    routed_latency REAL NOT NULL,
    routed_cost REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    claimed_at TEXT,
    error TEXT
);
"""


def prompt_hash(prompt: str) -> str:
    """Short, stable hash of a prompt for dedup/audit without storing raw text."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def init_db(path: Path = DB_FILE) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)


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


# --- verification job queue --------------------------------------------
#
# What decouples verification into a real separate worker process (matching
# the brief's "API service + a background worker for async verification"
# architecture) instead of an in-process thread pool: the API enqueues a
# row here and returns immediately; eval.verification_worker is a standalone
# process that polls this table and does the actual work.


def enqueue_verification_job(
    prompt: str, tier: int, routed_response, path: Path = DB_FILE
) -> int:
    """Queue a verification job for ``routed_response`` and return its id."""
    from datetime import datetime, timezone

    init_db(path)
    with sqlite3.connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO verification_jobs (
                created_at, prompt, tier, routed_model_id, routed_output_text,
                routed_input_tokens, routed_output_tokens, routed_latency, routed_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(), prompt, tier,
                routed_response.model_id, routed_response.output_text,
                routed_response.input_tokens, routed_response.output_tokens,
                routed_response.latency, routed_response.cost,
            ),
        )
        return cur.lastrowid


def claim_next_verification_job(path: Path = DB_FILE) -> dict | None:
    """Atomically claim one pending job (mark it 'claimed' and return it), or
    None if the queue is empty. Uses a single UPDATE...RETURNING so two
    workers polling concurrently can't both claim the same row."""
    from datetime import datetime, timezone

    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            UPDATE verification_jobs
            SET status = 'claimed', claimed_at = ?
            WHERE id = (
                SELECT id FROM verification_jobs
                WHERE status = 'pending'
                ORDER BY id LIMIT 1
            )
            RETURNING *
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def mark_job_done(job_id: int, path: Path = DB_FILE) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE verification_jobs SET status = 'done' WHERE id = ?", (job_id,))


def mark_job_error(job_id: int, error: str, path: Path = DB_FILE) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE verification_jobs SET status = 'error', error = ? WHERE id = ?",
            (error, job_id),
        )


def pending_job_count(path: Path = DB_FILE) -> int:
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM verification_jobs WHERE status IN ('pending', 'claimed')"
        ).fetchone()
    return count

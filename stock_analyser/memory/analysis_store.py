from __future__ import annotations

import json
import sqlite3
import datetime
from pathlib import Path

_DB_PATH = Path(__file__).parent / "analyses.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id          INTEGER PRIMARY KEY,
                user_id     TEXT    NOT NULL,
                ticker      TEXT    NOT NULL,
                date        TEXT    NOT NULL,
                recommendation TEXT,
                score       REAL,
                signals_json TEXT,
                created_at  TEXT    NOT NULL
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_ticker_date
            ON analyses (user_id, ticker, date)
        """)


_init_db()


def save_analysis(user_id: str, state: dict) -> None:
    signals: dict[str, dict] = {}
    for field in (
        "fundamentals_signal", "technical_signal",
        "sentiment_signal", "macro_signal", "risk_signal",
    ):
        sig = state.get(field)
        if sig is not None:
            signals[field] = sig.model_dump() if hasattr(sig, "model_dump") else sig

    now = datetime.datetime.now(datetime.timezone.utc)
    with _conn() as c:
        c.execute(
            """INSERT INTO analyses
               (user_id, ticker, date, recommendation, score, signals_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                (state.get("ticker") or "").upper(),
                now.strftime("%Y-%m-%d"),
                state.get("recommendation"),
                state.get("score"),
                json.dumps(signals, default=str),
                now.isoformat(),
            ),
        )


def get_recent_analyses(
    ticker: str,
    user_id: str | None = None,
    limit: int = 3,
) -> list[dict]:
    with _conn() as c:
        if user_id:
            rows = c.execute(
                """SELECT * FROM analyses
                   WHERE ticker = ? AND user_id = ?
                   ORDER BY date DESC, id DESC LIMIT ?""",
                (ticker.upper(), user_id, limit),
            ).fetchall()
        else:
            rows = c.execute(
                """SELECT * FROM analyses
                   WHERE ticker = ?
                   ORDER BY date DESC, id DESC LIMIT ?""",
                (ticker.upper(), limit),
            ).fetchall()
    return [dict(r) for r in rows]

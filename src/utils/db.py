from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_DB_PATH = "data/app.db"


def _ensure_parent_dir(db_path: str) -> None:
    p = Path(db_path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(str(text).split())


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize the SQLite database with required tables and indexes."""
    _ensure_parent_dir(db_path)
    
    with _connect(db_path) as conn:
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                lang TEXT CHECK(lang IN ('hi','en','hi-en')),
                predicted_label INTEGER CHECK(predicted_label IN (0,1,2)),
                score REAL NOT NULL,
                model_name TEXT NOT NULL,
                latency_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                macro_f1 REAL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                latency_p95_ms REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                lang TEXT,
                true_label INTEGER,
                source TEXT
            )
        """)
        
        # Add missing columns
        try:
            conn.execute("ALTER TABLE predictions ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except sqlite3.OperationalError:
            pass
        
        try:
            conn.execute("ALTER TABLE runs ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except sqlite3.OperationalError:
            pass
        
        # Create indexes with error handling
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC)")
        except sqlite3.OperationalError:
            pass
        
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(predicted_label)")
        except sqlite3.OperationalError:
            pass
        
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC)")
        except sqlite3.OperationalError:
            pass
        
        conn.commit()


def insert_prediction(
    text: str,
    lang: str,
    predicted_label: int,
    score: float,
    model_name: str,
    latency_ms: int,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Insert a prediction row and return its id."""
    if not isinstance(text, str) or text == "":
        raise ValueError("text must be a non-empty string")
    if not isinstance(lang, str) or lang == "":
        raise ValueError("lang must be a non-empty string")
    if not isinstance(predicted_label, int):
        raise ValueError("predicted_label must be int")
    if not isinstance(score, (int, float)):
        raise ValueError("score must be a number")
    if not isinstance(model_name, str) or model_name == "":
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(latency_ms, int) or latency_ms < 0:
        raise ValueError("latency_ms must be a non-negative int")

    _ensure_parent_dir(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (text, lang, predicted_label, score, model_name, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (text, lang, int(predicted_label), float(score), model_name, int(latency_ms)),
        )
        return int(cur.lastrowid)


def _rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def fetch_history(limit: int = 100, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Fetch recent prediction history as a list of dicts."""
    n = max(1, int(limit))
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT id, text, lang, predicted_label, score, model_name, latency_ms, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (n,),
        )
        return _rows_to_dicts(cur.fetchall())


def fetch_runs(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Fetch recent runs metadata as a list of dicts."""
    n = max(1, int(limit))
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT id, model_name, params, dataset, started_at, ended_at, duration_ms, samples, notes
            FROM runs
            ORDER BY started_at DESC, id DESC
            LIMIT ?
            """,
            (n,),
        )
        return _rows_to_dicts(cur.fetchall())


def delete_all_predictions(db_path: str = DEFAULT_DB_PATH) -> int:
    """Delete all prediction records and reset ID sequence. Returns count of deleted rows."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM predictions")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
        conn.commit()
        return cur.rowcount


def get_prediction_summary(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Aggregate prediction metrics for dashboard visualizations."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, text, lang, predicted_label, score, model_name, latency_ms, created_at
            FROM predictions
            ORDER BY datetime(created_at) ASC
            """
        ).fetchall()

    items = _rows_to_dicts(rows)
    total = len(items)
    total_words = sum(_word_count(item.get("text", "")) for item in items)

    by_day = defaultdict(lambda: {"searches": 0, "words": 0})
    label_counter: Counter[int] = Counter()
    for item in items:
        created_at = item.get("created_at") or ""
        day = created_at[:10] if created_at else "N/A"
        words = _word_count(item.get("text", ""))
        by_day[day]["searches"] += 1
        by_day[day]["words"] += words
        label_counter[int(item.get("predicted_label", 0))] += 1

    timeline = [
        {"date": day, "searches": stats["searches"], "words": stats["words"]}
        for day, stats in sorted(by_day.items())
        if day != "N/A"
    ]

    return {
        "total_searches": total,
        "total_words": total_words,
        "timeline": timeline,
        "label_counts": dict(label_counter),
        "recent": list(reversed(items[-10:])) if items else [],
    }


def reset_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """Drop and recreate all tables to reset IDs from 1."""
    _ensure_parent_dir(db_path)
    with _connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS predictions")
        conn.execute("DROP TABLE IF EXISTS runs")
        conn.execute("DROP TABLE IF EXISTS annotations")
        conn.commit()
    init_db(db_path)


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at: {DEFAULT_DB_PATH}")

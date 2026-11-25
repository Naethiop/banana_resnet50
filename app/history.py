"""FR9: prediction history in SQLite. Stores text only: when, category, confidence and two flags.
Never stores photographs, file names or anything about the person."""
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

SCHEMA = """CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, category TEXT NOT NULL,
    confidence REAL NOT NULL, low_confidence INTEGER, unsuitable INTEGER)"""


def init(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as con:
        con.execute(SCHEMA)
        con.commit()


def add(path, result):
    with closing(sqlite3.connect(path)) as con:
        con.execute("INSERT INTO predictions (created_at, category, confidence, low_confidence, unsuitable) VALUES (?,?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), result["label"], float(result["confidence"]),
                     None if result.get("low_confidence") is None else int(bool(result["low_confidence"])),
                     None if result.get("unsuitable") is None else int(bool(result["unsuitable"]))))
        con.commit()


def recent(path, limit=100):
    with closing(sqlite3.connect(path)) as con:
        rows = con.execute("SELECT created_at, category, confidence, low_confidence, unsuitable FROM predictions "
                           "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"when": r[0], "category": r[1], "confidence": r[2], "low_confidence": bool(r[3]), "unsuitable": bool(r[4])}
            for r in rows]


def count(path):
    with closing(sqlite3.connect(path)) as con:
        return con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


def clear(path):
    with closing(sqlite3.connect(path)) as con:
        con.execute("DELETE FROM predictions")
        con.commit()

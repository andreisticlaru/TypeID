"""SQLite-backed gallery store.

Per CLAUDE.md: at this project's scale a plain SQLite table is sufficient --
no ORM, no ANN index (FAISS/Annoy would only matter at thousands+ gallery
entries). Each row is one enrolled person's pooled embedding.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "gallery.db"


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create the gallery table if it doesn't already exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gallery (
                person_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding TEXT NOT NULL,
                enrolled_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def entry_exists(person_id: str) -> bool:
    """Whether this person_id is already enrolled. Enrolling over one destroys a template."""
    with _connect() as conn:
        return conn.execute("SELECT 1 FROM gallery WHERE person_id = ?", (person_id,)).fetchone() is not None


def add_entry(person_id: str, name: str, embedding: list[float]) -> dict:
    """Insert (or replace) a gallery entry and return the stored record."""
    enrolled_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO gallery (person_id, name, embedding, enrolled_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(person_id) DO UPDATE SET
                name = excluded.name,
                embedding = excluded.embedding,
                enrolled_at = excluded.enrolled_at
            """,
            (person_id, name, json.dumps(embedding), enrolled_at),
        )
        conn.commit()
    return {
        "person_id": person_id,
        "name": name,
        "embedding": embedding,
        "enrolled_at": enrolled_at,
    }


def get_entry(person_id: str) -> dict | None:
    """One gallery entry by id, or None. Verification compares against this one template."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT person_id, name, embedding, enrolled_at FROM gallery WHERE person_id = ?",
            (person_id,),
        ).fetchone()
    if row is None:
        return None
    person_id, name, embedding, enrolled_at = row
    return {
        "person_id": person_id,
        "name": name,
        "embedding": json.loads(embedding),
        "enrolled_at": enrolled_at,
    }


def get_all_entries() -> list[dict]:
    """Return every gallery entry with its embedding decoded back to a list of floats."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT person_id, name, embedding, enrolled_at FROM gallery"
        ).fetchall()
    return [
        {
            "person_id": person_id,
            "name": name,
            "embedding": json.loads(embedding),
            "enrolled_at": enrolled_at,
        }
        for person_id, name, embedding, enrolled_at in rows
    ]

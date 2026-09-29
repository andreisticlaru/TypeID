"""SQLite-backed gallery store.

Per CLAUDE.md: at this project's scale a plain SQLite table is sufficient --
no ORM, no ANN index (FAISS/Annoy would only matter at thousands+ gallery
entries). Each row is one enrolled person's pooled embedding.
"""

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "gallery.db"
# Committed starting point: the 500 held-out Aalto background people only (seed_aalto_gallery.py).
# gallery.db is gitignored, so each checkout enrolls into its own copy and this file never changes.
POPULATED_PATH = DB_PATH.with_name("gallery-populated.db")


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Start from the committed Aalto background on first run, then ensure the table exists."""
    if not DB_PATH.exists() and POPULATED_PATH.exists():
        shutil.copyfile(POPULATED_PATH, DB_PATH)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gallery (
                person_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding TEXT NOT NULL,
                enroll_prompts INTEGER NOT NULL,
                enrolled_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def entry_exists(person_id: str) -> bool:
    """Whether this person_id is already enrolled. Enrolling over one destroys a template."""
    with _connect() as conn:
        return conn.execute("SELECT 1 FROM gallery WHERE person_id = ?", (person_id,)).fetchone() is not None


def add_entry(person_id: str, name: str, embedding: list[float], enroll_prompts: int) -> dict:
    """Insert (or replace) a gallery entry and return the stored record.

    `enroll_prompts` is how many prompts were pooled into the template. Decision thresholds
    depend on it, so a gallery mixing 1-, 5- and 10-prompt people can't share one threshold.
    """
    enrolled_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO gallery (person_id, name, embedding, enroll_prompts, enrolled_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(person_id) DO UPDATE SET
                name = excluded.name,
                embedding = excluded.embedding,
                enroll_prompts = excluded.enroll_prompts,
                enrolled_at = excluded.enrolled_at
            """,
            (person_id, name, json.dumps(embedding), enroll_prompts, enrolled_at),
        )
        conn.commit()
    return {
        "person_id": person_id,
        "name": name,
        "embedding": embedding,
        "enroll_prompts": enroll_prompts,
        "enrolled_at": enrolled_at,
    }


def get_entry(person_id: str) -> dict | None:
    """One gallery entry by id, or None. Verification compares against this one template."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT person_id, name, embedding, enroll_prompts, enrolled_at FROM gallery WHERE person_id = ?",
            (person_id,),
        ).fetchone()
    if row is None:
        return None
    person_id, name, embedding, enroll_prompts, enrolled_at = row
    return {
        "person_id": person_id,
        "name": name,
        "embedding": json.loads(embedding),
        "enroll_prompts": enroll_prompts,
        "enrolled_at": enrolled_at,
    }


def get_all_entries() -> list[dict]:
    """Return every gallery entry with its embedding decoded back to a list of floats."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT person_id, name, embedding, enroll_prompts, enrolled_at FROM gallery"
        ).fetchall()
    return [
        {
            "person_id": person_id,
            "name": name,
            "embedding": json.loads(embedding),
            "enroll_prompts": enroll_prompts,
            "enrolled_at": enrolled_at,
        }
        for person_id, name, embedding, enroll_prompts, enrolled_at in rows
    ]

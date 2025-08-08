import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from .config import get_paths


SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  num_documents INTEGER,
  num_tokens INTEGER,
  languages TEXT,
  hash TEXT,
  output_dir TEXT
);

CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  finished_at TIMESTAMP,
  status TEXT,
  log_path TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    db_path = get_paths().db_path
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA)
    return conn


def insert_dataset(
    name: str,
    num_documents: int,
    num_tokens: int,
    languages: Iterable[str],
    dataset_hash: str,
    output_dir: Path,
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO datasets (name, num_documents, num_tokens, languages, hash, output_dir) VALUES (?, ?, ?, ?, ?, ?)",
        (name, num_documents, num_tokens, ",".join(sorted(languages)), dataset_hash, str(output_dir)),
    )
    conn.commit()
    rowid = cur.lastrowid
    conn.close()
    return int(rowid)

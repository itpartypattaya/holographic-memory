"""Store fixes from the review of 0.6.2–0.9.0: a failed COMMIT no longer wedges the connection, a forget
does not wait on a foreign reader, credentials are never stored, short CJK sentences are extracted, a missing space
after a full stop still splits sentences, user_id_alt names the user store, revert explains itself after a revert, and
a file written by a newer release is not opened for writing."""

from __future__ import annotations

import sqlite3
import time

import pytest

pytest.importorskip("numpy")

from holomem.engine.store import MemoryStore

DIM = 64


def _journal(store, mode: str) -> None:
    """Pin the journal mode for the test: sqlite_util picks WAL or DELETE by the SQLite it runs on."""
    assert store._conn.execute(f"PRAGMA journal_mode={mode}").fetchone()[0].lower() == mode.lower()


def _reader(db) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Another process in the middle of a long query: a statement left open holds its read lock."""
    other = sqlite3.connect(str(db), isolation_level=None)
    cur = other.execute("SELECT fact_id, content FROM facts")
    cur.fetchone()
    return other, cur


# --- a failed COMMIT ------------------------------------------------------------------------------------------------


def test_a_failed_commit_is_rolled_back_and_the_next_write_works(tmp_path):
    """Rollback-journal mode: a foreign SHARED lock makes COMMIT fail with "database is locked". The transaction
    must not stay open, or every later write would join it and nothing would ever be committed again."""
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    _journal(store, "DELETE")
    for i in range(5):  # several rows: Python's cursor steps one row ahead, and a finished statement holds no lock
        store.add_fact(f"Alice Smith lives in Paris, flat {i}")
    other, cur = _reader(db)
    store._conn.execute("PRAGMA busy_timeout = 200")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            store.add_fact("Bob Brown lives in Rome")
        assert not store._conn.in_transaction  # rolled back, not dangling
    finally:
        cur.close()
        other.close()
    fid = store.add_fact("Bob Brown lives in Rome")
    assert fid > 0 and store.count_facts() == 6
    assert sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 6  # committed for real
    store.close()


# --- forget and a foreign reader ------------------------------------------------------------------------------------


def test_forget_does_not_wait_for_a_foreign_reader(tmp_path):
    """WAL: the reader blocks only the checkpoint. The forget must report `pending` at once (every store
    operation queues behind it), not after the connection's 10 s busy timeout, and leave the timeout as it was."""
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    _journal(store, "WAL")
    for i in range(50):
        store.add_fact(f"Person{i} Surname{i} lives in city number {i}")
    timeout = store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    other, cur = _reader(db)
    try:
        started = time.monotonic()
        result = store.forget_fact(50)
        elapsed = time.monotonic() - started
    finally:
        cur.close()
        other.close()
    assert result["forgotten"] is True and result["scrub"] == "pending" and elapsed < 3.0
    assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == timeout
    assert store.read("SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH 'Surname49'")[0][0] == 0
    again = store.forget_fact(50)
    assert again["already"] is True and again["scrub"] == "done"
    assert store.read("SELECT COUNT(*) FROM meta WHERE key = 'scrub_pending'")[0][0] == 0
    store.close()

"""0.19.0: `fact_store merge` — two facts that say the same thing become one, without losing either history.
plans/2026-10-25-fact-merge.md (v3) is the contract these tests check.

Requirements, in the user's words:
- a duplicate merged into the fact that stays leaves search, recall and the list; its versions, sources and
  confirmations stay under its own id, and `history` of the keep shows them;
- the text of a merged copy does not come back as a new fact: `add`, auto-extraction, `change`, `update` and a resume
  answer with the keep;
- what held on a date before the merge is still answered, and the copy stops "holding" on the day it was merged;
- the keep keeps its own trust; the group's counts add up once; keywords are joined or replaced as asked;
- one person confirming the same claim twice (once on the copy) raises trust once;
- `forget` of the keep forgets the copies too, a tombstone each; `forget` of a copy leaves the keep as it is;
- an error anywhere in a merge leaves nothing written; the file keeps working after a failed commit;
- a scheduled job merges only what it stored in this session (contract addition of 2026-10-26);
- a file without merges keeps history schema 1; the first merge raises it to 2."""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("numpy")

from holomem.engine.history import HistoryMixin
from holomem.engine.store import MemoryStore

DIM = 256


class Clock:
    def __init__(self, day: str):
        self.day = day


@pytest.fixture
def clock(monkeypatch):
    c = Clock("2026-10-12")
    monkeypatch.setattr(HistoryMixin, "_now", staticmethod(lambda: f"{c.day} 12:00:00"))
    monkeypatch.setattr(HistoryMixin, "_today", lambda self: c.day)
    return c


def _snapshot(store) -> dict:
    """Everything a refused merge must leave as it was."""
    tables = ("facts", "fact_versions", "fact_entities", "entities", "fact_confirmations", "fact_merges", "meta",
              "fact_embeddings", "memory_banks", "entity_aliases", "forgotten")
    return {t: [tuple(r) for r in store.read(f"SELECT * FROM {t}")] for t in tables}


def _raw_bytes(db_path) -> bytes:
    data = b""
    for suffix in ("", "-wal"):
        try:
            with open(f"{db_path}{suffix}", "rb") as fh:
                data += fh.read()
        except FileNotFoundError:
            pass
    return data.lower()


# --- forget ------------------------------------------------------------------------------------------------------------------


def test_forget_of_the_keep_with_a_blocked_scrub_is_retried(tmp_path, clock):
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    assert store._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
    for i in range(30):
        store.add_fact(f"Person{i} Surname{i} lives in city number {i}")
    keep = store.add_fact("Alice Smith lives in Paris")
    copy = store.add_fact("Alice Smith is living in Paris zorblat")
    store.merge_facts(keep, [copy])
    other = sqlite3.connect(str(db), isolation_level=None)
    cur = other.execute("SELECT fact_id, content FROM facts")
    cur.fetchone()
    try:
        result = store.forget_fact(keep)
    finally:
        cur.close()
        other.close()
    assert result["forgotten"] is True and result["scrub"] == "pending" and result["merged_copies"] == [copy]
    assert store.read("SELECT COUNT(*) FROM forgotten")[0][0] == 2
    again = store.forget_fact(keep)
    assert again["already"] is True and again["scrub"] == "done"
    assert b"zorblat" not in _raw_bytes(db)
    store.close()


# --- a failure leaves nothing written ------------------------------------------------------------------------------------


def test_a_failed_commit_leaves_nothing_and_the_next_merge_works(tmp_path, clock):
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    assert store._conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0].lower() == "delete"
    for i in range(5):
        store.add_fact(f"Alice Smith lives in Paris, flat {i}")
    keep = store.add_fact("Alice Smith lives in Paris")
    copy = store.add_fact("Alice Smith is living in Paris zorblat")
    before = _snapshot(store)
    other = sqlite3.connect(str(db), isolation_level=None)
    cur = other.execute("SELECT fact_id, content FROM facts")
    cur.fetchone()
    store._conn.execute("PRAGMA busy_timeout = 200")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            store.merge_facts(keep, [copy])
        assert not store._conn.in_transaction
    finally:
        cur.close()
        other.close()
    assert _snapshot(store) == before
    assert sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM fact_merges").fetchone()[0] == 0
    assert store.merge_facts(keep, [copy])["merged"] == [copy]
    assert sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM fact_merges").fetchone()[0] == 1
    store.close()

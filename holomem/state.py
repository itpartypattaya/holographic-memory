"""The bridge's own bookkeeping, kept apart from the fact store: ``$HOLOMEM_HOME/state.db``.

A hook is a new process at every prompt, so what the plugin keeps in memory between turns lives here: the turn
number of each session and which revision of which fact was recalled at which turn (the repeat window,
``prefetch_repeat_turns``). A revision is (fact id, 12 hex of the text's sha1), as in the plugin: a reworded fact
is new. Sessions untouched for 14 days, and all but the 200 most recent, are dropped.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (session TEXT PRIMARY KEY, turn INTEGER NOT NULL, touched REAL NOT NULL);
CREATE TABLE IF NOT EXISTS recalled (
    session TEXT NOT NULL, fact_id INTEGER NOT NULL, rev TEXT NOT NULL, turn INTEGER NOT NULL,
    PRIMARY KEY (session, fact_id, rev));
"""
_KEEP_DAYS, _KEEP_SESSIONS, _KEEP_FACTS = 14, 200, 512


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), timeout=5.0, isolation_level=None)
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def next_turn(self, session: str) -> int:
        """This prompt's turn number in *session* (1, 2, ...), counted across processes."""
        now = time.time()
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute("SELECT turn FROM turns WHERE session = ?", (session,)).fetchone()
            turn = (row[0] if row else 0) + 1
            self._conn.execute("INSERT INTO turns (session, turn, touched) VALUES (?, ?, ?) ON CONFLICT(session) DO "
                               "UPDATE SET turn = excluded.turn, touched = excluded.touched", (session, turn, now))
            if turn == 1:
                self._prune(now)
        return turn

    def seen(self, session: str) -> dict[tuple[int, str], int]:
        rows = self._conn.execute("SELECT fact_id, rev, turn FROM recalled WHERE session = ?", (session,))
        return {(fact_id, rev): turn for fact_id, rev, turn in rows}

    def remember(self, session: str, revisions, turn: int) -> None:
        rows = [(session, fact_id, rev, turn) for fact_id, rev in revisions]
        if not rows:
            return
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.executemany("INSERT INTO recalled (session, fact_id, rev, turn) VALUES (?, ?, ?, ?) "
                                   "ON CONFLICT(session, fact_id, rev) DO UPDATE SET turn = excluded.turn", rows)
            self._conn.execute("DELETE FROM recalled WHERE session = ? AND rowid NOT IN (SELECT rowid FROM recalled "
                               "WHERE session = ? ORDER BY turn DESC LIMIT ?)", (session, session, _KEEP_FACTS))

    def forget_fact(self, fact_id: int) -> None:
        """A forgotten fact leaves no trace here either (its text hash)."""
        with self._conn:
            self._conn.execute("DELETE FROM recalled WHERE fact_id = ?", (fact_id,))

    def _prune(self, now: float) -> None:
        cutoff = now - _KEEP_DAYS * 86400
        self._conn.execute("DELETE FROM turns WHERE touched < ? OR session NOT IN (SELECT session FROM turns "
                           "ORDER BY touched DESC LIMIT ?)", (cutoff, _KEEP_SESSIONS))
        self._conn.execute("DELETE FROM recalled WHERE session NOT IN (SELECT session FROM turns)")

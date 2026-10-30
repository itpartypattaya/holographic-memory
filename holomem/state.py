"""The bridge's own bookkeeping, kept apart from the fact store: ``$HOLOMEM_HOME/state.db``.

A hook is a new process at every prompt, so what a long-running process would keep in memory between turns lives here: the turn
number of each session and which revision of which fact was recalled at which turn (the repeat window,
``prefetch_repeat_turns``). A revision is (fact id, 12 hex of the text's sha1): a reworded fact
is new. Sessions untouched for 14 days, and all but the 200 most recent, are dropped.

The pause after a failed call (``backoff``: until when, how long, the kind of failure — never an address or a text)
lives here too: each hook is a new process, and an endpoint's own backoff would end with it. Search by meaning pauses
for a fixed time (``pause``); rescoring for 60 s doubling to 900 s until an answer clears it (``fail`` / ``clear``),
keyed by the endpoint's fingerprint (rescore.py), with the last speeds it showed (``rerank_speed``). These writes wait
at most 2 x 50 ms like the journal's: a busy state.db loses a pause, never the recall. (A database from before held
the embeddings pause in ``pauses``: its rows move here when it is opened.)

A recall's own writes — the turn number before it, the repeat window after it — wait only as long as the caller
gives them (``wait``, from what its deadline leaves): another process holding state.db for longer costs this turn its
repeat-window bookkeeping, never its block. Without the lock the turn number is read, not counted.

The task-outcome journal (journal.py) links each attempt id to the session, client and database that recorded it
(``attempts``: ids and names only, never a fact's text), kept for the journal's 30 days. Writing a link waits at
most 2 x 50 ms for another process: it is optional, the recall is not.
"""
from __future__ import annotations

import contextlib
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (session TEXT PRIMARY KEY, turn INTEGER NOT NULL, touched REAL NOT NULL);
CREATE TABLE IF NOT EXISTS recalled (
    session TEXT NOT NULL, fact_id INTEGER NOT NULL, rev TEXT NOT NULL, turn INTEGER NOT NULL,
    PRIMARY KEY (session, fact_id, rev));
CREATE TABLE IF NOT EXISTS backoff (
    name TEXT PRIMARY KEY, until REAL NOT NULL, pause REAL NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS rerank_speed (name TEXT NOT NULL, sample REAL NOT NULL, at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS rerank_speed_name ON rerank_speed (name, at);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY, session TEXT NOT NULL, client TEXT NOT NULL, db TEXT NOT NULL, created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS attempts_db ON attempts (db, created);
"""
_KEEP_DAYS, _KEEP_SESSIONS, _KEEP_FACTS = 14, 200, 512
_ATTEMPT_DAYS = 30  # the engine's journal TTL (experience.TTL): a link outlives no receipt
_BUSY_MS, _LINK_BUSY_MS, _LINK_TRIES = 5000, 50, 2  # the connection's wait / a journal write's, as the engine's receipt
PAUSE_FIRST, PAUSE_MAX = 60.0, 900.0  # rescoring's backoff, as the engine's Rescorer (rerank.PAUSE_FIRST / PAUSE_MAX)
SPEED_SAMPLES = 5  # as the engine's Rescorer (rerank._SPEED_SAMPLES)
_BACKOFF_COLUMNS = ("until", "pause", "error")


def _busy(exc: sqlite3.OperationalError) -> bool:
    code = getattr(exc, "sqlite_errorcode", None)
    return (code & 0xff) == sqlite3.SQLITE_BUSY if code is not None else "locked" in str(exc)


@contextlib.contextmanager
def _waiting(conn: sqlite3.Connection, wait: float | None):
    """The connection's busy timeout set to *wait* seconds for a block (None: the default 5 s)."""
    if wait is None:
        yield
        return
    conn.execute(f"PRAGMA busy_timeout = {max(0, int(min(wait, _BUSY_MS / 1000) * 1000))}")
    try:
        yield
    finally:
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_MS}")


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread off: the MCP server uses it from its main thread and its vectors thread, one at a time
        # (under the server's lock)
        self._conn = sqlite3.connect(str(path), timeout=_BUSY_MS / 1000, isolation_level=None, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._move_pauses()

    def _move_pauses(self) -> None:
        """The embeddings pause of a state.db written before ``backoff``: moved over. The old table stays (empty): a
        server started before the update still writes there, and its rows come over at the next open."""
        if not self._conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pauses'").fetchone():
            return
        if not self._conn.execute("SELECT 1 FROM pauses LIMIT 1").fetchone():
            return
        now = time.time()
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute("INSERT INTO backoff (name, until, pause, error) SELECT name, until, MAX(until - ?, 0), "
                               "'' FROM pauses WHERE true ON CONFLICT(name) DO UPDATE SET until = MAX(backoff.until, "
                               "excluded.until)", (now,))
            self._conn.execute("DELETE FROM pauses")

    def close(self) -> None:
        self._conn.close()

    def next_turn(self, session: str, wait: float | None = None) -> int:
        """This prompt's turn number in *session* (1, 2, ...), counted across processes. When another process holds
        state.db past *wait* seconds: the number read, not counted (the next turn counts it again)."""
        now = time.time()
        try:
            with _waiting(self._conn, wait), self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute("SELECT turn FROM turns WHERE session = ?", (session,)).fetchone()
                turn = (row[0] if row else 0) + 1
                self._conn.execute("INSERT INTO turns (session, turn, touched) VALUES (?, ?, ?) ON CONFLICT(session) "
                                   "DO UPDATE SET turn = excluded.turn, touched = excluded.touched", (session, turn, now))
                if turn == 1:
                    self._prune(now)
            return turn
        except sqlite3.OperationalError as exc:
            if wait is None or not _busy(exc):
                raise
        row = self._conn.execute("SELECT turn FROM turns WHERE session = ?", (session,)).fetchone()
        return (row[0] if row else 0) + 1

    def seen(self, session: str) -> dict[tuple[int, str], int]:
        rows = self._conn.execute("SELECT fact_id, rev, turn FROM recalled WHERE session = ?", (session,))
        return {(fact_id, rev): turn for fact_id, rev, turn in rows}

    def remember(self, session: str, revisions, turn: int, wait: float | None = None) -> bool:
        """The revisions recalled at *turn* (the repeat window). False when another process held state.db past *wait*
        seconds: not written, and they may come again before their time."""
        rows = [(session, fact_id, rev, turn) for fact_id, rev in revisions]
        if not rows:
            return True
        try:
            with _waiting(self._conn, wait), self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.executemany("INSERT INTO recalled (session, fact_id, rev, turn) VALUES (?, ?, ?, ?) "
                                       "ON CONFLICT(session, fact_id, rev) DO UPDATE SET turn = excluded.turn", rows)
                self._conn.execute("DELETE FROM recalled WHERE session = ? AND rowid NOT IN (SELECT rowid FROM "
                                   "recalled WHERE session = ? ORDER BY turn DESC LIMIT ?)",
                                   (session, session, _KEEP_FACTS))
            return True
        except sqlite3.OperationalError as exc:
            if wait is None or not _busy(exc):
                raise
            return False

    def backoff(self, name: str) -> dict | None:
        """{until, pause, error} while *name* is paused, else None."""
        row = self._conn.execute("SELECT until, pause, error FROM backoff WHERE name = ? AND until > ?",
                                 (name, time.time())).fetchone()
        return dict(zip(_BACKOFF_COLUMNS, row)) if row else None

    def paused(self, name: str) -> bool:
        return self.backoff(name) is not None

    def pause(self, name: str, seconds: float) -> bool:
        """A fixed pause (search by meaning); False when state.db stayed busy (not written)."""
        return self._briefly("INSERT INTO backoff (name, until, pause, error) VALUES (?, ?, ?, '') ON CONFLICT(name) "
                             "DO UPDATE SET until = excluded.until, pause = excluded.pause",
                             (name, time.time() + seconds, seconds))

    def fail(self, name: str, error: str) -> bool:
        """A failed call: paused PAUSE_FIRST, twice the last pause while no answer cleared it, PAUSE_MAX at most.
        *error* is the failure's kind. False when state.db stayed busy (nothing written)."""
        doubled = f"MIN({PAUSE_MAX}, CASE WHEN backoff.pause > 0 THEN backoff.pause * 2 ELSE {PAUSE_FIRST} END)"
        return self._briefly(f"INSERT INTO backoff (name, until, pause, error) VALUES (:name, :now + {PAUSE_FIRST}, "
                             f"{PAUSE_FIRST}, :error) ON CONFLICT(name) DO UPDATE SET pause = {doubled}, "
                             f"until = :now + {doubled}, error = excluded.error",
                             {"name": name, "now": time.time(), "error": error})

    def clear(self, name: str) -> bool:
        """An answer: no pause, and the next failure starts again at PAUSE_FIRST."""
        return self._briefly("DELETE FROM backoff WHERE name = ?", (name,))

    def speeds(self, name: str) -> list[float]:
        """The last SPEED_SAMPLES samples of *name* (seconds per character of the pairs sent)."""
        rows = self._conn.execute("SELECT sample FROM rerank_speed WHERE name = ? ORDER BY at DESC, rowid DESC LIMIT ?",
                                  (name, SPEED_SAMPLES))
        return [r[0] for r in rows]

    def add_speed(self, name: str, sample: float) -> bool:
        if not self._briefly("INSERT INTO rerank_speed (name, sample, at) VALUES (?, ?, ?)", (name, sample, time.time())):
            return False
        self._briefly("DELETE FROM rerank_speed WHERE name = ? AND rowid NOT IN (SELECT rowid FROM rerank_speed WHERE "
                      "name = ? ORDER BY at DESC, rowid DESC LIMIT ?)", (name, name, SPEED_SAMPLES))
        return True

    # -- the task-outcome journal ----------------------------------------------------------------------------------

    def _briefly(self, sql: str, params=()) -> bool:
        """One optional write: False when another process held state.db through two 50 ms waits."""
        for _ in range(_LINK_TRIES):
            self._conn.execute(f"PRAGMA busy_timeout = {_LINK_BUSY_MS}")
            try:
                self._conn.execute(sql, params)
                return True
            except sqlite3.OperationalError as exc:
                if not _busy(exc):
                    raise
            finally:
                self._conn.execute(f"PRAGMA busy_timeout = {_BUSY_MS}")
        return False

    def note_attempt(self, attempt_id: str, session: str, client: str, db: str) -> bool:
        """Link a receipt the store committed to who recorded it; False when state.db stayed busy (no link)."""
        return self._briefly("INSERT OR REPLACE INTO attempts (attempt_id, session, client, db, created) "
                             "VALUES (?, ?, ?, ?, ?)", (attempt_id, session, client, db, time.time()))

    def attempt(self, attempt_id: str) -> dict | None:
        row = self._conn.execute("SELECT attempt_id, session, client, db, created FROM attempts WHERE attempt_id = ?",
                                 (attempt_id,)).fetchone()
        return dict(zip(("attempt_id", "session", "client", "db", "created"), row)) if row else None

    def recent_attempts(self, client: str, db: str, limit: int = 20) -> list[dict]:
        since = time.time() - _ATTEMPT_DAYS * 86400
        rows = self._conn.execute("SELECT attempt_id, session, client, db, created FROM attempts WHERE client = ? AND "
                                  "db = ? AND created > ? ORDER BY created DESC LIMIT ?", (client, db, since, limit))
        return [dict(zip(("attempt_id", "session", "client", "db", "created"), r)) for r in rows]

    def drop_attempts(self, db: str, alive: set[str], before: float) -> None:
        """After a forget: the links of *db*'s attempts the store no longer holds (its triggers removed them). Only
        links written before *before* (when the store was read): a newer one may belong to a receipt committed after."""
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            gone = [r[0] for r in self._conn.execute("SELECT attempt_id FROM attempts WHERE db = ? AND created <= ?",
                                                     (db, before)) if r[0] not in alive]
            self._conn.executemany("DELETE FROM attempts WHERE attempt_id = ?", [(a,) for a in gone])

    def forget_fact(self, fact_id: int) -> None:
        """A forgotten fact leaves no trace here either (its text hash)."""
        with self._conn:
            self._conn.execute("DELETE FROM recalled WHERE fact_id = ?", (fact_id,))

    def _prune(self, now: float) -> None:
        cutoff = now - _KEEP_DAYS * 86400
        self._conn.execute("DELETE FROM turns WHERE touched < ? OR session NOT IN (SELECT session FROM turns "
                           "ORDER BY touched DESC LIMIT ?)", (cutoff, _KEEP_SESSIONS))
        self._conn.execute("DELETE FROM recalled WHERE session NOT IN (SELECT session FROM turns)")
        self._conn.execute("DELETE FROM attempts WHERE created < ?", (now - _ATTEMPT_DAYS * 86400,))
        self._conn.execute("DELETE FROM backoff WHERE until < ?", (cutoff,))  # an endpoint no longer called
        self._conn.execute("DELETE FROM rerank_speed WHERE at < ?", (cutoff,))


def read_backoff(path: Path, name: str) -> dict | None:
    """{until, pause, error} while *name* is paused, read without writing (holomem status); None when not paused or
    state.db does not exist or predates the table."""
    if not path.exists():
        return None
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=_BUSY_MS / 1000)
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'backoff'").fetchone():
            return None
        row = conn.execute("SELECT until, pause, error FROM backoff WHERE name = ? AND until > ?",
                           (name, time.time())).fetchone()
    finally:
        conn.close()
    return dict(zip(_BACKOFF_COLUMNS, row)) if row else None


def read_attempts(path: Path, db: str, since: float) -> list[dict]:
    """The journal's links of *db* since *since*, oldest first, read without writing (no file is created, no schema
    added): holomem export --journal only reads."""
    if not path.exists():
        return []
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=_BUSY_MS / 1000)
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'attempts'").fetchone():
            return []
        rows = conn.execute("SELECT attempt_id, session, client, db, created FROM attempts WHERE db = ? AND "
                            "created > ? ORDER BY created, rowid", (db, since)).fetchall()
    finally:
        conn.close()
    return [dict(zip(("attempt_id", "session", "client", "db", "created"), r)) for r in rows]

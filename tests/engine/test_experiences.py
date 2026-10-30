"""Task outcomes describe relevance, never the truth of a stored claim."""
from pathlib import Path
import sqlite3
import time


from holomem.engine.store import MemoryStore


def _on_disk(db):
    wal = Path(str(db) + "-wal")
    return Path(db).read_bytes() + (wal.read_bytes() if wal.exists() else b"")


def test_retention_scrubs_removed_outcome_on_the_open_connection(tmp_path):
    """A long-open store: the receipt write whose retention removes an expired outcome also takes its correction text
    out of the main file and the WAL, without a reopen; a later retention call with nothing due leaves no mark behind
    (Codex review 2026-10-29: the mark was set, and nothing on this connection served it until the next open)."""
    from holomem.engine import experience as journal
    db = tmp_path / "review.db"
    store = MemoryStore(db, hrr_dim=64)
    try:
        assert store._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        fid = store.add_fact("The observatory opens at noon.")
        fact = dict(store.read("SELECT * FROM facts WHERE fact_id=?", (fid,))[0])
        aid = store.record_attempt("search", [fact], [], "session")
        marker = "cinnabar annex private correction marker"
        store.record_outcome(aid, "event1", "corrected", "other", correction=marker, session_key="session")
        store._conn.execute("UPDATE memory_attempts SET created_at=? WHERE attempt_id=?",
                            (time.time() - journal.TTL - 60, aid))
        store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert marker.encode() in _on_disk(db)
        store.record_attempt("search", [fact], [], "session")  # the normal receipt write runs retention
        assert store.read("SELECT * FROM memory_outcomes") == []
        assert marker.encode() not in _on_disk(db), f"scrub_pending={store._meta('scrub_pending')}"
        assert store.prune_experiences() == 0
        assert marker.encode() not in _on_disk(db)
        assert store._meta("scrub_pending") is None
    finally:
        store.close()


def test_every_statement_of_a_scrub_runs_without_waiting_on_another_process(tmp_path):
    """The scrub a receipt write runs sits on the search path: none of its statements may wait out the 10 s busy
    timeout, the closing checkpoint after the mark is cleared included (Codex review 2026-10-30: a reader opened
    after the first checkpoint made the second one wait 10 s inside search). The timeout is restored last."""
    store = MemoryStore(tmp_path / "trace.db", hrr_dim=64)
    try:
        store.add_fact("The observatory opens at noon.")
        store._set_meta("scrub_pending", "1")
        statements = []
        store._conn.set_trace_callback(statements.append)
        assert store.scrub()
        store._conn.set_trace_callback(None)
        timeout = None
        for sql in statements:
            words = sql.split()
            if sql.startswith("PRAGMA busy_timeout") and "=" in sql:
                timeout = int(words[-1].strip("= "))
            elif timeout is not None and (sql.startswith(("VACUUM", "PRAGMA wal_checkpoint", "DELETE"))):
                assert timeout == 0, f"{sql!r} ran with busy timeout {timeout}"
        assert statements[-1].replace(" ", "").startswith("PRAGMAbusy_timeout=") and timeout > 0
    finally:
        store.close()


def test_a_scrub_clears_only_the_mark_it_served(tmp_path):
    """Another process can delete text, and set the mark, while this scrub runs between its rewrite and clearing the
    mark; that mark must survive, or the other text stays on disk with nothing left to scrub it (Codex review
    2026-10-30). Stores of one path share a connection in a process, so the other process is a plain connection
    that writes its own mark, as its forget would, while a reader it does not control keeps its own rewrite back."""
    db = tmp_path / "race.db"
    store = MemoryStore(db, hrr_dim=64)
    reader, other = sqlite3.connect(db), sqlite3.connect(db, isolation_level=None)
    try:
        assert store._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        store.add_fact("The observatory opens at noon.")
        store._mark_scrub() if hasattr(store, "_mark_scrub") else store._set_meta("scrub_pending", "1")
        fired = []

        def before_clearing(sql):  # this scrub has rewritten and checkpointed the file
            if sql.startswith("DELETE FROM meta") and not fired:
                reader.execute("BEGIN")
                reader.execute("SELECT count(*) FROM facts").fetchone()
                other.execute("INSERT INTO meta (key, value) VALUES ('scrub_pending', 'other-forget') "
                              "ON CONFLICT(key) DO UPDATE SET value = excluded.value")
                fired.append(sql)
        store._conn.set_trace_callback(before_clearing)
        store.scrub()
        store._conn.set_trace_callback(None)
        assert fired
        assert store._meta("scrub_pending") == "other-forget", "the other process's mark was cleared"
        reader.rollback()
        assert store.scrub() and store._meta("scrub_pending") is None
    finally:
        reader.close()
        other.close()
        store.close()

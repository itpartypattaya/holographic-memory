"""Task outcomes describe relevance, never the truth of a stored claim (engine/experience.py).

Requirements, in the person's words:
- an outcome is the agent's report on an attempt: it changes no fact (trust, text, history), says it is the agent's
  and not a person's confirmation, and stays on the version the attempt used, after a change or an end of the fact;
- an outcome needs a receipt of this store and this session: an invented, expired or another store's attempt takes
  none; a revision supersedes the active outcome once, a repeated delivery is a no-op, a conflicting one an error;
  a task or correction that holds a credential, or a correction source that is not a version of this store, is
  refused;
- the receipt names what was emitted, also without the trace and also when nothing was found; a candidate another
  writer changed meanwhile is left out of the trace and counted, never named by a version it did not see;
- forget means forgotten: the receipt goes with any fact it names (emitted, a trace candidate, a correction source,
  a merged copy, an ended fact), its outcome text with it from the file; an older release deleting the fact row by
  SQL takes the receipt too;
- the journal never holds the answer: another writer's lock is waited for 50 ms per try and twice at most, a busy
  failure is retried once (and recorded once), a fact that changed between the tries or a failure that is not busy is
  not retried; the busy timeout is restored after;
- retention (30 days, 1000 receipts) also runs with the journal off, when holomem's server opens the store, and takes
  the outcome text out of the file and its WAL; a rewrite another process blocks is neither waited for nor lost;
- with the journal off nothing of it exists: no table, no receipt in a search, no tool, no line in the session block;
  on, the session block asks for an outcome when the person corrects or rejects an answer, and nothing else differs.

The engine's own behaviour is tested on MemoryStore; what lives in holomem's layer (the tools, recall, the session
block, the links in state.db, retention when the server opens the store) through holomem's server in this process."""
import json
import subprocess
import sys
import threading
from pathlib import Path
import re
import sqlite3
import time

import pytest

from holomem.engine import experience as journal_engine
from holomem.engine.store import MemoryStore

ROOT = Path(__file__).resolve().parents[2]
KEY = "a1b2c3d4"  # the session key the engine stores with a receipt (holomem: a hash of client, session, database)
ALICE = "Alice Smith works on the observatory project"


def _on_disk(db):
    wal = Path(str(db) + "-wal")
    return Path(db).read_bytes() + (wal.read_bytes() if wal.exists() else b"")


def _fact(store, fact_id) -> dict:
    return dict(store.read("SELECT * FROM facts WHERE fact_id = ?", (fact_id,))[0])


def _said(store, text) -> int:
    return store.add_fact_status(text, trust=store.start_trust("said"),
                                 source={"via": "tool", "session_id": "s1", "origin": "said"})[0]


def _seed(store) -> tuple[int, str]:
    """A fact and the receipt of a search that emitted it."""
    fid = _said(store, ALICE)
    return fid, store.record_attempt("search", [_fact(store, fid)], [], KEY)


def _attempts(store) -> list[str]:
    return [r[0] for r in store.read("SELECT attempt_id FROM memory_attempts")] if store.experience_ready() else []


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "facts.db", hrr_dim=64)
    yield s
    s.close()


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


# -- an outcome is a report, not a vote ------------------------------------------------------------------------------


def test_an_outcome_is_the_agents_report_and_changes_no_fact(store):
    fid, aid = _seed(store)
    before = store.history(fid), _fact(store, fid)
    result = store.record_outcome(aid, "event-1", "corrected", "wrong_entity",
                                  correction="Use the other Alice for this task.", session_key=KEY)
    assert result["recorded"] is True and result["human_verified"] is False
    assert (store.history(fid), _fact(store, fid)) == before  # trust, text and versions as they were
    report = store.experience_report(aid, KEY)
    assert report["outcomes"][0]["actor"] == "agent" and report["outcomes"][0]["human_verified"] is False
    assert report["emitted"][0]["fact_id"] == fid
    assert "Alice Smith works" not in json.dumps(report)


def test_a_revision_supersedes_the_active_outcome_once_and_a_delivery_again_is_a_no_op(store):
    _, aid = _seed(store)
    first = dict(attempt_id=aid, event_id="event-1", outcome="dead_end", reason="wrong_entity", session_key=KEY)
    assert store.record_outcome(**first)["recorded"] is True
    assert store.record_outcome(**first)["already"] is True
    with pytest.raises(ValueError, match="conflict"):
        store.record_outcome(**{**first, "outcome": "useful"})
    assert store.record_outcome(aid, "event-2", "useful", "other", supersedes="event-1",
                                session_key=KEY)["recorded"] is True
    with pytest.raises(ValueError, match="revision"):  # event-1 is no longer the active one
        store.record_outcome(aid, "event-3", "useful", "other", supersedes="event-1", session_key=KEY)
    outcomes = store.experience_report(aid, KEY)["outcomes"]
    assert [(o["event_id"], o["active"]) for o in outcomes] == [("event-1", False), ("event-2", True)]


def test_an_outcome_stays_on_the_version_the_attempt_used(store):
    fid, aid = _seed(store)
    old = store.experience_report(aid, KEY)["emitted"][0]["version_id"]
    store.change_fact(fid, "Alice Smith now works on the museum project")
    assert store.record_outcome(aid, "old-version", "dead_end", "stale_version", session_key=KEY)["recorded"]
    emitted = store.experience_report(aid, KEY)["emitted"][0]
    assert (emitted["version_id"], emitted["version_status"]) == (old, "changed")


def test_an_ended_fact_keeps_the_used_version_and_its_outcome(store):
    fid, aid = _seed(store)
    store.end_fact(fid)
    assert store.experience_report(aid, KEY)["status"] == "ok"
    assert store.record_outcome(aid, "ended-version", "dead_end", "stale_version", session_key=KEY)["recorded"]
    assert store.experience_report(aid, KEY)["emitted"][0]["version_status"] == "ended"


# -- no receipt, no outcome ---------------------------------------------------------------------------------------


def test_without_a_receipt_of_this_store_and_session_there_is_no_outcome(store, tmp_path):
    with pytest.raises(ValueError, match="unavailable"):  # no journal yet: nothing to report on
        store.record_outcome("invented", "fake", "useful", "other", session_key=KEY)
    _, aid = _seed(store)
    with pytest.raises(ValueError, match="unavailable"):
        store.record_outcome("invented", "fake", "useful", "other", session_key=KEY)
    with pytest.raises(ValueError, match="Human verification"):  # an agent cannot report a person's check
        store.record_outcome(aid, "fake-human", "useful", "other", actor="human", session_key=KEY)
    with pytest.raises(ValueError, match="unavailable"):  # another session's receipt
        store.record_outcome(aid, "other-session", "useful", "other", session_key="another-key")
    assert store.experience_report(aid, "another-key")["status"] == "not_recorded"
    other = MemoryStore(tmp_path / "other.db", hrr_dim=64)
    try:
        assert other.experience_report(aid)["status"] == "not_recorded"
        with pytest.raises(ValueError, match="unavailable"):
            other.record_outcome(aid, "cross-store", "useful", "other", session_key=KEY)
    finally:
        other.close()
    assert store.experience_report(aid, KEY)["outcomes"] == []


def test_a_secret_task_or_a_correction_source_of_no_version_is_refused(store):
    _, aid = _seed(store)
    with pytest.raises(ValueError, match="Task"):
        store.record_outcome(aid, "secret", "corrected", "wrong_entity", task="API_KEY=sk-" + "a" * 48,
                             session_key=KEY)
    with pytest.raises(ValueError, match="reference"):
        store.record_outcome(aid, "bad-ref", "corrected", "wrong_entity",
                             correction_refs=[{"fact_id": 999999, "version_id": 999999}], session_key=KEY)
    assert store.experience_report(aid, KEY)["outcomes"] == []
    assert b"sk-aaaa" not in _on_disk(store.db_path)


def test_an_answer_that_found_nothing_has_a_receipt_and_no_invented_outcome(store):
    _said(store, ALICE)
    aid = store.record_attempt("search", [], [], KEY)
    report = store.experience_report(aid, KEY)
    assert (report["status"], report["emitted"], report["outcomes"]) == ("ok", [], [])


def test_without_the_trace_the_emission_is_still_recorded(store):
    fid = _said(store, ALICE)
    fact = _fact(store, fid)
    aid = store.record_attempt("search", [fact], [("lexical_candidates", [fact])], KEY, trace=False)
    report = store.experience_report(aid, KEY)
    assert [e["fact_id"] for e in report["emitted"]] == [fid]
    assert (report["stages"], report["trace_status"]) == ([], "not_recorded")


def test_a_candidate_another_writer_changed_meanwhile_is_left_out_of_the_trace_and_counted(store):
    """A rescoring call can take seconds; another writer changes a candidate that was not emitted meanwhile. The
    receipt is kept (the emission is exact), the trace never names a version it did not see, and says how many it
    left out; no text of either version is in the report."""
    boris = _said(store, "Boris Petrov maintains the harbour crane schedule")
    tea = _said(store, "Boris Petrov prefers tea over coffee")
    bystander = _said(store, "The harbour crane schedule changes on Mondays")
    seen = {fid: _fact(store, fid) for fid in (boris, tea, bystander)}
    store.change_fact(bystander, "The harbour crane schedule changes on Tuesdays")
    aid = store.record_attempt("prefetch", [seen[boris]], [("lexical_candidates", list(seen.values()))], KEY,
                               trace=True)
    report = store.experience_report(aid, KEY)
    assert report["status"] == "ok" and [e["fact_id"] for e in report["emitted"]] == [boris]
    lexical = next(s for s in report["stages"] if s["stage"] == "lexical_candidates")
    assert [r["fact_id"] for r in lexical["facts"]] == [boris, tea] and lexical["changed"] == 1
    assert all(r["fact_id"] != bystander for s in report["stages"] for r in s["facts"])
    assert "Mondays" not in json.dumps(report) and "Tuesdays" not in json.dumps(report)


# -- forget takes the receipt --------------------------------------------------------------------------------------


def test_an_older_writer_deleting_the_fact_row_by_sql_erases_the_receipt(store):
    fid, aid = _seed(store)
    conn = sqlite3.connect(store.db_path)
    try:
        conn.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))  # a release without history: its remove
        conn.commit()
    finally:
        conn.close()
    assert store.experience_report(aid, KEY)["status"] == "not_recorded"
    assert _attempts(store) == []


def test_the_task_and_a_correction_source_go_with_either_fact(store):
    _, aid = _seed(store)
    correct = _said(store, "Alice Brown works on the observatory engineering team")
    version = store.history(correct)["versions"][-1]["version_id"]
    assert store.record_outcome(aid, "correct-source", "corrected", "wrong_entity",
                                task="Find the observatory engineer", correction="The correct source is Alice Brown.",
                                correction_refs=[{"fact_id": correct, "version_id": version}],
                                session_key=KEY)["recorded"]
    report = store.experience_report(aid, KEY)
    assert report["outcomes"][0]["task"] == "Find the observatory engineer"
    assert report["outcomes"][0]["correction_refs"][0]["fact_id"] == correct
    store.forget_fact(correct)  # the source, not the emitted fact
    assert store.experience_report(aid, KEY)["status"] == "not_recorded"
    assert b"observatory engineer" not in _on_disk(store.db_path)


def test_a_candidate_only_the_trace_saw_takes_the_receipt_with_it(store):
    fid = _said(store, ALICE)
    other = _said(store, "Boris works on the museum collection")
    facts = [_fact(store, fid), _fact(store, other)]
    aid = store.record_attempt("search", facts[:1], [("lexical_candidates", facts)], KEY, trace=True)
    store.forget_fact(other)
    assert store.experience_report(aid, KEY)["status"] == "not_recorded"


def test_forgetting_the_keep_after_a_merge_erases_the_receipts_of_the_copy(store):
    keep, _ = _seed(store)
    copy = _said(store, "Alice Smith is working on the observatory project")
    aid = store.record_attempt("search", [_fact(store, keep), _fact(store, copy)], [], KEY)
    assert store.record_outcome(aid, "before-merge", "useful", "other", session_key=KEY)["recorded"]
    assert store.merge_facts(keep, [copy])["merged"] == [copy]
    assert store.experience_report(aid, KEY)["status"] == "ok"
    assert store.forget_fact(keep)["merged_copies"] == [copy]
    assert store.experience_report(aid, KEY)["status"] == "not_recorded"


@pytest.mark.parametrize("left_present", ["end", "merge"])
def test_forgetting_a_fact_no_longer_current_erases_its_receipt(store, left_present):
    """An ended or merged fact has no `facts` row any more: only its versions tie the receipt to it."""
    fid, aid = _seed(store)
    assert store.record_outcome(aid, "before-forget", "corrected", "wrong_entity",
                                correction="The corrected destination is the cobalt depot.", session_key=KEY)["recorded"]
    if left_present == "end":
        store.end_fact(fid)
    else:
        keep = _said(store, "Alice Smith is working on the observatory project")
        assert store.merge_facts(keep, [fid])["merged"] == [fid]
    assert store.experience_report(aid, KEY)["status"] == "ok"
    assert store.forget_fact(fid)["forgotten"] is True
    assert store.experience_report(aid, KEY)["status"] == "not_recorded"
    for table in ("memory_attempts", "memory_attempt_refs", "memory_outcomes"):
        assert not store.read(f"SELECT 1 FROM {table} WHERE attempt_id = ?", (aid,)), table
    assert not store.read("SELECT 1 FROM memory_attempt_refs WHERE fact_id = ?", (fid,))
    assert b"cobalt depot" not in _on_disk(store.db_path)


# -- the journal never holds the answer ------------------------------------------------------------------------------
# Another process holds the database. Wall time is not the measure (SQLite's busy wait is a series of short sleeps the
# OS stretches): the bound is read where it is set, the busy timeout each lock statement of the journal runs under,
# and how many it runs. The hold ends long before the connection's own 10 s timeout would: a journal that waited for
# the holder instead of giving up could only return after the holder let go.
_HOLD_S = 5
_JOURNAL_BUSY_MS = 50  # a 50 ms busy timeout per lock acquisition, a busy failure retried once
_BUSY_SET = re.compile(r"PRAGMA\s+busy_timeout\s*=\s*(\d+)\s*$", re.IGNORECASE)


def _hold(db_path, *statements):
    """A thread with its own connection (as another process has) that runs *statements* in one transaction and
    holds it until `release` is set or _HOLD_S passes. Returns (thread, acquired, release, let_go)."""
    acquired, release, let_go = threading.Event(), threading.Event(), threading.Event()

    def holder():
        external = sqlite3.connect(db_path, isolation_level=None)
        try:
            for statement in statements:
                external.execute(statement).fetchall()
            acquired.set()
            release.wait(_HOLD_S)
            let_go.set()
            external.execute("ROLLBACK")
        finally:
            external.close()

    thread = threading.Thread(target=holder)
    thread.start()
    return thread, acquired, release, let_go


def _traced(store, run, traced=("BEGIN", "COMMIT")):
    """Run *run*; list each *traced* statement the store's connection ran with the busy timeout in force for it.
    Returns (statements, result or the exception raised)."""
    conn = store._conn
    busy = [conn.execute("PRAGMA busy_timeout").fetchone()[0]]
    seen = []

    def trace(statement):
        statement = statement.strip()
        assigned = _BUSY_SET.match(statement)
        if assigned:
            busy[0] = int(assigned.group(1))
        elif statement.upper().startswith(traced):
            seen.append((statement.upper(), busy[0]))

    conn.set_trace_callback(trace)
    try:
        result = run()
    except Exception as exc:  # noqa: BLE001 - returned for the test's asserts
        result = exc
    finally:
        conn.set_trace_callback(None)
    return seen, result


def test_another_writer_holding_the_store_bounds_the_journals_wait(store, caplog):
    fid, _ = _seed(store)
    fact = _fact(store, fid)
    thread, acquired, release, let_go = _hold(store.db_path, "BEGIN IMMEDIATE")
    try:
        assert acquired.wait(5)
        with caplog.at_level("WARNING", logger="holomem.engine.experience"):
            locks, error = _traced(store, lambda: store.record_attempt("search", [fact], [], KEY))
        assert isinstance(error, sqlite3.OperationalError), error
        assert not let_go.is_set(), "the journal waited until the other writer let go"
        assert [statement for statement, _ in locks] == ["BEGIN IMMEDIATE"] * 2, locks  # one try, one retry
        assert all(busy <= _JOURNAL_BUSY_MS for _, busy in locks), f"journal lock wait not bounded: {locks}"
        diagnostic = [r.getMessage() for r in caplog.records if "experience slow" in r.getMessage()]
        assert diagnostic and all(field in diagnostic[-1] for field in ("lock_ms=", "schema_ms=", "begin_ms=",
                                                                         "work_ms=", "commit_ms=", "total_ms="))
        assert "Alice" not in diagnostic[-1]
        assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000
        assert not store._conn.in_transaction
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert store.record_attempt("search", [fact], [], KEY)


def test_a_reader_in_rollback_mode_cannot_hold_the_journals_commit(store, caplog):
    fid, _ = _seed(store)
    fact = _fact(store, fid)
    assert store._conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    thread, acquired, release, let_go = _hold(store.db_path, "BEGIN", "SELECT count(*) FROM facts")
    try:
        assert acquired.wait(5)
        with caplog.at_level("WARNING", logger="holomem.engine.experience"):
            locks, error = _traced(store, lambda: store.record_attempt("search", [fact], [], KEY))
        assert isinstance(error, sqlite3.OperationalError), error
        assert not let_go.is_set(), "the journal's commit waited until the reader let go"
        # The reader blocks the COMMIT, not the BEGIN: each try gets as far as committing and gives up there.
        assert [statement for statement, _ in locks] == ["BEGIN IMMEDIATE", "COMMIT"] * 2, locks
        assert all(busy <= _JOURNAL_BUSY_MS for _, busy in locks), f"journal lock wait not bounded: {locks}"
        diagnostic = [r.getMessage() for r in caplog.records if "experience slow" in r.getMessage()]
        assert diagnostic and "commit_ms=-" not in diagnostic[-1]
        assert not store._conn.in_transaction
        assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert store.record_attempt("search", [fact], [], KEY)


def _release_at_second_begin(release, then=None):
    begins = []

    def trace(statement):
        if statement.upper().startswith("BEGIN IMMEDIATE"):
            begins.append(statement)
            if len(begins) == 2:
                release.set()
                if then is not None:
                    then()
    return begins, trace


def test_a_writer_that_lets_go_meanwhile_is_waited_for_once_more_and_the_receipt_recorded_once(store):
    fid, _ = _seed(store)
    fact, before = _fact(store, fid), len(_attempts(store))
    thread, acquired, release, _ = _hold(store.db_path, "BEGIN IMMEDIATE")
    begins, trace = _release_at_second_begin(release)
    try:
        assert acquired.wait(5)
        store._conn.set_trace_callback(trace)
        aid = store.record_attempt("search", [fact], [], KEY)
    finally:
        store._conn.set_trace_callback(None)
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert aid and len(begins) >= 2, "temporary contention lost the receipt"
    assert len(_attempts(store)) == before + 1 and aid in _attempts(store)


def test_a_commit_a_reader_blocked_is_retried_and_recorded_once(store):
    fid, _ = _seed(store)
    fact, before = _fact(store, fid), len(_attempts(store))
    assert store._conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    thread, acquired, release, _ = _hold(store.db_path, "BEGIN", "SELECT count(*) FROM facts")
    begins, trace = _release_at_second_begin(release)
    try:
        assert acquired.wait(5)
        store._conn.set_trace_callback(trace)
        aid = store.record_attempt("search", [fact], [], KEY)
    finally:
        store._conn.set_trace_callback(None)
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert aid and len(begins) >= 2, "a temporary reader lock lost the receipt"
    assert len(_attempts(store)) == before + 1
    assert not store._conn.in_transaction


def test_a_fact_that_changed_since_it_was_read_is_not_retried_nor_recorded(store):
    fid, _ = _seed(store)
    before = len(_attempts(store))
    fact = {**_fact(store, fid), "content": "Outdated content for this fact"}
    begins = []
    store._conn.set_trace_callback(lambda sql: begins.append(sql) if sql.upper().startswith("BEGIN IMMEDIATE") else None)
    try:
        with pytest.raises(ValueError, match="Memory changed during retrieval"):
            store.record_attempt("search", [fact], [], KEY)
    finally:
        store._conn.set_trace_callback(None)
    assert len(begins) == 1 and len(_attempts(store)) == before


def test_a_failure_that_is_not_busy_is_not_retried(store):
    fid, _ = _seed(store)  # the journal exists: the failure is the receipt's write itself
    fact = _fact(store, fid)
    begins = []
    store._conn.execute("PRAGMA query_only=ON")
    store._conn.set_trace_callback(lambda sql: begins.append(sql) if sql.upper().startswith("BEGIN IMMEDIATE") else None)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.record_attempt("search", [fact], [], KEY)
    finally:
        store._conn.set_trace_callback(None)
        store._conn.execute("PRAGMA query_only=OFF")
    assert len(begins) == 1
    assert not store._conn.in_transaction


_ANOTHER_PROCESS = """
import sys
sys.path.insert(0, sys.argv[1])
from holomem.engine.store import MemoryStore
store = MemoryStore(sys.argv[2], hrr_dim=64)
fact_id = int(sys.argv[3])
if sys.argv[4] == "change":
    store.change_fact(fact_id, "Alice Smith now works on the museum project")
else:
    store.forget_fact(fact_id)
store.close()
"""


@pytest.mark.parametrize("operation", ["change", "forget"])
def test_a_retry_refuses_a_fact_another_process_changed_or_forgot_between_the_tries(store, operation):
    """The first try finds the store held; before the second, another process changes (or forgets) the emitted fact.
    The retry checks the fact again in its own transaction: no receipt names a version the answer did not use."""
    fid, _ = _seed(store)
    fact, before = _fact(store, fid), _attempts(store)
    thread, acquired, release, _ = _hold(store.db_path, "BEGIN IMMEDIATE")
    done = []

    def another_process():
        thread.join(5)  # the holder has let go
        done.append(subprocess.run([sys.executable, "-I", "-c", _ANOTHER_PROCESS, str(ROOT), str(store.db_path),
                                    str(fid), operation], capture_output=True, timeout=60))

    begins, trace = _release_at_second_begin(release, then=another_process)
    try:
        assert acquired.wait(5)
        store._conn.set_trace_callback(trace)
        with pytest.raises(ValueError, match="Memory changed during retrieval"):
            store.record_attempt("search", [fact], [], KEY)
    finally:
        store._conn.set_trace_callback(None)
        release.set()
        thread.join(5)
    assert done and done[0].returncode == 0, done[0].stderr.decode(errors="replace")[-1500:] if done else "no run"
    assert len(begins) == 2
    assert [a for a in _attempts(store) if a not in before] == []


# -- retention -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("served_by", ["receipt", "open"])
def test_a_blocked_rewrite_after_retention_neither_holds_the_receipt_nor_stays_owed(store, monkeypatch, served_by):
    """The rewrite after a receipt's retention runs under busy timeout 0: another process reading the database makes
    it come back at once with the mark kept. A second receipt within the retry window does not rewrite the file again
    (a VACUUM per search while a reader stays would grow the WAL by the whole database each time). Once the reader is
    gone, a later receipt or a writer's open serves the mark although nothing new has expired."""
    assert store._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    fid, old = _seed(store)
    fact = _fact(store, fid)
    marker = "Use the cinnabar annex record instead."
    assert store.record_outcome(old, "old-result", "corrected", "wrong_entity", correction=marker,
                                session_key=KEY)["recorded"]
    store._conn.execute("UPDATE memory_attempts SET created_at = ? WHERE attempt_id = ?",
                        (time.time() - journal_engine.TTL - 60, old))
    store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    rewrite = ("VACUUM", "PRAGMA WAL_CHECKPOINT")
    thread, acquired, release, let_go = _hold(store.db_path, "BEGIN", "SELECT count(*) FROM facts")
    try:
        assert acquired.wait(5)
        statements, aid = _traced(store, lambda: store.record_attempt("search", [fact], [], KEY), rewrite)
        assert not let_go.is_set(), "the rewrite held the receipt until the reader let go"
        assert isinstance(aid, str), aid  # the receipt itself is recorded
        assert not store.read("SELECT 1 FROM memory_outcomes")
        assert any(s == "VACUUM" for s, _ in statements), f"no rewrite was tried after retention: {statements}"
        assert all(busy == 0 for _, busy in statements), f"the rewrite waited on the other process: {statements}"
        assert store._meta("scrub_pending") is not None  # blocked: the mark stays for a retry
        assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000
        statements, again = _traced(store, lambda: store.record_attempt("search", [fact], [], KEY), rewrite)
        assert isinstance(again, str) and not statements, f"rewrote the file again at once: {statements}"
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    if served_by == "receipt":
        monkeypatch.setattr(journal_engine, "_SCRUB_RETRY_S", 0)  # the retry window has passed
        assert store.record_attempt("search", [fact], [], KEY)
    else:
        assert store.prune_experiences() == 0  # what a writer's open runs: nothing is due, the mark is
    assert store._meta("scrub_pending") is None
    assert marker.encode() not in _on_disk(store.db_path)


def test_the_receipt_cap_applies_and_a_store_without_a_journal_gets_none(store, tmp_path):
    fid, newest = _seed(store)
    now = time.time()
    store._conn.executemany("INSERT INTO memory_attempts VALUES (?,?,?,?,?,?,?)",
                            [(f"older-{i}", now - 3600 - i, "search", KEY, "[]", "[]", "not_recorded")
                             for i in range(journal_engine.MAX_ATTEMPTS)])
    assert store.prune_experiences() == 1
    kept = set(_attempts(store))
    assert len(kept) == journal_engine.MAX_ATTEMPTS
    assert newest in kept and f"older-{journal_engine.MAX_ATTEMPTS - 1}" not in kept  # the oldest went
    fresh = MemoryStore(tmp_path / "never.db", hrr_dim=64)
    try:
        assert fresh.prune_experiences() == 0 and not fresh.experience_ready()
    finally:
        fresh.close()


# -- holomem's layer: the tools, recall, the session block, the server's open ---------------------------------------

EXP = "experience = true\n"


def test_with_the_journal_off_a_search_has_no_receipt_and_no_journal_is_made(tmp_path, holomem_server):
    server = holomem_server(tmp_path / "holo")
    added, _ = server.call("memory_add", content=ALICE, origin="said")
    found, is_error = server.call("memory_search", query="Alice observatory")
    assert not is_error and [f["fact_id"] for f in found["results"]] == [added["fact_id"]]
    assert "memory_attempt" not in found
    assert "memory_experience" not in {t["name"] for t in server.request("tools/list")["result"]["tools"]}
    block = server.recall("What does Alice Smith work on at the observatory?")
    assert f"#{added['fact_id']}" in block and "Memory attempt" not in block
    assert not server.store.experience_ready()


def test_a_busy_store_gives_the_search_without_its_receipt_and_says_why(tmp_path, holomem_server):
    server = holomem_server(tmp_path / "holo", EXP)
    server.call("memory_add", content=ALICE, origin="said")  # the store is open (its migration behind it)
    thread, acquired, release, let_go = _hold(server.store.db_path, "BEGIN IMMEDIATE")
    try:
        assert acquired.wait(5)
        found, is_error = server.call("memory_search", query="Alice observatory")
        assert not let_go.is_set(), "the search waited until the other writer let go"
        assert not is_error and found["results"] and "memory_attempt" not in found
        recent, _ = server.call("memory_experience", action="recent")
        assert recent["error_kind"].startswith("OperationalError")
    finally:
        release.set()
        thread.join(5)
    assert server.call("memory_search", query="Alice observatory")[0]["memory_attempt"]
    assert server.call("memory_experience", action="recent")[0]["error_kind"] == ""


def test_the_tool_takes_no_invented_attempt_and_no_persons_verification(tmp_path, holomem_server):
    server = holomem_server(tmp_path / "holo", EXP)
    server.call("memory_add", content=ALICE, origin="said")
    aid = server.call("memory_search", query="Alice observatory")[0]["memory_attempt"]
    args = {"action": "outcome", "event_id": "fake", "outcome": "useful", "reason": "other"}
    invented, is_error = server.call("memory_experience", attempt_id="invented", **args)
    assert is_error, invented
    human, is_error = server.call("memory_experience", attempt_id=aid, actor="human", **args)
    assert is_error and "actor" in human["error"], human
    shown, _ = server.call("memory_experience", action="explain", attempt_id=aid)
    assert shown["status"] == "ok" and shown["outcomes"] == []


def test_an_attempt_is_found_only_in_the_store_that_recorded_it(tmp_path, holomem_server):
    """Project scope: two projects, two stores, one state.db. An attempt of one project's store is not another's to
    explain or to report on, whatever its id."""
    home = tmp_path / "holo"
    settings = EXP + 'scope = "project"\n'
    first, second = tmp_path / "project-a", tmp_path / "project-b"
    first.mkdir()
    second.mkdir()
    a = holomem_server(home, settings, cwd=first)
    a.call("memory_add", content=ALICE, origin="said")
    aid = a.call("memory_search", query="Alice observatory")[0]["memory_attempt"]
    b = holomem_server(home, settings, cwd=second)
    b.call("memory_add", content="Boris works on the museum collection", origin="said")
    assert b.store.db_path != a.store.db_path
    assert b.call("memory_experience", action="explain", attempt_id=aid)[0]["status"] == "not_recorded"
    refused, is_error = b.call("memory_experience", action="outcome", attempt_id=aid, event_id="cross",
                               outcome="useful", reason="other")
    assert is_error, refused
    assert a.call("memory_experience", action="explain", attempt_id=aid)[0]["status"] == "ok"


def test_a_recall_that_emits_nothing_writes_nothing(tmp_path, holomem_server):
    """Nothing found, or found again inside the repeat window of the session: nothing is emitted, so no receipt and
    no write to the store at all."""
    server = holomem_server(tmp_path / "holo", EXP)
    server.call("memory_add", content=ALICE, origin="said")
    begins = []
    conn = server.store._conn
    conn.set_trace_callback(lambda sql: begins.append(sql) if sql.upper().startswith("BEGIN IMMEDIATE") else None)
    try:
        assert server.recall("unicorn zeppelin", session_id="quiet-task") == ""
        assert begins == [] and not server.store.experience_ready()
        assert "Memory attempt:" in server.recall("What does Alice Smith work on at the observatory?",
                                                  session_id="quiet-task")
        begins.clear()
        assert server.recall("What does Alice Smith work on at the observatory?", session_id="quiet-task") == ""
        assert begins == []
    finally:
        conn.set_trace_callback(None)
    assert len(server.call("memory_experience", action="recent")[0]["attempts"]) == 1


def test_the_servers_open_applies_retention_with_the_journal_off(tmp_path, holomem_server):
    """The journal was on, then turned off: nothing records any more, yet outcome text must not outlive the 30 days.
    The server's open removes the expired receipt with its refs and outcome, and the text leaves the file and WAL."""
    home = tmp_path / "holo"
    on = holomem_server(home, EXP)
    on.call("memory_add", content=ALICE, origin="said")
    old = on.call("memory_search", query="Alice observatory")[0]["memory_attempt"]
    recent = on.call("memory_search", query="Alice observatory")[0]["memory_attempt"]
    done, is_error = on.call("memory_experience", action="outcome", attempt_id=old, event_id="old-result",
                             outcome="corrected", reason="wrong_entity",
                             correction="Use the amber warehouse record instead.")
    assert not is_error and done["recorded"], done
    db = on.store.db_path
    on.store._conn.execute("UPDATE memory_attempts SET created_at = ? WHERE attempt_id = ?",
                           (time.time() - journal_engine.TTL - 60, old))
    on.close()  # the last connection closes: the correction is checkpointed into the main file
    assert b"amber warehouse" in _on_disk(db)
    off = holomem_server(home, "")
    assert not off.call("memory_stats")[1]  # the first call opens the store
    assert [r[0] for r in off.store.read("SELECT attempt_id FROM memory_attempts")] == [recent]
    assert not off.store.read("SELECT 1 FROM memory_outcomes")
    assert not off.store.read("SELECT 1 FROM memory_attempt_refs WHERE attempt_id = ?", (old,))
    assert b"amber warehouse" not in _on_disk(db)
    fresh = holomem_server(tmp_path / "never", "")
    assert not fresh.call("memory_stats")[1] and not fresh.store.experience_ready()  # never created by the open


def test_with_the_journal_on_the_session_block_asks_for_an_outcome_after_a_correction(tmp_path, holomem_server):
    from holomem.session import block

    server = holomem_server(tmp_path / "holo", EXP)
    fid = server.call("memory_add", content=ALICE, origin="said")[0]["fact_id"]
    before = server.call("memory_history", fact_id=fid)[0]
    lines = [line for line in block(server.server.cfg, server.server.memory).splitlines()
             if "memory_experience" in line]
    assert len(lines) == 1
    for said in ("corrects or rejects", "outcome", "corrected", "dead_end", "useful", "Memory attempt",
                 "Not every turn", "not their confirmation", "changes no fact", "memory_change or memory_correct"):
        assert said in lines[0], said
    # what the line names is what the agent sees and can send: the recall's attempt id, a useful with reason other
    text = server.recall("What does Alice Smith work on at the observatory?")
    aid = text.rsplit("Memory attempt: ", 1)[1].strip()
    assert server.call("memory_experience", action="outcome", attempt_id=aid, event_id="confirmed-1",
                       outcome="useful", reason="other")[0]["recorded"]
    assert server.call("memory_experience", action="outcome", attempt_id=aid, event_id="rejected-1",
                       outcome="corrected", reason="wrong_entity", correction="The person meant another Alice.",
                       supersedes="confirmed-1")[0]["recorded"]
    assert server.call("memory_history", fact_id=fid)[0] == before  # the outcome changed no fact


def test_with_the_journal_off_nothing_differs_but_its_line_and_its_tool(tmp_path, holomem_server):
    from holomem.session import block

    on = holomem_server(tmp_path / "on", EXP)
    off = holomem_server(tmp_path / "off", "")
    for server in (on, off):
        assert not server.call("memory_stats")[1]  # each store open, both empty
    on_block = block(on.server.cfg, on.server.memory)
    off_block = block(off.server.cfg, off.server.memory)
    assert not [line for line in off_block.splitlines() if "memory_experience" in line]
    assert [line for line in on_block.splitlines() if "memory_experience" not in line] == off_block.splitlines()
    on_tools = on.request("tools/list")["result"]["tools"]
    off_tools = off.request("tools/list")["result"]["tools"]
    assert json.dumps(off_tools, sort_keys=True) == \
        json.dumps([t for t in on_tools if t["name"] != "memory_experience"], sort_keys=True)

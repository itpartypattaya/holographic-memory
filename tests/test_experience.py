"""The task-outcome journal (experience = true): which recall offered which facts, and what the agent says came of it.

Requirements (an opt-in journal, with the bridge's own session link in state.db):
- with experience on, a recall that injects facts ends its block with "Memory attempt: <32 hex>", and the
  prefetch_max_tokens budget counts that line (a fact line that no longer fits is left out whole); a recall that
  injects nothing, and every recall with experience off, leaves no journal table in the store;
- memory_search answers with memory_attempt (also when it finds nothing); a search "as_of" a day records nothing;
- memory_experience (listed, and named in the session block, only with experience on): explain shows a recorded attempt
  (what was emitted, fact and version ids, never fact text), outcome records the agent's report against it — also for
  an attempt a hook (another process, the client's session) recorded — and changes no fact's trust or version; a
  repeated event is "already", a conflicting one an error, a revision supersedes; a credential in the correction or a
  task over 160 characters is refused; recent lists this client's attempts in this database;
- with experience_trace on, explain says why a candidate was not emitted: shown in the last turns (rejected_repeat), not
  covering the message (coverage_rejected), let in by one ordinary word that meaning does not back
  (rejected_lone_word);
- forgetting a fact an attempt emitted removes the attempt, its outcomes and their text from the database file and its
  WAL (also when another reader held the file at the forget) and its link in state.db; explain then says
  not_recorded and an outcome is refused; an attempt older than 30 days is "expired", refuses outcomes, and the next
  receipt removes it;
- the journal never costs the recall: a store write-locked by another process, or a state.db held by one, gives the
  block (or the search) without the attempt line at once; the receipt comes back once the lock is gone;
- nothing rewrites the database file while a prompt waits (a rewrite has no time bound): a hook neither prunes the
  journal nor records a receipt that would remove outcome text (the block goes out without its attempt line), and
  memory_recall leaves a rewrite a blocked forget owes to the next tool call; retention, also with experience off,
  runs at holomem status (and the MCP server's open), and then outcome text past 30 days is gone from the file;
- holomem export --journal lists the attempts (hook and search) with their outcomes, without fact text, an expired one
  as expired, also after the journal was turned off; it opens the database read-only: the file is byte for byte as
  before (vectors of a switched-off search by meaning and an owed rewrite untouched).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from holomem.recall import HEADER, approx_tokens
from test_export import cli as export_cli
from test_hooks import context, hook, prompt, seed
from test_mcp_server import Client
from test_meaning import CITY, FILLER, KEY, LONE_WORD, QUESTION, Endpoint, store_with_vectors
from test_meaning import cli as meaning_cli
from test_meaning import config as meaning_config

ROOT = Path(__file__).resolve().parent.parent
_LAUNCH = "import sys; sys.path.insert(0, sys.argv.pop(1)); from holomem.__main__ import main; sys.exit(main())"
EXP = "experience = true\n"
ATTEMPT = re.compile(r"\nMemory attempt: ([0-9a-f]{32})\Z")
DEPLOY = "The billing service deploys to the staging server with systemd"
SWITCH = "The billing service deploy uses a blue green switch"
ASK = "How does the billing service deploy?"
CORRECTION = "Речь шла о складе в Порту, не о складе в Лиссабоне"  # Cyrillic, as an agent may write it
DAY = 86400


@pytest.fixture
def home(tmp_path) -> Path:
    return tmp_path / "holo"


@pytest.fixture
def endpoint():
    """The local embeddings endpoint of test_meaning (search by meaning, for the lone-word stage)."""
    ep = Endpoint()
    yield ep
    ep.close()


def attempt_of(text: str) -> str:
    match = ATTEMPT.search(text)
    assert match, text
    return match.group(1)


def rows(path: Path, query: str, params=()) -> list:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def write(path: Path, query: str, params=()) -> None:
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute(query, params)
    finally:
        conn.close()


def holomem(home: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOLOMEM_HOME": str(home)}
    return subprocess.run([sys.executable, "-I", "-c", _LAUNCH, str(ROOT), *args], capture_output=True, env=env,
                          timeout=60)


def has_journal(home: Path) -> bool:
    return bool(rows(home / "memory_store.db", "SELECT 1 FROM sqlite_master WHERE name = 'memory_attempts'"))


def wal(home: Path) -> None:
    """The store in WAL, as the plugin's own Python (uv, SQLite 3.53) keeps it. A SQLite with the WAL-reset bug (the
    system one on Ubuntu 24.04, 3.45) leaves a new file in rollback mode, where another process mid-read blocks the
    forget itself, not only its rewrite; a database already in WAL stays in WAL (engine/sqlite_util.py)."""
    conn = sqlite3.connect(home / "memory_store.db")
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def file_bytes(home: Path) -> bytes:
    """The database file and its WAL, as a disk would show them."""
    data = (home / "memory_store.db").read_bytes()
    wal = home / "memory_store.db-wal"
    return data + (wal.read_bytes() if wal.exists() else b"")


def holds(data: bytes, text: str) -> bool:
    """*text* in the file's bytes in either form it may be stored in: UTF-8, or JSON's \\u escapes (the engine keeps an
    outcome as json.dumps of its payload)."""
    return text.encode("utf-8") in data or json.dumps(text)[1:-1].encode("ascii") in data


def outcome(client: Client, attempt: str, event: str = "e1", **extra) -> tuple[dict, bool]:
    args = {"action": "outcome", "attempt_id": attempt, "event_id": event, "outcome": "corrected",
            "reason": "wrong_entity", **extra}
    return client.call("memory_experience", **args)


def test_a_recall_ends_with_its_attempt_and_off_leaves_no_journal(home, tmp_path):
    (fid,) = seed(home, [(DEPLOY, [])], EXP)
    text = context(hook(home, "user-prompt", prompt(ASK)))
    attempt_of(text)
    assert text == f"{HEADER}\n- #{fid} [0.5] {DEPLOY}\nMemory attempt: {attempt_of(text)}"
    assert rows(home / "state.db", "SELECT attempt_id FROM attempts") == [(attempt_of(text),)]

    off = tmp_path / "off"
    seed(off, [(DEPLOY, [])])
    plain = context(hook(off, "user-prompt", prompt(ASK)))
    assert plain == f"{HEADER}\n- #{fid} [0.5] {DEPLOY}"
    assert not has_journal(off)


def test_a_recall_with_nothing_emitted_creates_no_journal(home):
    seed(home, [(DEPLOY, [])], EXP)
    proc = hook(home, "user-prompt", prompt("What is the weather on Mars tomorrow?"))
    assert proc.returncode == 0 and proc.stdout == b""
    assert not has_journal(home)


def test_search_records_an_attempt_that_explain_shows(home):
    (fid,) = seed(home, [(DEPLOY, [])], EXP)
    client = Client(home).start()
    try:
        found, _ = client.call("memory_search", query="billing deploy")
        assert [f["fact_id"] for f in found["results"]] == [fid] and re.fullmatch(r"[0-9a-f]{32}",
                                                                                found["memory_attempt"])
        shown, is_error = client.call("memory_experience", action="explain", attempt_id=found["memory_attempt"])
        assert not is_error and shown["status"] == "ok" and shown["kind"] == "search"
        (version,) = rows(home / "memory_store.db", "SELECT version_id FROM fact_versions WHERE fact_id = ? AND "
                                                   "status = 'current'", (fid,))
        assert [(e["fact_id"], e["version_id"]) for e in shown["emitted"]] == [(fid, version[0])]
        assert DEPLOY not in json.dumps(shown, ensure_ascii=False)  # ids and sources, never the fact's text

        empty, _ = client.call("memory_search", query="quantum teapots of Neptune")
        assert empty["count"] == 0 and "memory_attempt" in empty
        past, _ = client.call("memory_search", query="billing deploy", as_of="today")
        assert "memory_attempt" not in past
    finally:
        client.close()


def test_an_outcome_from_the_server_lands_on_the_hooks_attempt_and_changes_no_fact(home):
    (fid,) = seed(home, [(DEPLOY, [])], EXP)
    attempt = attempt_of(context(hook(home, "user-prompt", prompt(ASK, session="s1"))))
    before = rows(home / "memory_store.db", "SELECT trust_score FROM facts WHERE fact_id = ?", (fid,))
    versions = rows(home / "memory_store.db", "SELECT version_id, status, content FROM fact_versions")
    client = Client(home).start()
    try:
        done, is_error = outcome(client, attempt, correction=CORRECTION, task="deploy question")
        assert not is_error and done["recorded"] is True, done
        shown, _ = client.call("memory_experience", action="explain", attempt_id=attempt)
        assert shown["kind"] == "prefetch" and [e["fact_id"] for e in shown["emitted"]] == [fid]
        (event,) = shown["outcomes"]
        assert (event["outcome"], event["reason"], event["correction"], event["active"]) == \
            ("corrected", "wrong_entity", CORRECTION, True)
        recent, _ = client.call("memory_experience", action="recent")
        assert attempt in [a["attempt_id"] for a in recent["attempts"]]
    finally:
        client.close()
    assert rows(home / "memory_store.db", "SELECT trust_score FROM facts WHERE fact_id = ?", (fid,)) == before
    assert rows(home / "memory_store.db", "SELECT version_id, status, content FROM fact_versions") == versions


def test_a_repeated_event_is_already_a_conflicting_one_an_error_a_revision_supersedes(home):
    seed(home, [(DEPLOY, [])], EXP)
    attempt = attempt_of(context(hook(home, "user-prompt", prompt(ASK))))
    client = Client(home).start()
    try:
        assert outcome(client, attempt)[0]["recorded"] is True
        again, is_error = outcome(client, attempt)
        assert not is_error and again["already"] is True
        clash, is_error = outcome(client, attempt, reason="stale_version")
        assert is_error and "conflict" in clash["error"]
        revised, is_error = outcome(client, attempt, event="e2", outcome="useful", reason="other", supersedes="e1")
        assert not is_error and revised["recorded"] is True, revised
        shown, _ = client.call("memory_experience", action="explain", attempt_id=attempt)
        assert {e["event_id"]: e["active"] for e in shown["outcomes"]} == {"e1": False, "e2": True}
    finally:
        client.close()


def test_a_credential_in_the_correction_or_a_long_task_is_refused(home):
    seed(home, [(DEPLOY, [])], EXP)
    attempt = attempt_of(context(hook(home, "user-prompt", prompt(ASK))))
    client = Client(home).start()
    try:
        secret, is_error = outcome(client, attempt, correction="use ghp_0123456789abcdefghijklmnopqrstuvwxyzAB")
        assert is_error and "ghp_" not in secret["error"]
        long_task, is_error = outcome(client, attempt, event="e2", task="x" * 200)
        assert is_error, long_task
        shown, _ = client.call("memory_experience", action="explain", attempt_id=attempt)
        assert shown["outcomes"] == []
    finally:
        client.close()
    assert b"ghp_0123456789" not in file_bytes(home)


def test_the_trace_says_why_a_candidate_was_not_emitted(home):
    shown_before, switch, team = seed(home, [(DEPLOY, []), (SWITCH, []), ("The billing team meets on Fridays", [])],
                                      EXP + "experience_trace = true\n")
    attempt_of(context(hook(home, "user-prompt", prompt("Which server runs systemd for staging?"))))
    second = attempt_of(context(hook(home, "user-prompt", prompt(ASK))))
    client = Client(home).start()
    try:
        shown, _ = client.call("memory_experience", action="explain", attempt_id=second)
    finally:
        client.close()
    decisions = {d["fact_id"]: d["decision"] for d in shown["decisions"]}
    assert decisions == {switch: "emitted", shown_before: "rejected_repeat", team: "coverage_rejected"}, shown


def test_the_trace_names_a_fact_one_ordinary_word_let_in(home, endpoint):
    """A question about a person's age («сейчас»): the person's fact is emitted, the blog rule that holds «сейчас»
    is let in by the words and left out by meaning."""
    ids = seed(home, [(text, []) for text in FILLER + [LONE_WORD, CITY]] + [("Марк родился в марте", ["Марк"])],
               meaning_config(endpoint) + "embedding_prefetch_facts = 0\n" + EXP + "experience_trace = true\n")
    lone, person = ids[-3], ids[-1]
    (home / ".env").write_text(f"HOLO_TEST_EMBED_KEY={KEY}\n", encoding="utf-8")
    assert meaning_cli(home, "embed").returncode == 0
    text = context(hook(home, "user-prompt", prompt("Сколько сейчас лет Марку?")))
    assert f"#{person}" in text and f"#{lone}" not in text
    client = Client(home).start()
    try:
        shown, _ = client.call("memory_experience", action="explain", attempt_id=attempt_of(text))
    finally:
        client.close()
    stages = {s["stage"]: [f["fact_id"] for f in s["facts"]] for s in shown["stages"]}
    assert lone in stages["coverage_admitted"]  # the words let it in ...
    assert stages["rejected_lone_word"] == [lone]  # ... meaning did not back it
    assert {d["fact_id"]: d["decision"] for d in shown["decisions"]}[lone] == "rejected_lone_word"


def test_the_tool_and_the_session_line_exist_only_with_the_journal_on(home, tmp_path):
    start = {"session_id": "s1", "cwd": "", "source": "startup"}
    for on in (False, True):
        place = tmp_path / ("on" if on else "off")
        seed(place, [(DEPLOY, [])], EXP if on else "")
        client = Client(place).start()
        try:
            names = {t["name"] for t in client.request("tools/list")["result"]["tools"]}
            answer, is_error = client.call("memory_experience", action="recent")
        finally:
            client.close()
        block = context(hook(place, "session-start", start), "SessionStart")
        assert ("memory_experience" in names) is on and ("memory_experience outcome" in block) is on
        assert is_error is not on, answer


def test_the_budget_counts_the_attempt_line(home):
    long_one = DEPLOY + " and a health check after every step of the rollout"
    seed(home, [(long_one, []), (SWITCH, [])])
    plain = context(hook(home, "user-prompt", prompt(ASK, session="a")))
    lines = plain.splitlines()[1:]
    assert len(lines) == 2
    exact = approx_tokens(HEADER) + sum(approx_tokens("\n" + line) for line in lines)
    (home / "config.toml").write_text(f"prefetch_max_tokens = {exact}\n", encoding="utf-8")
    assert context(hook(home, "user-prompt", prompt(ASK, session="b"))) == plain  # exactly fits without the journal
    (home / "config.toml").write_text(f"prefetch_max_tokens = {exact}\n{EXP}", encoding="utf-8")
    text = context(hook(home, "user-prompt", prompt(ASK, session="c")))
    assert text == f"{HEADER}\n{lines[0]}\nMemory attempt: {attempt_of(text)}"
    assert approx_tokens(text) <= exact


def test_forget_removes_the_attempt_its_outcome_and_their_text_from_the_file(home):
    (fid,) = seed(home, [(DEPLOY, [])], EXP)
    wal(home)
    attempt = attempt_of(context(hook(home, "user-prompt", prompt(ASK))))
    client = Client(home).start()
    try:
        assert outcome(client, attempt, correction=CORRECTION)[0]["recorded"] is True
        assert holds(file_bytes(home), CORRECTION)
        reader = sqlite3.connect(home / "memory_store.db", isolation_level=None)
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM memory_outcomes").fetchone()  # another process mid-read: no rewrite now
        try:
            forgot, is_error = client.call("memory_forget", fact_id=fid)
        finally:
            reader.execute("COMMIT")
            reader.close()
        assert not is_error and forgot["scrub"] == "pending", forgot
        assert rows(home / "state.db", "SELECT 1 FROM attempts WHERE attempt_id = ?", (attempt,)) == []
        shown, _ = client.call("memory_experience", action="explain", attempt_id=attempt)
        assert shown["status"] == "not_recorded"
        refused, is_error = outcome(client, attempt, event="e2")
        assert is_error, refused
        client.call("memory_search", query="anything at all")  # the next write rewrites the file
        data = file_bytes(home)
        assert not holds(data, CORRECTION) and attempt.encode() not in data
    finally:
        client.close()


def test_an_expired_attempt_refuses_outcomes_and_the_next_receipt_removes_it(home):
    seed(home, [(DEPLOY, [])], EXP)
    attempt = attempt_of(context(hook(home, "user-prompt", prompt(ASK))))
    client = Client(home).start()
    try:
        client.call("memory_stats")  # the server has the store open (its open-time retention is behind it)
        write(home / "memory_store.db", "UPDATE memory_attempts SET created_at = created_at - ?", (31 * DAY,))
        shown, _ = client.call("memory_experience", action="explain", attempt_id=attempt)
        assert shown["status"] == "expired"
        refused, is_error = outcome(client, attempt)
        assert is_error, refused
        assert client.call("memory_search", query="billing deploy")[0].get("memory_attempt")
        assert rows(home / "memory_store.db", "SELECT 1 FROM memory_attempts WHERE attempt_id = ?", (attempt,)) == []
    finally:
        client.close()


def test_a_write_locked_store_gives_the_block_without_the_attempt_at_once(home):
    seed(home, [(DEPLOY, [])], EXP)
    client = Client(home).start()
    try:
        args = {"prompt": ASK, "cwd": ""}
        warm = client.request("tools/call", {"name": "memory_recall", "arguments": {**args, "session_id": "s1"}})
        assert "Memory attempt" in warm["result"]["content"][0]["text"]
        holder = sqlite3.connect(home / "memory_store.db", isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")  # another process mid-write
        try:
            started = time.monotonic()
            busy = client.request("tools/call", {"name": "memory_recall", "arguments": {**args, "session_id": "s2"}})
            elapsed = time.monotonic() - started
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        block = json.loads(busy["result"]["content"][0]["text"])["hookSpecificOutput"]["additionalContext"]
        assert DEPLOY in block and "Memory attempt" not in block
        assert elapsed < 3, elapsed  # well inside hook_deadline_seconds (8)
        after = client.request("tools/call", {"name": "memory_recall", "arguments": {**args, "session_id": "s3"}})
        attempt_of(json.loads(after["result"]["content"][0]["text"])["hookSpecificOutput"]["additionalContext"])
    finally:
        client.close()
    assert rows(home / "memory_store.db", "PRAGMA integrity_check") == [("ok",)]


def test_a_held_state_db_gives_the_search_without_the_attempt_and_leaves_an_orphan(home):
    seed(home, [(DEPLOY, [])], EXP)
    client = Client(home).start()
    try:
        linked = client.call("memory_search", query="billing deploy")[0]["memory_attempt"]
        holder = sqlite3.connect(home / "state.db", isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")  # another hook mid-write to state.db
        try:
            started = time.monotonic()
            found, is_error = client.call("memory_search", query="billing deploy")
            elapsed = time.monotonic() - started
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        assert not is_error and found["count"] == 1 and "memory_attempt" not in found
        assert elapsed < 2, elapsed
        # the receipt was committed to the store, its link was not: nobody can report against it
        (orphan,) = rows(home / "memory_store.db", "SELECT attempt_id FROM memory_attempts WHERE attempt_id != ?",
                         (linked,))[0]
        shown, _ = client.call("memory_experience", action="explain", attempt_id=orphan)
        assert shown["status"] == "not_recorded"
        assert outcome(client, orphan)[1] is True
    finally:
        client.close()


def _expired_outcome(home: Path) -> str:
    """An attempt with an outcome (Cyrillic correction), 31 days old, the journal then turned off."""
    seed(home, [(DEPLOY, [])], EXP)
    attempt = attempt_of(context(hook(home, "user-prompt", prompt(ASK))))
    client = Client(home).start()
    try:
        assert outcome(client, attempt, correction=CORRECTION)[0]["recorded"] is True
    finally:
        client.close()
    write(home / "memory_store.db", "UPDATE memory_attempts SET created_at = created_at - ?", (31 * DAY,))
    return attempt


def scrub_pending(home: Path) -> bool:
    """A file rewrite is owed (the engine's mark after text was deleted and the rewrite has not run)."""
    return bool(rows(home / "memory_store.db", "SELECT 1 FROM meta WHERE key = 'scrub_pending'"))


def test_a_hook_never_rewrites_the_file_for_journal_retention(home):
    """A rewrite of the whole database (VACUUM) has no bound on a large store or a slow disk: it never runs while a
    prompt waits. Expired outcome text goes at the next MCP server open, holomem status or export instead."""
    attempt = _expired_outcome(home)
    for settings in ("", EXP):  # the journal off (retention still owed), then on (a receipt would prune)
        (home / "config.toml").write_text(settings, encoding="utf-8")
        text = context(hook(home, "user-prompt", prompt("Which server runs systemd for staging?", session=settings)))
        assert DEPLOY in text and "Memory attempt" not in text  # the block goes out; its receipt waits for retention
        assert holds(file_bytes(home), CORRECTION) and not scrub_pending(home)  # no rewrite, none half-done
    assert holomem(home, "status").returncode == 0
    data = file_bytes(home)
    assert not holds(data, CORRECTION) and attempt.encode() not in data
    assert "Memory attempt" in context(hook(home, "user-prompt", prompt(ASK, session="after")))  # receipts again


def test_a_recall_through_the_server_leaves_a_pending_rewrite_to_a_tool_call(home):
    """memory_recall answers a prompt (a hook of type mcp_tool): a rewrite a blocked forget left is not done there,
    but by the next memory tool call."""
    deploy, other = seed(home, [(DEPLOY, []), ("Bob runs a bakery on the corner", [])], EXP)
    wal(home)
    client = Client(home).start()
    try:
        client.call("memory_stats")  # the store is open
        reader = sqlite3.connect(home / "memory_store.db", isolation_level=None)
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM facts").fetchone()  # another process mid-read: the forget cannot rewrite
        try:
            assert client.call("memory_forget", fact_id=other)[0]["scrub"] == "pending"
        finally:
            reader.execute("COMMIT")
            reader.close()
        answer = client.request("tools/call", {"name": "memory_recall",
                                               "arguments": {"prompt": ASK, "session_id": "s1", "cwd": ""}})
        block = json.loads(answer["result"]["content"][0]["text"])["hookSpecificOutput"]["additionalContext"]
        assert f"#{deploy}" in block and "Memory attempt" in block
        assert scrub_pending(home)  # not rewritten while the prompt waited
        client.call("memory_search", query="billing deploy")  # a tool call: the rewrite runs here
        assert not scrub_pending(home) and b"bakery on the corner" not in file_bytes(home)
    finally:
        client.close()


def test_status_removes_expired_outcome_text(home):
    _expired_outcome(home)
    (home / "config.toml").write_text("", encoding="utf-8")
    done = holomem(home, "status")
    assert done.returncode == 0, done.stderr[-800:]
    assert not holds(file_bytes(home), CORRECTION)


def test_export_journal_lists_attempts_and_outcomes_without_fact_text(home):
    seed(home, [(DEPLOY, [])], EXP)
    from_hook = attempt_of(context(hook(home, "user-prompt", prompt(ASK, session="session-one"))))
    client = Client(home).start()
    try:
        from_search = client.call("memory_search", query="billing deploy")[0]["memory_attempt"]
        assert outcome(client, from_hook, correction=CORRECTION, task="deploy question")[0]["recorded"] is True
        old = client.call("memory_search", query="systemd staging")[0]["memory_attempt"]
    finally:
        client.close()
    write(home / "memory_store.db", "UPDATE memory_attempts SET created_at = created_at - ? WHERE attempt_id = ?",
          (31 * DAY, old))
    (home / "config.toml").write_text("", encoding="utf-8")  # the journal turned off since: its data is still read
    done = export_cli(home, "--journal")
    assert done.returncode == 0, done.stderr.decode(errors="replace")[-800:]
    page = done.stdout.decode("utf-8")
    assert page.startswith("# Memory journal (3 attempts")
    sections = {next(a for a in (from_hook, from_search, old) if a in part): part for part in page.split("\n## ")[1:]}
    assert len(sections) == 3
    assert "prefetch" in sections[from_hook] and "claude-code" in sections[from_hook]
    assert "corrected" in sections[from_hook] and CORRECTION in sections[from_hook]
    assert "deploy question" in sections[from_hook]
    assert " search " in sections[from_search] and "corrected" not in sections[from_search]
    assert "expired" in sections[old]
    assert DEPLOY not in page and "\r\n" not in page


def test_export_journal_only_reads_the_database(home, endpoint):
    """holomem export --journal opens the store read-only: not even the work every writer open does (vectors of a
    switched-off search by meaning deleted, a rewrite a blocked forget owes) touches the file."""
    target = store_with_vectors(home, endpoint)
    (home / "config.toml").write_text(meaning_config(endpoint) + EXP, encoding="utf-8")
    attempt = attempt_of(context(hook(home, "user-prompt", prompt(QUESTION))))
    (home / "config.toml").write_text(EXP, encoding="utf-8")  # search by meaning off since: a writer would drop vectors
    write(home / "memory_store.db", "INSERT INTO meta (key, value) VALUES ('scrub_pending', 'a1b2c3d4')")
    before, vectors = file_bytes(home), rows(home / "memory_store.db", "SELECT COUNT(*) FROM fact_embeddings")
    done = export_cli(home, "--journal")
    assert done.returncode == 0, done.stderr.decode(errors="replace")[-800:]
    page = done.stdout.decode("utf-8")
    assert attempt in page and f"#{target}" in page
    assert rows(home / "memory_store.db", "SELECT COUNT(*) FROM fact_embeddings") == vectors != [(0,)]
    assert scrub_pending(home) and file_bytes(home) == before

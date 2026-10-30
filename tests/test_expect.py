"""A version precondition on every write (issue #2): `expect` {"<fact_id>": <version_id>}, as memory_history shows them.

A review shows a person a fact and applies their answer later (a nightly report, another session). Without a
precondition it had to read the history right before the write and compare: another session could change the fact in
between, and the answer landed on a text the person never saw. With `expect` the write applies only while every named
fact is still at that version, checked in the write's own transaction; otherwise nothing is written and the answer
names each fact's current version id. Without `expect` a write is as before."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from test_mcp_server import Client


@pytest.fixture
def home(tmp_path) -> Path:
    return tmp_path / "holo"


@pytest.fixture
def mcp(home):
    client = Client(home).start()
    yield client
    client.close()


def add(mcp, content: str) -> int:
    answer, is_error = mcp.call("memory_add", content=content, origin="said", valid_from="2026-01-05")
    assert not is_error and answer["status"] == "added", answer
    return answer["fact_id"]


def versions(mcp, fact_id: int) -> list[dict]:
    return mcp.call("memory_history", fact_id=fact_id)[0]["versions"]


def head(mcp, fact_id: int) -> int:
    return [v for v in versions(mcp, fact_id) if v["status"] != "corrected"][-1]["version_id"]


def store_row(home: Path, fact_id: int):
    conn = sqlite3.connect(home / "memory_store.db")
    try:
        return conn.execute("SELECT content, trust_score, tags, category FROM facts WHERE fact_id = ?",
                            (fact_id,)).fetchone()
    finally:
        conn.close()


def moved_on(mcp, text: str) -> tuple[int, int, int]:
    """A fact the person saw at one version, which another session then changed: (fact_id, seen, now)."""
    fact_id = add(mcp, text)
    seen = head(mcp, fact_id)
    changed, is_error = mcp.call("memory_change", fact_id=fact_id, content=text + " again", valid_from="2026-02-01")
    assert not is_error, changed
    return fact_id, seen, head(mcp, fact_id)


WRITES = {
    "memory_change": {"content": "Lena lives in Porto", "valid_from": "2026-03-01"},
    "memory_correct": {"content": "Lena lives in Braga"},
    "memory_end": {"valid_to": "2026-03-01"},
    "memory_forget": {},
    "memory_feedback": {"action": "unhelpful"},
    "memory_update": {"keywords": ["relocation"], "category": "project"},
}


@pytest.mark.parametrize("tool", list(WRITES))
def test_a_stale_expect_writes_nothing_and_names_the_current_version(mcp, home, tool):
    fact_id, seen, now = moved_on(mcp, "Lena lives in Lisbon")
    before_versions, before_row = versions(mcp, fact_id), store_row(home, fact_id)
    answer, is_error = mcp.call(tool, fact_id=fact_id, expect={str(fact_id): seen}, **WRITES[tool])
    assert is_error, answer
    assert answer["changed_since"] == {str(fact_id): now}, answer
    assert versions(mcp, fact_id) == before_versions  # no version written, none closed
    assert store_row(home, fact_id) == before_row  # text, trust, keywords, category as they were


@pytest.mark.parametrize("tool", list(WRITES))
def test_a_current_expect_writes_as_before(mcp, home, tool):
    fact_id, _, now = moved_on(mcp, "Lena lives in Lisbon")
    before_row = store_row(home, fact_id)
    answer, is_error = mcp.call(tool, fact_id=fact_id, expect={str(fact_id): now}, **WRITES[tool])
    assert not is_error, answer
    assert store_row(home, fact_id) != before_row  # the write happened (a forgotten fact has no row)
    if tool == "memory_forget":
        assert answer["forgotten"] is True and answer["scrub"] == "done", answer  # rewritten after its transaction


def test_another_named_fact_that_moved_on_stops_the_write_too(mcp, home):
    """expect may name more than the fact written: the person decided on both texts together."""
    lena = add(mcp, "Lena lives in Lisbon")
    mark, seen, now = moved_on(mcp, "Mark works at the harbour")
    answer, is_error = mcp.call("memory_end", fact_id=lena, valid_to="2026-03-01",
                                expect={str(lena): head(mcp, lena), str(mark): seen})
    assert is_error and answer["changed_since"] == {str(mark): now}, answer
    assert versions(mcp, lena)[-1]["status"] == "current"


def test_a_forgotten_fact_has_no_version(mcp):
    fact_id = add(mcp, "Lena lives in Lisbon")
    seen = head(mcp, fact_id)
    other = add(mcp, "Mark works at the harbour")
    assert not mcp.call("memory_forget", fact_id=fact_id)[1]
    answer, is_error = mcp.call("memory_feedback", fact_id=other, action="helpful", expect={str(fact_id): seen})
    assert is_error and answer["changed_since"] == {str(fact_id): None}, answer


def test_merge_names_the_current_versions_too(mcp):
    keep, copy = add(mcp, "Lena lives in Lisbon"), add(mcp, "Lena's home is in Lisbon")
    seen = head(mcp, copy)
    assert not mcp.call("memory_change", fact_id=copy, content="Lena's home is Lisbon", valid_from="2026-02-01")[1]
    answer, is_error = mcp.call("memory_merge", fact_id=keep, **{"from": [copy]}, expect={str(copy): seen})
    assert is_error and answer["changed_since"] == {str(copy): head(mcp, copy)}, answer


@pytest.mark.parametrize("expect", [[1, 2], {"one": 3}, {"1": "3"}, {"1": True}, "1:3"])
def test_expect_must_be_fact_ids_to_version_ids(mcp, expect):
    fact_id = add(mcp, "Lena lives in Lisbon")
    answer, is_error = mcp.call("memory_end", fact_id=fact_id, valid_to="2026-03-01", expect=expect)
    assert is_error and "expect" in answer["error"], answer
    assert versions(mcp, fact_id)[-1]["status"] == "current"


def test_a_text_an_older_release_wrote_is_a_version_the_person_did_not_see(mcp, home):
    """An older release (or another program) changed the text in place, without a version: the history still shows
    the old text under the version the person saw. The write must not land on the new text as if it were that one."""
    fact_id = add(mcp, "Lena lives in Lisbon")
    seen = head(mcp, fact_id)
    conn = sqlite3.connect(home / "memory_store.db")
    with conn:
        conn.execute("UPDATE facts SET content = 'Lena lives in Porto' WHERE fact_id = ?", (fact_id,))
    conn.close()
    answer, is_error = mcp.call("memory_correct", fact_id=fact_id, content="Lena lives in Braga",
                                expect={str(fact_id): seen})
    assert is_error, answer
    now = head(mcp, fact_id)
    assert now != seen and answer["changed_since"] == {str(fact_id): now}, answer  # the version the answer names exists
    assert versions(mcp, fact_id)[-1]["content"] == "Lena lives in Porto"
    assert store_row(home, fact_id)[0] == "Lena lives in Porto"


def test_the_same_fact_named_twice_is_refused(mcp):
    """"1" and "01" are one fact: keeping only one of the two versions would drop a condition the caller set."""
    fact_id, seen, now = moved_on(mcp, "Lena lives in Lisbon")
    for expect in ({str(fact_id): seen, "0" + str(fact_id): now}, {str(fact_id): now, f" {fact_id} ": seen}):
        answer, is_error = mcp.call("memory_end", fact_id=fact_id, valid_to="2026-03-01", expect=expect)
        assert is_error and "expect" in answer["error"], answer
        assert versions(mcp, fact_id)[-1]["status"] == "current"

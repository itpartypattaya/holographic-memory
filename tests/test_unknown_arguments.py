"""An argument a tool does not know is refused, and nothing is written. Before, it was dropped without a word: an
agent that passed `expect` to a release without it (0.4.0) got its write applied unchecked, and a misspelt key
(`limt`, `valid_form`) did nothing while the call looked fine."""
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


def store_row(home: Path, fact_id: int):
    conn = sqlite3.connect(home / "memory_store.db")
    try:
        return conn.execute("SELECT content, trust_score, tags, category FROM facts WHERE fact_id = ?",
                            (fact_id,)).fetchone()
    finally:
        conn.close()


def test_a_write_with_an_unknown_argument_writes_nothing_and_names_it(mcp, home):
    fact_id = add(mcp, "Lena lives in Lisbon")
    before = store_row(home, fact_id)
    answer, is_error = mcp.call("memory_end", fact_id=fact_id, valid_to="2026-03-01", expected={str(fact_id): 1})
    assert is_error and "expected" in answer["error"], answer
    assert store_row(home, fact_id) == before
    history = mcp.call("memory_history", fact_id=fact_id)[0]["versions"]
    assert history[-1]["status"] == "current"


def test_a_misspelt_argument_of_a_read_is_refused_too(mcp):
    add(mcp, "Lena lives in Lisbon")
    answer, is_error = mcp.call("memory_search", query="Lena", limt=1)
    assert is_error and "limt" in answer["error"], answer


def test_every_tool_says_in_its_schema_that_it_takes_nothing_else(mcp):
    """memory_recall aside: the hook's own entry, which takes what the client's hook sends and never fails a prompt."""
    tools = [t for t in mcp.request("tools/list", {})["result"]["tools"] if t["name"] != "memory_recall"]
    assert tools and all(t["inputSchema"].get("additionalProperties") is False for t in tools), \
        [t["name"] for t in tools if t["inputSchema"].get("additionalProperties") is not False]

"""memory_contradict: the stored facts that contradict each other, through the MCP server.

The requirement: asked "what in memory contradicts itself" (or before a clean-up), the agent gets every pair of
current facts that make different claims about the same person or thing — both ids and texts, the score and the
names they share — and no pair of facts about different subjects, of facts that only add detail to each other, or of
different topics about one person; the call changes nothing, and once the person settled a pair (one fact ended or
corrected) it is no longer listed.

The engine's own reading of a conflict is tested in tests/engine (test_holographic_v020 / v021); these tests hold the
tool: its schema, its answer, its limits and that it is a read.
"""
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


def add(mcp, content: str, entities: list[str], category: str = "general") -> int:
    answer, is_error = mcp.call("memory_add", content=content, entities=entities, category=category, origin="said")
    assert not is_error and answer["status"] == "added", answer
    return answer["fact_id"]


def pairs_of(answer: dict) -> list[set[int]]:
    return [{p["fact_a"]["fact_id"], p["fact_b"]["fact_id"]} for p in answer["pairs"]]


def a_store(mcp) -> dict[str, int]:
    """Two facts that contradict each other, among facts that must not be paired with them or with each other."""
    return {
        "lisbon": add(mcp, "Lena lives in Lisbon", ["Lena"]),
        "porto": add(mcp, "Lena lives in Porto", ["Lena"]),
        # same subject, another topic
        "nurse": add(mcp, "Lena works as a nurse at the city hospital", ["Lena"]),
        # same claim, another subject
        "oscar": add(mcp, "Oscar lives in Lisbon", ["Oscar"]),
        # one only adds detail to the other
        "tea": add(mcp, "Oscar drinks green tea", ["Oscar"]),
        "tea_detail": add(mcp, "Oscar drinks green tea with honey", ["Oscar"]),
        # no names at all
        "deploy": add(mcp, "The billing service deploys to staging with systemd", []),
    }


def facts_rows(home: Path) -> list[tuple]:
    conn = sqlite3.connect(home / "memory_store.db")
    try:
        return conn.execute("SELECT fact_id, content, category, tags, trust_score, retrieval_count, helpful_count, "
                            "created_at, updated_at FROM facts ORDER BY fact_id").fetchall()
    finally:
        conn.close()


def test_it_is_listed_as_a_read_with_a_closed_schema(mcp):
    tools = {t["name"]: t for t in mcp.request("tools/list")["result"]["tools"]}
    assert "memory_contradict" in tools
    tool = tools["memory_contradict"]
    assert tool["annotations"]["readOnlyHint"] is True
    assert tool["annotations"]["destructiveHint"] is False
    schema = tool["inputSchema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["limit"] == {"type": "integer", "minimum": 1, "maximum": 100}
    assert not schema.get("required")
    # the description says when to call it and what to do with a pair
    for word in ("contradict", "memory_correct", "memory_end"):
        assert word in tool["description"], word


def test_the_pair_that_contradicts_is_reported_and_nothing_else(mcp):
    ids = a_store(mcp)
    answer, is_error = mcp.call("memory_contradict")
    assert not is_error, answer
    assert pairs_of(answer) == [{ids["lisbon"], ids["porto"]}], answer
    assert answer["count"] == 1
    pair = answer["pairs"][0]
    assert {pair["fact_a"]["content"], pair["fact_b"]["content"]} == {"Lena lives in Lisbon", "Lena lives in Porto"}
    assert pair["shared_entities"] == ["lena"]
    assert 0 < pair["contradiction_score"] <= 1
    assert "memory_correct" in answer["hint"] and "memory_end" in answer["hint"]


def test_a_flipped_preference_is_a_contradiction_too(mcp):
    likes = add(mcp, "Oscar likes jazz concerts", ["Oscar"])
    not_likes = add(mcp, "Oscar does not like jazz concerts", ["Oscar"])
    add(mcp, "Oscar does not like cooking", ["Oscar"])  # another topic, negated: no conflict with the jazz
    answer, is_error = mcp.call("memory_contradict")
    assert not is_error, answer
    assert pairs_of(answer) == [{likes, not_likes}], answer


def test_an_empty_or_consistent_store_has_no_pairs_and_no_hint(mcp):
    answer, is_error = mcp.call("memory_contradict")
    assert not is_error and answer == {"pairs": [], "count": 0}, answer
    add(mcp, "Lena lives in Lisbon", ["Lena"])
    add(mcp, "Lena works as a nurse at the city hospital", ["Lena"])
    answer, is_error = mcp.call("memory_contradict")
    assert not is_error and answer == {"pairs": [], "count": 0}, answer


def test_the_call_changes_nothing(mcp, home):
    a_store(mcp)
    before = facts_rows(home)
    for _ in range(2):
        answer, is_error = mcp.call("memory_contradict")
        assert not is_error and answer["count"] == 1, answer
    assert facts_rows(home) == before


def test_a_settled_pair_is_no_longer_listed(mcp):
    ids = a_store(mcp)
    ended, is_error = mcp.call("memory_end", fact_id=ids["lisbon"])
    assert not is_error, ended
    answer, is_error = mcp.call("memory_contradict")
    assert not is_error and answer["pairs"] == [], answer

    second = add(mcp, "Oscar lives in Madrid", ["Oscar"])
    answer = mcp.call("memory_contradict")[0]
    assert pairs_of(answer) == [{ids["oscar"], second}], answer
    corrected, is_error = mcp.call("memory_correct", fact_id=second, content="Oscar visits Madrid every spring")
    assert not is_error, corrected
    answer, is_error = mcp.call("memory_contradict")
    assert not is_error and answer["pairs"] == [], answer


def test_limit_keeps_the_strongest_pairs(mcp):
    ids = a_store(mcp)
    budget = add(mcp, "The Atlas project budget is 500 euros", ["Atlas"], category="project")
    budget2 = add(mcp, "The Atlas project budget is 600 euros", ["Atlas"], category="project")
    everything = mcp.call("memory_contradict")[0]
    assert sorted(map(sorted, pairs_of(everything))) == sorted([sorted([ids["lisbon"], ids["porto"]]),
                                                                sorted([budget, budget2])]), everything
    scores = [p["contradiction_score"] for p in everything["pairs"]]
    assert scores == sorted(scores, reverse=True)
    one, is_error = mcp.call("memory_contradict", limit=1)
    assert not is_error and one["count"] == 1 and one["pairs"] == everything["pairs"][:1], one


def test_limit_out_of_range_is_held_to_the_range_as_in_other_tools(mcp):
    a_store(mcp)
    for given in (0, -5):
        answer, is_error = mcp.call("memory_contradict", limit=given)
        assert not is_error and answer["count"] == 1, (given, answer)
    answer, is_error = mcp.call("memory_contradict", limit=10_000)
    assert not is_error and answer["count"] == 1, answer
    answer, is_error = mcp.call("memory_contradict", limit="many")  # not a number: an error, nothing listed
    assert is_error and "pairs" not in answer, answer


def test_category_keeps_the_pairs_of_that_category(mcp):
    ids = a_store(mcp)
    budget = add(mcp, "The Atlas project budget is 500 euros", ["Atlas"], category="project")
    budget2 = add(mcp, "The Atlas project budget is 600 euros", ["Atlas"], category="project")
    project, is_error = mcp.call("memory_contradict", category="project")
    assert not is_error and pairs_of(project) == [{budget, budget2}], project
    general, is_error = mcp.call("memory_contradict", category="general")
    assert not is_error and pairs_of(general) == [{ids["lisbon"], ids["porto"]}], general
    tools = mcp.call("memory_contradict", category="tool")[0]
    assert tools == {"pairs": [], "count": 0}, tools


def test_min_trust_leaves_out_facts_below_it(mcp, home):
    ids = a_store(mcp)
    for _ in range(4):  # Porto turned out outdated each time it was used: its trust falls
        answer, is_error = mcp.call("memory_feedback", fact_id=ids["porto"], action="unhelpful")
        assert not is_error, answer
    trust = {r[0]: r[4] for r in facts_rows(home)}
    floor = (trust[ids["porto"]] + trust[ids["lisbon"]]) / 2
    assert trust[ids["porto"]] < floor < trust[ids["lisbon"]], trust
    answer, is_error = mcp.call("memory_contradict", min_trust=floor)
    assert not is_error and answer["pairs"] == [], answer
    answer, is_error = mcp.call("memory_contradict")  # by default every fact counts, the doubtful ones too
    assert not is_error and pairs_of(answer) == [{ids["lisbon"], ids["porto"]}], answer


def test_an_argument_it_does_not_take_is_refused(mcp):
    a_store(mcp)
    answer, is_error = mcp.call("memory_contradict", treshold=0.1)
    assert is_error and "treshold" in answer["error"], answer

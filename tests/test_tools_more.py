"""The tools added after the first release: merge, keywords (update/enrich), names (entities/entity), reason, review.

Requirements:
- memory_merge: the facts in 'from' become part of fact_id; search then shows one fact; the merged text added again
  is answered "merged" and not stored as new; history of the kept fact lists what was merged; keywords of both are
  kept (the kept fact's first); 'from' that is not a list of integers is refused with nothing changed;
- memory_update gives a fact keywords: a word that is only a keyword then finds it; memory_enrich lists facts without
  keywords and stops listing one that has them; a new text is refused (that is change/correct);
- memory_entity alias: a fact stored about one name is found by probing the other; delete removes a name, not facts;
- memory_reason returns only the facts that involve every named entity;
- memory_review lists a near-duplicate and points to memory_merge; nothing is changed by it.
"""
from __future__ import annotations

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


def call(mcp, tool: str, arguments: dict) -> tuple[dict, bool]:
    """Client.call with arguments that may be named 'name' (its own first parameter)."""
    import json

    result = mcp.request("tools/call", {"name": tool, "arguments": arguments})["result"]
    return json.loads(result["content"][0]["text"]), result["isError"]


def add(mcp, content: str, **extra) -> int:
    answer, is_error = mcp.call("memory_add", content=content, origin="said", **extra)
    assert not is_error and answer["status"] == "added", answer
    return answer["fact_id"]


def test_merge_keeps_one_fact_and_both_histories(mcp):
    keep = add(mcp, "The bakery on Third Street opens at seven", keywords=["breakfast"])
    copy = add(mcp, "Third Street's bakery starts selling bread at 7 am", keywords=["pastry"])
    merged, is_error = mcp.call("memory_merge", fact_id=keep, **{"from": [copy]})
    assert not is_error and merged["merged"] == [copy]
    assert merged["keywords"] == ["breakfast", "pastry"]
    found, _ = mcp.call("memory_search", query="Third Street bakery")
    assert [r["fact_id"] for r in found["results"]] == [keep]
    again, _ = mcp.call("memory_add", content="Third Street's bakery starts selling bread at 7 am", origin="said")
    assert again["status"] == "merged" and again["stored"] is False and again["fact_id"] == keep
    history, _ = mcp.call("memory_history", fact_id=keep)
    assert copy in [m["fact_id"] for m in history.get("merged_from", [])]


def test_merge_refuses_ids_that_are_not_integers(mcp):
    keep = add(mcp, "The bakery on Third Street opens at seven")
    other = add(mcp, "Third Street's bakery starts selling bread at 7 am")
    answer, is_error = mcp.call("memory_merge", fact_id=keep, **{"from": [str(other)]})
    assert is_error and "from" in answer["error"]
    found, _ = mcp.call("memory_search", query="Third Street bakery")
    assert {r["fact_id"] for r in found["results"]} == {keep, other}


def test_keywords_make_a_fact_findable_and_leave_the_enrich_list(mcp):
    fid = add(mcp, "Mark gets around town on a blue Vespa")
    listed, _ = mcp.call("memory_enrich")
    assert fid in [f["fact_id"] for f in listed["facts"]]
    before, _ = mcp.call("memory_search", query="скутер")
    assert before["count"] == 0
    updated, is_error = mcp.call("memory_update", fact_id=fid, keywords=["скутер", "motorbike"])
    assert not is_error and updated == {"updated": True, "keywords": ["скутер", "motorbike"]}
    after, _ = mcp.call("memory_search", query="скутер")
    assert [r["fact_id"] for r in after["results"]] == [fid]
    listed, _ = mcp.call("memory_enrich")
    assert fid not in [f["fact_id"] for f in listed["facts"]]


def test_update_does_not_take_a_new_text(mcp):
    fid = add(mcp, "Mark gets around town on a blue Vespa")
    # with valid keywords beside it, so a refusal can only come from the text (review: a bare content was refused
    # merely for missing keywords, and text plus keywords was accepted with the text silently dropped)
    answer, is_error = mcp.call("memory_update", fact_id=fid, content="Mark walks everywhere", keywords=["scooter"])
    assert is_error and "content" in answer["error"]
    found, _ = mcp.call("memory_search", query="blue Vespa")
    assert found["results"][0]["content"] == "Mark gets around town on a blue Vespa"
    listed, _ = mcp.call("memory_enrich")
    assert fid in [f["fact_id"] for f in listed["facts"]]  # nothing changed: still no keywords


def test_an_alias_finds_the_facts_of_the_other_name(mcp):
    fid = add(mcp, "Robert runs the bicycle repair shop", entities=["Robert"])
    answer, is_error = call(mcp, "memory_entity", {"action": "alias", "name": "Robert", "other": "Bob"})
    assert not is_error, answer
    probed, _ = mcp.call("memory_probe", entity="Bob")
    assert [r["fact_id"] for r in probed["results"]] == [fid]
    names, _ = mcp.call("memory_entities")
    assert any(e.get("name") == "Robert" and "Bob" in str(e.get("aliases")) for e in names["entities"])


def test_deleting_a_name_keeps_its_facts(mcp):
    fid = add(mcp, "Robert runs the bicycle repair shop", entities=["Robert"])
    answer, is_error = mcp.call("memory_entity", action="delete", names=["Robert"])
    assert not is_error, answer
    names, _ = mcp.call("memory_entities")
    assert not any(e.get("name") == "Robert" for e in names["entities"])
    found, _ = mcp.call("memory_search", query="bicycle repair shop")
    assert [r["fact_id"] for r in found["results"]] == [fid]


def test_reason_joins_every_named_entity(mcp):
    both = add(mcp, "Robert and Lena moved to Lisbon together", entities=["Robert", "Lena"])
    add(mcp, "Robert runs the bicycle repair shop", entities=["Robert"])
    found, is_error = mcp.call("memory_reason", entities=["Robert", "Lena"])
    assert not is_error and [r["fact_id"] for r in found["results"]] == [both]


def test_review_points_a_near_duplicate_to_merge(mcp):
    first = add(mcp, "The bakery opens at seven every morning")
    second = add(mcp, "The bakery opens at seven every morning daily")
    review, is_error = mcp.call("memory_review")
    assert not is_error
    reasons = {c["fact_id"]: c["reason"] for c in review["candidates"]}
    assert set(reasons) & {first, second} and any(r.startswith("near-duplicate") for r in reasons.values())
    assert "memory_merge" in review["hint"]
    found, _ = mcp.call("memory_search", query="bakery opens seven")
    assert {r["fact_id"] for r in found["results"]} == {first, second}


def test_dates_take_today_and_yesterday_as_the_tools_promise(mcp):
    """The tools' schema says a date is "YYYY-MM-DD, 'today' or 'yesterday'"; an agent that believes it must not get
    an error on its first write (seen 2026-10-30: valid_from "today" was refused). Any case, spaces around."""
    import datetime

    today = datetime.date.today()
    yesterday, earlier = today - datetime.timedelta(days=1), (today - datetime.timedelta(days=3)).isoformat()

    def versions(fact_id):
        return mcp.call("memory_history", fact_id=fact_id)[0]["versions"]

    added, is_error = mcp.call("memory_add", content="Robert started a pottery course", valid_from="today",
                               origin="said")
    assert not is_error, added
    assert versions(added["fact_id"])[-1]["valid_from"] == today.isoformat()

    moved = add(mcp, "Lena lives in Lisbon", valid_from=earlier)
    changed, is_error = mcp.call("memory_change", fact_id=moved, content="Lena lives in Porto", valid_from=" Today ")
    assert not is_error, changed
    assert versions(moved)[-1]["valid_from"] == today.isoformat()

    ended = add(mcp, "Mark rents a flat on the river", valid_from=earlier)
    answer, is_error = mcp.call("memory_end", fact_id=ended, valid_to="YESTERDAY")
    assert not is_error, answer
    assert versions(ended)[-1]["valid_to"] == yesterday.isoformat()

    found, is_error = mcp.call("memory_search", query="pottery course", as_of="today")
    assert not is_error and [r["fact_id"] for r in found["results"]] == [added["fact_id"]]
    refused, is_error = mcp.call("memory_add", content="Robert likes green tea", valid_from="tomorrow", origin="said")
    assert is_error and "YYYY-MM-DD" in refused["error"]

"""Keywords on memory_change and memory_correct (ported 2026-11-02 from the Hermes plugin's #74, issue #73, as
holographic-memory's own change).

- memory_change without keywords dropped the old text's keywords without a word: the agent believed its search aids
  were still there. They still go (a word for the old text may mislead for the new: "lisbon" after a move to Porto),
  and the answer names them under keywords_cleared, with how to give the new text its own.
- memory_correct took no keywords at all (an unknown argument): it takes them now, and they replace the fact's, as on
  memory_update; without them the fact keeps its own. A correction of the date alone writes them too; a correction
  that changes nothing writes nothing and says so (keywords_not_set).
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")


@pytest.fixture
def tools(tmp_path, monkeypatch):
    from holomem import config
    from holomem.memory import Context, open_memory
    from holomem.state import State
    from holomem.tools import Tools

    monkeypatch.setenv("HOLOMEM_HOME", str(tmp_path / "holo"))
    cfg = config.load()
    ctx = Context(client="claude-code", session_id="s1")
    memory = open_memory(cfg, ctx)
    yield Tools(memory, ctx, State(cfg.state_path))
    memory.store.close()


def call(tools, name: str, **args) -> dict:
    answer, is_error = tools.call(name, args)
    assert not is_error, answer
    return answer


def kept(tools, fact_id: int) -> list[str]:
    from holomem.tools import split_tags

    return split_tags(tools.store.fact_tags(fact_id))[1]


def found(tools, query: str) -> list[int]:
    return [r["fact_id"] for r in call(tools, "memory_search", query=query).get("results", [])]


def test_a_change_without_keywords_names_those_it_dropped(tools):
    fact = call(tools, "memory_add", content="Lena lives in Lisbon", keywords=["home", "portugal"], origin="said")["fact_id"]
    answer = call(tools, "memory_change", fact_id=fact, content="Lena lives in Porto", valid_from="2026-02-01")
    assert answer["keywords_cleared"] == ["home", "portugal"], answer
    assert "memory_update" in answer["hint"], answer
    assert kept(tools, fact) == []


def test_a_change_with_keywords_clears_nothing(tools):
    fact = call(tools, "memory_add", content="Lena lives in Lisbon", keywords=["home"], origin="said")["fact_id"]
    answer = call(tools, "memory_change", fact_id=fact, content="Lena lives in Porto", keywords=["home", "north"])
    assert "keywords_cleared" not in answer and kept(tools, fact) == ["home", "north"], answer


def test_a_change_of_a_fact_without_keywords_says_nothing_of_them(tools):
    fact = call(tools, "memory_add", content="Lena lives in Lisbon", origin="said")["fact_id"]
    assert "keywords_cleared" not in call(tools, "memory_change", fact_id=fact, content="Lena lives in Porto")


def test_a_correction_with_keywords_replaces_them(tools):
    fact = call(tools, "memory_add", content="Lena lives in Lisbon", keywords=["home"], origin="said")["fact_id"]
    answer = call(tools, "memory_correct", fact_id=fact, content="Lena lives in Lagos", keywords=["algarve", "beach"])
    assert answer["status"] == "corrected" and answer["keywords"] == ["algarve", "beach"], answer
    assert kept(tools, fact) == ["algarve", "beach"]
    assert found(tools, "algarve") == [fact]


def test_a_correction_without_keywords_keeps_them(tools):
    fact = call(tools, "memory_add", content="Lena lives in Lisbon", keywords=["home"], origin="said")["fact_id"]
    call(tools, "memory_correct", fact_id=fact, content="Lena lives in Lagos")
    assert kept(tools, fact) == ["home"]


def test_a_correction_of_the_date_alone_writes_the_keywords(tools):
    fact = call(tools, "memory_add", content="Lena lives in Lisbon", keywords=["home"], origin="said")["fact_id"]
    answer = call(tools, "memory_correct", fact_id=fact, valid_from="2025-03-01", keywords=["capital"])
    assert answer["status"] == "corrected", answer
    assert kept(tools, fact) == ["capital"] and found(tools, "capital") == [fact]


def test_a_correction_that_changes_nothing_writes_no_keywords_and_says_so(tools):
    fact = call(tools, "memory_add", content="Lena lives in Lisbon", keywords=["home"], origin="said")["fact_id"]
    answer = call(tools, "memory_correct", fact_id=fact, content="Lena lives in Lisbon", keywords=["capital"])
    assert answer["status"] == "unchanged" and "memory_update" in answer["keywords_not_set"], answer
    assert kept(tools, fact) == ["home"]

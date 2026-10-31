"""Keywords a write does not keep are named in its answer (`keywords_dropped`), not cut without a word.

A fact keeps at most 12 keywords of up to 40 characters. Before, the 13th and later and any longer one were dropped
silently: the agent believed its search aids were stored. A repeat of a word kept is not a loss and is not named."""
from __future__ import annotations

from pathlib import Path

import pytest

from test_mcp_server import Client

FOURTEEN = [f"word{i}" for i in range(1, 15)]
TOO_LONG = "x" * 41


@pytest.fixture
def home(tmp_path) -> Path:
    return tmp_path / "holo"


@pytest.fixture
def mcp(home):
    client = Client(home).start()
    yield client
    client.close()


def add(mcp, content: str, **extra) -> dict:
    answer, is_error = mcp.call("memory_add", content=content, origin="said", **extra)
    assert not is_error and answer["status"] == "added", answer
    return answer


def test_an_add_names_the_keywords_past_the_twelfth_and_the_too_long(mcp):
    answer = add(mcp, "Lena lives in Lisbon", keywords=FOURTEEN + [TOO_LONG, "word1"])
    assert answer["keywords"] == FOURTEEN[:12]
    assert answer["keywords_dropped"] == FOURTEEN[12:] + [TOO_LONG], answer  # the repeated word1 is no loss


def test_an_add_that_keeps_every_keyword_says_nothing_of_dropping(mcp):
    answer = add(mcp, "Lena lives in Lisbon", keywords=["city", "home", "city"])
    assert "keywords_dropped" not in answer, answer


def test_a_change_and_an_update_name_them_too(mcp):
    fact_id = add(mcp, "Lena lives in Lisbon")["fact_id"]
    changed, is_error = mcp.call("memory_change", fact_id=fact_id, content="Lena lives in Porto",
                                 keywords=FOURTEEN, valid_from="2026-02-01")
    assert not is_error and changed["keywords_dropped"] == FOURTEEN[12:], changed
    updated, is_error = mcp.call("memory_update", fact_id=fact_id, keywords=[TOO_LONG, "porto"])
    assert not is_error and updated["keywords"] == ["porto"] and updated["keywords_dropped"] == [TOO_LONG], updated


def test_a_merge_names_what_the_joined_keywords_leave_out(mcp):
    keep = add(mcp, "Lena lives in Lisbon", keywords=FOURTEEN[:8])["fact_id"]
    copy = add(mcp, "Lena's home is Lisbon", keywords=FOURTEEN[6:])["fact_id"]  # 6 of its 8 are new to the keep
    merged, is_error = mcp.call("memory_merge", fact_id=keep, **{"from": [copy]})
    assert not is_error and merged["keywords"] == FOURTEEN[:12], merged
    assert merged["keywords_dropped"] == FOURTEEN[12:], merged


def test_keywords_given_as_one_comma_separated_string_are_counted_word_by_word(mcp):
    """The schema takes a list; a string of comma-separated words is read word by word, and so is what it drops."""
    keep = add(mcp, "Lena lives in Lisbon")["fact_id"]
    copy = add(mcp, "Lena's home is Lisbon")["fact_id"]
    merged, is_error = mcp.call("memory_merge", fact_id=keep, keywords=", ".join(FOURTEEN), **{"from": [copy]})
    assert not is_error and merged["keywords"] == FOURTEEN[:12], merged
    assert merged["keywords_dropped"] == FOURTEEN[12:], merged

"""0.8.0: fact history (change / end / correct / revert / forget, as_of), provenance, tombstones. docs/history.md
is the contract these tests check."""

from __future__ import annotations


import pytest

pytest.importorskip("numpy")

from holomem.engine.history import HistoryMixin, HistoryUnavailable, version_state
from holomem.engine.retrieval import FactRetriever
from holomem.engine.store import MemoryStore

DIM = 512


class Clock:
    """The store's "now": dates in these tests are fixed, whatever day the suite runs."""

    def __init__(self, day: str):
        self.day = day


@pytest.fixture
def clock(monkeypatch):
    c = Clock("2026-10-12")
    monkeypatch.setattr(HistoryMixin, "_now", staticmethod(lambda: f"{c.day} 12:00:00"))
    monkeypatch.setattr(HistoryMixin, "_today", lambda self: c.day)
    return c


# --- read-only contexts, mirror, scopes, stats --------------------------------------------------------------------


def test_version_state_rules():
    row = {"status": "changed", "valid_from": None, "valid_to": "2026-04-10", "valid_to_exact": 0,
           "recorded_on": "2026-03-10"}
    assert version_state(row, "2026-03-09")[0] == "undetermined"  # before it was recorded
    assert version_state(row, "2026-03-10")[0] == "holds"  # the day it was recorded
    assert version_state(row, "2026-03-11")[0] == "undetermined"  # changed at some point before the bound
    assert version_state(row, "2026-04-10")[0] == "no"
    assert version_state({**row, "status": "corrected"}, "2026-03-10")[0] == "no"


def test_history_unavailable_is_explicit(tmp_path, clock):
    store = MemoryStore(str(tmp_path / "x.db"), hrr_dim=DIM, migrate=False)
    store._conn.execute("DELETE FROM meta WHERE key = 'history_schema'")
    with pytest.raises(HistoryUnavailable):
        store.history(1)
    with pytest.raises(HistoryUnavailable):
        store.versions_on("2026-01-01")
    store.close()


def test_retriever_search_without_history_tables_ready_is_unchanged(tmp_path, clock):
    store = MemoryStore(str(tmp_path / "y.db"), hrr_dim=DIM)
    retriever = FactRetriever(store, hrr_dim=DIM)
    store.add_fact("Vera Lin lives in Minsk")
    assert [f["content"] for f in retriever.search("Vera Minsk")] == ["Vera Lin lives in Minsk"]
    store.close()

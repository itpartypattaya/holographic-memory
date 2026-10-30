"""0.17.0: an alias "X aka Y" in a fact's text gave is a hint for text, never the target of an action someone takes with
a name (HANDOFF Open work, review 0.16.2). Before: after facts about Alice Smith, "Alice Smith aka Bob Brown" made Bob
Brown her alias, and entities=["Bob Brown"] then filed the second person's facts under her; rename and delete_entity by
"Bob Brown" renamed and deleted her. Names here are made up."""
from __future__ import annotations


import pytest

from holomem.engine.store import MemoryStore

DIM = 64


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _names(store, fact_id):
    return sorted(r[0] for r in store.read("SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
                                           "WHERE fe.fact_id = ?", (fact_id,)))


def _aliases(store):
    return sorted((r[0], r[1]) for r in store.read(
        "SELECT ea.alias, e.name FROM entity_aliases ea JOIN entities e ON e.entity_id = ea.entity_id"))


def _assert_consistent(store):
    """Links and aliases point at what exists, no two entities share a key, and both alias representations (the
    entity_aliases rows and the comma-separated column older versions read) say the same."""
    assert not store.read("SELECT 1 FROM fact_entities WHERE entity_id NOT IN (SELECT entity_id FROM entities)")
    assert not store.read("SELECT 1 FROM fact_entities WHERE fact_id NOT IN (SELECT fact_id FROM facts)")
    assert not store.read("SELECT 1 FROM entity_aliases WHERE entity_id NOT IN (SELECT entity_id FROM entities)")
    assert not store.read("SELECT name_key FROM entities WHERE name_key IS NOT NULL GROUP BY name_key HAVING COUNT(*) > 1")
    from holomem.engine.store import entity_key
    for r in store.read("SELECT entity_id, name, aliases FROM entities"):
        rows = {entity_key(a[0]) for a in store.read("SELECT alias FROM entity_aliases WHERE entity_id = ?", (r[0],))}
        column = {entity_key(a) for a in (r[2] or "").split(",") if a.strip()} - {entity_key(r[1])}
        assert rows == column, (r[1], rows, column)


# -- declarations that were dropped before 0.17.0 -------------------------------------------------------------------


def test_auto_extracted_text_gives_a_hint_a_declared_name_still_splits(store):
    store.add_fact_status("we decided that Alice Smith aka Bob Brown will handle it")
    assert _aliases(store) == [("Bob Brown", "Alice Smith")]
    with store.collect_splits() as splits:
        fact_id, _ = store.add_fact_status("He works at the bank", entities=["Bob Brown"])
    assert _names(store, fact_id) == ["Bob Brown"] and splits["Bob Brown"]["from"] == "Alice Smith"

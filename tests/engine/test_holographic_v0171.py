"""0.17.1: what reviews of 0.17.0 found in splitting an alias "X aka Y" gave (Codex CLI; names are made up)."""
from __future__ import annotations


import pytest

from holomem.engine.store import MemoryStore
from test_holographic_v0170 import _aliases, _assert_consistent, _names

DIM = 64


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _backed(store):
    """Every alias a text gave is said by a current fact linked to its entity (none hangs on nothing)."""
    for alias, owner in _aliases(store):
        row = [r for r in store.unconfirmed_text_aliases(limit=1000) if r["alias"] == alias]
        assert not row or row[0]["facts"], (alias, owner)


def test_a_chain_fact_leaves_a_declared_owner(store):
    """Codex P1.1: Alice declared, "Alice aka Bob", "Bob aka Carl", split Bob: the fact "Bob aka Carl" stayed on
    Alice and Bob, and no list said so."""
    store.add_fact_status("Alice Smith lives here", entities=["Alice Smith"])
    store.add_fact_status("Alice Smith aka Bob Brown")
    f, _ = store.add_fact_status("Bob Brown aka Carl Cole")
    out = store.split_alias("Bob Brown")
    assert _names(store, f) == ["Bob Brown"]
    assert _aliases(store) == [("Carl Cole", "Bob Brown")]
    assert f in out["facts_moved"] and out["facts_to_check"] == []
    for _ in range(2):
        store.rebuild_entity_index()
        assert _names(store, f) == ["Bob Brown"]
    _assert_consistent(store)


def test_a_long_chain_follows_the_split_to_its_end(store):
    """Codex P1.2: "Alice aka Bob", "Bob aka Carl", "Carl aka Dan", split Bob: one pass left Dan with Alice."""
    store.add_fact_status("Alice Smith aka Bob Brown")
    store.add_fact_status("Bob Brown aka Carl Cole")
    store.add_fact_status("Carl Cole aka Dan Dale")
    tea, _ = store.add_fact_status("Dan Dale likes green tea")
    store.split_alias("Bob Brown")
    assert _aliases(store) == [("Carl Cole", "Bob Brown"), ("Dan Dale", "Bob Brown")]
    assert _names(store, tea) == ["Bob Brown"] and store.entity_ids("Dan Dale") == store.entity_ids("Bob Brown")
    _backed(store)
    _assert_consistent(store)


def test_a_chain_stored_backwards_is_read_as_adding_it_would(store):
    """The facts saying "X aka Y" are read first, in the order they were stored."""
    tea, _ = store.add_fact_status("Carl Cole likes green tea")
    store.add_fact_status("Alice Smith aka Bob Brown")
    store.add_fact_status("Bob Brown aka Carl Cole")
    store.split_alias("Bob Brown")
    assert _names(store, tea) == ["Carl Cole"]  # Carl was a name of its own before any "aka": it stays one
    _backed(store)
    _assert_consistent(store)


def test_a_cycle_ends(store):
    store.add_fact_status("Alice Smith aka Bob Brown")
    store.add_fact_status("Bob Brown aka Alice Smith")
    out = store.split_alias("Bob Brown")
    assert out["entity"] == "Bob Brown" and _aliases(store) == []
    _assert_consistent(store)


def test_the_report_is_read_at_the_end(store):
    """Codex P2.3: facts_shared listed a fact whose links ended on Bob alone."""
    a, _ = store.add_fact_status("Alice Smith aka Bob Brown")
    b, _ = store.add_fact_status("Bob Brown aka Carl Cole")
    out = store.split_alias("Bob Brown")
    final = {f: _names(store, f) for f in (a, b)}
    assert final == {a: ["Alice Smith", "Bob Brown"], b: ["Bob Brown"]}
    assert out["facts_shared"] == [a] and out["facts_moved"] == [b]


def test_a_rolled_back_split_is_not_reported(store):
    store.add_fact_status("Alice Smith aka Bob Brown")
    with store.collect_splits() as splits:
        with pytest.raises(RuntimeError):
            with store._tx():
                store.add_fact_status("He works at the bank", entities=["Bob Brown"])
                raise RuntimeError("later step failed")
    assert splits == {} and _aliases(store) == [("Bob Brown", "Alice Smith")]


def test_a_declaration_on_a_stored_text_reencodes(store):
    """Codex P2.5: banks stayed, and without numpy the old vector stayed."""
    tea, _ = store.add_fact_status("He likes green tea")
    store._conn.execute("INSERT OR REPLACE INTO memory_banks (bank_name, vector, dim, fact_count) "
                        "VALUES ('cat:general', x'00', 1, 1)")
    store._conn.commit()
    store.add_fact_status("He likes green tea", entities=["Bob Brown"])
    assert _names(store, tea) == ["Bob Brown"] and not store.read("SELECT 1 FROM memory_banks")


def test_a_split_lifts_a_tombstone_of_the_name(store):
    """Codex P2.6: a tombstoned alias, split, stayed tombstoned: declared and never found in text."""
    store.add_fact_status("Alice Smith aka Bob Brown")
    for digest in store._name_digests("Bob Brown"):
        store._conn.execute("INSERT OR IGNORE INTO entity_tombstones (digest) VALUES (?)", (digest,))
    store._conn.commit()
    store.split_alias("Bob Brown")
    assert not store._tombstoned("Bob Brown")


def test_update_without_touching_still_declares(store):
    """Codex P3.8: update_fact(entities=..., touch=False) returned True and declared nothing."""
    tea, _ = store.add_fact_status("He likes green tea")
    assert store.update_fact(tea, entities=["Bob Brown"], touch=False)
    assert _names(store, tea) == ["Bob Brown"]


# -- review 0.17.0 by an agent on Claude Fable 5.1 ------------------------------------------------------------------


def test_deleting_x_does_not_tombstone_a_name_a_text_gave_it(store):
    """Fable P2.1: delete_entity Alice put a tombstone on Bob too; "Bob Brown lives in Berlin" then linked nobody."""
    store.add_fact_status("Alice Smith lives in Paris")
    store.add_fact_status("Alice Smith aka Bob Brown")
    out = store.delete_entity("Alice Smith")
    assert out["text_aliases_dropped"] == ["Bob Brown"]
    assert store._tombstoned("Alice Smith") and not store._tombstoned("Bob Brown")
    berlin, _ = store.add_fact_status("Bob Brown lives in Berlin")
    assert _names(store, berlin) == ["Bob Brown"]
    _assert_consistent(store)


def test_deleting_x_still_tombstones_an_alias_someone_gave(store):
    store.add_fact_status("Alice Smith lives in Paris")
    store.add_alias("Alice Smith", "Bob Brown")
    out = store.delete_entity("Alice Smith")
    assert "text_aliases_dropped" not in out and store._tombstoned("Bob Brown")


def test_rename_into_a_text_alias_of_another_is_refused(store):
    """Fable P2.2: the refusal's hint (alias Carol <- Alice) merged Alice into Carol."""
    store.add_fact_status("Alice Smith lives in Paris")
    store.add_fact_status("Carol King aka Bob Brown")
    with pytest.raises(ValueError, match=r"split \{entity: 'Bob Brown'\}"):
        store.rename_entity("Alice Smith", "Bob Brown")
    assert store.entity_ids("Alice Smith") and store.entity_ids("Alice Smith") != store.entity_ids("Carol King")


def test_rename_into_its_own_text_alias_is_allowed(store):
    store.add_fact_status("Alice Smith aka Bob Brown")
    out = store.rename_entity("Alice Smith", "Bob Brown")
    assert out["name"] == "Bob Brown" and out["previous"] == "Alice Smith"


def test_splitting_a_given_alias_keeps_its_owner(store):
    """Fable P2.3: the owner alias created went with its last fact, and the answer named an entity that was gone."""
    store.add_alias("Alice Smith", "Bob Brown")
    f, _ = store.add_fact_status("Bob Brown lives in Berlin")
    out = store.split_alias("Bob Brown")
    assert out["from"] == "Alice Smith" and out["facts_moved"] == [f] and "from_dropped" not in out
    assert store.entity_ids("Alice Smith") and _names(store, f) == ["Bob Brown"]
    _assert_consistent(store)


def test_a_split_says_when_the_old_entity_went(store):
    """An alias of unknown origin (written before 0.16.0): nobody is known to have named its owner."""
    store.add_alias("Alice Smith", "Bob Brown")
    store._conn.execute("UPDATE entity_aliases SET source = NULL")
    store._conn.commit()
    f, _ = store.add_fact_status("Bob Brown lives in Berlin")
    out = store.split_alias("Bob Brown")
    assert out["was"] == "legacy" and out["from_dropped"] and out["facts_moved"] == [f]
    assert not store.entity_ids("Alice Smith")
    _assert_consistent(store)

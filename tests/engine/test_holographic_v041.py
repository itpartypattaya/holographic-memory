"""0.4.1: fixes from the second independent review (Codex CLI) — user isolation, attribution, thread safety, per-store
counts, alias/merge migrations, cache invalidation, MATCH safety, relevance gate, polarity, revisions."""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("numpy")

from holomem.engine.retrieval import FactRetriever
from holomem.engine.store import MemoryStore

DIM = 512


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _r(store):
    return FactRetriever(store, hrr_dim=DIM)


# --- entities and migrations --------------------------------------------------------------------------------------


def test_legacy_aliases_survive_a_key_merge(tmp_path):
    db = tmp_path / "facts.db"
    s = MemoryStore(str(db), hrr_dim=DIM)
    s.close()
    raw = sqlite3.connect(str(db))
    raw.execute("DELETE FROM meta")  # look like a store an older version wrote
    raw.execute("INSERT INTO entities (name, aliases) VALUES ('Atlas Agent', 'Old Name'), ('atlas-agent', 'Lost Name')")
    raw.commit()
    raw.close()
    s = MemoryStore(str(db), hrr_dim=DIM)
    ids = {name: s.entity_ids(name) for name in ("Atlas Agent", "Old Name", "Lost Name")}
    assert len({tuple(v) for v in ids.values()}) == 1 and all(ids.values())
    s.close()


def test_aliases_of_entities_an_older_version_deleted_do_not_resolve(tmp_path):
    db = tmp_path / "facts.db"
    s = MemoryStore(str(db), hrr_dim=DIM)
    fact = s.add_fact("Alice Smith lives in Paris")
    s.add_alias("Alice Smith", "Ally Smith")
    s.close()
    raw = sqlite3.connect(str(db))  # v0.1.1 removing the fact: links and the orphan entity go, our alias row stays
    raw.execute("DELETE FROM fact_entities WHERE fact_id = ?", (fact,))
    raw.execute("DELETE FROM facts WHERE fact_id = ?", (fact,))
    raw.execute("DELETE FROM entities")
    raw.commit()
    raw.close()
    s = MemoryStore(str(db), hrr_dim=DIM)
    assert s.entity_ids("Ally Smith") == []
    tea = s.add_fact("Ally Smith likes tea")
    linked = s.read("SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id WHERE fe.fact_id = ?", (tea,))
    assert [r[0] for r in linked] == ["Ally Smith"]
    s.close()


def test_an_alias_owned_by_another_entity_is_not_stolen(store):
    store.add_fact("Alice Smith lives in Paris")
    store.add_fact("Carol Jones lives in Rome")
    store.add_alias("Alice Smith", "Ally Smith")
    with pytest.raises(ValueError, match="already an alias"):
        store.add_alias("Carol Jones", "Ally Smith")
    merged = store.add_alias("Carol Jones", "Alice Smith")  # an explicit merge keeps the merged name findable
    assert merged["merged"]
    assert store.entity_ids("Alice Smith") == store.entity_ids("Ally Smith") == store.entity_ids("Carol Jones")


def test_merge_reencodes_vectors_and_keeps_declared(store):
    paris = store.add_fact_status("Ally Smith lives in Paris")[0]
    store.add_fact_status("Alice Smith likes tea", entities=["Alice Smith"])
    store.add_alias("Alice Smith", "Ally Smith")
    assert paris in {f["fact_id"] for f in _r(store).related("Alice Smith")}


def test_entity_cache_does_not_survive_a_rollback(store):
    store.add_fact("seed fact")
    with pytest.raises(RuntimeError):
        with store._tx():
            store._resolve_entity("Alice Smith")
            store.mentioned_entity_ids("alice smith")  # cache built inside the transaction
            raise RuntimeError("rollback")
    bob = store._resolve_entity("Bob Brown")  # may reuse the rolled-back id
    fact = store.add_fact("alice smith likes coffee")
    linked = {r[0] for r in store.read("SELECT entity_id FROM fact_entities WHERE fact_id = ?", (fact,))}
    assert bob not in linked


def test_vectors_of_another_size_are_reencoded_on_open(tmp_path):
    db = tmp_path / "facts.db"
    s = MemoryStore(str(db), hrr_dim=DIM)
    fact = s.add_fact("Alice Smith lives in Paris")
    s.close()
    raw = sqlite3.connect(str(db))
    raw.execute("UPDATE facts SET hrr_vector = ? WHERE fact_id = ?", (b"HRR1" + b"\0" * 4 * 128, fact))
    raw.commit()
    raw.close()
    s = MemoryStore(str(db), hrr_dim=DIM)
    assert len(s.read("SELECT hrr_vector FROM facts")[0][0]) == 4 + 4 * DIM
    s.close()


def test_entities_only_update_links_and_rewording_keeps_declared(store):
    fact = store.add_fact("She swims every Saturday")
    assert store.update_fact(fact, entities=["Masha"])
    assert [f["fact_id"] for f in _r(store).probe("Masha")] == [fact]
    store.update_fact(fact, content="She swims every Sunday")
    assert [f["fact_id"] for f in _r(store).probe("Masha")] == [fact]


# --- search and the recall gate -----------------------------------------------------------------------------------


@pytest.mark.parametrize("query", ['foo"bar walk', 'he said "hi', "a*b (c) ^d:e", '"" ""'])
def test_user_queries_never_produce_invalid_fts(store, caplog, query):
    store.add_fact("Alice Smith lives in Paris")
    with caplog.at_level("WARNING"):
        _r(store).search(query, min_trust=0)
    assert not [r for r in caplog.records if "FTS5 search failed" in r.getMessage()]


@pytest.mark.parametrize("query", ['foo"bar walk', 'he said "hi'])
def test_queries_are_safe_without_the_stemmer(store, caplog, monkeypatch, query):
    from holomem.engine import text
    monkeypatch.setattr(text, "_STEMMERS", {})
    text.stem.cache_clear()
    store.add_fact("Alice Smith lives in Paris")
    with caplog.at_level("WARNING"):
        _r(store).search(query, min_trust=0)
    text.stem.cache_clear()
    assert not [r for r in caplog.records if "FTS5 search failed" in r.getMessage()]


def test_prefix_matches_are_validated(store):
    store.add_fact("The cart has wheels")
    assert _r(store).search("car", min_trust=0) == []


def test_same_root_word_ranks_below_the_word_itself(store):
    # 0.9.0: "walker" is a word of the same root as "walk" (a bare prefix match before 0.4.1, then dropped); it is
    # found, below a fact with the word itself.
    walker = store.add_fact("The walker has boots")
    walk = store.add_fact("A long walk every Sunday")
    found = [f["fact_id"] for f in _r(store).search("the walk", min_trust=0)]  # English: roots need the language
    assert found == [walk, walker]


def test_reason_mixes_known_identity_and_unknown_words(store):
    store.add_fact("Alice Smith met masha at the market")
    store.add_fact("Alice Smith lives in Paris")
    store.add_alias("Alice Smith", "Ally Smith")
    assert [f["content"] for f in _r(store).reason(["Ally Smith", "masha"])] == ["Alice Smith met masha at the market"]

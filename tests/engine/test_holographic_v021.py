"""0.2.1: entity recall from the entity index (strict AND for reason), trust filters on every path, atomic writes,
honest add/update statuses, cleaner entity extraction, read-only agent contexts, the mirror contract, setup defaults."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("numpy")

from holomem.engine.retrieval import FactRetriever
from holomem.engine.store import DuplicateContentError, MemoryStore

DIM = 1024


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _retriever(store):
    return FactRetriever(store, hrr_dim=DIM)


# --- entity recall ----------------------------------------------------------------------------------------------


def test_reason_never_returns_facts_about_only_one_entity(store):
    store.add_fact("Alice Smith lives in Paris")
    store.add_fact("Bob Brown lives in Rome")
    assert _retriever(store).reason(["Alice Smith", "Bob Brown"]) == []
    both = store.add_fact("Alice Smith and Bob Brown share a flat")
    assert [r["fact_id"] for r in _retriever(store).reason(["alice smith", "BOB BROWN"])] == [both]


def test_reason_with_an_unknown_name_still_requires_every_name(store):
    store.add_fact("Alice Smith met masha at the market")
    store.add_fact("Alice Smith lives in Paris")
    results = _retriever(store).reason(["Alice Smith", "masha"])
    assert [r["content"] for r in results] == ["Alice Smith met masha at the market"]


def test_probe_of_an_unknown_name_returns_nothing_unrelated(store):
    store.add_fact("Alice Smith lives in Paris")
    assert _retriever(store).probe("Unknown Person 19293") == []


def test_probe_matches_cyrillic_names_in_any_case(store):
    fact = store.add_fact("Мария Иванова работает в Сбере")
    assert [r["fact_id"] for r in _retriever(store).probe("МАРИЯ ИВАНОВА")] == [fact]


def test_trust_filter_applies_to_probe_related_reason_and_contradict(store):
    low = store.add_fact("Alice Smith lives in Paris")
    store.add_fact("Alice Smith lives in London")
    store._write("UPDATE facts SET trust_score = 0.1 WHERE fact_id = ?", (low,))
    r = _retriever(store)
    for results in (r.probe("Alice Smith"), r.related("Alice Smith"), r.reason(["Alice Smith"])):
        assert low not in {x["fact_id"] for x in results}
    assert r.contradict(min_trust=0.3) == []


def test_related_has_a_noise_floor(store):
    store.add_fact("Alice Smith lives in Paris")
    assert _retriever(store).related("Unknown Person 19293") == []


# --- contradict -------------------------------------------------------------------------------------------------


def test_contradict_ignores_a_claim_that_only_adds_detail(store):
    store.add_fact("Alice Smith likes apples")
    store.add_fact("Alice Smith likes apples and bananas")
    assert _retriever(store).contradict() == []


# --- entity extraction ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("don't stop it's fine", []),
    ("Alice Smith aka Bob Brown works here", ["Alice Smith", "Bob Brown"]),
    ('run with "--flag" and "key=value"', []),
    ("So Apple released a phone", []),
    ("Yesterday Alice Smith called", ["Alice Smith"]),
    ("server at '10.0.0.1'", []),
    ("user42 aka bob", ["user42", "bob"]),
])
def test_entity_extraction_rejects_noise(store, text, expected):
    assert store._extract_entities(text) == expected


# --- writes -----------------------------------------------------------------------------------------------------


def test_update_rejects_blank_content(store):
    fact = store.add_fact("Alice Smith lives in Paris")
    with pytest.raises(ValueError):
        store.update_fact(fact, content="   ")
    with pytest.raises(DuplicateContentError):
        store.update_fact(store.add_fact("Bob Brown lives in Rome"), content="Alice Smith lives in Paris")


def test_failed_update_leaves_the_fact_untouched(store, monkeypatch):
    fact = store.add_fact("Alice Smith lives in Paris")
    monkeypatch.setattr(MemoryStore, "_compute_hrr_vector", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        store.update_fact(fact, content="Alice Smith lives in Rome")
    monkeypatch.undo()
    assert [f["content"] for f in store.list_facts()] == ["Alice Smith lives in Paris"]
    assert [r["fact_id"] for r in _retriever(store).search("Paris", min_trust=0)] == [fact]


def test_removing_the_last_fact_of_an_entity_drops_the_entity(store):
    fact = store.add_fact("Alice Smith lives in Paris")
    store.remove_fact(fact)
    assert store.read("SELECT COUNT(*) FROM entities")[0][0] == 0


def test_record_retrievals_counts_integers_only(store):
    fact = store.add_fact("Alice Smith lives in Paris")
    assert store.record_retrievals([1.9, True, "x", None]) == 0
    assert store.record_retrievals([fact, str(fact)]) == 1


def test_counter_updates_do_not_touch_the_fts_index(store):
    sql = store.read("SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'facts_au'")[0][0]
    assert "UPDATE OF content, tags" in sql


def test_old_fts_trigger_is_narrowed_on_open(tmp_path):
    db = tmp_path / "old.db"
    s = MemoryStore(str(db), hrr_dim=DIM)
    s._write("DROP TRIGGER facts_au")
    s._write("CREATE TRIGGER facts_au AFTER UPDATE ON facts BEGIN "
             "INSERT INTO facts_fts(facts_fts, rowid, content, tags) VALUES ('delete', old.fact_id, old.content, old.tags); "
             "INSERT INTO facts_fts(rowid, content, tags) VALUES (new.fact_id, new.content, new.tags); END")
    s.close()
    s = MemoryStore(str(db), hrr_dim=DIM)
    assert "UPDATE OF" in s.read("SELECT sql FROM sqlite_master WHERE name = 'facts_au'")[0][0]
    s.close()


def test_writes_drop_the_category_bank_instead_of_rebuilding_it(store):
    store._write("INSERT INTO memory_banks (bank_name, vector, dim) VALUES ('cat:general', x'00', 1)")
    store.add_fact("Alice Smith lives in Paris")
    assert store.read("SELECT COUNT(*) FROM memory_banks")[0][0] == 0


def test_unreadable_vector_is_skipped_not_fatal(store):
    good = store.add_fact("Alice Smith lives in Paris")
    bad = store.add_fact("Alice Smith lives in Rome")
    store._write("UPDATE facts SET hrr_vector = x'0102' WHERE fact_id = ?", (bad,))
    r = _retriever(store)
    assert {x["fact_id"] for x in r.search("Alice lives", min_trust=0)} == {good, bad}
    assert good in {x["fact_id"] for x in r.related("Alice Smith")}


def test_hyphenated_identifiers_are_searchable(store):
    fact = store.add_fact("Backups run on pve-01 every night")
    assert [r["fact_id"] for r in _retriever(store).search("status of pve-01", min_trust=0)] == [fact]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_database_is_owner_only(tmp_path):
    s = MemoryStore(str(tmp_path / "p.db"), hrr_dim=DIM)
    s.add_fact("Alice Smith lives in Paris")
    s.close()
    MemoryStore(str(tmp_path / "p.db"), hrr_dim=DIM).close()
    assert (tmp_path / "p.db").stat().st_mode & 0o077 == 0

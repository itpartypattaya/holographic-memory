"""0.2.0 behaviour: entity recall that finds the entity's facts, a conflict check that pairs similar claims about
the same subject, non-Latin entity names, recall counting, no mirror of built-in memory, logged FTS failures."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("numpy")

from holomem.engine.retrieval import FactRetriever
from holomem.engine.store import MemoryStore, _capitalized_phrases

DIM = 1024


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _set_trust(store, fact_id, trust):
    store._write("UPDATE facts SET trust_score = ? WHERE fact_id = ?", (trust, fact_id))


# --- probe / reason ---------------------------------------------------------------------------------------------


def _people_store(store):
    alice = [store.add_fact("Alice Smith plays guitar on weekends"),
             store.add_fact("Alice Smith works at the bakery"),
             store.add_fact("Alice Smith and Bob Brown run a coffee shop")]
    others = [store.add_fact("Carol King repairs old radios"),
              store.add_fact("David Lee grows tomatoes"),
              store.add_fact("Bob Brown teaches yoga classes")]
    for fact_id in others:  # the most trusted facts are about someone else
        _set_trust(store, fact_id, 1.0)
    for fact_id in alice:
        _set_trust(store, fact_id, 0.3)
    return alice, others


def test_probe_returns_the_entitys_facts_not_the_most_trusted_ones(store):
    alice, _ = _people_store(store)
    results = FactRetriever(store, hrr_dim=DIM).probe("Alice Smith", limit=10)
    assert {r["fact_id"] for r in results} == set(alice)


def test_probe_is_case_insensitive_and_honours_category(store):
    alice, _ = _people_store(store)
    in_category = store.add_fact("Alice Smith prefers tea", category="user_pref")
    results = FactRetriever(store, hrr_dim=DIM).probe("alice smith", category="user_pref")
    assert [r["fact_id"] for r in results] == [in_category]


def test_probe_falls_back_to_keyword_search_for_an_unlinked_name(store):
    _people_store(store)
    plain = store.add_fact("Masha likes coffee in the morning")  # one-word name: never linked as an entity
    results = FactRetriever(store, hrr_dim=DIM).probe("Masha")
    assert [r["fact_id"] for r in results] == [plain]


def test_reason_returns_facts_linking_all_entities(store):
    alice, _ = _people_store(store)
    results = FactRetriever(store, hrr_dim=DIM).reason(["Alice Smith", "Bob Brown"], limit=5)
    assert results and results[0]["fact_id"] == alice[2]
    assert all("Alice Smith" in r["content"] and "Bob Brown" in r["content"] for r in results)


# --- contradict -------------------------------------------------------------------------------------------------


def test_contradict_pairs_similar_claims_about_the_same_subject(store):
    paris, london = store.add_fact("Alice Smith lives in Paris"), store.add_fact("Alice Smith lives in London")
    store.add_fact("Alice Smith plays guitar on weekends with friends")  # same subject, different topic
    store.add_fact("Bob Brown lives in Paris")  # same claim, different subject
    found = FactRetriever(store, hrr_dim=DIM).contradict()
    assert [{c["fact_a"]["fact_id"], c["fact_b"]["fact_id"]} for c in found] == [{paris, london}]
    assert found[0]["shared_entities"] == ["alice smith"]


def test_contradict_skips_identical_claims_and_needs_no_vectors(store):
    store.add_fact("Alice Smith lives in Paris")
    store.add_fact("Alice Smith  lives in Paris.")  # same words, distinct row
    store._write("UPDATE facts SET hrr_vector = NULL")
    assert FactRetriever(store, hrr_dim=DIM).contradict() == []
    store.add_fact("Alice Smith lives in London")
    assert len(FactRetriever(store, hrr_dim=DIM).contradict()) == 2


# --- entity extraction ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("Alice Smith works at Google", ["Alice Smith"]),
    ("Мария Иванова работает в Сбере", ["Мария Иванова"]),
    ("Анна-Мария Петрова пришла", ["Анна-Мария Петрова"]),
    ("Alice Smith, Bob Brown", ["Alice Smith", "Bob Brown"]),
    ("Встреча. Мария Иванова пришла", ["Мария Иванова"]),
    ("NASA and iPhone are not names", []),
    ("Alice works at Google", []),
])
def test_capitalized_phrases_in_any_cased_script(text, expected):
    assert _capitalized_phrases(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Клиент «Сбер Бизнес» платит вовремя", ["Сбер Бизнес"]),
    ("Project “Phoenix” ships Friday", ["Phoenix"]),
    ('Mom said: "Masha comes on Friday after school"', []),  # a quotation, not a name
    ("Маша aka Мария Петровна", ["Мария Петровна", "Маша"]),
])
def test_entity_extraction_quotes_and_aliases(store, text, expected):
    assert store._extract_entities(text) == expected


# --- retrieval counting -----------------------------------------------------------------------------------------


def _counts(store):
    return {r["fact_id"]: (r["retrieval_count"], r["updated_at"])
            for r in store._conn.execute("SELECT fact_id, retrieval_count, updated_at FROM facts")}


def test_record_retrievals_counts_once_and_keeps_updated_at(store):
    a, b = store.add_fact("Alice Smith lives in Paris"), store.add_fact("Bob Brown lives in Rome")
    before = _counts(store)
    assert store.record_retrievals([a, a, str(b), None, True, "x", 999]) == 2
    after = _counts(store)
    assert after[a] == (1, before[a][1]) and after[b] == (1, before[b][1])


# --- FTS failure ------------------------------------------------------------------------------------------------


def test_fts_failure_is_logged_not_silent(store, caplog):
    store.add_fact("Alice Smith lives in Paris")
    store._write("DROP TABLE facts_fts")  # a damaged index
    with caplog.at_level(logging.WARNING):
        FactRetriever(store, hrr_dim=DIM).search("Paris")  # 0.5.0: a one-word query is also matched word by word
    assert any("FTS5 search failed" in r.getMessage() for r in caplog.records)


def test_unsearchable_query_skips_fts_quietly(store, caplog):
    store.add_fact("Alice Smith lives in Paris")
    with caplog.at_level(logging.WARNING):
        assert FactRetriever(store, hrr_dim=DIM).search('" the of') == []
    assert not caplog.records

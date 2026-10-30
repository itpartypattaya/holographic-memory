"""0.3.0: word forms, a relevance gate and no repeats for recall before the turn, entity mentions, write-time
duplicate/conflict hints with supersedes, a write validator, stats/export/review, buffered recall counts, and a
store that survives an unreadable database."""

from __future__ import annotations


import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from holomem.engine.retrieval import FactRetriever
from holomem.engine.store import MemoryStore

DIM = 1024


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


# --- word forms and stopwords -----------------------------------------------------------------------------------


@pytest.mark.parametrize("fact,query", [
    ("Маша любит кофе по утрам", "что любит Маши"),
    ("Олег ездит на байке Honda PCX", "какой у Олега байк"),
    ("Созвоны лучше ставить после обеда", "когда ставить созвон"),
    ("We deploy the site with GitHub Actions", "how are deploys done"),
])
def test_search_finds_other_word_forms(store, fact, query):
    fact_id = store.add_fact(fact)
    assert [r["fact_id"] for r in FactRetriever(store, hrr_dim=DIM).search(query, min_trust=0)] == [fact_id]


@pytest.mark.parametrize("query", ["да, это так", "ok thanks", "ну да, конечно же", "спасибо!"])
def test_queries_without_informative_words_search_nothing(query):
    assert FactRetriever._sanitize_fts_query(query) == ""


# --- recall before the turn -------------------------------------------------------------------------------------


def test_undeclared_one_word_entities_do_not_attach_to_every_fact(store):
    store._write("INSERT INTO entities (name) VALUES ('Done')")  # junk from an older version
    fact = store.add_fact("The migration is done")
    assert store.read("SELECT COUNT(*) FROM fact_entities WHERE fact_id = ?", (fact,))[0][0] == 0

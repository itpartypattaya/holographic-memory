"""0.4.0: entity identity keys and aliases, the recorded HRR dimension, unsegmented scripts, per-user stores."""

from __future__ import annotations


import pytest

pytest.importorskip("numpy")

from holomem.engine.retrieval import FactRetriever
from holomem.engine.store import MemoryStore, entity_key

DIM = 512


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _r(store):
    return FactRetriever(store, hrr_dim=DIM)


# --- entity identity --------------------------------------------------------------------------------------------


def test_entity_key_unifies_spelling_not_meaning():
    assert entity_key("Atlas Agent") == entity_key("atlas-agent") == entity_key("ATLAS  AGENT") == entity_key("Atlas_Agent")
    assert entity_key("co-op") != entity_key("coop")
    assert entity_key("Мария Иванова") == entity_key("МАРИЯ ИВАНОВА")


def test_spelling_variants_are_one_entity(store):
    a = store.add_fact("Atlas Agent runs on the server")
    b = store.add_fact('The repo "atlas-agent" moved to a new org')
    assert store.read("SELECT COUNT(*) FROM entities")[0][0] == 1
    assert {f["fact_id"] for f in _r(store).probe("ATLAS-AGENT")} == {a, b}


def test_aka_makes_an_alias_not_a_second_entity(store):
    fact = store.add_fact("Маша aka Мария Петровна ходит в бассейн")
    assert [f["fact_id"] for f in _r(store).probe("Мария Петровна")] == [fact]
    entities = store.list_entities()
    assert [(e["name"], e["aliases"]) for e in entities] == [("Маша", ["Мария Петровна"])]
    later = store.add_fact("Мария Петровна любит плавать")  # the alias links later facts too
    assert {f["fact_id"] for f in _r(store).probe("Маша")} == {fact, later}


def test_entities_written_by_an_older_version_are_reconciled_on_open(tmp_path):
    db = tmp_path / "facts.db"
    s = MemoryStore(str(db), hrr_dim=DIM)
    first = s.add_fact("Atlas Agent runs on the server")
    # what an older version leaves behind: a second entity of the same name, no key, linked to its own fact
    second = s.add_fact("deploys are weekly")
    s._write("INSERT INTO entities (name) VALUES ('atlas-agent')")
    s._write("INSERT INTO fact_entities (fact_id, entity_id) VALUES (?, last_insert_rowid())", (second,))
    s.close()
    s = MemoryStore(str(db), hrr_dim=DIM)
    assert s.read("SELECT COUNT(*) FROM entities")[0][0] == 1
    assert {f["fact_id"] for f in _r(s).probe("Atlas Agent")} == {first, second}
    s.close()


# --- HRR dimension ----------------------------------------------------------------------------------------------


def test_changed_hrr_dim_reencodes_instead_of_failing(tmp_path):
    db = tmp_path / "facts.db"
    s = MemoryStore(str(db), hrr_dim=256)
    fact = s.add_fact("Alice Smith lives in Paris")
    s.close()
    s = MemoryStore(str(db), hrr_dim=DIM)
    assert s.read("SELECT value FROM meta WHERE key = 'hrr_dim'")[0][0] == str(DIM)
    assert [f["fact_id"] for f in _r(s).related("Alice Smith")] == [fact]
    s.close()


# --- unsegmented scripts ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("fact,query", [
    ("ร้านกาแฟเปิดตอนเจ็ดโมงเช้า", "ร้านกาแฟเปิดกี่โมง"),         # Thai: when does the coffee shop open
    ("我们每周五下午开会", "开会是什么时候"),                         # Chinese: when is the meeting
    ("学校は八時に始まります", "学校は何時から"),                       # Japanese: when does school start
    ("학교에서 수영을 배워요", "학교 수영"),                            # Korean: particle attached to the noun
])
def test_unsegmented_scripts_are_searchable(store, fact, query):
    store.add_fact("Alice Smith lives in Paris")
    fact_id = store.add_fact(fact)
    assert [f["fact_id"] for f in _r(store).search(query, min_trust=0)][:1] == [fact_id]

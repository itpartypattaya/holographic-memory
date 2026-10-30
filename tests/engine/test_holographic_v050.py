"""0.5.0: Spanish, Hindi, Arabic, French, Bengali, Portuguese and Indonesian — per-language stems and function words,
folding, elision and hyphens, Arabic clitics, Indic scripts the FTS5 index splits, Indonesian prefixes, greetings,
negation, and a deterministic order for an entity's facts."""

from __future__ import annotations


import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from holomem.engine.retrieval import FactRetriever
from holomem.engine.store import MemoryStore
from holomem.engine.text import analyze, fold, is_greeting, is_identifier, query_terms, words

DIM = 512


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _r(store):
    return FactRetriever(store, hrr_dim=DIM)


def _found(store, query, **kw):
    return [f["content"] for f in _r(store).search(query, min_trust=0, **kw)]


# -- text ----------------------------------------------------------------------------------------------------------

def test_latin_language_follows_the_function_words():
    assert analyze("¿Dónde está la gata de Ana?").langs == ("es",)
    assert analyze("Où est la chatte d'Ana ?").langs == ("fr",)
    assert analyze("Onde está a gata da Ana?").langs == ("pt",)
    assert analyze("Di mana kucing yang hilang itu?").langs == ("id",)
    assert analyze("Where is the cat?").langs == ("en",)
    # no function word: every Latin stemmer (0.6.0: + de), not the four of 0.11.0 without their own letters
    assert analyze("Proxmox cluster upgrade").langs == ("en", "es", "fr", "pt", "id", "de")


def test_a_function_word_of_another_language_stays_informative():
    # "son" is French for "his"; in English it is a word to search for
    assert [t.word for t in query_terms("where does my son study")] == ["son", "study"]


def test_folding():
    assert fold("está") == "esta" and fold("ação") == "acao"
    assert fold("й") == "й"  # Cyrillic keeps its breve
    assert fold("إيرما") == fold("ايرما") and fold("قطة") == "قطه" and fold("مرحبًا") == "مرحبا"
    assert fold("कॉफ़ी") == fold("कॉफी")  # nukta
    assert fold("২০২৭") == "2027" and fold("٢٠٢٧") == "2027"


def test_elision_inversion_and_hyphens():
    assert words("le rendez-vous d'Irma") == ["le", "rendez-vous", "irma"]  # a noun, not an inverted verb
    assert {"rendez", "vous"} <= analyze("le rendez-vous d'Irma").bag
    assert words("Comment Oscar prend-il son café ?") == ["comment", "oscar", "prend", "son", "cafe"]
    assert words("Quand Irma a-t-elle natation") == ["quand", "irma", "a", "natation"]
    assert words("anak-anak bermain") == ["anak", "bermain"]
    assert words("hr-vps-এর ব্যাকআপ") == ["hr-vps", "এর", "ব্যাকআপ"]
    assert words("IT-тусовка в пятницу") == ["it", "тусовка", "в", "пятницу"]
    assert "sextas" in analyze("nas sextas-feiras").bag
    assert is_identifier("pve-01") and is_identifier("v0.21.6") and not is_identifier("кто-то")


def test_unicode_punctuation_is_stripped():
    assert words("¿Qué hace QX24؟") == ["que", "hace", "qx24"]
    assert words("यह क्या है।") == ["यह", "क्या", "है"]


@pytest.mark.parametrize("text", ["¡Buenos días!", "Bonjour !", "Bom dia!", "Selamat pagi", "صباح الخير",
                                  "सुप्रभात!", "শুভ সকাল", "muchas gracias", "شكرًا", "ধন্যবাদ"])
def test_greetings_in_every_language(text):
    assert is_greeting(text)


# -- search --------------------------------------------------------------------------------------------------------

def test_spanish_word_forms_and_accents(store):
    store.add_fact("La gata de la familia se llama Nala")
    store.add_fact("El vuelo sale mañana")
    assert _found(store, "¿Cómo se llama el gato?")[0] == "La gata de la familia se llama Nala"
    assert _found(store, "manana vuelo")[0] == "El vuelo sale mañana"  # typed without the tilde


def test_arabic_clitics(store):
    store.add_fact("قطة العائلة اسمها ميا")
    store.add_fact("تدرس إيرما في مدرسة دولية")
    assert _found(store, "ما اسم القطة؟")[0] == "قطة العائلة اسمها ميا"  # article on one side only
    assert _found(store, "مدرسة لإيرما")[0] == "تدرس إيرما في مدرسة دولية"  # preposition glued to the name


def test_arabic_and_indic_queries_never_reach_fts():
    assert FactRetriever._sanitize_fts_query("ما اسم القطة") == ""
    assert FactRetriever._sanitize_fts_query("बिल्ली किस नस्ल की है") == ""
    assert FactRetriever._sanitize_fts_query("QX24 क्या करती है").startswith('("qx24" OR')  # the identifier still does


def test_hindi_in_a_store_where_letters_are_everywhere(store):
    # unicode61 indexes Devanagari letter by letter; with many facts sharing letters, FTS ranking was noise
    for i in range(60):
        store.add_fact(f"पटाया में शुक्रवार की पार्टी नंबर {i} रात को होती है")
    target = store.add_fact("परिवार की बिल्ली ब्रिटिश शॉर्टहेयर नस्ल की है")
    assert _r(store).search("बिल्ली किस नस्ल की है?", min_trust=0, limit=3)[0]["fact_id"] == target
    coffee = store.add_fact("एंटोन ब्लैक कॉफ़ी पीते हैं")
    assert _r(store).search("एंटोन कॉफी", min_trust=0, limit=1)[0]["fact_id"] == coffee  # written without the nukta


def test_bengali_case_endings_and_digits(store):
    store.add_fact("পরিবার ২০২৭ সালের বসন্তে হুয়া হিনে চলে যাবে")
    store.add_fact("প্রতি শনিবার সকালে ইরমার সাঁতারের ক্লাস হয়")
    assert _found(store, "হুয়া হিন কবে")[0].startswith("পরিবার")
    assert _found(store, "2027")[0].startswith("পরিবার")
    assert _found(store, "ইরমা সাঁতার")[0].startswith("প্রতি")


def test_indonesian_prefixes(store):
    store.add_fact("Skrip yang tidak terkait Atlas dijalankan dengan Python 3.12")
    store.add_fact("Kucing keluarga bernama Nala")
    assert _found(store, "bisa jalankan skrip apa?")[0].startswith("Skrip")
    assert _found(store, "kucingnya siapa namanya?")[0].startswith("Kucing")


def test_reason_with_an_unknown_arabic_word(store):
    store.add_fact_status("إيرما لديها حساسية من الفول السوداني", entities=["إيرما"])[0]
    store.add_fact_status("تدرس إيرما في مدرسة دولية", entities=["إيرما"])[0]
    assert [f["content"] for f in _r(store).reason(["إيرما", "الحساسية"])] == ["إيرما لديها حساسية من الفول السوداني"]


# -- gate, conflicts, entity facts ---------------------------------------------------------------------------------


def test_entity_facts_come_newest_first_when_timestamps_tie(store):
    ids = [store.add_fact_status(f"Irma fact number {i}", entities=["Irma"])[0] for i in range(4)]
    store._conn.execute("UPDATE facts SET updated_at = '2026-01-01 00:00:00'")
    group = store.entity_ids("Irma")
    assert [f["fact_id"] for f in _r(store)._linked_facts([group], None, 0.0, 2)] == [ids[3], ids[2]]


# -- the independent review of 0.5.0 (Codex CLI) ---------------------------------------------------------------------

def test_a_function_word_is_told_by_the_other_words():
    assert [t.word for t in query_terms("son")] == ["son"]  # Spanish/French function word, English noun
    assert [t.word for t in query_terms("bisa")] == ["bisa"]
    assert query_terms("ok") == []  # no other evidence: the English list, as before 0.5.0
    assert [t.word for t in query_terms("Il boit son café")] == ["boit", "cafe"]


@pytest.mark.parametrize("fact,query", [
    ("a ação foi rápida", "ações"), ("o coração bate", "corações"), ("el lápiz rojo", "lápices"),
    ("le cœur de la ville", "coeur"), ("إيرما تدرس هنا", "ولإيرما"), ("إيرما تدرس هنا", "وبإيرما"),
    ("قهوته سوداء", "القهوة")])
def test_forms_of_one_word_meet(fact, query):
    assert any(t.found_in(analyze(fact)) for t in query_terms(query))


@pytest.mark.parametrize("fact,query", [
    ("contrast in the photos", "contract"), ("كتاب جديد", "تاب"), ("মাটি ভেজা", "মা")])
def test_unrelated_words_do_not_meet(fact, query):
    assert not any(t.found_in(analyze(fact)) for t in query_terms(query))


def test_latin_names_are_matched_as_written(store):
    store.add_fact_status("Jean lives in Lyon", entities=["Jean"])
    store.add_fact_status("Rose runs the cafe", entities=["Rose"])
    assert store.mentioned_entity_ids("I bought jeans and the roses bloom") == set()
    assert store.mentioned_entity_ids("Is that Jean's car?") == set(store.entity_ids("Jean"))


def test_indonesian_prefix_found_from_a_one_word_query(store):
    store.add_fact("Skrip dijalankan dengan Python")
    assert _found(store, "jalankan")


def test_reason_applies_every_constraint_before_the_cap(store):
    old = store.add_fact_status("needle in the old haystack", trust=1.0)[0]
    for i in range(505):
        store.add_fact_status(f"needle number {i}", trust=0.3)
    assert old in [f["fact_id"] for f in _r(store).reason(["needle"], limit=10)]


def test_native_digits_inside_an_identifier(store):
    store.add_fact("QX২৪ has a new server")
    assert _found(store, "QX24")


def test_control_characters_never_reach_fts(store, caplog):
    store.add_fact("alpha beta")
    with caplog.at_level("WARNING"):
        assert _found(store, "a\x00b alpha")
    assert not [r for r in caplog.records if "FTS5 search failed" in r.getMessage()]


def test_an_english_store_is_never_scanned(store, monkeypatch):
    store.add_fact("Deploys run on the production server nightly")
    retriever = _r(store)
    monkeypatch.setattr(retriever, "_texts", lambda *a, **k: pytest.fail("an English store was scanned"))
    found = retriever.search("deploys", min_trust=0)  # one word: no language determined, might be Indonesian
    assert found and retriever.relevant("deploys", found)

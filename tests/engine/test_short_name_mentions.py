"""A short Russian name in another case mentions its entity (2026-10-31): «Расскажи про Иру» found no Ира.

A word of three letters gets no stem, so «иру» met none of «Ира»'s keys (ир, ира) while «Машу» met «Маша» by its stem.
The case ending tells the name (text.name_lemmas, as names_fit reads it), and only for such an unstemmed word: an
ordinary short word or a stopword still mentions no one."""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from holomem.engine.store import MemoryStore

DIM = 512


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


@pytest.mark.parametrize("text", ["Расскажи про Иру", "Что с Ирой?", "Подарок для Иры", "Ира, привет"])
def test_a_short_name_in_any_case_mentions_it(store, text):
    store.add_fact_status("Ира учится в пятом классе", entities=["Ира"])
    assert store.mentioned_entity_ids(text) == set(store.entity_ids("Ира"))


def test_a_case_ending_tells_two_short_names_apart(store):
    store.add_fact_status("Ян играет в шахматы", entities=["Ян"])
    store.add_fact_status("Яна поёт в хоре", entities=["Яна"])
    (yan,), (yana,) = store.entity_ids("Ян"), store.entity_ids("Яна")
    assert store.mentioned_entity_ids("Встреча с Яной") == {yana}
    assert store.mentioned_entity_ids("Встреча с Яном") == {yan}
    assert store.mentioned_entity_ids("Напиши Яну") == {yan, yana}  # of «Ян» or of «Яна»: the ending cannot tell


def test_ordinary_short_words_mention_no_one(store):
    store.add_fact_status("Ира учится в пятом классе", entities=["Ира"])
    store.add_fact_status("Маша любит кошек", entities=["Маша"])
    assert store.mentioned_entity_ids("Его нет, ему всё равно, её тоже") == set()
    assert store.mentioned_entity_ids("Я купил новую машину") == set()


def test_a_short_verb_does_not_read_as_a_name(store):
    """Codex review 2026-10-31: «рою» read by its ending is «Рой», «даю» is «Дай». A name is written with a capital;
    a word in lower case keeps only its own keys."""
    store.add_fact_status("Рой работает в порту", entities=["Рой"])
    store.add_fact_status("Дай держит кафе у моря", entities=["Дай"])
    assert store.mentioned_entity_ids("Я рою яму и даю тебе слово") == set()
    assert store.mentioned_entity_ids("Позвони Рою завтра") == set(store.entity_ids("Рой"))


def test_a_verb_opening_a_sentence_does_not_read_as_a_name(store):
    """Codex second pass 2026-10-31: a sentence starts with a capital, so «Рою яму» read as «Рой». A name in another
    case at the start of a sentence waits for its nominative; mid-sentence it is read."""
    store.add_fact_status("Рой работает в порту", entities=["Рой"])
    store.add_fact_status("Дай держит кафе у моря", entities=["Дай"])
    assert store.mentioned_entity_ids("Рою яму у дома. Даю тебе слово!") == set()
    for opening in ("План: Рою яму", "* Рою яму", "- Рою яму", "Он сказал: «Рою яму»", "1) Рою яму"):
        assert store.mentioned_entity_ids(opening) == set(), opening  # Codex third pass
    assert store.mentioned_entity_ids("Вчера. Позвони Рою") == set(store.entity_ids("Рой"))


def test_a_name_with_yo_is_still_found_as_written(store):
    """Codex second pass 2026-10-31: the ending read «ёся» as «еся» and dropped the word's own key."""
    store.add_fact_status("Ёся любит чай", entities=["Ёся"])
    assert store.mentioned_entity_ids("Ёся любит чай") == set(store.entity_ids("Ёся"))
    assert store.mentioned_entity_ids("Позови Ёсю") == set(store.entity_ids("Ёся"))

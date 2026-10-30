"""Case forms of Russian nouns that share no key (text._RU_FORM_GROUPS: an experiment, off by default).

Snowball cuts «чай» to «ча», «мае» to «ма», «дня» to «дн»; a stem under three letters is no key (as a prefix «ча*»
would find «час» and «чат»), so «чая» never met «чай» and «какие планы на май?» never found «В мае едем на дачу». A
dropped vowel does the same to longer words: «подарки» / «подарок», «отца» / «отец». One key for every two-letter
stem was rejected: it joins different words («ель» / «ели» of «есть», «рой» / «рою» of «рыть», «уха» / «ухо»). The
experiment lists a few common nouns with their own case forms, each form checked, and a word meets another only
through its own noun's list. The facts are synthetic."""

from __future__ import annotations


import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from holomem.engine import text


def _clear():
    text._keys.cache_clear()
    text.analyze.cache_clear()
    if hasattr(text, "_ru_form_keys"):
        text._ru_form_keys.cache_clear()


@pytest.fixture
def groups(monkeypatch):
    # raising=False: on 0.22.1 the switch does not exist, setting it changes nothing, and these tests show its behaviour
    monkeypatch.setattr(text, "_RU_FORM_GROUPS", True, raising=False)
    _clear()
    yield
    monkeypatch.undo()
    _clear()


def test_a_form_written_with_e_for_yo_is_the_same_form(groups):
    assert text.analyze("ребенка").words[0].keys & text.analyze("ребёнок").words[0].keys
    assert text.analyze("подарки").words[0].keys & text.analyze("подарок").words[0].keys


def test_a_light_word_is_not_listed(groups):
    # «дни рождения» would meet every «День рождения …»: «день» keeps its own keys (measured, see text._RU_FORMS)
    assert text.analyze("дни").words[0].keys.isdisjoint(text.analyze("день").words[0].keys)


# -- Codex review 2026-10-29 ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("form, noun", [
    ("матери", "мать"),  # also «матери!», the imperative of «материть»
    ("потолок", "потолка"),  # also the past tense of «потолочь»
])
def test_a_form_that_is_also_a_verb_joins_no_group(groups, form, noun):
    assert text.analyze(form).keys.isdisjoint(text.analyze(noun).keys)

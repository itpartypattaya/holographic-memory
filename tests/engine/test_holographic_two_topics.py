"""Two questions in one message (prefetch_clause_coverage: an experiment, off by default).

The coverage gate weighs a fact against every informative word of the message. A message that asks two things at once
(«что Злате нужно обязательно взять с собой на плавание в четверг и какой чай она любит?») gives each answer only its
half of the words: the fact about tea holds «Злате», «чай» and «любит», every word of its question, and still covers
0.24 of the message, under the gate's 0.3, so it is left out. With prefetch_clause_coverage on, a fact the gate left
out gets a second look against each of the message's two questions alone. It is let in only when it holds at least two
words of that question itself and covers the question's words (a word no fact holds counts against it, as in the gate),
and when the question is about a person — named in it, or «она» / «он» of a name in the other question — only if the
fact names that person: «Тимур любит зелёный чай» answers «какой чай она любит» only when «она» is Тимур. Whose answer
it is must be told for sure (Codex review 2026-10-29): «она» of one of two people, a name the store does not know, a word
that is also a known name («чай» when a dog is called «Чай») — nothing is let in. A rescued fact comes after the facts
the gate kept and, with the reranker on too, only takes a place the others leave free, so it never takes the place of
one of them. A message is split only outside quotes and only into two questions that each stand alone. The facts are
synthetic; the household is invented."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")


TWO = "что Злате нужно обязательно взять с собой на плавание в четверг и какой чай она любит?"


# -- what stays as it was ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("message, parts", [
    (TWO, ("что Злате нужно обязательно взять с собой на плавание в четверг", "какой чай она любит?")),
    ("Что Злате взять в четверг? И какой чай она любит?", ("Что Злате взять в четверг", "какой чай она любит?")),
    ("напомни, что Тимуру взять в субботу, и во сколько у него футбол",
     ("напомни, что Тимуру взять в субботу", "во сколько у него футбол")),
    ("What should Dina take on Tuesday and which tea does she like?",
     ("What should Dina take on Tuesday", "which tea does she like?")),
    ("какой чай и кофе она любит?", None),  # «и» between two words of one question
    ("Злата любит чай и как это понимать", None),  # the first part asks nothing
    ("Что взять? Какой чай? Когда день рождения?", None),  # three questions: not two
    ("я выпью чай и пойду спать", None),
    # one question about two things: "which tea" alone asks nothing, the verb is in the second half (Codex review)
    ("which tea and which coffee does she like?", None),
    ("какой зелёный чай и какой кофе она любит?", None),
    # a quoted line is what is asked about, not two questions of the message (Codex review)
    ("Что означает у Златы длинная надпись про плавание очки шапочку и четверг „Что упаковать? Какой чай она любит?“?",
     None),
    ("Что значит надпись «что взять в четверг и какой чай она любит»?", None),
])
def test_the_split_finds_exactly_two_questions(message, parts):
    from holomem.engine.text import two_questions
    assert two_questions(message) == parts


# -- the cost of the split (finding 6) ------------------------------------------------------------------------------

def test_the_split_is_linear_in_a_long_run_of_spaces():
    # a pasted message with a long run of spaces: a regular expression that looks ahead from every space took about
    # 0.35 s for 4 000 of them and grows with the square; one pass takes milliseconds. The bound is generous.
    from holomem.engine.text import two_questions
    message = "what" + " " * 20_000 + "nothing"
    start = time.perf_counter()
    assert two_questions(message) is None
    assert time.perf_counter() - start < 1.0

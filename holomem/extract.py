"""Auto-extraction (``auto_extract = true``, off by default): sentences of the person's prompt kept as facts.

The user's sentences that state a preference or a decision are stored. holomem applies the engine's patterns
(``engine/extract_patterns.py``) to each prompt as the hook receives it: there the client hands over exactly what the
person typed, while its transcript mixes in instruction files, system notes and tool results under the same "user"
role. A sentence is skipped when it is a question, too short or too long, fails the engine's write screen
(instructions, credentials, markup), was forgotten or is a past version of a fact, or restates a stored fact.
Stored at ``default_trust - 0.1``: unconfirmed, below hand-added facts.
"""
from __future__ import annotations

import logging
import re

from .memory import Context, Memory

logger = logging.getLogger("holomem")

# A reply to the assistant is not a preference (hermes-agent#22907, ported 2026-11-02): "I like that, sounds good",
# "ok, I love it", "I prefer this one" like whatever the assistant just said, and stored they say nothing on their own.
# Their object is "it", or this / that / these / those and at most one word, and the clause ends there (a degree word
# may follow: "better", "a lot"). "I like it when …", "I like that you …" and "I use this laptop for work" go on, and
# are kept; so are "them", «его», «их»: more often people than a proposal. The clause ends at a mark, a dash or an
# emoji, not at a dot inside a name ("this Node.js library") nor at a colon, which says what the object is.
_REACTION_END = (r'(?:\s+(?:better|more|most|best|too|a\s+lot|very\s+much|so\s+much))*\s*'
                 r'(?:[,;…)—–]|[.!](?!\w)|-(?=\s)|[☀-➿🌀-🫿]|$)')
_RE_REACTION = re.compile(
    r'\bI\s+(?:really\s+|also\s+|just\s+|totally\s+|do\s+)?(?:prefer|like|love|use)\s+'
    r'(?:it|(?:this|that|these|those)(?:\s+[^\W\d_]+)?)' + _REACTION_END +
    r'|(?:^|\s)я\s+(?:тоже\s+|очень\s+)?(?:предпочитаю|люблю|использую)\s+'
    r'(?:это|то|такое|(?:этот|эта|это|эту|эти|тот|та|ту|те)\s+[^\W\d_]+)' + _REACTION_END,
    re.IGNORECASE)


_RE_THIS_NOUN = re.compile(r'\b(?:this|that|these|those)\s+[^\W\d_]+|(?:этот|эта|это|эту|эти|тот|та|ту|те)\s+[^\W\d_]+',
                           re.IGNORECASE)
_REPLY_WORD = (r'(?:thanks|thank\s+you|great|perfect|awesome|nice|cool|sure|ok|okay|yes|yep|yeah|agreed|exactly'
               r'|excellent|wonderful|fantastic|amazing|brilliant|lovely|good|fine|deal|done|sounds\s+(?:good|great)'
               r'|love\s+it|good\s+job|well\s+done|спасибо|отлично|супер|класс|круто|согласен|согласна|да|хорошо'
               r'|понятно|прекрасно|замечательно|договорились)')
_RE_REPLY_WORDS = re.compile(_REPLY_WORD + r'(?:[\s,!.]+' + _REPLY_WORD + r')*', re.IGNORECASE)


def _names_its_object(sentence: str, match: re.Match) -> bool:
    """The thing is named right after, between two marks: "I prefer this database — PostgreSQL — for all my
    projects", "I use this editor, VS Code, for everything" (Codex review 2026-11-02). A name starts with a capital or
    a digit, so "I like this idea, thanks, let's go" stays a reply; so does a reply word with a capital ("I like this
    idea — Great!", "…, Thanks.": _RE_REPLY_WORDS, review 2026-11-03)."""
    said = match.group(0).rstrip()
    if not said or said[-1] not in "—–-," or not _RE_THIS_NOUN.search(said):
        return False
    close = re.escape("—–-") if said[-1] in "—–-" else ","
    named = re.match(rf'\s*([A-ZА-ЯЁ\d][^{close}]{{0,40}}?)(?:[{close}]\s*\w|[.!?]?\s*$)', sentence[match.end():])
    return named is not None and _RE_REPLY_WORDS.fullmatch(named.group(1).strip(" .!?…")) is None


def _reaction(sentence: str, categories) -> bool:
    """A sentence that states no preference but a reply to the assistant (_RE_REACTION), with nothing left that
    would be extracted once the reply is taken out ("I like that, I always use tabs" keeps its second half)."""
    found = list(_RE_REACTION.finditer(sentence))
    if not found or any(_names_its_object(sentence, m) for m in found):
        return False
    rest = _RE_REACTION.sub(" ", sentence)
    return not any(p.search(rest) for patterns, _ in categories for p in patterns)


def extract(memory: Memory, ctx: Context, text: str) -> list[int]:
    """Store the sentences of *text* that state a preference or decision; returns the new fact ids."""
    from .engine.extract_patterns import _EXTRACT_CATEGORIES
    from .engine.screen import (_EXTRACT_MAX_CHARS, _RE_QUESTION, _RE_SENTENCE, _extract_floor, _fact_problem,
                                _not_a_fact, _role_prompt)

    store, retriever = memory.store, memory.retriever
    trust = max(0.0, memory.cfg.default_trust - 0.1)
    stored: list[int] = []
    if not isinstance(text, str) or len(text) < _extract_floor(text) or _role_prompt(text):
        return stored  # a prompt pasted whole: its sentences pass one by one, and none is the person's
    for sentence in (m.group(0).strip() for m in _RE_SENTENCE.finditer(text)):
        if len(sentence) < _extract_floor(sentence) or len(sentence) > _EXTRACT_MAX_CHARS \
                or _fact_problem(sentence) or _not_a_fact(sentence, person=True) or _RE_QUESTION.search(sentence) \
                or _reaction(sentence, _EXTRACT_CATEGORIES):
            continue
        for patterns, category in _EXTRACT_CATEGORIES:
            if any(p.search(sentence) for p in patterns):
                try:
                    if store.forgotten_match(sentence) or store.merged_match(sentence) or store.past_version(
                            sentence, statuses=("changed", "ended", "corrected")):
                        break
                    if retriever.near_duplicate(sentence) is not None:
                        break
                    fact_id, created = store.add_fact_status(sentence, category=category, trust=trust,
                                                             source=ctx.source("auto_extract", origin="said"))
                    if created:
                        stored.append(fact_id)
                except Exception as exc:  # noqa: BLE001 - the next sentence may still go in
                    logger.warning("holomem: auto-extraction could not store a sentence (%s)", type(exc).__name__)
                break
    return stored

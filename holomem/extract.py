"""Auto-extraction (``auto_extract = true``, off by default): sentences of the person's prompt kept as facts.

The user's sentences that state a preference or a decision are stored. The bridge applies the engine's patterns
(``engine/extract_patterns.py``) to each prompt as the hook receives it: there the client hands over exactly what the
person typed, while its transcript mixes in instruction files, system notes and tool results under the same "user"
role. A sentence is skipped when it is a question, too short or too long, fails the engine's write screen
(instructions, credentials, markup), was forgotten or is a past version of a fact, or restates a stored fact.
Stored at ``default_trust - 0.1``: unconfirmed, below hand-added facts.
"""
from __future__ import annotations

import logging

from .memory import Context, Memory

logger = logging.getLogger("holomem")


def extract(memory: Memory, ctx: Context, text: str) -> list[int]:
    """Store the sentences of *text* that state a preference or decision; returns the new fact ids."""
    from .engine.extract_patterns import _EXTRACT_CATEGORIES
    from .engine.screen import _EXTRACT_MAX_CHARS, _RE_QUESTION, _RE_SENTENCE, _extract_floor, _fact_problem

    store, retriever = memory.store, memory.retriever
    trust = max(0.0, memory.cfg.default_trust - 0.1)
    stored: list[int] = []
    if not isinstance(text, str) or len(text) < _extract_floor(text):
        return stored
    for sentence in (m.group(0).strip() for m in _RE_SENTENCE.finditer(text)):
        if len(sentence) < _extract_floor(sentence) or len(sentence) > _EXTRACT_MAX_CHARS \
                or _fact_problem(sentence) or _RE_QUESTION.search(sentence):
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

"""The memory tools the MCP server offers: the fact store's actions, one tool each, over the engine.

The checks: text that is not a fact (an instruction, a credential, markup) is refused by the engine's
write screen; a text the person asked to forget, a past version or a copy merged into another fact is not stored
again as new; a new fact comes back with ``similar`` / ``possible_conflicts`` for the agent to decide on; a name that
cannot be an entity is listed under ``entities_not_linked`` by position, never quoted. Answers are JSON text; a
failure nobody meant is named by its kind, never by a message that may quote a fact. With ``embeddings`` on,
memory_search also finds facts by meaning (``match: "meaning"``). With ``rerank`` on, the facts
its words found are rescored (rescore.py) before those by meaning are added after them: what the endpoint did not score
(beyond rerank_top_k, or not to be sent) keeps its order behind; a failure keeps the order of rerank off. A search
"as_of" a day is never rescored. memory_export (``export.py``) gives the store as one Markdown page.
With ``experience`` on, memory_search answers with the receipt of what it offered (``memory_attempt``) and
memory_experience — listed only then — explains an attempt and records the agent's report of how it turned out
(``journal.py``).

Not in the bridge (yet): contradict.
"""
from __future__ import annotations

import json
import logging

from . import journal
from .memory import Context, Memory

logger = logging.getLogger("holomem")

_KW_MARK = "keywords: "  # the marker: keywords live in the indexed tags column after it
_MAX_KEYWORDS, _MAX_KEYWORD_CHARS, _MAX_LIMIT = 12, 40, 100
_EXPORT_LIMIT, _EXPORT_MAX = 100, 1000  # memory_export: a page of facts, not a search answer

_ID = {"type": "integer", "description": "The fact id (#id in recalled lines and results)."}
_ENTITIES = {"type": "array", "items": {"type": "string"},
             "description": "Who or what the fact is about; needed for one-word names (a first name, a pet) so later "
                            "facts in any word form link to it."}
_KEYWORDS = {"type": "array", "items": {"type": "string"},
             "description": "Up to 12 words a person might ask with that the text does not contain (synonyms, other "
                            "roots, translations). Never shown in recall."}
_DAY = {"type": "string", "description": "YYYY-MM-DD, 'today' or 'yesterday'."}
_DAY_KEYS = ("valid_from", "valid_to", "as_of")  # the arguments that take _DAY (Tools._days)
_LIMIT = {"type": "integer", "minimum": 1, "maximum": _MAX_LIMIT}
_CATEGORIES = ("user_pref", "project", "tool", "general")
_EXPECT = {"type": "object", "additionalProperties": {"type": "integer"},
           "description": "Optional {\"<fact_id>\": <version_id>} as memory_history shows them: write only while each "
                          "named fact is still at that version (otherwise nothing is written: changed_since)."}

SCHEMAS = [
    {"name": "memory_add", "description": "Store a durable fact as one short standalone sentence (local store). The "
     "answer lists similar facts and possible conflicts: if one changed into this, use memory_change on it instead.",
     "inputSchema": {"type": "object", "properties": {
         "content": {"type": "string"}, "category": {"type": "string",
                                                     "enum": ["user_pref", "project", "tool", "general"]},
         "entities": _ENTITIES, "keywords": _KEYWORDS, "valid_from": _DAY,
         "origin": {"type": "string", "enum": ["said", "confirmed", "inferred"],
                    "description": "said: the person stated it; confirmed: they agreed; inferred: your conclusion."}},
         "required": ["content"]}},
    {"name": "memory_search", "description": "Search stored facts by words, in any word form and language; as_of: what "
     "held on that day.", "readOnly": True,
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_LIMIT},
         "category": {"type": "string"}, "min_trust": {"type": "number"}, "as_of": _DAY}, "required": ["query"]}},
    {"name": "memory_probe", "description": "Everything stored about a person or thing, by name.", "readOnly": True,
     "inputSchema": {"type": "object", "properties": {"entity": {"type": "string"},
                                                      "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_LIMIT}},
                     "required": ["entity"]}},
    {"name": "memory_related", "description": "Facts structurally related to a person or thing (HRR), beyond "
     "the ones that name it.", "readOnly": True,
     "inputSchema": {"type": "object", "properties": {"entity": {"type": "string"},
                                                      "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_LIMIT}},
                     "required": ["entity"]}},
    {"name": "memory_change", "description": "The fact changed in the world: the new text becomes current, the old "
     "one is kept as history.",
     "inputSchema": {"type": "object", "properties": {"fact_id": _ID, "content": {"type": "string"},
                                                      "valid_from": _DAY, "entities": _ENTITIES, "keywords": _KEYWORDS,
                                                      "expect": _EXPECT},
                     "required": ["fact_id", "content"]}},
    {"name": "memory_correct", "description": "The fact was wrong: the right text (and/or start date) replaces it; the "
     "wrong one never comes back as the past.",
     "inputSchema": {"type": "object", "properties": {"fact_id": _ID, "content": {"type": "string"},
                                                      "valid_from": _DAY, "entities": _ENTITIES, "expect": _EXPECT},
                     "required": ["fact_id"]}},
    {"name": "memory_end", "description": "The fact no longer holds and nothing replaces it.",
     "inputSchema": {"type": "object", "properties": {"fact_id": _ID, "valid_to": _DAY, "expect": _EXPECT},
                     "required": ["fact_id"]}},
    {"name": "memory_revert", "description": "Undo the last change or correction of a fact.",
     "inputSchema": {"type": "object", "properties": {"fact_id": _ID}, "required": ["fact_id"]}},
    {"name": "memory_forget", "description": "Delete a fact with its whole history, when the person asks to forget it.",
     "inputSchema": {"type": "object", "properties": {"fact_id": _ID, "expect": _EXPECT}, "required": ["fact_id"]}},
    {"name": "memory_history", "description": "A fact's versions: what it said when, where each was written.",
     "readOnly": True, "inputSchema": {"type": "object", "properties": {"fact_id": _ID}, "required": ["fact_id"]}},
    {"name": "memory_list", "description": "Stored facts, most trusted first.", "readOnly": True,
     "inputSchema": {"type": "object", "properties": {
         "category": {"type": "string"}, "min_trust": {"type": "number"},
         "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_LIMIT}}}},
    {"name": "memory_stats", "description": "What is in the store: counts, categories, origins, the database file.",
     "readOnly": True, "inputSchema": {"type": "object", "properties": {}}},
    {"name": "memory_export", "description": "The store as one Markdown page for a review: each fact with trust, "
     "category, names and where it was written, most trusted first; facts merged into them below. Facts recall keeps "
     "out of prompts are included, marked (withheld from recall). The whole page enters the context (up to 'limit' "
     "facts of 500 characters each): ask for it only when the person wants to review the memory.", "readOnly": True,
     "inputSchema": {"type": "object", "properties": {
         "category": {"type": "string"}, "min_trust": {"type": "number"},
         "limit": {"type": "integer", "minimum": 1, "maximum": _EXPORT_MAX,
                   "description": f"Facts on the page (default {_EXPORT_LIMIT})."}}}},
    {"name": "memory_feedback", "description": "Rate a fact after using it: helpful (accurate), unhelpful (outdated "
     "or wrong), or confirmed (the person said again, or agreed, that it is so).",
     "inputSchema": {"type": "object", "properties": {
         "fact_id": _ID, "action": {"type": "string", "enum": ["helpful", "unhelpful", "confirmed"]},
         "expect": _EXPECT},
         "required": ["fact_id", "action"]}},
    {"name": "memory_merge", "description": "Two or more facts say the same thing (a duplicate, a retelling): the "
     "facts in 'from' become part of fact_id; their history and sources stay under it, search shows one fact. Not for "
     "something the person wants gone: that is memory_forget.",
     "inputSchema": {"type": "object", "properties": {
         "fact_id": {"type": "integer", "description": "The fact that stays."},
         "from": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 20,
                  "description": "The facts that become part of fact_id."},
         "keywords": {**_KEYWORDS, "description": "The keywords fact_id keeps (without: its own and theirs, its own "
                                                  "first)."},
         "expect": _EXPECT},
         "required": ["fact_id", "from"]}},
    {"name": "memory_update", "description": "Give a fact keywords (words a person might ask with that its text does "
     "not contain) or another category. Not for a new text: memory_change or memory_correct.",
     "inputSchema": {"type": "object", "properties": {"fact_id": _ID, "keywords": _KEYWORDS,
                                                      "category": {"type": "string", "enum": list(_CATEGORIES)},
                                                      "expect": _EXPECT},
                     "required": ["fact_id"]}},
    {"name": "memory_enrich", "description": "Facts stored without keywords, the most recalled first: give each 3-8 "
     "with memory_update.", "readOnly": True,
     "inputSchema": {"type": "object", "properties": {"limit": _LIMIT}}},
    {"name": "memory_reason", "description": "Facts that involve ALL of the named people or things.", "readOnly": True,
     "inputSchema": {"type": "object", "properties": {
         "entities": {"type": "array", "items": {"type": "string"}, "minItems": 1}, "category": {"type": "string"},
         "limit": _LIMIT}, "required": ["entities"]}},
    {"name": "memory_review", "description": "Clean-up candidates, each with a reason (low trust, never recalled, "
     "near-duplicates to merge), and names nobody confirmed. Advisory: nothing is changed.", "readOnly": True,
     "inputSchema": {"type": "object", "properties": {"limit": _LIMIT}}},
    {"name": "memory_entities", "description": "Known names with aliases and fact counts; 'provisional' = guessed from "
     "one fact, nobody confirmed.", "readOnly": True,
     "inputSchema": {"type": "object", "properties": {
         "status": {"type": "string", "enum": ["all", "established", "provisional"]}, "limit": _LIMIT}}},
    {"name": "memory_experience", "description": "The task-outcome journal, separate from trust and truth. recent: "
     "the latest attempts (a recall's 'Memory attempt: <id>', a search's memory_attempt); explain: what an attempt "
     "offered (fact and version ids, never text) and, with the trace, why a candidate was left out; outcome: your "
     "report of how the answer that used it turned out (never a confirmation by the person; changes no fact). Revise "
     "with a new event_id and supersedes=<the active event_id>.",
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["recent", "explain", "outcome"]},
         "attempt_id": {"type": "string"},
         "event_id": {"type": "string", "description": "A new id of your choosing for this report (1-80 of A-Z a-z "
                                                       "0-9 _ -); the same id again is a no-op."},
         "outcome": {"type": "string", "enum": ["useful", "dead_end", "corrected", "unknown"]},
         "reason": {"type": "string", "enum": ["wrong_entity", "stale_version", "irrelevant", "missing_evidence",
                                               "answer_error", "other"]},
         "correction": {"type": "string", "maxLength": 400, "description": "One short line: what was right."},
         "task": {"type": "string", "maxLength": 160, "description": "A short task label, not the conversation."},
         "supersedes": {"type": "string"},
         "correction_refs": {"type": "array", "maxItems": 20, "items": {"type": "object", "properties": {
             "fact_id": {"type": "integer"}, "version_id": {"type": "integer"}},
             "required": ["fact_id", "version_id"], "additionalProperties": False}}},
         "required": ["action"]}},
    {"name": "memory_entity", "description": "Maintain names. alias: one person or thing, two names (name stays, other "
     "becomes its alias; merges them if both exist). rename: other is the new name. delete: names that are not names "
     "(a phrase, a command): the facts stay, text never creates them again. split: two people under one name: the "
     "alias given as name becomes an entity of its own.",
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["alias", "rename", "delete", "split"]},
         "name": {"type": "string"},
         "other": {"type": "string", "description": "alias: the other name; rename: the new name."},
         "names": {"type": "array", "items": {"type": "string"}, "maxItems": 100,
                   "description": "delete: several names at once."}},
         "required": ["action"]}},
]
READ_ONLY = {s["name"] for s in SCHEMAS if s.pop("readOnly", False)}
NAMES = [s["name"] for s in SCHEMAS]
JOURNAL = "memory_experience"  # listed only with experience on


def listed(cfg) -> list[dict]:
    """The schemas a client is offered: the journal's tool only while it records."""
    return [s for s in SCHEMAS if s["name"] != JOURNAL or cfg.experience]


class Refusal(ValueError):
    """An answer to the agent, not a fault."""


def keywords(value) -> list[str]:
    """A list of strings or one comma-separated string -> distinct, trimmed (12 of 40 chars)."""
    items = value if isinstance(value, list) else str(value or "").split(",")
    out: list[str] = []
    for item in items:
        word = " ".join(item.split()).strip(" ,;") if isinstance(item, str) else ""
        if word and len(word) <= _MAX_KEYWORD_CHARS and word.casefold() not in {o.casefold() for o in out}:
            out.append(word)
    return out[:_MAX_KEYWORDS]


def split_tags(stored: str | None) -> tuple[str, list[str]]:
    text = stored or ""
    if text.startswith(_KW_MARK):
        return "", keywords(text[len(_KW_MARK):])
    head, mark, tail = text.rpartition("\n" + _KW_MARK)
    return (head, keywords(tail)) if mark else (text, [])


def join_tags(tags: str | None, words: list[str]) -> str:
    base = (tags or "").strip()
    if not words:
        return base
    return (base + "\n" if base else "") + _KW_MARK + ", ".join(words)


def _clean(value):
    """JSON-safe: vectors (bytes) out, everything else as is or as text."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if not isinstance(v, (bytes, bytearray, memoryview))}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _entities(a: dict) -> list[str]:
    value = a.get("entities") or []
    return [value] if isinstance(value, str) else [e for e in value if isinstance(e, str)]


def _limit(a: dict, default: int = 10) -> int:
    return max(1, min(_MAX_LIMIT, int(a.get("limit", default))))


class Tools:
    def __init__(self, memory: Memory, ctx: Context, state=None):
        self.memory, self.ctx, self.state = memory, ctx, state
        self.store, self.retriever, self.cfg = memory.store, memory.retriever, memory.cfg

    # -- dispatch --------------------------------------------------------------------------------------------------

    def call(self, name: str, args: dict) -> tuple[dict, bool]:
        """(answer, is_error)."""
        from .engine.history import ChangedSince, HistoryTooNew, NotFound
        from .engine.store import DuplicateContentError

        handler = getattr(self, name, None) if name in NAMES else None
        if handler is None:
            return {"error": f"Unknown tool: {name}"}, True
        if not isinstance(args, dict):
            return {"error": "arguments must be an object"}, True
        try:
            return _clean(handler(self._days(args))), False
        except Refusal as exc:
            return {"error": str(exc)}, True
        except NotFound as exc:
            return {"error": f"{exc.args[0] if exc.args else 'not found'}; nothing was changed."}, True
        except KeyError as exc:
            return {"error": f"Missing required argument: {exc.args[0] if exc.args else exc}"}, True
        except DuplicateContentError as exc:
            return {"updated": False, "duplicate_of": exc.fact_id,
                    "reason": "another fact already says exactly this; remove one of them instead"}, True
        except ChangedSince as exc:  # nothing was written: each named fact's current version id (null: none)
            return {"error": "a fact changed since it was read; nothing was written; read its history again",
                    "changed_since": {str(f): v for f, v in exc.current.items()}}, True
        except UnicodeError as exc:
            logger.warning("holomem: %s failed (%s)", name, type(exc).__name__)
            return {"error": f"{name} failed ({type(exc).__name__})"}, True
        except (ValueError, TypeError, HistoryTooNew, RuntimeError) as exc:
            # the engine's own refusals carry a message meant for the agent; a TypeError is a wrong argument type
            if isinstance(exc, TypeError):
                return {"error": f"{name}: an argument has the wrong type"}, True
            return {"error": str(exc)}, True
        except Exception as exc:  # noqa: BLE001 - named by kind: SQLite may quote a value in its message
            logger.warning("holomem: %s failed (%s)", name, type(exc).__name__)
            return {"error": f"{name} failed ({type(exc).__name__})"}, True

    def _days(self, a: dict) -> dict:
        """'today' / 'yesterday' in a date argument, as _DAY promises, made the store's date (its timezone): the
        engine reads only YYYY-MM-DD. Anything else goes on as given, for the engine to accept or refuse."""
        words = {k: str(a[k]).strip().lower() for k in _DAY_KEYS if isinstance(a.get(k), str)}
        if not any(w in ("today", "yesterday") for w in words.values()):
            return a
        import datetime

        today = datetime.date.fromisoformat(self.store.today())
        given = {"today": today, "yesterday": today - datetime.timedelta(days=1)}
        return {**a, **{k: given[w].isoformat() for k, w in words.items() if w in given}}

    @staticmethod
    def _expect(a: dict, name: str) -> dict[int, int] | None:
        """*a*'s 'expect' as {fact_id: version_id}, None when not given; a Refusal names the shape otherwise."""
        expect = a.get("expect")
        if expect is None:
            return None
        if not isinstance(expect, dict) or not all(
                str(k).strip().lstrip("-").isdigit() and isinstance(v, int) and not isinstance(v, bool)
                for k, v in expect.items()):
            raise Refusal(f"{name}: 'expect' must be an object {{\"<fact_id>\": <version_id>}}")
        seen = {int(k): v for k, v in expect.items()}
        if len(seen) != len(expect):  # "1" and "01" are one fact: keeping one of the two would drop a condition
            raise Refusal(f"{name}: 'expect' names the same fact twice")
        return seen or None

    def _expecting(self, a: dict, name: str):
        """The block a write runs in: with 'expect', the store's conditional transaction (issue #2: the write applies
        only while each named fact is at the version given, checked in the write's own transaction); without, as
        before."""
        import contextlib

        expect = self._expect(a, name)
        return self.store.expecting(expect) if expect else contextlib.nullcontext()

    def _fact(self, a: dict) -> int:
        try:
            return int(a["fact_id"])
        except (TypeError, ValueError):
            raise Refusal("fact_id must be an integer") from None

    def _screen(self, text: str, what: str) -> None:
        from .engine.screen import _fact_problem

        problem = _fact_problem(text)
        if problem:
            raise Refusal(f"Not {what}: {problem}.")

    def _keyword_check(self, a: dict, what: str) -> list[str]:
        words = keywords(a.get("keywords"))
        if words:
            from .engine.screen import _fact_problem

            problem = _fact_problem(", ".join(words))
            if problem:
                raise Refusal(f"Not {what}: keywords: {problem}.")
        return words

    def _unlinked(self, result: dict, a: dict) -> dict:
        from .engine.store import unusable_names

        names = _entities(a)
        bad = unusable_names(names) if names else []
        if bad and "error" not in result:
            result["entities_not_linked"] = bad
        return result

    # -- writes ----------------------------------------------------------------------------------------------------

    def memory_add(self, a: dict) -> dict:
        content = a["content"]
        if not isinstance(content, str) or not content.strip():
            raise Refusal("content must be a non-empty sentence")
        self._screen(content, "stored")
        words = self._keyword_check(a, "stored")
        content = content.strip()
        store = self.store
        if not store.read("SELECT 1 FROM facts WHERE content = ?", (content,)):
            merged = store.merged_match(content)
            if merged is not None:
                return {"fact_id": merged.get("fact_id"), "status": "merged", "stored": False,
                        "hint": "This text was merged into the fact named here, which says the same."}
            past = store.past_version(content)
            if past is not None:
                until = past["valid_to"] if past["valid_to_exact"] else f"some point before {past['valid_to']}"
                return {"fact_id": past["fact_id"], "status": "past_version", "stored": False,
                        "hint": f"Fact #{past['fact_id']} said exactly this until {until}. If it holds again, "
                                f"memory_change {{fact_id: {past['fact_id']}, content}} keeps one history."}
        origin = a.get("origin") or None
        fact_id, created = store.add_fact_status(content, category=a.get("category") or "general",
                                                 tags=join_tags("", words), entities=_entities(a),
                                                 valid_from=a.get("valid_from"), trust=store.start_trust(origin),
                                                 source=self.ctx.source("tool", origin=origin or ""))
        result: dict = {"fact_id": fact_id, "status": "added" if created else "exists"}
        if created:
            if words:
                result["keywords"] = words
            check = self.retriever.write_check(fact_id)
            result.update({k: v for k, v in check.items() if v})
            if check.get("possible_conflicts") or check.get("similar") or check.get("past_versions"):
                result["hint"] = (f"If one of these changed into the new fact: memory_forget {{fact_id: {fact_id}}} "
                                  "and memory_change {fact_id: <old id>, content} instead (one history); if the old "
                                  "one was wrong, memory_correct it; if both hold, keep both.")
        return self._unlinked(result, a)

    def memory_change(self, a: dict) -> dict:
        content = a["content"]
        if not isinstance(content, str) or not content.strip():
            raise Refusal("memory_change needs the new text in 'content'")
        self._screen(content, "changed")
        words = self._keyword_check(a, "changed")
        fact_id = self._fact(a)
        with self._expecting(a, "memory_change"):
            stored = self.store.fact_tags(fact_id)
            tags = join_tags(split_tags(stored)[0], words)
            if stored is None and a.get("keywords") is None:
                tags = None  # an ended fact resumes with its last version's tags
            result = self.store.change_fact(fact_id, content, valid_from=a.get("valid_from"), tags=tags,
                                            entities=_entities(a), source=self.ctx.source("tool"))
        return self._unlinked(dict(result), a)

    def memory_correct(self, a: dict) -> dict:
        if a.get("content") is None and not a.get("valid_from"):
            raise Refusal("memory_correct needs the right text in 'content' and/or the right start in 'valid_from'")
        if a.get("content") is not None:
            self._screen(a["content"], "corrected")
        fact_id = self._fact(a)
        with self._expecting(a, "memory_correct"):
            result = self.store.correct_fact(fact_id, a.get("content"), valid_from=a.get("valid_from"),
                                             entities=_entities(a), source=self.ctx.source("tool"))
        return self._unlinked(dict(result), a)

    def memory_end(self, a: dict) -> dict:
        fact_id = self._fact(a)
        with self._expecting(a, "memory_end"):
            return dict(self.store.end_fact(fact_id, valid_to=a.get("valid_to")))

    def memory_revert(self, a: dict) -> dict:
        return dict(self.store.revert_fact(self._fact(a)))

    def memory_forget(self, a: dict) -> dict:
        fact_id = self._fact(a)
        with self._expecting(a, "memory_forget"):
            result = dict(self.store.forget_fact(fact_id))
        if result.get("scrub") == "pending":  # a rewrite cannot run inside the conditional transaction: now
            result["scrub"] = "done" if self.store.scrub() else "pending"
        for gone in (fact_id, *result.get("merged_copies", ())):
            self.retriever.drop_cached(gone)
            if self.state is not None:
                self.state.forget_fact(gone)
        try:  # the engine's triggers took the attempts that offered it: their links go too
            journal.forget_links(self.memory, self.state)
        except Exception as exc:  # noqa: BLE001 - a stale link only makes explain say not_recorded
            logger.warning("holomem: journal links not cleared after a forget (%s)", type(exc).__name__)
        result["hint"] = ("Deleted from the fact store with its history. If the client's own memory (CLAUDE.md, "
                          "MEMORY.md, AGENTS.md) says the same, remove it there too.")
        return result

    def memory_merge(self, a: dict) -> dict:
        """Merge: keywords given replace the keep's; none given: the keep's own first, then each copy's,
        distinct, up to the limit. The tags are joined inside the store's transaction, from what it holds then."""
        keep = self._fact(a)
        given = a.get("from")
        ids = [given] if isinstance(given, int) and not isinstance(given, bool) else given
        if not isinstance(ids, list) or not ids or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
            raise Refusal("memory_merge needs 'from': the ids (integers) of the facts that become part of fact_id")
        words = self._keyword_check(a, "merged") if a.get("keywords") is not None else None
        expect = self._expect(a, "memory_merge")
        kept: list[str] = []

        def joined(keep_tags: str, copy_tags: list) -> str | None:
            user_tags, own = split_tags(keep_tags)
            if words is not None:
                kept[:] = words
                return join_tags(user_tags, words)
            candidates = list(own)
            for tags in copy_tags:
                candidates += split_tags(tags)[1]
            kept[:] = keywords(candidates)
            return join_tags(user_tags, kept) if candidates else None

        result = dict(self.store.merge_facts(keep, ids, tags=joined, expect=expect))
        for gone in (keep, *result.get("merged", ())):
            self.retriever.drop_cached(gone)
        result["keywords"] = kept
        result["hint"] = (f"The merged facts are kept under #{keep}: memory_history {{fact_id: {keep}}} shows their "
                          "versions and sources, and their text is not added as a new fact again."
                          if result.get("merged") else f"These facts were part of #{keep} already; nothing changed.")
        return result

    def memory_update(self, a: dict) -> dict:
        fact_id = self._fact(a)
        category = a.get("category")
        if a.get("content") is not None:
            raise Refusal("memory_update takes no 'content': a new text is memory_change (it changed) or "
                          "memory_correct (it was wrong); nothing was changed")
        if a.get("keywords") is None and category is None:
            raise Refusal("memory_update needs 'keywords' and/or 'category'")
        if category is not None and category not in _CATEGORIES:
            raise Refusal(f"category must be one of: {', '.join(_CATEGORIES)}")
        tags = None
        words = self._keyword_check(a, "updated") if a.get("keywords") is not None else []
        with self._expecting(a, "memory_update"):
            if a.get("keywords") is not None:
                stored = self.store.fact_tags(fact_id)
                if stored is None:
                    raise Refusal(f"fact_id {fact_id} is not a current fact; nothing was changed")
                tags = join_tags(split_tags(stored)[0], words)
            # keywords alone are a search aid, not a change of the fact: its age (decay, "recently changed") stays
            updated = self.store.update_fact(fact_id, tags=tags, category=category, touch=category is not None,
                                             source=self.ctx.source("tool"))
        self.retriever.drop_cached(fact_id)
        result: dict = {"updated": updated}
        if updated and a.get("keywords") is not None:
            result["keywords"] = words
        return result

    def memory_feedback(self, a: dict) -> dict:
        action = a["action"]
        if action not in ("helpful", "unhelpful", "confirmed"):
            raise Refusal("action must be 'helpful', 'unhelpful' or 'confirmed'")
        fact_id = self._fact(a)
        with self._expecting(a, "memory_feedback"):
            if action == "confirmed":
                return dict(self.store.record_confirmation(fact_id, self.ctx.source("feedback")))
            return dict(self.store.record_feedback(fact_id, helpful=action == "helpful"))

    # -- reads -----------------------------------------------------------------------------------------------------

    def _with_origins(self, facts: list[dict]) -> list[dict]:
        origins = self.store.origins([f["fact_id"] for f in facts])
        for fact in facts:
            info = origins.get(fact["fact_id"]) or {}
            if info.get("origin"):
                fact["origin"] = info["origin"]
            if info.get("confirmations"):
                fact["confirmations"] = info["confirmations"]
        return facts

    def memory_search(self, a: dict) -> dict:
        query = a["query"]
        if not isinstance(query, str):
            raise Refusal("query must be a string")
        floor = float(a.get("min_trust", self.cfg.min_trust_threshold))
        category = a.get("category")
        if a.get("as_of"):
            from .engine.history import parse_day

            day = parse_day(a["as_of"], field="as_of", today=self.store.today())
            found = dict(self.retriever.search_as_of(query, day, category=a.get("category"), min_trust=floor,
                                                     limit=_limit(a)))
            found["count"] = len(found.get("results", []))
            return found
        limit, meaning = _limit(a), self.memory.meaning_of(query, before_turn=False)
        results = list(self.retriever.search(query, category=category, min_trust=floor, limit=limit, meaning=meaning))
        trace = journal.Trace(self.cfg.experience and self.cfg.experience_trace)
        trace.observe("lexical_candidates", results)
        if self.memory.rescoring is not None and results:
            results = self._rescored(query, results)
        if meaning is not None:  # the splice: by meaning after every fact that holds the words, then those
            from .engine import embeddings  # closer in meaning than anything the words found

            words, min_score = list(results), self.memory.embedding.min_score
            if len(results) < limit:
                for fact in self.retriever.by_meaning(meaning, exclude={f["fact_id"] for f in results},
                                                      category=category, min_trust=floor, limit=limit - len(results),
                                                      margin=embeddings.SEARCH_MARGIN, min_score=min_score):
                    fact["match"] = "meaning"
                    results.append(fact)
            if words:
                for fact in self.retriever.beyond_words(meaning, query, words, exclude={f["fact_id"] for f in results},
                                                        category=category, min_trust=floor, k=embeddings.SEARCH_TOP_K,
                                                        beat=embeddings.SEARCH_BEAT_WORDS,
                                                        z_min=embeddings.SEARCH_Z_FLOOR, min_score=min_score):
                    fact["match"] = "meaning"
                    results.append(fact)
        results = self._with_origins(results)
        answer: dict = {"results": results, "count": len(results)}
        attempt = journal.record(self.memory, self.state, self.ctx, "search", results, trace)  # also when empty
        if attempt:
            answer["memory_attempt"] = attempt
        return answer

    def _rescored(self, query: str, results: list[dict]) -> list[dict]:
        """Rescoring of a search: the facts the words found in the endpoint's order, the ones it did not
        score behind in theirs; facts that may not be sent are shown, never sent. Any failure: *results* as they are."""
        from .engine.screen import _sendable

        rescoring = self.memory.rescoring
        try:
            positions = [i for i, f in enumerate(results) if _sendable(f.get("content") or "")]
            ranked = rescoring.scores(self.state, query, [results[i].get("content") or "" for i in positions]) \
                if positions else None
            if ranked is None:
                logger.debug("holomem: rescoring a search: %s", "nothing sendable" if not positions else rescoring.last)
                return results
            order = [positions[i] for i, _ in ranked if 0 <= i < len(positions)]
            scored = set(order)
            return [results[i] for i in order + [j for j in range(len(results)) if j not in scored]]
        except Exception as exc:  # noqa: BLE001 - as a failed call: paused, the order of rerank off
            rescoring.failed(self.state, f"search: {type(exc).__name__}")
            return results

    def memory_probe(self, a: dict) -> dict:
        results = self._with_origins(list(self.retriever.probe(a["entity"], limit=_limit(a),
                                                                min_trust=self.cfg.min_trust_threshold)))
        return {"results": results, "count": len(results)}

    def memory_related(self, a: dict) -> dict:
        results = list(self.retriever.related(a["entity"], limit=_limit(a), min_trust=self.cfg.min_trust_threshold))
        return {"results": results, "count": len(results)}

    def memory_reason(self, a: dict) -> dict:
        names = _entities(a)
        if not names:
            raise Refusal("memory_reason needs 'entities': the names to join")
        results = self._with_origins(list(self.retriever.reason(names, category=a.get("category"), limit=_limit(a),
                                                                min_trust=self.cfg.min_trust_threshold)))
        return {"results": results, "count": len(results)}

    def memory_enrich(self, a: dict) -> dict:
        facts = [{"fact_id": r["fact_id"], "content": r["content"]}
                 for r in self.store.facts_without_keywords(_KW_MARK, _limit(a))]
        result: dict = {"facts": facts, "count": len(facts)}
        if facts:
            result["hint"] = ("Give each 3-8 keywords with memory_update {fact_id, keywords}: synonyms, other roots and "
                              "translations someone might ask with.")
        return result

    def memory_review(self, a: dict) -> dict:
        facts = self.retriever.review(limit=_limit(a))
        names = self.store.stale_provisional_entities(limit=_limit(a))
        aliases = self.store.unconfirmed_text_aliases(limit=_limit(a))
        result: dict = {"candidates": facts, "count": len(facts), "entity_candidates": names,
                        "alias_candidates": aliases}
        hints = []
        if any(str(f.get("reason", "")).startswith("near-duplicate") for f in facts):
            hints.append("A near-duplicate: memory_merge {fact_id: <keep>, from: [...]} keeps one fact and both "
                         "histories.")
        if names:
            hints.append("entity_candidates: memory_entity {action: delete, names: [...]} for those that are not "
                         "names (the facts stay).")
        if aliases:
            hints.append("alias_candidates: names a fact's text gave as another name. One person: memory_entity "
                         "{action: alias, name, other}; two: {action: split, name: <alias>}.")
        if hints:
            result["hint"] = " ".join(hints)
        return result

    def memory_entities(self, a: dict) -> dict:
        status = a.get("status")
        rows = self.store.list_entities(limit=_limit(a, 50), status=None if status in (None, "", "all") else status)
        counts = self.store.entity_counts()
        result: dict = {"entities": rows, "count": len(rows), "provisional_entities": counts["provisional"]}
        if counts["text_aliases"]:
            result["text_aliases"] = counts["text_aliases"]
        return result

    def memory_entity(self, a: dict) -> dict:
        action, name, other = a["action"], a.get("name"), a.get("other")
        if action in ("alias", "rename"):
            if not isinstance(name, str) or not isinstance(other, str):
                raise Refusal(f"memory_entity {action} needs 'name' and 'other'")
            call = self.store.add_alias if action == "alias" else self.store.rename_entity
            return dict(call(name, other))
        if action == "split":
            if not isinstance(name, str):
                raise Refusal("memory_entity split needs 'name': the alias that becomes an entity of its own")
            return dict(self.store.split_alias(name))
        if action == "delete":
            names = a.get("names")
            names = [names] if isinstance(names, str) else list(names or [])
            if isinstance(name, str) and name:
                names.append(name)
            if not names or not all(isinstance(n, str) for n in names):
                raise Refusal("memory_entity delete needs 'name' or 'names'")
            return dict(self.store.delete_entities(names))
        raise Refusal("action must be one of: alias, rename, delete, split")

    def memory_experience(self, a: dict) -> dict:
        """The task-outcome journal over the bridge's links (journal.py): an attempt is found by its id in this
        database, whichever session of it recorded the attempt."""
        if not self.cfg.experience:
            raise Refusal("the task-outcome journal is off (experience = false in config.toml)")
        action, store = a.get("action"), self.store
        if action == "recent":
            attempts = []
            for link in (self.state.recent_attempts(self.ctx.client, str(self.memory.path)) if self.state else []):
                report = store.experience_report(link["attempt_id"], journal.key_of(link))
                item = {"attempt_id": link["attempt_id"], "kind": report.get("kind"), "created_at": link["created"],
                        "status": report["status"]}
                if report["status"] == "ok":
                    item["emitted"] = [e["fact_id"] for e in report["emitted"]]
                    active = [o for o in report["outcomes"] if o["active"]]
                    item["outcome"] = active[0]["outcome"] if active else None
                attempts.append(item)
            return {"attempts": attempts, "count": len(attempts), "error_kind": self.memory.journal_error}
        if action not in ("explain", "outcome"):
            raise Refusal("action must be one of: recent, explain, outcome")
        attempt_id = a["attempt_id"]
        link = journal.link_of(self.memory, self.state, attempt_id)
        if action == "explain":
            return store.experience_report(attempt_id, journal.key_of(link)) if link else {"status": "not_recorded"}
        if link is None:
            raise Refusal("Attempt unavailable: not recorded in this memory, or forgotten since")
        return dict(store.record_outcome(attempt_id, a["event_id"], a["outcome"], a["reason"],
                                         correction=a.get("correction", ""), supersedes=a.get("supersedes"),
                                         session_key=journal.key_of(link), task=a.get("task", ""),
                                         correction_refs=a.get("correction_refs")))

    def memory_history(self, a: dict) -> dict:
        return dict(self.store.history(self._fact(a)))

    def memory_list(self, a: dict) -> dict:
        facts = self.store.list_facts(category=a.get("category"), min_trust=float(a.get("min_trust", 0.0)),
                                      limit=_limit(a, 50))
        facts = self._with_origins(list(facts))
        return {"results": facts, "count": len(facts)}

    def memory_export(self, a: dict) -> dict:
        from .export import TOOL_CHARS, render

        limit = max(1, min(_EXPORT_MAX, int(a.get("limit", _EXPORT_LIMIT))))
        return render(self.store, category=a.get("category"), min_trust=float(a.get("min_trust", 0.0)), limit=limit,
                      chars=TOOL_CHARS)

    def memory_stats(self, a: dict) -> dict:
        read = self.store.read
        ids = [r[0] for r in read("SELECT fact_id FROM facts")]
        origins = self.store.origins(ids)
        counts: dict[str, int] = {}
        for fact_id in ids:
            key = (origins.get(fact_id) or {}).get("origin") or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return {"facts": self.store.count_facts(),
                "entities": read("SELECT COUNT(*) FROM entities")[0][0],
                "by_category": {r[0] or "general": r[1] for r in read("SELECT category, COUNT(*) FROM facts "
                                                                       "GROUP BY category")},
                "recalled_at_least_once": read("SELECT COUNT(*) FROM facts WHERE retrieval_count > 0")[0][0],
                "low_trust": read("SELECT COUNT(*) FROM facts WHERE trust_score < ?",
                                  (self.cfg.min_trust_threshold,))[0][0],
                "origin": counts, "history": self.store.history_counts() or "not set up yet",
                "search_by_meaning": self._meaning_stats(), "rescoring": self._rescoring_stats(),
                "database": str(self.memory.path), "scope": self.cfg.scope}

    def _meaning_stats(self) -> dict | str:
        if self.memory.embedder is None:
            return "off"
        return {"endpoint": self.memory.embedder.status(), **self.memory.pending_vectors()}


    def _rescoring_stats(self) -> dict | str:
        rescoring = self.memory.rescoring
        if rescoring is None:
            return "off"
        settings = rescoring.settings
        return {"model": settings.model, "endpoint": rescoring.status(self.state), "top_k": settings.top_k,
                "weight": settings.weight, "window": settings.window, "floor": settings.floor, "scale": settings.scale}


def dumps(answer: dict) -> str:
    return json.dumps(answer, ensure_ascii=False, default=str)

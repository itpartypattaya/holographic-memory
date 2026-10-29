"""The memory tools the MCP server offers: the plugin's fact_store actions, one tool each, over the engine.

The checks are the plugin's: text that is not a fact (an instruction, a credential, markup) is refused by the engine's
write screen; a text the person asked to forget, a past version or a copy merged into another fact is not stored
again as new; a new fact comes back with ``similar`` / ``possible_conflicts`` for the agent to decide on; a name that
cannot be an entity is listed under ``entities_not_linked`` by position, never quoted. Answers are JSON text; a
failure nobody meant is named by its kind, never by a message that may quote a fact.

Not in the bridge (yet): merge, aliases and entity maintenance, keywords enrichment by a nightly job, task outcomes.
"""
from __future__ import annotations

import json
import logging

from .memory import Context, Memory

logger = logging.getLogger("holomem")

_KW_MARK = "keywords: "  # the plugin's marker: keywords live in the indexed tags column after it
_MAX_KEYWORDS, _MAX_KEYWORD_CHARS, _MAX_LIMIT = 12, 40, 100

_ID = {"type": "integer", "description": "The fact id (#id in recalled lines and results)."}
_ENTITIES = {"type": "array", "items": {"type": "string"},
             "description": "Who or what the fact is about; needed for one-word names (a first name, a pet) so later "
                            "facts in any word form link to it."}
_KEYWORDS = {"type": "array", "items": {"type": "string"},
             "description": "Up to 12 words a person might ask with that the text does not contain (synonyms, other "
                            "roots, translations). Never shown in recall."}
_DAY = {"type": "string", "description": "YYYY-MM-DD, 'today' or 'yesterday'."}

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
                                                      "valid_from": _DAY, "entities": _ENTITIES, "keywords": _KEYWORDS},
                     "required": ["fact_id", "content"]}},
    {"name": "memory_correct", "description": "The fact was wrong: the right text (and/or start date) replaces it; the "
     "wrong one never comes back as the past.",
     "inputSchema": {"type": "object", "properties": {"fact_id": _ID, "content": {"type": "string"},
                                                      "valid_from": _DAY, "entities": _ENTITIES},
                     "required": ["fact_id"]}},
    {"name": "memory_end", "description": "The fact no longer holds and nothing replaces it.",
     "inputSchema": {"type": "object", "properties": {"fact_id": _ID, "valid_to": _DAY}, "required": ["fact_id"]}},
    {"name": "memory_revert", "description": "Undo the last change or correction of a fact.",
     "inputSchema": {"type": "object", "properties": {"fact_id": _ID}, "required": ["fact_id"]}},
    {"name": "memory_forget", "description": "Delete a fact with its whole history, when the person asks to forget it.",
     "inputSchema": {"type": "object", "properties": {"fact_id": _ID}, "required": ["fact_id"]}},
    {"name": "memory_history", "description": "A fact's versions: what it said when, where each was written.",
     "readOnly": True, "inputSchema": {"type": "object", "properties": {"fact_id": _ID}, "required": ["fact_id"]}},
    {"name": "memory_list", "description": "Stored facts, most trusted first.", "readOnly": True,
     "inputSchema": {"type": "object", "properties": {
         "category": {"type": "string"}, "min_trust": {"type": "number"},
         "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_LIMIT}}}},
    {"name": "memory_stats", "description": "What is in the store: counts, categories, origins, the database file.",
     "readOnly": True, "inputSchema": {"type": "object", "properties": {}}},
    {"name": "memory_feedback", "description": "Rate a fact after using it: helpful (accurate), unhelpful (outdated "
     "or wrong), or confirmed (the person said again, or agreed, that it is so).",
     "inputSchema": {"type": "object", "properties": {
         "fact_id": _ID, "action": {"type": "string", "enum": ["helpful", "unhelpful", "confirmed"]}},
         "required": ["fact_id", "action"]}},
]
READ_ONLY = {s["name"] for s in SCHEMAS if s.pop("readOnly", False)}
NAMES = [s["name"] for s in SCHEMAS]


class Refusal(ValueError):
    """An answer to the agent, not a fault."""


def keywords(value) -> list[str]:
    """The plugin's rule: a list of strings or one comma-separated string -> distinct, trimmed (12 of 40 chars)."""
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
            return _clean(handler(args)), False
        except Refusal as exc:
            return {"error": str(exc)}, True
        except NotFound as exc:
            return {"error": f"{exc.args[0] if exc.args else 'not found'}; nothing was changed."}, True
        except KeyError as exc:
            return {"error": f"Missing required argument: {exc.args[0] if exc.args else exc}"}, True
        except DuplicateContentError as exc:
            return {"updated": False, "duplicate_of": exc.fact_id,
                    "reason": "another fact already says exactly this; remove one of them instead"}, True
        except ChangedSince:
            return {"error": "a fact changed since it was read; read its history again"}, True
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
        result = self.store.correct_fact(self._fact(a), a.get("content"), valid_from=a.get("valid_from"),
                                         entities=_entities(a), source=self.ctx.source("tool"))
        return self._unlinked(dict(result), a)

    def memory_end(self, a: dict) -> dict:
        return dict(self.store.end_fact(self._fact(a), valid_to=a.get("valid_to")))

    def memory_revert(self, a: dict) -> dict:
        return dict(self.store.revert_fact(self._fact(a)))

    def memory_forget(self, a: dict) -> dict:
        fact_id = self._fact(a)
        result = dict(self.store.forget_fact(fact_id))
        for gone in (fact_id, *result.get("merged_copies", ())):
            self.retriever.drop_cached(gone)
            if self.state is not None:
                self.state.forget_fact(gone)
        result["hint"] = ("Deleted from the fact store with its history. If the client's own memory (CLAUDE.md, "
                          "MEMORY.md, AGENTS.md) says the same, remove it there too.")
        return result

    def memory_feedback(self, a: dict) -> dict:
        action = a["action"]
        if action not in ("helpful", "unhelpful", "confirmed"):
            raise Refusal("action must be 'helpful', 'unhelpful' or 'confirmed'")
        fact_id = self._fact(a)
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
        if a.get("as_of"):
            from .engine.history import parse_day

            day = parse_day(a["as_of"], field="as_of", today=self.store.today())
            found = dict(self.retriever.search_as_of(query, day, category=a.get("category"), min_trust=floor,
                                                     limit=_limit(a)))
            found["count"] = len(found.get("results", []))
            return found
        results = self.retriever.search(query, category=a.get("category"), min_trust=floor, limit=_limit(a))
        results = self._with_origins(list(results))
        return {"results": results, "count": len(results)}

    def memory_probe(self, a: dict) -> dict:
        results = self._with_origins(list(self.retriever.probe(a["entity"], limit=_limit(a),
                                                                min_trust=self.cfg.min_trust_threshold)))
        return {"results": results, "count": len(results)}

    def memory_related(self, a: dict) -> dict:
        results = list(self.retriever.related(a["entity"], limit=_limit(a), min_trust=self.cfg.min_trust_threshold))
        return {"results": results, "count": len(results)}

    def memory_history(self, a: dict) -> dict:
        return dict(self.store.history(self._fact(a)))

    def memory_list(self, a: dict) -> dict:
        facts = self.store.list_facts(category=a.get("category"), min_trust=float(a.get("min_trust", 0.0)),
                                      limit=_limit(a, 50))
        facts = self._with_origins(list(facts))
        return {"results": facts, "count": len(facts)}

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
                "database": str(self.memory.path), "scope": self.cfg.scope}


def dumps(answer: dict) -> str:
    return json.dumps(answer, ensure_ascii=False, default=str)

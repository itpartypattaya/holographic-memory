"""Rescoring (``rerank = true``, off by default): a second stage, a cross-encoder at a Cohere-shaped
``/v1/rerank`` endpoint (llama-server with ``--reranking``) that reads the message and each candidate together and
reorders them. Recall uses it in ``recall.py`` (one ranked pool of the three streams), memory_search between the words
and the facts found by meaning (``tools.py``). Any failure gives exactly what rerank off gives.

The engine's ``Rescorer`` makes the call (no key, no redirects, plain http only to this machine or the local network,
a query holding a credential or over 500 characters never sent, the answer checked index by index). What a long-running
process would keep in memory between turns a hook cannot — it is a new process at every prompt — so here:
- One authority over the pause: state.db (``backoff``), shared by the hooks and the MCP server, keyed by a fingerprint
  of the endpoint (sha256 of scheme://host:port/path, 16 hex), never its address. A failed call pauses the endpoint
  60 s, doubling to 900 s, until an answer clears it; while paused no process calls it, and a call not made is not a
  new failure. A Rescorer is made for each call, so its own in-memory pause never answers in state.db's place: a warm
  server whose Rescorer had paused itself would skip calls state.db already allows.
- The speed the endpoint showed: the last 5 samples (seconds per character of the pairs) in state.db, and the bridge
  decides how many of the first documents a call sends (the engine's ``fitting``, over every process's samples): as
  many as the slowest sample scores in half the call's timeout, two at least; top_k before any sample. A sample is the
  wall time of an answered call, or the timeout of one that ran out after the connection was up (a lower bound; a
  connect or name lookup that outlasts the call says nothing of how fast the endpoint scores, and the engine's
  Rescorer, which sees the connection, keeps a sample of its own only then). Wall time includes setting up the
  connection — through an SSH tunnel its round trips — so the estimate is slower than the engine's own, measured from
  the connection: fewer documents go, never more.
- The deadline: a call takes ``min(rerank_timeout, what the deadline leaves after RESERVE seconds)``, the reserve being
  what the turn needs after it (the fallback search, the lines, state.db); under 0.5 s the call is not made. The engine
  further keeps rerank_timeout within 7.5 s minus embedding_timeout.
- A rerank_url with credentials or a query is refused by config.py: neither reaches a request, a log or state.db.
Logs name a failure by its kind, never a fact's text.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
import urllib.parse

logger = logging.getLogger("holomem")

NAME_PREFIX = "rerank:"
RESERVE = 0.8  # seconds of a deadline kept for the rest of the turn after the call
MIN_TIMEOUT = 0.5  # a call with less than this is not made
_WINDOW_MIN = 2  # the window never cuts below two scored facts
_INTERNAL = ("paused", "unavailable")  # the Rescorer's own skips: never a new failure


def fingerprint(url: str) -> str:
    """``rerank:<16 hex>``: the endpoint as scheme://host:port/path, without anything else of the address."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
        port = parts.port or {"http": 80, "https": 443}.get(parts.scheme, 0)
    except ValueError:
        parts, host, port = urllib.parse.urlsplit(""), "", 0
    path = parts.path.rstrip("/")
    if not path.endswith("/rerank"):
        path += "/rerank"
    return NAME_PREFIX + hashlib.sha256(f"{parts.scheme}://{host}:{port}{path}".encode()).hexdigest()[:16]


def pair_chars(query: str, document: str) -> int:
    """What one pair costs the endpoint, as sent: the query and the document cut to what the query leaves of
    RERANK_PAIR_CHARS (the engine's _pair_document)."""
    from .engine.rerank import RERANK_PAIR_CHARS

    return len(query) + min(len(document), max(RERANK_PAIR_CHARS - len(query), 0))


def bottom_cut(scores: list[float], window: float) -> list[int]:
    """The ranked positions within *window* of the best score (normalised to [0, 1])."""
    if not scores:
        return []
    top = scores[0]
    return [i for i, s in enumerate(scores) if top - s <= window]


def window_keep(keep: list[int], ranked_len: int, need: int = _WINDOW_MIN) -> list[int]:
    """The cut topped up to *need* scored positions (the best missing ranks first)."""
    if not keep:
        return keep
    out = set(keep)
    for i in range(ranked_len):
        if len(out) >= min(need, ranked_len):
            break
        out.add(i)
    return sorted(out)


def blend_union(union: list[dict], ranked: list[tuple[int, float]], weight: float) -> list[dict]:
    """The blend (rerank_weight > 0): every fact of the pool gets (1 - weight) x its first score +
    weight x the endpoint's score min-max normalised; a fact the endpoint did not score gets the bottom of that scale.
    Sorts *union* in place and returns it."""
    raw = [s for _, s in ranked]
    lo, hi = min(raw), max(raw)
    span = (hi - lo) or 1.0
    norm = {i: (s - lo) / span for i, s in ranked if i < len(union)}
    for i, fact in enumerate(union):
        score = fact.get("score") or 0.0
        fact["score"] = (1.0 - weight) * score + weight * norm.get(i, 0.0)
    union.sort(key=lambda f: f.get("score") or 0.0, reverse=True)
    return union


class Rescoring:
    """The endpoint of one configuration: its settings, its fingerprint, the calls."""

    def __init__(self, config: dict):
        from .engine import rerank

        self._config = config
        self.settings = rerank.settings_from(config)
        if self.settings is None:
            raise ValueError("rerank is off")
        self.name = fingerprint(self.settings.url)
        self.unavailable = rerank.Rescorer(self.settings).unavailable()  # a refused address (plain http to a public host)
        self.last = "not_called"  # the last call's outcome: answered / failed: ... / skipped...

    def describe(self, held: dict | None) -> str:
        """ready / paused (kind); tried again in N s / unavailable: why — never the address."""
        if self.unavailable:
            return f"unavailable: {self.unavailable}"
        if held:
            return f"paused ({held['error'] or 'failed'}); tried again in {int(held['until'] - time.time()) + 1} s"
        return "ready"

    def status(self, state) -> str:
        return self.describe(state.backoff(self.name) if state is not None else None)

    def timeout(self, budget) -> float:
        """This call's timeout: rerank_timeout, cut to what *budget* leaves after RESERVE."""
        timeout = float(self.settings.timeout)
        return timeout if budget is None else min(timeout, budget.left() - RESERVE)

    def callable(self, state, budget=None) -> bool:
        """Whether a call may be made now: the address accepted, no pause in state.db, time enough."""
        return not self.unavailable and (state is None or state.backoff(self.name) is None) \
            and self.timeout(budget) >= MIN_TIMEOUT

    def fitting(self, state, query: str, documents: list[str], timeout: float) -> int:
        """How many of the first *documents* a call sends (the engine's fitting, over the samples in state.db)."""
        from .engine.rerank import RERANK_TIME_SHARE

        count = min(int(self.settings.top_k), len(documents))
        slowest = max(state.speeds(self.name), default=0.0) if state is not None else 0.0
        if slowest <= 0.0:
            return count
        budget, used = RERANK_TIME_SHARE * timeout / slowest, 0
        for i, document in enumerate(documents[:count]):
            used += pair_chars(query, document)
            if used > budget:
                return max(i, min(2, count))
        return count

    def scores(self, state, query: str, documents: list[str], budget=None) -> list[tuple[int, float]] | None:
        """``(index into documents, score)`` pairs, best first, for the documents sent (the first fitting() of them);
        None when no call was made or it failed (``last`` says which), and the caller keeps the order of rerank off."""
        from .engine import rerank
        from .engine.screen import _secret_in

        if not documents:
            self.last = "not_called"
            return None
        if self.unavailable:
            self.last = "skipped: unavailable"
            return None
        if state is not None and state.backoff(self.name) is not None:
            self.last = "skipped: paused"
            return None
        timeout = self.timeout(budget)
        if timeout < MIN_TIMEOUT:
            self.last = "skipped: budget"
            return None
        settings = rerank.Settings(self._config)
        settings.timeout = min(settings.timeout, timeout)
        rescorer = rerank.Rescorer(settings, secret_check=_secret_in)
        text = (query or "").strip()
        sent = documents[:self.fitting(state, text, documents, settings.timeout)]
        chars = sum(pair_chars(text, d) for d in sent)
        started = time.perf_counter()
        ranked = rescorer.scores(query, sent)
        took = time.perf_counter() - started
        self.last = rescorer.last_status
        # the engine samples a timeout only once the connection was up (net: "connected"); its samples are read here
        # as that witness (a private field: if it goes, no timeout sample — fewer documents never follow from it)
        connected = bool(getattr(rescorer, "_speeds", None))
        self._settle(state, rescorer.last_status, took / chars if chars else 0.0,
                     settings.timeout / chars if chars and connected else 0.0)
        return ranked

    def failed(self, state, why: str) -> None:
        """A failure outside the call (the stage that maps the answer): paused like a failed call."""
        self.last = f"failed: {why}"
        logger.warning("holomem: rescoring failed (%s); recall as with rerank off", why)
        self._write(state, "fail", why)

    def _settle(self, state, status: str, answered_sample: float, timeout_sample: float) -> None:
        if status == "answered":
            self._write(state, "clear")
            if answered_sample > 0:
                self._write(state, "add_speed", answered_sample)
            return
        if not status.startswith("failed: "):
            return  # skipped_single / skipped_secret / skipped_long / skipped_busy: no call, no failure
        why = status[len("failed: "):]
        if why.startswith(_INTERNAL):
            return
        kind = "malformed answer" if why.startswith("malformed answer") else why
        self._write(state, "fail", kind)
        if why == "no answer (timeout)" and timeout_sample > 0:
            self._write(state, "add_speed", timeout_sample)

    def _write(self, state, action: str, *args) -> None:
        if state is None:
            return
        try:
            done = getattr(state, action)(self.name, *args)
        except sqlite3.Error as exc:
            done = False
            logger.warning("holomem: rescoring state not written (%s)", type(exc).__name__)
        if done is False:
            logger.warning("holomem: rescoring state not written (state.db busy)")

    def normalized(self, ranked: list[tuple[int, float]]) -> list[float]:
        """Scores in [0, 1] for the window (rerank_scale); a single score as it is."""
        from .engine.rerank import normalized

        return [s for _, s in ranked] if len(ranked) < 2 else normalized(ranked, self.settings.scale)

    def arrange(self, pool: list[dict], ranked: list[tuple[int, float]]) -> list[dict]:
        """The pool in its new order, *ranked* holding (position in pool, score): weight 0 — the scored by score, the
        rest (not sent, not sendable) behind in their order, and with rerank_window the scored tail below the window
        cut (never below two); weight > 0 — every member blended (blend_union), no window. Before either, with
        rerank_floor a scored fact below it on the 0..1 scale goes, the last one too and the window's two as well:
        the window is relative to the best candidate, so a lone unrelated one always went in (2026-10-31)."""
        weight, window = float(self.settings.weight), float(self.settings.window)
        floor = float(self.settings.floor)
        if floor > 0.0:
            from .engine.rerank import normalized

            low = {i for (i, _), s in zip(ranked, normalized(ranked, self.settings.scale)) if s < floor}
            ranked = [(i, s) for i, s in ranked if i not in low]
            pool = [f for i, f in enumerate(pool) if i not in low]
            ranked = [(i - sum(1 for j in low if j < i), s) for i, s in ranked]
            if not ranked:  # nothing scored is left: the facts that were not scored, in their order
                return pool
        if weight > 0.0:
            return blend_union(list(pool), ranked, weight)
        order = [i for i, _ in ranked if i < len(pool)]
        scored = set(order)
        arranged = [pool[i] for i in order + [i for i in range(len(pool)) if i not in scored]]
        if window > 0.0:
            keep = window_keep(bottom_cut(self.normalized(ranked), window), len(ranked))
            arranged = [arranged[i] for i in keep] + arranged[len(ranked):]
        return arranged

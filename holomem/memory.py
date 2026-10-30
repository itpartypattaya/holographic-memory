"""Opening the fact store for a session: which database (config scope), the engine's store and retriever, and, with
``embeddings`` on, the endpoint of search by meaning (the engine's Embedder; the key from ``$HOLOMEM_HOME/.env`` or the
environment), and with ``rerank`` on the rescoring endpoint (rescore.py)."""
from __future__ import annotations

import datetime
import hashlib
import logging
import os
import re
import time
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path

from .config import Config, project_key, project_root

logger = logging.getLogger("holomem")

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_RESERVE = 0.3  # seconds of a catch-up budget kept for storing the last answer

CLIENTS = ("claude-code", "codex")


@dataclass
class Context:
    """Who is asking: the client (claude-code / codex), its session, the directory it runs in."""
    client: str = ""
    session_id: str = ""
    cwd: str = ""

    @property
    def project(self) -> str:
        return project_key(project_root(Path(self.cwd))) if self.cwd else ""

    def source(self, via: str, origin: str = "") -> dict:
        """Where a version is written, in the engine's source shape: the client as the platform, the project as the
        chat (so recall_source can say "this project" / "another project"), the client's session."""
        return {"via": via, "platform": self.client, "chat_type": "project" if self.cwd else "", "chat_id": self.project,
                "thread_id": "", "session_id": self.session_id, "user_id": "", "user_name": "", "origin": origin}


def secrets(path: Path) -> dict[str, str]:
    """NAME=value lines of a .env file (comments, blank lines and anything else skipped; quotes around a value
    removed). Never logged or shown: the values are keys."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        match = _ENV_LINE.match(line)
        if match and not line.lstrip().startswith("#"):
            value = match.group(2)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            out[match.group(1)] = value
    return out


class LocalTime(datetime.tzinfo):
    """The machine's timezone with its clock changes, read from the system at each date: an offset taken once at start
    would stay wrong in an MCP server running through a change (Codex review 2026-10-31)."""

    @staticmethod
    def _stamps(wall: datetime.datetime) -> list[float]:
        """The moments a local wall time names: one, two in the hour an autumn change repeats (earlier first), none in
        the hour a spring change skips."""
        found = set()
        for isdst in (0, 1):
            try:
                stamp = time.mktime(wall.timetuple()[:8] + (isdst,))
            except (OverflowError, ValueError, OSError):
                continue
            if time.localtime(stamp)[:6] == wall.timetuple()[:6]:
                found.add(stamp)
        return sorted(found)

    def fromutc(self, dt: datetime.datetime) -> datetime.datetime:
        stamp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        local = dt + datetime.timedelta(seconds=time.localtime(stamp).tm_gmtoff)
        twice = self._stamps(local.replace(tzinfo=None))
        return local.replace(fold=1) if len(twice) == 2 and stamp == twice[1] else local

    def utcoffset(self, dt: datetime.datetime | None) -> datetime.timedelta:
        stamps = self._stamps(dt.replace(tzinfo=None, fold=0)) if dt is not None else []
        if not stamps:  # now, a skipped hour, or a date the system cannot place: the offset of the moment it names
            try:
                stamps = [time.mktime(dt.replace(tzinfo=None).timetuple()) if dt is not None else time.time()]
            except (OverflowError, ValueError, OSError):
                stamps = [time.time()]
        stamp = stamps[-1] if dt is not None and dt.fold and len(stamps) == 2 else stamps[0]
        return datetime.timedelta(seconds=time.localtime(stamp).tm_gmtoff)

    def dst(self, dt: datetime.datetime | None) -> None:
        return None

    def tzname(self, dt: datetime.datetime | None) -> str:
        return time.strftime("%z", time.localtime())


def environ(cfg: Config):
    """Where the embeddings key is looked up: $HOLOMEM_HOME/.env first, then the environment."""
    return ChainMap(secrets(cfg.secrets_path), os.environ)


class Memory:
    """The engine's MemoryStore and FactRetriever for one database, and the Embedder when search by meaning is on."""

    def __init__(self, cfg: Config, db_path: Path):
        from .engine import embeddings
        from .engine.retrieval import FactRetriever  # numpy and the dictionaries: only once a store is needed
        from .engine.store import MemoryStore

        self.cfg, self.path = cfg, db_path
        self.journal_error = ""  # the last receipt failure, by kind (memory_experience recent)
        # dates in this machine's timezone ("today", recorded_on): without it they were UTC, and from midnight to 07:00
        # at UTC+7 "today" was yesterday (2026-10-31)
        self.store = MemoryStore(db_path, default_trust=cfg.default_trust, hrr_dim=cfg.hrr_dim, tz=LocalTime())
        self.retriever = FactRetriever(self.store, temporal_decay_half_life=cfg.temporal_decay_half_life,
                                       hrr_weight=cfg.hrr_weight, hrr_dim=cfg.hrr_dim)
        self.embedding = embeddings.settings_from(cfg.embedding_config())
        self.embedder = embeddings.Embedder(self.embedding, environ=environ(cfg)) if self.embedding else None
        if cfg.rerank:
            from .rescore import Rescoring

            self.rescoring = Rescoring(cfg.rerank_config())
        else:
            self.rescoring = None
        if self.embedding is None:  # off: no vector made by an outside model stays in the file
            try:
                dropped = self.store.drop_vectors()
                if dropped:
                    logger.warning("holomem: search by meaning is off; deleted %d stored vectors", dropped)
            except Exception as exc:  # noqa: BLE001 - a busy file: the next open tries again
                logger.debug("holomem: vectors not dropped (%s)", type(exc).__name__)

    def close(self) -> None:
        self.store.close()

    def meaning_of(self, query: str, *, before_turn: bool, whole: str | None = None):
        """How close each fact with a vector is to *query* (engine Meaning), or None: off, unusable, a message not worth
        a call or holding a credential, no fact vector yet. *whole*: the message *query* was cut from — it is what is
        screened, so a key past the cut still keeps the message from the endpoint. One request at most, within
        embedding_timeout; any failure is None."""
        from .engine import embeddings
        from .engine.screen import _secret_in

        if self.embedder is None or _secret_in(query or "") or (whole is not None and _secret_in(whole)) \
                or not embeddings.worth_asking(query, before_turn=before_turn):
            return None
        try:
            tag = self.embedding.tag
            if not self.embedder.usable() or not self.retriever.vector_index(tag).ready():
                return None
            vector = self.embedder.query(query)
            return self.retriever.meaning(vector, tag, self.embedding.weight) if vector is not None else None
        except Exception as exc:  # noqa: BLE001 - named by kind: a message may quote a fact
            logger.warning("holomem: search by meaning skipped (%s)", type(exc).__name__)
            return None

    def pending_vectors(self) -> dict:
        """{facts_with_vector, waiting, never_sent} for the current model; {} when search by meaning is off."""
        from .engine.screen import _sendable

        if self.embedding is None:
            return {}
        return self.retriever.vector_index(self.embedding.tag).counts(_sendable)

    def catch_up(self, budget: float, *, lock=None, stop=None) -> int:
        """Vectors for facts that have none (or whose text changed): requests until *budget* seconds are spent or
        nothing waits, each as many facts as the endpoint's measured speed fits (a catch-up). Facts recall
        would not inject, or that look secret, are never sent. *lock* (the MCP server's): held while the store is read
        or written, never during a request; *stop* (an Event): set when the store is closing, checked under the lock.
        Returns how many vectors were stored."""
        from .engine import embeddings
        from .engine.screen import _sendable

        if self.embedder is None:
            return 0
        deadline = time.monotonic() + budget
        tag, stored, first = self.embedding.tag, 0, True
        index = self.retriever.vector_index(tag)
        while self.embedder.usable():
            with lock if lock is not None else _NoLock():
                if stop is not None and stop.is_set():
                    break
                todo = index.pending(embeddings.BATCH, allowed=_sendable)
            left = deadline - time.monotonic() - _RESERVE
            if not todo or left < embeddings.MIN_REQUEST_SECONDS:
                break
            todo = todo[:self.embedder.fact_batch([len(content) for _, content, _ in todo], left, probe=first)]
            if not todo:
                break
            vectors = self.embedder.embed_facts([content for _, content, _ in todo], budget=left, decisive=first)
            first = False
            if vectors is None:
                break
            with lock if lock is not None else _NoLock():
                if stop is not None and stop.is_set():
                    break
                stored += self.store.put_vectors(tag, [(fact_id, digest, embeddings.to_blob(vector))
                                                       for (fact_id, _, digest), vector in zip(todo, vectors)])
        return stored


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def open_memory(cfg: Config, ctx: Context) -> Memory | None:
    """The store for this context, or None: memory off (config problems) or no database in this scope."""
    if not cfg.usable:
        return None
    path = cfg.database_for(ctx.cwd or None)
    if path is None:
        return None
    return Memory(cfg, path)


def revision(fact: dict) -> tuple[int, str]:
    """(fact_id, content hash): a reworded fact is a new revision even within the same second."""
    return int(fact["fact_id"]), hashlib.sha1((fact.get("content") or "").encode()).hexdigest()[:12]

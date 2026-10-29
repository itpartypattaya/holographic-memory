"""Opening the fact store for a session: which database (config scope), the engine's store and retriever."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .config import Config, project_key, project_root

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
        """Where a version is written, in the plugin's source shape: the client as the platform, the project as the
        chat (so recall_source can say "this project" / "another project"), the client's session."""
        return {"via": via, "platform": self.client, "chat_type": "project" if self.cwd else "", "chat_id": self.project,
                "thread_id": "", "session_id": self.session_id, "user_id": "", "user_name": "", "origin": origin}


class Memory:
    """The engine's MemoryStore and FactRetriever for one database."""

    def __init__(self, cfg: Config, db_path: Path):
        from .engine.retrieval import FactRetriever  # numpy and the dictionaries: only once a store is needed
        from .engine.store import MemoryStore

        self.cfg, self.path = cfg, db_path
        self.store = MemoryStore(db_path, default_trust=cfg.default_trust, hrr_dim=cfg.hrr_dim)
        self.retriever = FactRetriever(self.store, temporal_decay_half_life=cfg.temporal_decay_half_life,
                                       hrr_weight=cfg.hrr_weight, hrr_dim=cfg.hrr_dim)

    def close(self) -> None:
        self.store.close()


def open_memory(cfg: Config, ctx: Context) -> Memory | None:
    """The store for this context, or None: memory off (config problems) or no database in this scope."""
    if not cfg.usable:
        return None
    path = cfg.database_for(ctx.cwd or None)
    if path is None:
        return None
    return Memory(cfg, path)


def revision(fact: dict) -> tuple[int, str]:
    """(fact_id, content hash): a reworded fact is a new revision even within the same second (as the plugin)."""
    return int(fact["fact_id"]), hashlib.sha1((fact.get("content") or "").encode()).hexdigest()[:12]

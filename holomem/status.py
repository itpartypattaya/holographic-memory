"""holomem status: settings, which database, how many facts, search by meaning and rescoring (never a key or the
rescoring address), whether the clients have the hooks and the server.

Opening the store here runs the task-outcome journal's retention (also with the journal off): outcome text past its
30 days goes from the file."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import config, install
from .memory import Context, open_memory, secrets
from .state import read_backoff


def _claude_state() -> str:
    path = Path.home() / ".claude" / "settings.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "no user settings"
    hooks = json.dumps(settings.get("hooks", {}))
    return "hooks installed (user)" if install._HOOK_MARK in hooks else "hooks not installed (user)"


def _codex_state() -> str:
    path = Path.home() / ".codex" / "config.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "no config.toml"
    return "installed (user)" if install.MARK_BEGIN in text else "not installed (user)"


def _meaning(cfg: config.Config, memory) -> str:
    """off, or the model, where its key comes from (never the key), the endpoint's state and the vectors."""
    if memory.embedder is None:
        return "off"
    from .engine.embeddings import _RE_ENV_NAME, _RE_KEY_PREFIX

    name = cfg.embedding_key_env
    if not _RE_ENV_NAME.fullmatch(name) or _RE_KEY_PREFIX.match(name):
        # a key pasted where its variable's name goes: never echoed (the engine refuses it the same way)
        return f"on ({cfg.embedding_model}); embedding_key_env is not the name of a variable (not shown)"
    where = ("in " + str(cfg.secrets_path) if name in secrets(cfg.secrets_path)
             else "in the environment" if os.environ.get(name) else "not set")
    counts = memory.pending_vectors()
    return (f"on ({cfg.embedding_model}); key {name}: {where}; endpoint: {memory.embedder.status()}; facts with a "
            f"vector {counts.get('facts_with_vector', 0)}, waiting {counts.get('waiting', 0)}")


def _rescoring(cfg: config.Config, memory) -> str:
    """off, or the model and the endpoint's state (ready / paused (kind) / unavailable: why), read from state.db."""
    rescoring = memory.rescoring
    if rescoring is None:
        return "off"
    try:
        held = read_backoff(cfg.state_path, rescoring.name)
    except Exception as exc:  # noqa: BLE001 - a busy or broken state.db: said by kind
        return f"on ({rescoring.settings.model}); endpoint: unknown (state.db: {type(exc).__name__})"
    return f"on ({rescoring.settings.model}); endpoint: {rescoring.describe(held)}"


def _journal(cfg: config.Config, memory) -> str:
    """on/off, the trace, and how many attempts and outcomes the store holds (also when off: data left behind)."""
    store = memory.store
    counts = (store.read("SELECT (SELECT COUNT(*) FROM memory_attempts), (SELECT COUNT(*) FROM memory_outcomes)")[0]
              if store.experience_ready() else (0, 0))
    state = f"on (trace {'on' if cfg.experience_trace else 'off'})" if cfg.experience else "off"
    return f"{state}; attempts {counts[0]}, outcomes {counts[1]} (kept 30 days)"


def main() -> int:
    cfg = config.load()
    print(f"settings: {cfg.home / 'config.toml'}" + ("" if (cfg.home / "config.toml").exists() else " (none: defaults)"))
    for problem in cfg.problems:
        print(f"  problem: {problem}")
    for warning in cfg.warnings:
        print(f"  warning: {warning}")
    print(f"scope: {cfg.scope}; auto_extract: {cfg.auto_extract}; store_policy: {cfg.store_policy}")
    memory = open_memory(cfg, Context(client="status", cwd=os.getcwd()))
    if memory is None:
        print("database: none here (memory off or no project)")
    else:
        try:
            from .journal import prune

            prune(memory)
            print(f"database: {memory.path}; facts: {memory.store.count_facts()}")
            print(f"search by meaning: {_meaning(cfg, memory)}")
            print(f"rescoring: {_rescoring(cfg, memory)}")
            print(f"task-outcome journal: {_journal(cfg, memory)}")
        finally:
            memory.close()
    print(f"python: {install.python_path()}")
    print(f"claude code: {_claude_state()}")
    print(f"codex: {_codex_state()}")
    return 0 if cfg.usable else 1


if __name__ == "__main__":
    sys.exit(main())

"""holomem status: settings, which database, how many facts, whether the clients have the hooks and the server."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import config, install
from .memory import Context, open_memory, secrets


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
            print(f"database: {memory.path}; facts: {memory.store.count_facts()}")
            print(f"search by meaning: {_meaning(cfg, memory)}")
        finally:
            memory.close()
    print(f"python: {install.python_path()}")
    print(f"claude code: {_claude_state()}")
    print(f"codex: {_codex_state()}")
    return 0 if cfg.usable else 1


if __name__ == "__main__":
    sys.exit(main())

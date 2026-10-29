"""holomem status: settings, which database, how many facts, whether the clients have the hooks and the server."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import config, install
from .memory import Context, open_memory


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
        finally:
            memory.close()
    print(f"python: {install.python_path()}")
    print(f"claude code: {_claude_state()}")
    print(f"codex: {_codex_state()}")
    return 0 if cfg.usable else 1


if __name__ == "__main__":
    sys.exit(main())

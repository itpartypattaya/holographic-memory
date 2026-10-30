"""holomem embed [--seconds N]: make the vectors of search by meaning now (after an import, or to fill a store at once).

The MCP server does this in the background during a session; this command does it in one go, within *N* seconds. Facts
recall would not inject, or that look secret, are never sent (the engine's rule).
"""
from __future__ import annotations

import os
import sys

from . import config
from .memory import Context, open_memory


def main(args, out=sys.stdout) -> int:
    cfg = config.load()
    if cfg.problems:
        print("memory is off: " + "; ".join(cfg.problems), file=out)
        return 1
    if not cfg.embeddings:
        print(f"search by meaning is off: set embeddings = true in {cfg.home / 'config.toml'}", file=out)
        return 1
    memory = open_memory(cfg, Context(client="embed", cwd=os.getcwd()))
    if memory is None:
        print("no database here (project scope without a project)", file=out)
        return 1
    try:
        why = memory.embedder.unavailable()
        if why:
            print(f"search by meaning cannot run: {why}", file=out)
            return 1
        stored = memory.catch_up(float(args.seconds))
        counts = memory.pending_vectors()
        print(f"vectors made: {stored}; facts with a vector: {counts.get('facts_with_vector', 0)}; waiting: "
              f"{counts.get('waiting', 0)}; never sent (secret or not recallable): {counts.get('never_sent', 0)}; "
              f"endpoint: {memory.embedder.status()}", file=out)
        return 0 if not counts.get("waiting") else 1
    finally:
        memory.close()

"""Holographic memory for Claude Code and Codex CLI: a local fact store behind hooks and an MCP server."""

import os

__version__ = "0.11.2"

# numpy's OpenBLAS commits ~32 MB per CPU thread when numpy is imported: ~225 MB of the ~260 MB an MCP server held on an
# 8-thread laptop, in every session. holomem's math (a matrix-vector product over the facts, FFTs) needs no more than one
# thread. Set before anything imports numpy: every entry point imports this package first. A count the person set stays.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

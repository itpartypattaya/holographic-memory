"""holomem — holographic memory for Claude Code and Codex CLI.

    holomem hook user-prompt|session-start --client claude-code|codex   (stdin: the client's hook JSON)
    holomem mcp --client claude-code|codex                              (stdio MCP server)
    holomem install claude-code|codex [--scope user|project] [--dry-run] [--uninstall]
    holomem import claude-memory|codex-memories [--dry-run]
    holomem embed [--seconds N]                                         (search by meaning: vectors now)
    holomem status
"""
from __future__ import annotations

import argparse
import sys

from .memory import CLIENTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="holomem", description="Holographic memory for Claude Code and Codex CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    hook = sub.add_parser("hook", help="run a hook (stdin: the client's JSON)")
    hook.add_argument("event", choices=["user-prompt", "session-start"])
    hook.add_argument("--client", choices=CLIENTS, default="claude-code")
    mcp = sub.add_parser("mcp", help="run the stdio MCP server")
    mcp.add_argument("--client", choices=CLIENTS, default="claude-code")
    inst = sub.add_parser("install", help="add the hooks and the MCP server to a client's settings")
    inst.add_argument("client", choices=CLIENTS)
    inst.add_argument("--scope", choices=["user", "project"], default="user")
    inst.add_argument("--project", default=".", help="the project directory for --scope project")
    inst.add_argument("--dry-run", action="store_true", help="show the change, write nothing")
    inst.add_argument("--uninstall", action="store_true", help="remove what install added")
    inst.add_argument("--claude-cli", help="the claude executable, when it is not on PATH")
    inst.add_argument("--recall", choices=["command", "mcp"], default="command",
                      help="Claude Code: recall by a new process per prompt (command) or by the running MCP server "
                           "(mcp: no Python start per prompt; nothing recalled while the server is not connected)")
    inst.add_argument("--approval", choices=["auto", "writes", "prompt"], default="auto",
                      help="Codex: when its memory tools ask for approval (default: never)")
    imp = sub.add_parser("import", help="copy a client's own memory files into the store, as facts")
    imp.add_argument("source", choices=["claude-memory", "codex-memories"])
    imp.add_argument("--dry-run", action="store_true")
    imp.add_argument("--path", help="read this file or directory instead of the default")
    emb = sub.add_parser("embed", help="make the vectors of search by meaning now (embeddings on)")
    emb.add_argument("--seconds", type=float, default=120.0, help="time to spend at most (default 120)")
    sub.add_parser("status", help="settings, database, facts")
    args = parser.parse_args(argv)

    if args.command not in ("hook", "mcp"):  # those two write bytes; the rest print notes in any language
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):  # not a text stream that can be reconfigured: leave it
                pass
    if args.command == "hook":
        from .hooks import main as hook_main

        return hook_main(args.event, args.client, sys.stdin.buffer.read())
    if args.command == "mcp":
        from .mcp_server import serve

        return serve(args.client)
    if args.command == "install":
        from .install import main as install_main

        return install_main(args)
    if args.command == "import":
        from .importers import main as import_main

        return import_main(args)
    if args.command == "embed":
        from .embed import main as embed_main

        return embed_main(args)
    from .status import main as status_main

    return status_main()


if __name__ == "__main__":
    sys.exit(main())

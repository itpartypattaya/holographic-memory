"""A stdio MCP server with the memory tools (holomem mcp --client claude-code|codex).

Newline-delimited JSON-RPC 2.0 on stdin/stdout, standard library only: initialize, ping, tools/list, tools/call.
Nothing but protocol messages goes to stdout. The store opens on the first tool call and stays open for the session,
so a call costs milliseconds after the first. Which database: the config's scope, with the project taken from
CLAUDE_PROJECT_DIR (set by Claude Code) or the server's working directory.
"""
from __future__ import annotations

import json
import logging
import os
import sys

from . import __version__, config
from .memory import Context, open_memory
from .tools import NAMES, READ_ONLY, SCHEMAS, Tools, dumps

logger = logging.getLogger("holomem")

PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
INSTRUCTIONS = ("Long-term memory kept in a local store (holographic-memory). Facts relevant to a message arrive with "
                "it; memory_search finds more. Store durable facts as short standalone sentences.")


class Server:
    def __init__(self, client: str, cwd: str | None = None):
        self.cfg = config.load()
        self.ctx = Context(client=client, session_id=f"mcp-{os.getpid()}",
                           cwd=cwd or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
        self.tools: Tools | None = None
        self.memory = self.state = None

    def _tools(self) -> Tools | None:
        if self.tools is None:
            self.memory = open_memory(self.cfg, self.ctx)
            if self.memory is None:
                return None
            from .state import State

            self.state = State(self.cfg.state_path)
            self.tools = Tools(self.memory, self.ctx, self.state)
        return self.tools

    def close(self) -> None:
        if self.memory is not None:
            self.memory.close()
        if self.state is not None:
            self.state.close()

    def handle(self, message: dict) -> dict | None:
        """The response to one JSON-RPC message, None for a notification."""
        method, msg_id = message.get("method"), message.get("id")
        if msg_id is None:  # a notification (initialized, cancelled): never answered
            return None
        params = message.get("params") or {}
        if method == "initialize":
            asked = params.get("protocolVersion")
            return self._ok(msg_id, {"protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[1],
                                     "capabilities": {"tools": {"listChanged": False}},
                                     "serverInfo": {"name": "holographic-memory", "version": __version__},
                                     "instructions": INSTRUCTIONS})
        if method == "ping":
            return self._ok(msg_id, {})
        if method == "tools/list":
            return self._ok(msg_id, {"tools": [
                {**schema, "annotations": {"readOnlyHint": schema["name"] in READ_ONLY,
                                           "destructiveHint": schema["name"] == "memory_forget",
                                           "openWorldHint": False}} for schema in SCHEMAS]})
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            if name not in NAMES:
                return self._error(msg_id, -32602, f"Unknown tool: {name}")
            answer, is_error = self._call(name, args)
            return self._ok(msg_id, {"content": [{"type": "text", "text": dumps(answer)}], "isError": is_error})
        return self._error(msg_id, -32601, f"Method not found: {method}")

    def _call(self, name: str, args) -> tuple[dict, bool]:
        if self.cfg.problems:
            return {"error": f"memory is off: {'; '.join(self.cfg.problems)}; fix {self.cfg.home / 'config.toml'}"}, True
        try:
            tools = self._tools()
        except Exception as exc:  # noqa: BLE001 - the store could not open: say so by kind, keep serving
            self.tools = None
            logger.warning("holomem: the store could not open (%s)", type(exc).__name__)
            return {"error": f"the memory store could not be opened ({type(exc).__name__})"}, True
        if tools is None:
            return {"error": "project memory is off here: no project directory"}, True
        return tools.call(name, args)

    @staticmethod
    def _ok(msg_id, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def serve(client: str, stdin=None, stdout=None) -> int:
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(message)s")
    server = Server(client)
    try:
        for raw in stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                response = Server._error(None, -32700, "Parse error")
            else:
                if isinstance(message, list):  # a batch (2025-03-26): answer each request in it
                    replies = [r for r in (server.handle(m) for m in message if isinstance(m, dict)) if r]
                    response = replies or None
                elif isinstance(message, dict):
                    response = server.handle(message)
                else:
                    response = Server._error(None, -32600, "Invalid Request")
            if response is not None:
                stdout.write(json.dumps(response, ensure_ascii=False, default=str).encode("utf-8") + b"\n")
                stdout.flush()
    finally:
        server.close()
    return 0

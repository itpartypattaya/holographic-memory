"""A stdio MCP server with the memory tools (holomem mcp --client claude-code|codex).

Newline-delimited JSON-RPC 2.0 on stdin/stdout, standard library only: initialize, ping, tools/list, tools/call.
Nothing but protocol messages goes to stdout. The store opens on the first tool call and stays open for the session,
so a call costs milliseconds after the first. Which database: the config's scope, with the project taken from
CLAUDE_PROJECT_DIR (set by Claude Code) or the server's working directory.

With ``embeddings`` on, a background thread makes the vectors facts are missing (the plugin's catch-up after a turn):
once the session starts, again after every write, and every few minutes for facts other sessions stored. It holds the
server's lock only while it reads or writes the store, never while a request is out.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading

from . import __version__, config
from .hooks import answer
from .memory import Context, open_memory
from .recall import recall, trivial
from .tools import NAMES, READ_ONLY, SCHEMAS, Tools, dumps

# For a UserPromptSubmit hook of type mcp_tool (Claude Code: input {"prompt": "${prompt}", "session_id":
# "${session_id}", "cwd": "${cwd}"}): its text is the hook's JSON answer, read like a command hook's stdout.
RECALL = {"name": "memory_recall",
          "description": "Used by the memory hook before each message: the stored facts relevant to it, as the hook's "
                         "JSON answer (empty when nothing is relevant). Not needed in conversation: use memory_search.",
          "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"},
                                                           "session_id": {"type": "string"},
                                                           "cwd": {"type": "string"}}, "required": ["prompt"]}}

logger = logging.getLogger("holomem")

PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
WRITES = frozenset({"memory_add", "memory_change", "memory_correct", "memory_revert", "memory_merge"})
_CATCH_UP_SECONDS = 20.0  # one pass of the vectors thread
_CATCH_UP_EVERY = 300.0  # an idle pass: facts another session or a hook stored
INSTRUCTIONS = ("Long-term memory kept in a local store (holographic-memory). Facts relevant to a message arrive with "
                "it; memory_search finds more. Store durable facts as short standalone sentences.")


class Server:
    def __init__(self, client: str, cwd: str | None = None):
        self.cfg = config.load()
        self.ctx = Context(client=client, session_id=f"mcp-{os.getpid()}",
                           cwd=cwd or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
        self.tools: Tools | None = None
        self.memory = self.state = None
        self.lock = threading.Lock()  # the store and the retriever: tool calls and the vectors thread
        self._wake, self._stop = threading.Event(), threading.Event()
        self._vectors: threading.Thread | None = None

    def _tools(self) -> Tools | None:
        if self.tools is None:
            self.memory = open_memory(self.cfg, self.ctx)
            if self.memory is None:
                return None
            from .state import State

            self.state = State(self.cfg.state_path)
            self.tools = Tools(self.memory, self.ctx, self.state)
        return self.tools

    def start_vectors(self) -> None:
        """The vectors thread, when search by meaning is on (decided from the settings: no store is opened for it).

        The store, numpy and the TLS and socket modules are loaded here, on the main thread, before it reads stdin:
        on Windows an extension module (a DLL) loaded by another thread waits for as long as the main thread's
        synchronous read of the stdin pipe lasts — a whole idle session — so the thread made no vector until the
        agent's next tool call."""
        if self.cfg.problems or not self.cfg.embeddings or self._vectors is not None:
            return
        import http.client  # noqa: F401 - loaded now, see above
        import socket  # noqa: F401
        import ssl  # noqa: F401

        try:
            with self.lock:
                self._tools()
        except Exception as exc:  # noqa: BLE001 - the first tool call says why; the thread tries again
            self.tools = None
            logger.warning("holomem: the store could not open (%s)", type(exc).__name__)
        self._vectors = threading.Thread(target=self._vectors_loop, name="holomem-vectors", daemon=True)
        self._vectors.start()

    def _vectors_loop(self) -> None:
        while not self._stop.is_set():
            again = False
            try:
                with self.lock:
                    if self._stop.is_set():  # closed meanwhile: never open the store again
                        return
                    tools = self._tools()
                memory = tools.memory if tools is not None else None
                if memory is None or memory.embedder is None:
                    return
                memory.catch_up(_CATCH_UP_SECONDS, lock=self.lock, stop=self._stop)
                with self.lock:
                    if self._stop.is_set():
                        return
                    again = memory.embedder.usable() and memory.pending_vectors().get("waiting", 0) > 0
            except Exception as exc:  # noqa: BLE001 - named by kind; the next wake tries again
                logger.warning("holomem: making fact vectors stopped (%s)", type(exc).__name__)
            if not again:
                self._wake.wait(_CATCH_UP_EVERY)
            self._wake.clear()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        with self.lock:
            self._close()

    def _close(self) -> None:
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
                                           "openWorldHint": False}} for schema in (*SCHEMAS, RECALL)]})
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            if name == RECALL["name"]:  # never an error: a failing hook must not stand in the prompt's way
                text = self._recall(args if isinstance(args, dict) else {})
                return self._ok(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})
            if name not in NAMES:
                return self._error(msg_id, -32602, f"Unknown tool: {name}")
            with self.lock:
                answer, is_error = self._call(name, args)
            if name in WRITES and not is_error:
                self._wake.set()
            return self._ok(msg_id, {"content": [{"type": "text", "text": dumps(answer)}], "isError": is_error})
        return self._error(msg_id, -32601, f"Method not found: {method}")

    def _recall(self, args: dict) -> str:
        """The UserPromptSubmit hook's answer, computed in this warm process (a hook of type mcp_tool): the same gates
        and recall as ``holomem hook user-prompt``, without starting Python, numpy and the store for each prompt."""
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or trivial(prompt) or self.cfg.problems:
            return ""
        try:
            from .engine.text import is_greeting

            with self.lock:
                if is_greeting(prompt) or self._tools() is None:
                    return ""
            ctx = Context(client=self.ctx.client, session_id=str(args.get("session_id") or ""),
                          cwd=str(args.get("cwd") or self.ctx.cwd))
            with self.lock:
                return answer("user-prompt", recall(self.memory, self.state, ctx, prompt))
        except Exception as exc:  # noqa: BLE001 - logged by kind; the prompt goes on without memory
            logger.warning("holomem: memory_recall failed (%s)", type(exc).__name__)
            return ""

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
    server.start_vectors()
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

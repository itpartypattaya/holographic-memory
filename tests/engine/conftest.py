"""Bootstrap for the engine's own tests (``holomem.engine``).

The engine and these tests were ported from hermes-plugin-holographic v0.25.2 (MIT), now this repo's own: they are
edited here, together. Only the tests of the engine came over: the ones of the agent integration that wrapped it
there test code this repo does not have.
"""
import os
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test gets its own HOME; nothing touches the real home folder or a store kept there."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


# PRAGMA synchronous=OFF on every connection the suite opens: no test checks what survives a power cut. Where the
# linked SQLite has the WAL-reset bug every store runs in DELETE journal mode and each commit waits on two fsyncs; on a
# Windows disk that makes the suite many times slower. The journal mode, locking and every write stay as they are;
# only the wait for the disk goes. Patched at import, not in a fixture, so that stores the module fixtures open get it
# too.
_real_connect = sqlite3.connect


def _connect_without_fsync(*args, **kwargs):
    conn = _real_connect(*args, **kwargs)
    try:
        conn.execute("PRAGMA synchronous=OFF")
    except sqlite3.Error:
        pass  # a file a test made unreadable on purpose fails where that test expects it to, not here
    return conn


sqlite3.connect = _connect_without_fsync


class InProcessServer:
    """holomem's MCP server (holomem.mcp_server.Server, the code `holomem mcp` runs) in this process: tools/call and
    memory_recall go through its own handle(), as the client's JSON-RPC requests do, without a process per test."""

    def __init__(self, server):
        self.server, self._id = server, 0

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        return self.server.handle({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})

    def call(self, name: str, **arguments) -> tuple[dict, bool]:
        import json

        result = self.request("tools/call", {"name": name, "arguments": arguments})["result"]
        return json.loads(result["content"][0]["text"]), result["isError"]

    def recall(self, prompt: str, session_id: str = "s1") -> str:
        """The block memory_recall gives a prompt ("" when nothing), the hook's additionalContext."""
        import json

        reply = self.request("tools/call", {"name": "memory_recall",
                                            "arguments": {"prompt": prompt, "session_id": session_id, "cwd": ""}})
        text = reply["result"]["content"][0]["text"]
        return json.loads(text)["hookSpecificOutput"]["additionalContext"] if text else ""

    @property
    def store(self):
        return self.server.memory.store

    def close(self) -> None:
        self.server.close()


@pytest.fixture
def holomem_server(monkeypatch):
    """Start holomem's server on a home folder with the given config.toml text. Some requirements of the engine's
    tests live in holomem's own layer (its tools, recall, the journal's links in state.db): they are tested there."""
    started = []

    def start(home: Path, settings: str = "", cwd: Path | None = None) -> InProcessServer:
        from holomem.mcp_server import Server

        home.mkdir(parents=True, exist_ok=True)
        (home / "config.toml").write_text(settings, encoding="utf-8")
        monkeypatch.setenv("HOLOMEM_HOME", str(home))
        server = InProcessServer(Server("claude-code", cwd=str(cwd or home)))
        started.append(server)
        return server

    yield start
    for server in started:
        server.close()


def _symlinks_supported(tmp_dir: Path) -> bool:
    target, link = tmp_dir / "target", tmp_dir / "link"
    target.write_text("x", encoding="utf-8")
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        return False
    return True


def pytest_configure(config):
    config.addinivalue_line("markers", "require_symlinks: skip when symbolic links cannot be created here")


def pytest_runtest_setup(item):
    if item.get_closest_marker("require_symlinks"):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            if not _symlinks_supported(Path(tmp)):
                pytest.skip("symbolic links are not supported here (Windows needs developer mode)")

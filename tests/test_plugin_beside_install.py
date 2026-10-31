"""The Claude Code plugin and `holomem install claude-code` on one machine.

Requirement: a person who has both — the installer's hooks and server in Claude Code's settings, and the plugin — gets
one set of recalled facts per message, one session block and one set of memory tools. The plugin (its commands carry
--plugin) stands aside then: its hooks print nothing, its server lists no tools and says why in its instructions. The
installer is seen where Claude Code reads it: the user's settings.json (in CLAUDE_CONFIG_DIR when set) or, after an
install with --scope project, the project's .claude/settings.local.json, from the project's folder or one below it.
Without the installer's hooks (none, another tool's hooks only, an unreadable settings file) the plugin works in full;
the installer's own hooks never change because the plugin is there.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _launch import LAUNCH as _LAUNCH
from holomem import install
from test_hooks import context, prompt, seed
from test_mcp_server import Client

ROOT = Path(__file__).resolve().parent.parent
FACT = "The billing service deploys to the staging server with systemd"
ASK = "How does the billing service deploy to staging?"


@pytest.fixture
def home(tmp_path) -> Path:
    seed(tmp_path / "holo", [(FACT, [])])
    return tmp_path / "holo"


@pytest.fixture
def claude_dir(tmp_path) -> Path:
    folder = tmp_path / "claude-config"
    folder.mkdir()
    return folder


def _installer_settings(path: Path, recall: str = "command") -> None:
    """The settings file exactly as `holomem install claude-code` writes it, beside what was there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    before = {"model": "opus", "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "notify-send done"}]}]}}
    path.write_text(json.dumps(install.claude_settings(before, recall=recall), indent=2), encoding="utf-8")


def run_hook(home: Path, claude_dir: Path, event: str, payload: dict, *, plugin: bool) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOLOMEM_HOME": str(home), "CLAUDE_CONFIG_DIR": str(claude_dir)}
    argv = ["hook", event, "--client", "claude-code", *(["--plugin"] if plugin else [])]
    return subprocess.run([sys.executable, "-I", "-c", _LAUNCH, str(ROOT), *argv],
                          input=json.dumps(payload).encode("utf-8"), capture_output=True, env=env, timeout=60)


def start(cwd: str = "") -> dict:
    return {"session_id": "s1", "cwd": cwd, "hook_event_name": "SessionStart", "source": "startup"}


def test_the_plugin_alone_recalls_and_starts_the_session(home, claude_dir):
    assert FACT in context(run_hook(home, claude_dir, "user-prompt", prompt(ASK), plugin=True))
    block = context(run_hook(home, claude_dir, "session-start", start(), plugin=True), "SessionStart")
    assert block.startswith("## Holographic Memory\nActive")


@pytest.mark.parametrize("recall", ["command", "mcp"])
def test_the_plugins_hooks_are_silent_beside_a_user_install(home, claude_dir, recall):
    _installer_settings(claude_dir / "settings.json", recall)
    for event, payload in (("user-prompt", prompt(ASK)), ("session-start", start())):
        proc = run_hook(home, claude_dir, event, payload, plugin=True)
        assert proc.returncode == 0 and proc.stdout == b"", (event, proc.stdout, proc.stderr[-800:])


def test_the_installers_hooks_are_unchanged_beside_the_plugin(home, claude_dir):
    _installer_settings(claude_dir / "settings.json")
    assert FACT in context(run_hook(home, claude_dir, "user-prompt", prompt(ASK), plugin=False))
    block = context(run_hook(home, claude_dir, "session-start", start(), plugin=False), "SessionStart")
    assert block.startswith("## Holographic Memory\nActive")


def test_a_project_install_is_seen_from_its_folder_and_below(home, claude_dir, tmp_path):
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / "src").mkdir()
    _installer_settings(project / ".claude" / "settings.local.json")
    for cwd in (project, project / "src"):
        proc = run_hook(home, claude_dir, "user-prompt", prompt(ASK, cwd=str(cwd)), plugin=True)
        assert proc.returncode == 0 and proc.stdout == b"", cwd
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    assert FACT in context(run_hook(home, claude_dir, "user-prompt", prompt(ASK, cwd=str(elsewhere)), plugin=True))


@pytest.mark.parametrize("text", [
    json.dumps({"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "other-tool recall"}]}]}}),
    "{ not json",
    "",
])
def test_settings_without_the_installers_hooks_leave_the_plugin_on(home, claude_dir, text):
    (claude_dir / "settings.json").write_text(text, encoding="utf-8")
    assert FACT in context(run_hook(home, claude_dir, "user-prompt", prompt(ASK), plugin=True))


def _server(home: Path, claude_dir: Path) -> Client:
    return Client(home, extra=("--plugin",), env={"CLAUDE_CONFIG_DIR": str(claude_dir)})


def test_the_plugins_server_lists_no_tools_beside_a_user_install(home, claude_dir):
    _installer_settings(claude_dir / "settings.json")
    client = _server(home, claude_dir)
    try:
        hello = client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                              "clientInfo": {"name": "test", "version": "0"}})["result"]
        assert "holomem install claude-code" in hello["instructions"]
        client.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert client.request("tools/list")["result"]["tools"] == []
        answer, is_error = client.call("memory_search", query="billing")
        assert is_error and "holomem install claude-code" in answer["error"]
    finally:
        client.close()


def test_the_plugins_server_serves_alone(home, claude_dir):
    client = _server(home, claude_dir).start()
    try:
        names = {t["name"] for t in client.request("tools/list")["result"]["tools"]}
        assert {"memory_add", "memory_search", "memory_recall"} <= names
        answer, is_error = client.call("memory_search", query="billing staging")
        assert not is_error and FACT in json.dumps(answer)
    finally:
        client.close()

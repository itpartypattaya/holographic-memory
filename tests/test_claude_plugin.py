"""The Claude Code plugin stays in step with the package.

Requirement: the marketplace points at the plugin folder; the plugin's version is the package's; every hook command
and the MCP server's arguments are ones the holomem CLI accepts (a renamed subcommand or flag would make the plugin's
hooks fail silently on every prompt); the server name has no "__" (Claude Code's permission rules split on it).
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

import holomem
from holomem.__main__ import main

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "claude-plugin"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_marketplace_points_at_the_plugin():
    market = _json(ROOT / ".claude-plugin" / "marketplace.json")
    (entry,) = market["plugins"]
    assert (ROOT / entry["source"]).resolve() == PLUGIN.resolve()
    assert entry["name"] == _json(PLUGIN / ".claude-plugin" / "plugin.json")["name"]


def test_plugin_version_is_the_package_version():
    assert _json(PLUGIN / ".claude-plugin" / "plugin.json")["version"] == holomem.__version__


def _parses(argv: list[str], monkeypatch) -> None:
    """argv is accepted by the CLI parser: the command is replaced by a stub that records it ran."""
    import holomem.hooks
    import holomem.mcp_server

    ran = []
    monkeypatch.setattr(holomem.hooks, "main", lambda event, client, stdin: ran.append((event, client)) or 0)
    monkeypatch.setattr(holomem.mcp_server, "serve", lambda client: ran.append(("mcp", client)) or 0)
    monkeypatch.setattr("sys.stdin", type("S", (), {"buffer": type("B", (), {"read": lambda self: b"{}"})()})())
    assert main(argv) == 0 and ran


@pytest.mark.parametrize("event", ["UserPromptSubmit", "SessionStart"])
def test_hook_commands_are_valid_cli_calls(event, monkeypatch):
    groups = _json(PLUGIN / "hooks" / "hooks.json")["hooks"][event]
    (command,) = [h["command"] for g in groups for h in g["hooks"]]
    program, *argv = shlex.split(command)
    assert program == "holomem"
    _parses(argv, monkeypatch)


def test_mcp_server_arguments_are_valid(monkeypatch):
    servers = _json(PLUGIN / ".mcp.json")["mcpServers"]
    (name,) = servers
    assert "__" not in name
    assert servers[name]["command"] == "holomem"
    _parses(servers[name]["args"], monkeypatch)

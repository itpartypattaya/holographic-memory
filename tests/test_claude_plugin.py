"""The Claude Code plugin stays in step with the package.

Requirement: the plugin folder is the repository root, so a person who installs the plugin gets the engine with it: every
hook and the MCP server run this folder's own package through `uv run --locked --project ${CLAUDE_PLUGIN_ROOT}`, never
a program they had to install from elsewhere (the Claude plugin directory installs only the plugin folder). The
marketplace points at that folder; the plugin's version is the package's; uv.lock holds the package's version and
dependencies (with --locked, a lock left behind a release makes uv refuse to run, and every hook fails); every hook
command and the MCP server's arguments are ones the holomem CLI accepts (a renamed subcommand or flag would make the
plugin's hooks fail silently on every prompt); the server name has no "__" (Claude Code's permission rules split on it).
"""
from __future__ import annotations

import json
import shlex
import tomllib
from pathlib import Path

import pytest

import holomem
from holomem.__main__ import main

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT
MANIFEST = PLUGIN / ".claude-plugin" / "plugin.json"
LAUNCH = ["uv", "run", "--locked", "--quiet", "--project", "${CLAUDE_PLUGIN_ROOT}", "holomem"]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_marketplace_points_at_the_plugin():
    market = _json(ROOT / ".claude-plugin" / "marketplace.json")
    (entry,) = market["plugins"]
    assert (ROOT / entry["source"]).resolve() == PLUGIN.resolve()
    assert entry["name"] == _json(MANIFEST)["name"]


def test_plugin_version_is_the_package_version():
    assert _json(MANIFEST)["version"] == holomem.__version__


def test_lock_holds_the_package_as_released():
    lock = tomllib.loads((PLUGIN / "uv.lock").read_text(encoding="utf-8"))
    project = tomllib.loads((PLUGIN / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    (own,) = [p for p in lock["package"] if p["name"] == project["name"]]
    assert own["version"] == holomem.__version__
    assert own["source"] == {"editable": "."}
    locked = {(d["name"], d.get("specifier", "")) for d in own["metadata"]["requires-dist"]}
    declared = set()
    for requirement in project["dependencies"]:
        name, _, specifier = requirement.partition(">")
        declared.add((name.strip(), ">" + specifier.strip() if specifier else ""))
    assert locked == declared


def _parses(argv: list[str], monkeypatch) -> None:
    """argv is accepted by the CLI parser: the command is replaced by a stub that records it ran."""
    import holomem.hooks
    import holomem.mcp_server

    ran = []
    monkeypatch.setattr(holomem.hooks, "main", lambda event, client, stdin: ran.append((event, client)) or 0)
    monkeypatch.setattr(holomem.mcp_server, "serve", lambda client: ran.append(("mcp", client)) or 0)
    monkeypatch.setattr("sys.stdin", type("S", (), {"buffer": type("B", (), {"read": lambda self: b"{}"})()})())
    assert main(argv) == 0 and ran


def _own_package(argv: list[str]) -> list[str]:
    """The arguments after the launcher, which must run this folder's package and nothing from the PATH."""
    assert argv[:len(LAUNCH)] == LAUNCH
    return argv[len(LAUNCH):]


@pytest.mark.parametrize("event", ["UserPromptSubmit", "SessionStart"])
def test_hook_commands_run_the_plugins_own_package(event, monkeypatch):
    groups = _json(PLUGIN / "hooks" / "hooks.json")["hooks"][event]
    (command,) = [h["command"] for g in groups for h in g["hooks"]]
    _parses(_own_package(shlex.split(command)), monkeypatch)


def test_mcp_server_runs_the_plugins_own_package(monkeypatch):
    servers = _json(MANIFEST)["mcpServers"]
    (name,) = servers
    assert "__" not in name
    _parses(_own_package([servers[name]["command"], *servers[name]["args"]]), monkeypatch)

"""The Claude Code plugin stays in step with the package.

Requirement: the plugin folder is the repository root, so a person who installs the plugin gets the engine with it: every
hook and the MCP server run this folder's own package through `uv run --locked --project ${CLAUDE_PLUGIN_ROOT}` and the
plugin's own script (scripts/holomem_plugin.py, a file the Claude plugin directory reads), never a program they had to install
from elsewhere (the directory installs only the plugin folder). The
marketplace points at that folder; the plugin's version is the package's; uv.lock holds the package's version and
dependencies (with --locked, a lock left behind a release makes uv refuse to run, and every hook fails); every hook
command and the MCP server's arguments are ones the holomem CLI accepts (a renamed subcommand or flag would make the
plugin's hooks fail silently on every prompt) and say they run as the plugin (--plugin: test_plugin_beside_install.py);
the server name has no "__" (Claude Code's permission rules split on it). The embeddings key reaches the plugin only
through its sensitive userConfig option, handed to the MCP server under the name the engine reads (memory.PLUGIN_KEY;
hooks get it from Claude Code under that name). The manifest holds only keys Claude Code reads: the directory warned of
its listing-only ones (icon, documentationUrl, supportUrl, privacyPolicyUrl), and a listing keeps what it imported.
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
LAUNCH = ["uv", "run", "--locked", "--quiet", "--project", "${CLAUDE_PLUGIN_ROOT}", "python",
          "${CLAUDE_PLUGIN_ROOT}/scripts/holomem_plugin.py"]
# the keys of plugin.json that Claude Code reads (plugins reference, "Fields")
READ_KEYS = {"$schema", "name", "displayName", "version", "description", "author", "homepage", "repository", "license",
             "keywords", "metadata", "defaultEnabled", "dependencies", "settings", "userConfig", "channels", "skills",
             "commands", "agents", "hooks", "mcpServers", "lspServers", "outputStyles", "workflows", "experimental"}


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
    monkeypatch.setattr(holomem.hooks, "main",
                        lambda event, client, stdin, plugin: ran.append((event, client, plugin)) or 0)
    monkeypatch.setattr(holomem.mcp_server, "serve", lambda client, plugin: ran.append(("mcp", client, plugin)) or 0)
    monkeypatch.setattr("sys.stdin", type("S", (), {"buffer": type("B", (), {"read": lambda self: b"{}"})()})())
    assert main(argv) == 0
    (call,) = ran
    assert call[-1] is True  # run as the plugin: it stands aside beside `holomem install claude-code`


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


def test_the_script_runs_this_folders_package():
    import subprocess
    import sys

    script = PLUGIN / "scripts" / "holomem_plugin.py"
    done = subprocess.run([sys.executable, str(script), "hook", "--help"], capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONPATH": str(ROOT)}, timeout=60)
    assert done.returncode == 0, done.stderr[-800:]
    assert "user-prompt" in done.stdout and "--plugin" in done.stdout


def test_the_embeddings_key_comes_only_from_the_plugins_option():
    from holomem.memory import PLUGIN_KEY

    manifest = _json(MANIFEST)
    options = manifest["userConfig"]
    assert options["embedding_api_key"]["sensitive"] is True
    assert not options["embedding_api_key"].get("required")  # search by meaning is off until turned on
    (server,) = manifest["mcpServers"].values()
    assert server["env"] == {PLUGIN_KEY: "${user_config.embedding_api_key}"}
    assert PLUGIN_KEY == "CLAUDE_PLUGIN_OPTION_" + "embedding_api_key".upper()  # the name hooks get it under


def test_the_manifest_holds_only_keys_claude_code_reads():
    assert set(_json(MANIFEST)) <= READ_KEYS

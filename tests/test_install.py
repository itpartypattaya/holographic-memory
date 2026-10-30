"""The installer edits the clients' settings, and nothing else in them.

Requirements:
- Claude Code: the two hooks and the permission rule are added; everything already there (other hooks of the same
  event, other permissions, unrelated keys) stays; installing twice leaves one copy; uninstall removes exactly what
  install added; the MCP server goes in through the claude CLI (remove, then add), never by writing ~/.claude.json;
- Codex: one marked block at the end of config.toml; the file still parses, existing tables (other MCP servers, hook
  trust state) are untouched; installing twice leaves one block; uninstall restores the original text;
- --dry-run shows the diff and writes nothing; a real write leaves a backup of the old file; a settings file that is
  not valid JSON is refused, not overwritten;
- the commands written run this Python isolated (-I) and call the hooks for the right client.
"""
from __future__ import annotations

import io
import json
import sys
import tomllib
from pathlib import Path

from holomem import install

CODEX_BEFORE = '''model = "gpt"
notify = ["node", "notify.js"]

[mcp_servers.other]
command = "node"
args = ["server.js"]

[hooks.state."plugin:hooks/hooks.json:stop:0:0"]
trusted_hash = "sha256:abc"

[memories]
use_memories = true
'''

CLAUDE_BEFORE = {
    "model": "claude-opus-5-5",
    "permissions": {"allow": ["Bash(gh pr merge:*)"]},
    "hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "echo other"}]}]},
}


def _claude_cli(tmp_path: Path) -> tuple[str, Path]:
    """A stand-in for the claude executable that records its arguments."""
    log = tmp_path / "claude-calls.jsonl"
    script = tmp_path / "fake_claude.py"
    script.write_text("import json, sys\nwith open(sys.argv[1], 'a', encoding='utf-8') as f:\n"
                      "    f.write(json.dumps(sys.argv[2:]) + '\\n')\n", encoding="utf-8")
    if sys.platform == "win32":
        launcher = tmp_path / "claude.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" "{log}" %*\n', encoding="utf-8")
    else:
        launcher = tmp_path / "claude"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "{log}" "$@"\n', encoding="utf-8")
        launcher.chmod(0o755)
    return str(launcher), log


def _run_claude(tmp_path, project, **kw) -> tuple[int, str]:
    out = io.StringIO()
    rc = install.install_claude("project", project, out=out, **kw)
    return rc, out.getvalue()


def test_claude_install_adds_ours_and_keeps_the_rest(tmp_path):
    project = tmp_path / "p"
    path = project / ".claude" / "settings.local.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(CLAUDE_BEFORE), encoding="utf-8")
    cli, log = _claude_cli(tmp_path)
    rc, _ = _run_claude(tmp_path, project, remove=False, dry_run=False, cli=cli)
    assert rc == 0
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["model"] == "claude-opus-5-5"
    assert after["permissions"]["allow"] == ["Bash(gh pr merge:*)", "mcp__holographic__*"]
    prompt_cmds = [h["command"] for g in after["hooks"]["UserPromptSubmit"] for h in g["hooks"]]
    assert prompt_cmds[0] == "echo other" and len(prompt_cmds) == 2
    assert prompt_cmds[1].endswith("-I -m holomem hook user-prompt --client claude-code")
    assert Path(sys.executable).as_posix() in prompt_cmds[1]
    start_cmds = [h["command"] for g in after["hooks"]["SessionStart"] for h in g["hooks"]]
    assert start_cmds == [prompt_cmds[1].replace("user-prompt", "session-start")]
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert calls[0][:4] == ["mcp", "remove", "--scope", "local"]
    assert calls[1][:7] == ["mcp", "add", "--transport", "stdio", "--scope", "local", "holographic"]
    assert calls[1][-6:] == ["-I", "-m", "holomem", "mcp", "--client", "claude-code"]
    assert list(path.parent.glob("settings.local.json.bak-*"))


def test_claude_install_twice_then_uninstall_restores(tmp_path):
    project = tmp_path / "p"
    path = project / ".claude" / "settings.local.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(CLAUDE_BEFORE), encoding="utf-8")
    cli, _ = _claude_cli(tmp_path)
    _run_claude(tmp_path, project, remove=False, dry_run=False, cli=cli)
    once = path.read_text(encoding="utf-8")
    _run_claude(tmp_path, project, remove=False, dry_run=False, cli=cli)
    assert path.read_text(encoding="utf-8") == once
    _run_claude(tmp_path, project, remove=True, dry_run=False, cli=cli)
    assert json.loads(path.read_text(encoding="utf-8")) == CLAUDE_BEFORE


def test_claude_dry_run_writes_nothing(tmp_path):
    project = tmp_path / "p"
    cli, log = _claude_cli(tmp_path)
    rc, shown = _run_claude(tmp_path, project, remove=False, dry_run=True, cli=cli)
    assert rc == 0 and "+++" in shown and "would run:" in shown
    assert not (project / ".claude").exists() and not log.exists()


def test_claude_refuses_a_broken_settings_file(tmp_path):
    project = tmp_path / "p"
    path = project / ".claude" / "settings.local.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"model": "x",', encoding="utf-8")
    cli, log = _claude_cli(tmp_path)
    rc, shown = _run_claude(tmp_path, project, remove=False, dry_run=False, cli=cli)
    assert rc == 1 and "not valid JSON" in shown
    assert path.read_text(encoding="utf-8") == '{"model": "x",' and not log.exists()


def _run_codex(project, **kw) -> tuple[int, str]:
    out = io.StringIO()
    rc = install.install_codex("project", project, out=out, **kw)
    return rc, out.getvalue()


def test_codex_install_appends_one_block_and_keeps_the_rest(tmp_path):
    project = tmp_path / "p"
    path = project / ".codex" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text(CODEX_BEFORE, encoding="utf-8")
    rc, shown = _run_codex(project, remove=False, dry_run=False)
    assert rc == 0 and "/hooks" in shown
    text = path.read_text(encoding="utf-8")
    assert text.startswith(CODEX_BEFORE.rstrip("\n"))
    data = tomllib.loads(text)
    assert data["mcp_servers"]["other"]["command"] == "node" and data["memories"]["use_memories"] is True
    assert "plugin:hooks/hooks.json:stop:0:0" in data["hooks"]["state"]
    server = data["mcp_servers"]["holographic"]
    assert server["command"] == Path(sys.executable).as_posix()
    assert server["args"] == ["-I", "-m", "holomem", "mcp", "--client", "codex"]
    assert server["default_tools_approval_mode"] == "auto"
    hook = data["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    assert hook["command"].endswith("hook user-prompt --client codex") and hook["command_windows"] == hook["command"]
    assert data["hooks"]["SessionStart"][0]["hooks"][0]["command"].endswith("hook session-start --client codex")
    _run_codex(project, remove=False, dry_run=False)
    assert path.read_text(encoding="utf-8").count(install.MARK_BEGIN) == 1
    _run_codex(project, remove=True, dry_run=False)
    assert path.read_text(encoding="utf-8") == CODEX_BEFORE


def test_codex_dry_run_writes_nothing(tmp_path):
    project = tmp_path / "p"
    rc, shown = _run_codex(project, remove=False, dry_run=True)
    assert rc == 0 and "[mcp_servers.holographic]" in shown
    assert not (project / ".codex").exists()


def test_codex_refuses_a_result_that_would_not_parse(tmp_path):
    project = tmp_path / "p"
    path = project / ".codex" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text("[mcp_servers.holographic]\ncommand = \"x\"\n", encoding="utf-8")  # a hand-made entry: a duplicate
    rc, shown = _run_codex(project, remove=False, dry_run=False)
    assert rc == 1 and "would not parse" in shown
    assert path.read_text(encoding="utf-8") == "[mcp_servers.holographic]\ncommand = \"x\"\n"


def test_claude_recall_mcp_writes_an_mcp_tool_hook_and_switches_cleanly(tmp_path):
    """--recall mcp: UserPromptSubmit calls the running server's memory_recall with the prompt; SessionStart stays a
    command hook (mcp_tool hooks do not run at launch). Switching modes leaves one hook; uninstall removes it."""
    project = tmp_path / "p"
    path = project / ".claude" / "settings.local.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(CLAUDE_BEFORE), encoding="utf-8")
    cli, _ = _claude_cli(tmp_path)
    _run_claude(tmp_path, project, remove=False, dry_run=False, cli=cli)
    out = io.StringIO()
    install.install_claude("project", project, remove=False, dry_run=False, cli=cli, out=out, recall="mcp")
    after = json.loads(path.read_text(encoding="utf-8"))
    prompt_hooks = [h for g in after["hooks"]["UserPromptSubmit"] for h in g["hooks"]]
    assert prompt_hooks[0]["command"] == "echo other" and len(prompt_hooks) == 2
    assert prompt_hooks[1] == {"type": "mcp_tool", "server": "holographic", "tool": "memory_recall",
                               "input": {"prompt": "${prompt}", "session_id": "${session_id}", "cwd": "${cwd}"},
                               "timeout": 15}
    assert [h["type"] for g in after["hooks"]["SessionStart"] for h in g["hooks"]] == ["command"]
    _run_claude(tmp_path, project, remove=True, dry_run=False, cli=cli)
    assert json.loads(path.read_text(encoding="utf-8")) == CLAUDE_BEFORE

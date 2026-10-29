"""holomem install claude-code|codex [--scope user|project] [--dry-run] [--uninstall].

What it writes, and only that:

Claude Code
- hooks UserPromptSubmit and SessionStart in ``~/.claude/settings.json`` (user) or ``<project>/.claude/settings.local.json``
  (project), and the permission rule ``mcp__holographic__*`` so the memory tools run without a prompt each;
- the MCP server through Claude Code's own CLI: ``claude mcp add --scope user|local holographic -- <python> -I -m holomem
  mcp --client claude-code`` (``~/.claude.json`` is written by every running session; the CLI is the safe way in).

Codex CLI
- one marked block at the end of ``~/.codex/config.toml`` (user) or ``<project>/.codex/config.toml``:
  ``[mcp_servers.holographic]`` and ``[[hooks.UserPromptSubmit]]`` / ``[[hooks.SessionStart]]``. Codex runs a hook only
  after it is trusted: open ``/hooks`` in Codex once and trust the two.

Every command names this Python by its full path, ``-I`` (isolated: nothing from the working directory is imported).
A file is backed up next to itself before it is changed, the new content must parse, and ``--dry-run`` shows the diff
and writes nothing. Install again replaces what an earlier install added; ``--uninstall`` removes exactly that.
"""
from __future__ import annotations

import difflib
import json
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

SERVER = "holographic"
PERMISSION = f"mcp__{SERVER}__*"
MARK_BEGIN = "# >>> holographic-memory (holomem install codex) >>>"
MARK_END = "# <<< holographic-memory <<<"
_HOOK_MARK = "-m holomem hook"
_HOOK_TIMEOUT = 15  # seconds; the hook gives up on its own at hook_deadline_seconds (8)


def python_path() -> str:
    return Path(sys.executable).as_posix()


def command(*args: str) -> str:
    """One command line for both clients' shells: the interpreter quoted, forward slashes."""
    return " ".join([f'"{python_path()}"', "-I", "-m", "holomem", *args])


# -- Claude Code ---------------------------------------------------------------------------------------------------


def claude_settings_path(scope: str, project: Path) -> Path:
    return Path.home() / ".claude" / "settings.json" if scope == "user" else project / ".claude" / "settings.local.json"


def _without_ours(groups: list) -> list:
    kept = []
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, dict):
            kept.append(group)
            continue
        hooks = [h for h in group.get("hooks", []) if not (isinstance(h, dict)
                                                         and _HOOK_MARK in str(h.get("command", "")))]
        if hooks:
            kept.append({**group, "hooks": hooks})
        elif not group.get("hooks"):
            kept.append(group)
    return kept


def claude_settings(settings: dict, *, remove: bool = False) -> dict:
    """*settings* with our hooks and permission rule (or without them)."""
    out = json.loads(json.dumps(settings))  # a deep copy
    hooks = out.get("hooks") if isinstance(out.get("hooks"), dict) else {}
    for event, name in (("UserPromptSubmit", "user-prompt"), ("SessionStart", "session-start")):
        groups = _without_ours(hooks.get(event, []))
        if not remove:
            groups.append({"hooks": [{"type": "command", "command": command("hook", name, "--client", "claude-code"),
                                      "timeout": _HOOK_TIMEOUT}]})
        if groups:
            hooks[event] = groups
        else:
            hooks.pop(event, None)
    if hooks:
        out["hooks"] = hooks
    else:
        out.pop("hooks", None)
    permissions = out.get("permissions") if isinstance(out.get("permissions"), dict) else {}
    allow = [rule for rule in permissions.get("allow", []) if rule != PERMISSION]
    if not remove:
        allow.append(PERMISSION)
    if allow:
        permissions["allow"] = allow
    else:
        permissions.pop("allow", None)
    if permissions:
        out["permissions"] = permissions
    else:
        out.pop("permissions", None)
    return out


def claude_mcp_commands(scope: str, cli: str, *, remove: bool = False) -> list[list[str]]:
    cli_scope = "user" if scope == "user" else "local"
    drop = [cli, "mcp", "remove", "--scope", cli_scope, SERVER]
    if remove:
        return [drop]
    return [drop, [cli, "mcp", "add", "--transport", "stdio", "--scope", cli_scope, SERVER, "--",
                   python_path(), "-I", "-m", "holomem", "mcp", "--client", "claude-code"]]


# -- Codex ---------------------------------------------------------------------------------------------------------


def codex_config_path(scope: str, project: Path) -> Path:
    return Path.home() / ".codex" / "config.toml" if scope == "user" else project / ".codex" / "config.toml"


def _toml_str(value: str) -> str:
    return json.dumps(value)  # a TOML basic string is a JSON string for everything we write


def codex_block(approval: str = "auto") -> str:
    lines = [MARK_BEGIN, f"[mcp_servers.{SERVER}]", f"command = {_toml_str(python_path())}",
             "args = " + json.dumps(["-I", "-m", "holomem", "mcp", "--client", "codex"]),
             "startup_timeout_sec = 20", f"default_tools_approval_mode = {_toml_str(approval)}", ""]
    for event, name in (("UserPromptSubmit", "user-prompt"), ("SessionStart", "session-start")):
        cmd = command("hook", name, "--client", "codex")
        lines += [f"[[hooks.{event}]]", f"[[hooks.{event}.hooks]]", 'type = "command"', f"command = {_toml_str(cmd)}",
                  f"command_windows = {_toml_str(cmd)}", f"timeout = {_HOOK_TIMEOUT}",
                  'statusMessage = "holographic memory"', ""]
    lines.append(MARK_END)
    return "\n".join(lines) + "\n"


def codex_text(current: str, *, remove: bool = False, approval: str = "auto") -> str:
    """*current* with our block (replacing an earlier one) or without it."""
    text = current
    start = text.find(MARK_BEGIN)
    if start != -1:
        end = text.find(MARK_END, start)
        end = len(text) if end == -1 else end + len(MARK_END)
        text = (text[:start].rstrip("\n") + "\n" + text[end:].lstrip("\n")) if text[:start].strip() else text[end:].lstrip("\n")
    if not remove:
        text = (text.rstrip("\n") + "\n\n" if text.strip() else "") + codex_block(approval)
    return text


# -- writing -------------------------------------------------------------------------------------------------------


def _diff(path: Path, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), str(path), str(path)))


def _write(path: Path, old: str, new: str, dry_run: bool, out) -> bool:
    diff = _diff(path, old, new)
    if not diff:
        print(f"{path}: no change", file=out)
        return False
    print(diff, file=out)
    if dry_run:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
        print(f"backup: {backup}", file=out)
    path.write_text(new, encoding="utf-8", newline="\n")
    return True


def install_claude(scope: str, project: Path, *, remove: bool, dry_run: bool, cli: str | None, out=sys.stdout) -> int:
    path = claude_settings_path(scope, project)
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        settings = json.loads(old) if old.strip() else {}
    except ValueError:
        print(f"{path} is not valid JSON; fix it first, nothing was changed", file=out)
        return 1
    if not isinstance(settings, dict):
        print(f"{path} is not a JSON object; nothing was changed", file=out)
        return 1
    new = json.dumps(claude_settings(settings, remove=remove), indent=2, ensure_ascii=False) + "\n"
    if new.strip() == "{}" and not old.strip():
        new = old
    _write(path, old, new, dry_run, out)
    cli = cli or shutil.which("claude")
    commands = claude_mcp_commands(scope, cli or "claude", remove=remove)
    for cmd in commands:
        shown = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        if dry_run or not cli:
            print(("would run: " if cli else "run this yourself (claude CLI not found): ") + shown, file=out)
            continue
        result = subprocess.run(cmd, cwd=str(project), capture_output=True, text=True, timeout=60)
        if result.returncode != 0 and "remove" not in cmd:  # removing what is not there is fine
            print(f"failed ({result.returncode}): {shown}\n{result.stderr.strip()}", file=out)
            return 1
        print(f"ran: {shown}", file=out)
    return 0


def install_codex(scope: str, project: Path, *, remove: bool, dry_run: bool, approval: str = "auto",
                  out=sys.stdout) -> int:
    path = codex_config_path(scope, project)
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    new = codex_text(old, remove=remove, approval=approval)
    try:
        tomllib.loads(new)
    except tomllib.TOMLDecodeError as exc:
        print(f"{path}: the result would not parse ({type(exc).__name__}); nothing was changed", file=out)
        return 1
    written = _write(path, old, new, dry_run, out)
    if written and not remove:
        print("Codex runs a hook only once it is trusted: open /hooks in Codex and trust the two holographic hooks.",
              file=out)
    return 0


def main(args) -> int:
    project = Path(args.project).resolve()
    if args.client == "claude-code":
        return install_claude(args.scope, project, remove=args.uninstall, dry_run=args.dry_run,
                              cli=getattr(args, "claude_cli", None))
    return install_codex(args.scope, project, remove=args.uninstall, dry_run=args.dry_run,
                         approval=getattr(args, "approval", "auto"))

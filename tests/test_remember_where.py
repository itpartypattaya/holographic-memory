"""Where "remember this" goes when Claude Code's own auto memory is on too (decided 2026-10-31).

Claude Code's auto memory is on by default, and on "remember that…" Claude saves there
(code.claude.com/docs/en/memory). With holographic memory installed the same request could go to either store, or to both, and the two drift apart.
The requirement: in Claude Code with auto memory on, the session block says the person is asked where it goes (this
memory, Claude Code's auto memory, or both) unless they said; with auto memory off (CLAUDE_CODE_DISABLE_AUTO_MEMORY,
or autoMemoryEnabled: false in the local, project or user settings, the nearest one deciding) and in Codex, whose
memories are generated from past chats rather than on request, nothing is added.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _launch import LAUNCH

ROOT = Path(__file__).resolve().parent.parent
ASKED = "asked first where it goes"


@pytest.fixture
def user(tmp_path) -> Path:
    path = tmp_path / "user"
    (path / ".claude").mkdir(parents=True)
    return path


def session_block(tmp_path: Path, user: Path, *, client: str = "claude-code", cwd: str = "", **env: str) -> str:
    base = {k: v for k, v in os.environ.items()
            if k not in ("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "CLAUDE_CONFIG_DIR")}
    base.update({"HOLOMEM_HOME": str(tmp_path / "holo"), "HOME": str(user), "USERPROFILE": str(user), **env})
    proc = subprocess.run([sys.executable, "-I", "-c", LAUNCH, str(ROOT), "hook", "session-start", "--client", client],
                          input=json.dumps({"session_id": "s1", "cwd": cwd, "source": "startup"}).encode(),
                          capture_output=True, env=base, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-1500:]
    return json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]


def settings(folder: Path, name: str, value) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps({"autoMemoryEnabled": value}), encoding="utf-8")


def test_with_auto_memory_on_by_default_the_person_is_asked_where_it_goes(tmp_path, user):
    block = session_block(tmp_path, user)
    assert ASKED in block and "auto memory" in block, block


def test_with_auto_memory_off_in_the_user_settings_nothing_is_asked(tmp_path, user):
    settings(user / ".claude", "settings.json", False)
    assert ASKED not in session_block(tmp_path, user)


def test_with_auto_memory_off_by_the_environment_nothing_is_asked(tmp_path, user):
    assert ASKED not in session_block(tmp_path, user, CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")


def test_the_settings_nearest_the_project_decide(tmp_path, user):
    project = tmp_path / "work"
    (project / ".git").mkdir(parents=True)
    settings(user / ".claude", "settings.json", False)
    settings(project / ".claude", "settings.json", True)  # the project turns it on again
    assert ASKED in session_block(tmp_path, user, cwd=str(project / "src"))
    settings(project / ".claude", "settings.local.json", False)  # and the local file off
    assert ASKED not in session_block(tmp_path, user, cwd=str(project / "src"))


def test_claude_config_dir_holds_the_user_settings(tmp_path, user):
    config = tmp_path / "other-config"
    settings(config, "settings.json", False)
    assert ASKED not in session_block(tmp_path, user, CLAUDE_CONFIG_DIR=str(config))


def test_codex_is_not_asked(tmp_path, user):
    assert ASKED not in session_block(tmp_path, user, client="codex")

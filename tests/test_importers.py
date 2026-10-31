"""Importing the clients' own memory notes.

Requirements:
- Claude Code auto memory: one fact per topic file, the note before its **Why:** / **How to apply:** lines, category
  from the file's type; MEMORY.md (the index) is not a fact; a file with an empty body gives its description; works
  for the projects folder, one project, or one memory folder;
- Codex memory_summary.md: one fact per bullet, the trailing [tag] dropped, preferences headings give user_pref;
- every candidate passes the write screen (a credential or an instruction is not imported), a too long note is
  skipped, a text already stored or repeated is not stored twice; imported facts start at trust 0.4 with an import
  source; --dry-run stores nothing; running the import twice adds nothing the second time.
"""
from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest

from holomem import importers

TOPIC = """---
name: deploy-staging
description: How billing reaches staging
metadata:
  type: project
---

The billing service deploys to staging with systemd, never by hand.
Each deploy runs the smoke tests first.

**Why:** a manual deploy broke staging twice.
**How to apply:** use the deploy script.
"""

USER_TOPIC = """---
name: tabs
description: Prefers tabs in Python
metadata:
  type: user
---
"""

SUMMARY = """v1

## User profile

Some prose that is not a bullet.

## User preferences

- Default to short answers and lead with the outcome. [ad-hoc note]
- Never paste the API key sk-live-0123456789abcdefABCDEF into chat logs.

## Projects

- The bakery site is built with Astro and deployed by a GitHub action.
- """ + "A very long note " * 40 + """
- The bakery site is built with Astro and deployed by a GitHub action.
"""


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "holo"
    monkeypatch.setenv("HOLOMEM_HOME", str(path))
    return path


def _claude_tree(tmp_path: Path) -> Path:
    mem = tmp_path / "projects" / "C--work-billing" / "memory"
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text("- [Deploy](deploy-staging.md) — staging\n", encoding="utf-8")
    (mem / "deploy-staging.md").write_text(TOPIC, encoding="utf-8")
    (mem / "tabs.md").write_text(USER_TOPIC, encoding="utf-8")
    return tmp_path / "projects"


def _facts(home: Path) -> list[tuple]:
    conn = sqlite3.connect(home / "memory_store.db")
    try:
        return conn.execute("SELECT content, category, trust_score FROM facts ORDER BY fact_id").fetchall()
    finally:
        conn.close()


def test_claude_topic_files_become_facts(tmp_path):
    root = _claude_tree(tmp_path)
    for where in (root, root / "C--work-billing", root / "C--work-billing" / "memory"):
        got = sorted((c.text, c.category) for c in importers.claude_candidates(where))
        assert got == [("Prefers tabs in Python", "user_pref"),
                       ("The billing service deploys to staging with systemd, never by hand. Each deploy runs the "
                        "smoke tests first.", "project")]


def test_codex_bullets_become_facts(tmp_path):
    path = tmp_path / "memory_summary.md"
    path.write_text(SUMMARY, encoding="utf-8")
    got = [(c.text, c.category) for c in importers.codex_candidates(path)]
    assert got[0] == ("Default to short answers and lead with the outcome.", "user_pref")
    assert ("The bakery site is built with Astro and deployed by a GitHub action.", "general") in got
    assert not any("prose" in text for text, _ in got)


# The real file (Codex 0.163) ends with "What's in Memory": an index of past tasks per working directory — a keyword
# line per task, with "desc:" / "learnings:" sub-bullets. A task log, not facts about the person or their work.
SUMMARY_WITH_INDEX = """v1

## User preferences

- Default to short answers and lead with the outcome. [ad-hoc note]

## General Tips

- Quote Windows paths with spaces. [ad-hoc note]

## What's in Memory

### C:\\work\\bakery

#### 2026-10-28

- Bakery site release: `v1.2`, PR #10, `deploy.yml`
  - desc: Review-to-release workflow for the bakery site; use before the next release.
  - learnings: Run the smoke tests before tagging; CI alone missed a broken page.

### Older Memory Topics

#### Gmail workflow

- Mail triage: `newer_than:7d`, `next_page_token`
  - desc: Read-only triage of important mail.
"""


def test_codex_task_index_is_not_imported(tmp_path):
    path = tmp_path / "memory_summary.md"
    path.write_text(SUMMARY_WITH_INDEX, encoding="utf-8")
    got = [(c.text, c.category) for c in importers.codex_candidates(path)]
    assert got == [("Default to short answers and lead with the outcome.", "user_pref"),
                   ("Quote Windows paths with spaces.", "general")]


def test_import_screens_dedupes_and_marks_trust(tmp_path, home):
    path = tmp_path / "memory_summary.md"
    path.write_text(SUMMARY, encoding="utf-8")
    out = io.StringIO()
    counts = importers.import_candidates(importers.codex_candidates(path), "codex", dry_run=False, out=out)
    assert counts["added"] == 2 and counts["refused"] == 1 and counts["too_long"] == 1 and counts["exists"] == 1
    facts = _facts(home)
    assert [f[0] for f in facts] == ["Default to short answers and lead with the outcome.",
                                     "The bakery site is built with Astro and deployed by a GitHub action."]
    assert all(trust == pytest.approx(0.4) for _, _, trust in facts)
    assert not any("sk-live" in f[0] for f in facts)
    again = importers.import_candidates(importers.codex_candidates(path), "codex", dry_run=False, out=io.StringIO())
    assert again["added"] == 0 and len(_facts(home)) == 2


def test_dry_run_stores_nothing(tmp_path, home):
    out = io.StringIO()
    counts = importers.import_candidates(importers.claude_candidates(_claude_tree(tmp_path)), "claude-code",
                                         dry_run=True, out=out)
    assert counts["added"] == 2 and "would add [project] The billing service" in out.getvalue()
    assert _facts(home) == []


def test_imported_facts_say_where_they_came_from(tmp_path, home):
    importers.import_candidates(importers.claude_candidates(_claude_tree(tmp_path)), "claude-code", dry_run=False,
                                out=io.StringIO())
    from holomem.engine.store import MemoryStore

    store = MemoryStore(home / "memory_store.db")
    try:
        ids = [r[0] for r in store.read("SELECT fact_id FROM facts")]
        sources = list(store.current_sources(ids).values())
    finally:
        store.close()
    assert len(sources) == 2
    assert all(s["via"].startswith("import:") and s["platform"] == "claude-code" for s in sources)


def test_the_cli_prints_any_text_on_a_windows_console(tmp_path, home):
    """A note with "×" or Cyrillic printed by --dry-run must not crash on a console whose encoding cannot hold it
    (cp1252 / cp866 on Windows; the pipe a client reads is the same)."""
    import os
    import subprocess
    import sys

    path = tmp_path / "memory_summary.md"
    path.write_text("## Notes\n\n- The oven runs at 220 °C × 2 hours for the rye loaf, сказал пекарь.\n",
                    encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    env.update({"HOLOMEM_HOME": str(home), "PYTHONIOENCODING": "cp1252"})
    from _launch import LAUNCH as launch
    proc = subprocess.run([sys.executable, "-c", launch, str(Path(__file__).resolve().parent.parent), "import",
                           "codex-memories", "--path", str(path), "--dry-run"], capture_output=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-800:]
    assert b"would add 1" in proc.stdout


# -- where the clients keep their memory, and files that are not what the parser expects (2026-10-31) ---------------
# Claude Code (code.claude.com/docs/en/memory, sub-agents): auto memory in <config>/projects/<project>/memory/, or the
# folder autoMemoryDirectory names (absolute or ~/); <config> is ~/.claude or CLAUDE_CONFIG_DIR; a subagent's in
# <config>/agent-memory/<name>/. Codex: <CODEX_HOME or ~/.codex>/memories/memory_summary.md, and codex-cli 0.166 keeps
# a memories_v2/ of the same format beside it. Before, only ~/.claude/projects and ~/.codex/memories were read, and a
# file the parser could not read was skipped without a word ("nothing to import", exit 0).

NOTE = """---
name: {name}
description: {text}
metadata:
  type: project
---

{text}
"""


@pytest.fixture
def user_home(tmp_path, monkeypatch, home) -> Path:
    """A home of its own: ~ and the clients' variables point inside tmp_path."""
    path = tmp_path / "user"
    path.mkdir()
    for name in ("HOME", "USERPROFILE"):  # what Path.home() and ~ read, on Windows and elsewhere
        monkeypatch.setenv(name, str(path))
    for name in ("CLAUDE_CONFIG_DIR", "CODEX_HOME"):
        monkeypatch.delenv(name, raising=False)
    return path


def _note(folder: Path, name: str, text: str, encoding: str = "utf-8") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.md"
    path.write_bytes(NOTE.format(name=name, text=text).encode(encoding))
    return path


def _run(source: str, capsys, path: str | None = None) -> tuple[int, str]:
    from types import SimpleNamespace

    code = importers.main(SimpleNamespace(source=source, path=path, dry_run=True))
    return code, capsys.readouterr().out


def test_claude_reads_the_folder_auto_memory_directory_names(user_home, capsys):
    (user_home / ".claude").mkdir()
    (user_home / ".claude" / "settings.json").write_text('{"autoMemoryDirectory": "~/notes/memory"}', encoding="utf-8")
    _note(user_home / "notes" / "memory", "deploy", "The billing service deploys with systemd.")
    _note(user_home / ".claude" / "projects" / "C--work" / "memory", "oven", "The bakery oven runs at 220 degrees.")
    code, out = _run("claude-memory", capsys)
    assert code == 0, out
    assert "would add [project] The billing service deploys with systemd." in out, out
    assert "The bakery oven runs at 220 degrees." in out, out  # notes from before the setting are still read


def test_claude_reads_under_claude_config_dir(user_home, tmp_path, monkeypatch, capsys):
    config = tmp_path / "other-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    _note(config / "projects" / "C--work" / "memory", "deploy", "The billing service deploys with systemd.")
    code, out = _run("claude-memory", capsys)
    assert code == 0 and "The billing service deploys with systemd." in out, out


def test_claude_reads_a_subagents_memory_and_says_whose(user_home, capsys):
    folder = user_home / ".claude" / "agent-memory" / "reviewer"
    _note(folder, "style", "The billing service code uses four-space indents.")
    (folder / "MEMORY.md").write_text("- [Style](style.md)\n", encoding="utf-8")
    found = importers.claude_sources()
    cands = [c for root, prefix in found for c in importers.claude_candidates(root, origin=prefix)]
    assert [(c.text, c.origin) for c in cands] == [("The billing service code uses four-space indents.",
                                                    "agent-memory/reviewer/style.md")]
    code, out = _run("claude-memory", capsys)
    assert code == 0 and "would add 1" in out, out


def test_codex_reads_under_codex_home(user_home, tmp_path, monkeypatch, capsys):
    codex = tmp_path / "codex-home"
    (codex / "memories").mkdir(parents=True)
    (codex / "memories" / "memory_summary.md").write_text("## Notes\n\n- The bakery site is built with Astro.\n",
                                                          encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex))
    code, out = _run("codex-memories", capsys)
    assert code == 0 and "The bakery site is built with Astro." in out, out


def test_codex_reads_memories_v2_beside_memories(user_home, capsys):
    for folder, text in (("memories", "The bakery site is built with Astro."),
                         ("memories_v2", "The billing service deploys with systemd.")):
        (user_home / ".codex" / folder).mkdir(parents=True)
        (user_home / ".codex" / folder / "memory_summary.md").write_text(f"## Notes\n\n- {text}\n", encoding="utf-8")
    code, out = _run("codex-memories", capsys)
    assert code == 0, out
    assert "The bakery site is built with Astro." in out and "The billing service deploys with systemd." in out, out
    assert "memories_v2" in out and "memories" + __import__("os").sep + "memory_summary.md" in out, out  # each path named


def test_a_codex_file_in_a_format_not_recognised_is_an_error_not_nothing(user_home, capsys):
    (user_home / ".codex" / "memories").mkdir(parents=True)
    path = user_home / ".codex" / "memories" / "memory_summary.md"
    path.write_text("## Notes\n\n1. The bakery site is built with Astro.\n2. It deploys on push.\n", encoding="utf-8")
    code, out = _run("codex-memories", capsys)
    assert code == 1, out
    assert f"format not recognized: {path}" in out, out


def test_a_note_with_a_byte_order_mark_keeps_its_type_and_no_frontmatter(tmp_path):
    folder = tmp_path / "memory"
    _note(folder, "deploy", "The billing service deploys with systemd.", encoding="utf-8-sig")
    got = [(c.text, c.category) for c in importers.claude_candidates(folder)]
    assert got == [("The billing service deploys with systemd.", "project")], got


def test_a_note_whose_frontmatter_never_closes_is_reported_not_stored_as_text(user_home, capsys):
    folder = user_home / ".claude" / "projects" / "C--work" / "memory"
    folder.mkdir(parents=True)
    cut = folder / "cut.md"
    cut.write_text("---\nname: cut\ndescription: The billing service deploys\nmetadata:\n  type: proj", encoding="utf-8")
    _note(folder, "oven", "The bakery oven runs at 220 degrees.")
    code, out = _run("claude-memory", capsys)
    assert code == 1, out  # the rest is imported; the exit code says a file was left
    assert "name:" not in out and "---" not in out.replace(f"frontmatter not closed: {cut}", ""), out
    assert f"frontmatter not closed: {cut}" in out and "would add 1" in out, out


def test_a_note_not_in_utf8_is_named_not_skipped_silently(user_home, capsys):
    folder = user_home / ".claude" / "projects" / "C--work" / "memory"
    old = _note(folder, "old", "Сервис биллинга деплоится через systemd.", encoding="cp1251")
    _note(folder, "oven", "The bakery oven runs at 220 degrees.")
    code, out = _run("claude-memory", capsys)
    assert code == 1, out
    assert f"not UTF-8: {old}" in out and "would add 1" in out, out

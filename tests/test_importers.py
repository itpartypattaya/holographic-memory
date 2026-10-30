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
    launch = "import sys; sys.path.insert(0, sys.argv.pop(1)); from holomem.__main__ import main; sys.exit(main())"
    proc = subprocess.run([sys.executable, "-c", launch, str(Path(__file__).resolve().parent.parent), "import",
                           "codex-memories", "--path", str(path), "--dry-run"], capture_output=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-800:]
    assert b"would add 1" in proc.stdout

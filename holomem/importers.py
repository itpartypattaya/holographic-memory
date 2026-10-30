"""holomem import claude-memory|codex-memories [--path P] [--dry-run]: a client's own memory notes, as facts.

- claude-memory: Claude Code's auto memory, ``~/.claude/projects/*/memory/*.md``. One fact per topic file: its body up
  to the first ``**Why:**`` / ``**How to apply:**`` line (the note itself, not its reasoning), or its description when
  the body is empty. MEMORY.md is the index and is skipped. The file's type gives the category (user and feedback ->
  user_pref, project -> project, the rest general).
- codex-memories: Codex's ``~/.codex/memories/memory_summary.md``: one fact per top-level bullet; a bullet under a
  heading that names preferences is user_pref. A trailing ``[ad-hoc note]``-style tag is dropped. The file's last
  section, "What's in Memory", indexes past tasks (keywords, desc, learnings) and is not imported.

Every candidate goes through the engine's write screen (an instruction or a credential is not stored), longer than 400
characters or shorter than the engine's floor is skipped, a text already stored or nearly so is skipped. Imported facts
start below hand-added ones (``default_trust - 0.1``) with the source ``import``; nothing is deleted from the client.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config
from .memory import Context, open_memory

_MAX_CHARS = 400
_CATEGORY = {"user": "user_pref", "feedback": "user_pref", "project": "project"}
_TAG = re.compile(r"\s*\[[a-z][a-z -]{1,30}\]\s*$")
_WHY = re.compile(r"^\s*\*\*(why|how to apply)\s*:?\*\*", re.IGNORECASE)
# Codex's index of past tasks ("What's in Memory": keyword lines, desc/learnings sub-bullets): a task log, not facts
_INDEX = re.compile(r"^##\s+what[’']?s in memory\s*$", re.IGNORECASE)


@dataclass
class Candidate:
    text: str
    category: str
    origin: str  # the file it came from, for the report


def _frontmatter(text: str) -> tuple[dict, str]:
    """({key: value} of a leading --- block, flat and nested keys alike), the body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict = {}
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1:])
        match = re.match(r"^\s*([A-Za-z_]+):\s*(.*)$", line)
        if match:
            meta.setdefault(match.group(1), match.group(2).strip().strip('"'))
    return {}, text


def claude_candidates(root: Path) -> list[Candidate]:
    """*root*: ~/.claude/projects (every project's memory/), one project's folder (its memory/), or a memory folder."""
    files = sorted(root.glob("*/memory/*.md")) or sorted(root.glob("memory/*.md")) or sorted(root.glob("*.md"))
    out = []
    for path in files:
        if path.name.upper() == "MEMORY.MD":
            continue
        try:
            meta, body = _frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            continue
        note = []
        for line in body.splitlines():
            if _WHY.match(line):
                break
            note.append(line)
        text = " ".join(" ".join(note).split()) or meta.get("description", "")
        if text:
            out.append(Candidate(text, _CATEGORY.get(meta.get("type", ""), "general"), path.name))
    return out


def codex_candidates(path: Path) -> list[Candidate]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []
    out, heading = [], ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if _INDEX.match(stripped):
                break  # the task index runs to the end of the file
            heading = stripped.lstrip("#").strip().lower()
            continue
        if line[:2] in ("- ", "* "):
            text = _TAG.sub("", " ".join(stripped[2:].split()))
            if text:
                out.append(Candidate(text, "user_pref" if "prefer" in heading else "general", path.name))
    return out


def import_candidates(candidates: list[Candidate], client: str, *, dry_run: bool, out=sys.stdout) -> dict:
    from .engine.screen import _extract_floor, _fact_problem

    cfg = config.load()
    memory = open_memory(cfg, Context(client=client))
    if memory is None:
        print("memory is off: " + ("; ".join(cfg.problems) or "no database in this scope"), file=out)
        return {"error": "memory off"}
    counts = {"added": 0, "exists": 0, "too_long": 0, "too_short": 0, "refused": 0, "similar": 0}
    trust = max(0.0, cfg.default_trust - 0.1)
    seen: set[str] = set()
    try:
        for cand in candidates:
            text = cand.text
            if len(text) > _MAX_CHARS:
                counts["too_long"] += 1
                continue
            if len(text) < _extract_floor(text):
                counts["too_short"] += 1
                continue
            if _fact_problem(text):
                counts["refused"] += 1
                continue
            if text in seen or memory.store.read("SELECT 1 FROM facts WHERE content = ?", (text,)):
                counts["exists"] += 1
                continue
            seen.add(text)
            if memory.retriever.near_duplicate(text) is not None:
                counts["similar"] += 1
                continue
            if dry_run:
                print(f"would add [{cand.category}] {text}", file=out)
                counts["added"] += 1
                continue
            _, created = memory.store.add_fact_status(text, category=cand.category, trust=trust,
                                                      source=Context(client=client).source(f"import:{cand.origin}"))
            counts["added" if created else "exists"] += 1
    finally:
        memory.close()
    verb = "would add" if dry_run else "added"
    print(f"{verb} {counts['added']}; already stored {counts['exists']}; nearly the same as a stored fact "
          f"{counts['similar']}; refused by the screen {counts['refused']}; too long {counts['too_long']}; too short "
          f"{counts['too_short']}", file=out)
    return counts


def main(args) -> int:
    if args.source == "claude-memory":
        root = Path(args.path).expanduser() if args.path else Path.home() / ".claude" / "projects"
        candidates = claude_candidates(root)
        client = "claude-code"
    else:
        path = Path(args.path).expanduser() if args.path else Path.home() / ".codex" / "memories" / "memory_summary.md"
        candidates = codex_candidates(path)
        client = "codex"
    if not candidates:
        print("nothing to import")
        return 0
    result = import_candidates(candidates, client, dry_run=args.dry_run)
    return 1 if "error" in result else 0

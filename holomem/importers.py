"""holomem import claude-memory|codex-memories [--path P] [--dry-run]: a client's own memory notes, as facts.

- claude-memory: Claude Code's auto memory. Without --path every place it keeps notes (code.claude.com/docs/en/memory,
  sub-agents): ``<config>/projects/*/memory/``, the folder ``autoMemoryDirectory`` names in ``<config>/settings.json``
  and each subagent's ``<config>/agent-memory/<name>/``, where <config> is ``CLAUDE_CONFIG_DIR`` or ``~/.claude``. One
  fact per topic file: its body up to the first ``**Why:**`` / ``**How to apply:**`` line (the note itself, not its
  reasoning), or its description when the body is empty. MEMORY.md is the index and is skipped. The file's type gives
  the category (user and feedback -> user_pref, project -> project, the rest general).
- codex-memories: ``memory_summary.md`` of every ``memories*/`` folder in ``CODEX_HOME`` or ``~/.codex`` (codex-cli
  0.166 keeps a ``memories_v2/`` beside ``memories/``): one fact per top-level bullet; a bullet under a heading that
  names preferences is user_pref. A trailing ``[ad-hoc note]``-style tag is dropped. The file's last section, "What's
  in Memory", indexes past tasks (keywords, desc, learnings) and is not imported.

A file that cannot be read is named, not skipped without a word (2026-10-31): not UTF-8, a frontmatter that never
closes, or a non-empty file nothing could be read from ("format not recognized": the client changed its format). The
rest is imported and the exit code is 1.

Every candidate goes through the engine's write screen (an instruction or a credential is not stored), longer than 400
characters or shorter than the engine's floor is skipped, a text already stored or nearly so is skipped. Imported facts
start below hand-added ones (``default_trust - 0.1``) with the source ``import``; nothing is deleted from the client.
"""
from __future__ import annotations

import json
import os
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


def _frontmatter(text: str) -> tuple[dict, str] | None:
    """({key: value} of a leading --- block, flat and nested keys alike), the body; None when the block never closes
    (a write cut off: where the note starts cannot be told)."""
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
    return None


def _read(path: Path, problems: list | None) -> str | None:
    """The file's text, a UTF-8 byte order mark dropped (it hid the frontmatter, which then became the fact's text);
    None when it cannot be read, named in *problems*."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeError:
        _problem(problems, f"not UTF-8: {path}")
    except OSError as exc:
        _problem(problems, f"cannot read ({type(exc).__name__}): {path}")
    return None


def _problem(problems: list | None, line: str) -> None:
    if problems is not None:
        problems.append(line)


def claude_sources() -> list[tuple[Path, str]]:
    """(folder, origin prefix) of every place Claude Code keeps auto memory: <config>/projects, the folder
    autoMemoryDirectory names in <config>/settings.json (absolute or ~/), each subagent's <config>/agent-memory/<name>/.
    <config> is CLAUDE_CONFIG_DIR or ~/.claude. A project's own settings may name another folder: that one by --path."""
    config = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude").expanduser()
    out = [(config / "projects", "")]
    try:
        chosen = json.loads((config / "settings.json").read_text(encoding="utf-8-sig")).get("autoMemoryDirectory")
    except (OSError, ValueError, AttributeError):
        chosen = None
    if isinstance(chosen, str) and chosen.strip():
        out.append((Path(chosen.strip()).expanduser(), ""))
    agents = config / "agent-memory"
    if agents.is_dir():
        out += [(folder, f"agent-memory/{folder.name}/") for folder in sorted(agents.iterdir()) if folder.is_dir()]
    return out


def claude_candidates(root: Path, origin: str = "", problems: list | None = None) -> list[Candidate]:
    """*root*: ~/.claude/projects (every project's memory/), one project's folder (its memory/), or a memory folder.
    *origin* goes before each file name in the report; a file that cannot be read is named in *problems*."""
    files = sorted(root.glob("*/memory/*.md")) or sorted(root.glob("memory/*.md")) or sorted(root.glob("*.md"))
    out = []
    for path in files:
        if path.name.upper() == "MEMORY.MD":
            continue
        raw = _read(path, problems)
        if raw is None:
            continue
        parsed = _frontmatter(raw)
        if parsed is None:
            _problem(problems, f"frontmatter not closed: {path}")
            continue
        meta, body = parsed
        note = []
        for line in body.splitlines():
            if _WHY.match(line):
                break
            note.append(line)
        text = " ".join(" ".join(note).split()) or meta.get("description", "")
        if text:
            out.append(Candidate(text, _CATEGORY.get(meta.get("type", ""), "general"), origin + path.name))
        elif raw.strip():
            _problem(problems, f"format not recognized: {path}")
    return out


def codex_sources() -> list[Path]:
    """memory_summary.md of every memories*/ folder in CODEX_HOME or ~/.codex: codex-cli 0.166 keeps a memories_v2/
    beside memories/ and which one it writes is not documented, so both (a text in both is stored once)."""
    codex = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
    return sorted(p for p in codex.glob("memories*/memory_summary.md") if p.is_file())


def codex_candidates(path: Path, problems: list | None = None) -> list[Candidate]:
    raw = _read(path, problems)
    if raw is None:
        return []
    lines = raw.splitlines()
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
    if not out and raw.strip() and not any(_INDEX.match(line.strip()) for line in lines):
        _problem(problems, f"format not recognized: {path}")  # no bullet and no task index: not the format we know
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
    problems: list[str] = []
    candidates: list[Candidate] = []
    if args.source == "claude-memory":
        client = "claude-code"
        if args.path:
            candidates = claude_candidates(Path(args.path).expanduser(), problems=problems)
        else:
            for root, origin in claude_sources():
                candidates += claude_candidates(root, origin, problems)
    else:
        client = "codex"
        for path in [Path(args.path).expanduser()] if args.path else codex_sources():
            found = codex_candidates(path, problems)
            if found and not args.path:
                print(f"{len(found)} from {path}")
            candidates += found
    for line in problems:
        print(line)
    if not candidates:
        print("nothing to import")
        return 1 if problems else 0
    result = import_candidates(candidates, client, dry_run=args.dry_run, out=sys.stdout)
    if problems:
        print(f"{len(problems)} file(s) not read: see above")
    return 1 if "error" in result or problems else 0

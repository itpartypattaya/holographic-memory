"""Export: the whole store as one readable Markdown page (memory_export, holomem export).

A revision view, for reading by eye, Obsidian or another tool: each fact with its trust, category, names and where its
current version was written, most trusted first; the facts merged into the ones shown, with their text and the day of
the merge. A fact recall refuses to inject (a credential, an instruction, hidden markup: the engine's
``_recall_problem``) is listed too, marked "(withheld from recall)": this is where it is seen and cleaned up. Not a
backup: no history of versions, and the tool cuts long texts and the list.

    holomem export [--out FILE] [--category C] [--min-trust X] [--limit N]

    holomem export --journal [--days N] [--out FILE]

--journal is the task-outcome journal instead (journal.py), for the weekly review of what memory offered: each attempt
of the last N days (30 at most, the journal's life) linked to this database, oldest first — when, recall or search,
which client and session, what it emitted (fact and version ids, the version's state and source, never a fact's
text), with the trace why a candidate was left out, and the outcomes the agent reported (their own short text). It only
reads: the database is opened read-only (SQLite mode=ro), without the engine's store, whose every open writes (a
migration, a file rewrite a forget owes, deleting vectors when search by meaning is off); nothing is pruned or
extended, and what a journal turned off since has left is read. The facts' page opens the store as usual and runs the
journal's retention, as holomem status does.

Without --out the page goes to stdout; with it, to that file. Either way UTF-8 with LF line ends, whatever the
console's code page. A file is written beside its target and then replaces it, so a failure leaves an existing file
as it was. A failure is named by its kind, never by a message that may quote a fact.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import config
from .memory import Context, open_memory
from .recall import one_line

TOOL_CHARS = 500  # the tool's cut of a fact's text
WITHHELD = " (withheld from recall)"


def source_line(src: dict | None) -> str:
    """Where a version was written, as the export's line in the bridge's source shape:
    "tool · said · claude-code project <key> · session <id> · recorded 2026-10-30"."""
    if not src:
        return "source unknown"
    where = " ".join(p for p in (src.get("platform"), src.get("chat_type"), src.get("chat_id")) if p)
    parts = [src.get("via", ""), src.get("origin", ""), where,
             f"session {src['session_id']}" if src.get("session_id") else "",
             f"recorded {src['recorded_on']}" if src.get("recorded_on") else ""]
    return " · ".join(p for p in parts if p)


def render(store, *, category: str | None = None, min_trust: float = 0.0, limit: int | None = None,
           chars: int | None = None) -> dict:
    """{markdown, count[, merged]}: up to *limit* facts (None: every one), texts cut at *chars* (None: whole)."""
    from .engine.screen import _recall_problem

    total = store.count_facts()
    facts = store.list_facts(category=category, min_trust=min_trust,
                             limit=max(total, 1) if limit is None else limit)
    shown = [f["fact_id"] for f in facts]
    names: dict[int, list[str]] = {}
    for i in range(0, len(shown), 500):
        chunk = shown[i:i + 500]
        for fact_id, name in store.read("SELECT fe.fact_id, e.name FROM fact_entities fe JOIN entities e "
                                        f"ON e.entity_id = fe.entity_id WHERE fe.fact_id IN ({','.join('?' * len(chunk))})",
                                        chunk):
            names.setdefault(fact_id, []).append(name)
    sources = store.current_sources(shown)
    cut = (lambda text: one_line(text, chars)) if chars else (lambda text: " ".join((text or "").split()))
    lines = [f"# Holographic facts ({len(facts)} of {total})", ""]
    for f in facts:
        linked = ", ".join(sorted(names.get(f["fact_id"], [])))
        lines.append(f"- #{f['fact_id']} [{f['trust_score']:.2f}, {f['category']}] {cut(f['content'])}"
                     + (WITHHELD if _recall_problem(f["content"] or "") else "")
                     + (f" — {linked}" if linked else "")
                     + f"\n  _{source_line(sources.get(f['fact_id']))}_")
    result: dict = {"markdown": "\n".join(lines), "count": len(facts)}
    # facts merged into the ones shown: what they said and where it lives now (the engine's private check: a file
    # no writer of fact_merges has touched has no such table)
    merged = [dict(r) for r in store.read(
        "SELECT m.from_fact_id, m.into_fact_id, m.merged_at, v.content FROM fact_merges m JOIN fact_versions v "
        "ON v.fact_id = m.from_fact_id AND v.status = 'merged' WHERE m.into_fact_id IN "
        f"({','.join('?' * len(shown))}) ORDER BY m.into_fact_id, m.from_fact_id", sorted(shown))] \
        if shown and store._merges_ready() else []
    if merged:
        lines += ["", "## Merged into the facts above", ""]
        lines += [f"- #{m['from_fact_id']} → #{m['into_fact_id']} ({str(m['merged_at'])[:10]}): {cut(m['content'])}"
                  for m in merged]
        result["markdown"] = "\n".join(lines)
        result["merged"] = [{"fact_id": m["from_fact_id"], "into": m["into_fact_id"], "merged_at": m["merged_at"]}
                            for m in merged]
    return result


def journal_page(store, links: list[dict], days: int) -> dict:
    """{markdown, count}: the attempts of *links* (state.db's, oldest first) as the engine reports them."""
    from .journal import key_of

    lines = [f"# Memory journal ({len(links)} attempts, last {days} days)", ""]
    for link in links:
        report = store.experience_report(link["attempt_id"], key_of(link))
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(link["created"]))
        session = link["session"][:12] or "-"
        lines.append(f"## {when} · {report.get('kind', '-')} · {link['client']} · session {session}")
        lines.append(f"attempt `{link['attempt_id']}` · {report['status']}")
        if report["status"] == "ok":
            for e in report["emitted"]:
                src = e.get("source")
                where = source_line(src) if isinstance(src, dict) else str(src or "source unknown")
                lines.append(f"- emitted #{e['fact_id']} (version {e['version_id']}, {e.get('version_status')}"
                             + (f", {e['origin']}" if e.get("origin") else "") + f") · {where}")
            for d in report["decisions"]:
                if d["decision"] != "emitted":
                    lines.append(f"- left out #{d['fact_id']} (version {d['version_id']}): {d['decision']}")
            for o in report["outcomes"]:
                parts = [f"{o['outcome']} / {o['reason']}", "active" if o["active"] else "superseded"]
                if o.get("task"):
                    parts.append(f"task: {o['task']}")
                if o.get("correction"):
                    parts.append(f"correction: {o['correction']}")
                lines.append(f"- outcome `{o['event_id']}`: " + " · ".join(parts))
        lines.append("")
    return {"markdown": "\n".join(lines).rstrip("\n"), "count": len(links)}


def _journal_reader(conn: sqlite3.Connection):
    """Just what the engine's experience_report reads (ExperienceMixin, HistoryMixin._version), over *conn*."""
    from .engine.experience import ExperienceMixin
    from .engine.history import HistoryMixin

    class Reader:
        experience_ready = ExperienceMixin.experience_ready
        experience_report = ExperienceMixin.experience_report
        _version = HistoryMixin._version

        def __init__(self):
            self._conn, self._lock = conn, threading.RLock()

        def _one(self, sql: str, params=()):
            return self._conn.execute(sql, params).fetchone()

    return Reader()


def journal_export(cfg, days: int) -> dict:
    """The journal page of this directory's database, read without writing: state.db's links and the store, both
    opened read-only. Raises LookupError without a database here."""
    from .state import read_attempts

    path = cfg.database_for(os.getcwd())
    if path is None:
        raise LookupError("no database here")
    links = read_attempts(cfg.state_path, str(path), time.time() - days * 86400)
    if not links:
        return journal_page(None, [], days)
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        return journal_page(_journal_reader(conn), links, days)
    finally:
        conn.close()


def write_file(path: Path, data: bytes) -> None:
    """*data* into *path* whole or not at all: a temporary file beside it, then a replace; removed on any failure."""
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def main(args, out=None) -> int:
    """holomem export: 0 when the page was written, 1 (and a note on stderr, by kind) when it was not."""
    out = out if out is not None else sys.stdout.buffer
    cfg = config.load()
    if cfg.problems:
        print("memory is off: " + "; ".join(cfg.problems), file=sys.stderr)
        return 1
    if args.limit is not None and args.limit < 1:
        print("export: --limit must be 1 or more", file=sys.stderr)
        return 1
    if args.journal and not 1 <= args.days <= 30:
        print("export: --days must be between 1 and 30", file=sys.stderr)
        return 1
    try:
        if args.journal:
            try:
                page = journal_export(cfg, args.days)
            except LookupError:
                print("no database here (project scope without a project)", file=sys.stderr)
                return 1
        else:
            memory = open_memory(cfg, Context(client="export", cwd=os.getcwd()))
            if memory is None:
                print("no database here (project scope without a project)", file=sys.stderr)
                return 1
            try:
                from .journal import prune

                prune(memory)
                page = render(memory.store, category=args.category, min_trust=args.min_trust, limit=args.limit)
            finally:
                memory.close()
        data = (page["markdown"] + "\n").encode("utf-8")
        if args.out:
            write_file(Path(args.out).expanduser(), data)
        else:
            out.write(data)
            out.flush()
    except Exception as exc:  # noqa: BLE001 - named by kind: SQLite may quote a value, a path error a file name
        print(f"export failed ({type(exc).__name__}); nothing was written", file=sys.stderr)
        return 1
    if args.out:
        print(f"exported {page['count']} {'attempts' if args.journal else 'facts'} to {args.out}", file=sys.stderr)
    return 0

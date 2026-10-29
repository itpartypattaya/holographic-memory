"""The vendored engine runs where Hermes core is not installed.

Requirement: the copy of the plugin's engine in ``holomem.engine`` (unedited, see tools/sync_engine.py) opens a
database at a path it is given, stores facts and finds them, and opens the same file again; it opens a database an
older plugin made (no ``hrr_vector`` column); it never switches an on-disk WAL database to DELETE, and never turns WAL
on where the linked SQLite has the WAL-reset bug; with no path it says a path is needed instead of failing on a
missing Hermes module. Where a real Hermes helper is importable, that one is used, not the stand-in.

Each case runs in a fresh interpreter (``-I``) with every Hermes module blocked, as Claude Code or Codex would start
a hook or the MCP server: nothing of the test process (its sys.path, its imports) leaks in.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_HERMES = ["agent", "tools", "utils", "hermes_cli", "hermes_constants", "hermes_state_wal", "hermes_time", "hermes_yaml"]

_RUNNER = textwrap.dedent('''
    import importlib.abc, json, sqlite3, sys
    root, case, blocked, db, extra = sys.argv[1], sys.argv[2], set(json.loads(sys.argv[3])), sys.argv[4], sys.argv[5]

    class _NoHermes(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in blocked:  # name= as the import system sets it for a module that is not there
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _NoHermes())
    sys.path.insert(0, root)
    if extra:
        sys.path.insert(0, extra)
    out = {}

    def mode(path):
        c = sqlite3.connect(path)
        try:
            return c.execute("PRAGMA journal_mode").fetchone()[0].lower()
        finally:
            c.close()

    try:
        from holomem.engine.store import MemoryStore
        from holomem.engine.retrieval import FactRetriever

        if case == "add_search_reopen":
            store = MemoryStore(db, default_trust=0.5, hrr_dim=256)
            fid, created = store.add_fact_status("The billing service deploys to the staging server with systemd",
                                                 category="project")
            hits = FactRetriever(store, hrr_dim=256).search("how does the billing service deploy", limit=5)
            out["created"], out["found"] = created, [h["fact_id"] for h in hits] == [fid]
            store.close()
            again = MemoryStore(db, default_trust=0.5, hrr_dim=256)  # the next session opens the same file
            out["count_after_reopen"] = again.count_facts()
            again.close()
            out["mode"] = mode(db)
        elif case == "pre_hrr_db":
            c = sqlite3.connect(db)  # the 0.1.x facts table: no hrr_vector column
            c.executescript("""
                CREATE TABLE facts (
                    fact_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL UNIQUE,
                    category TEXT DEFAULT 'general', tags TEXT DEFAULT '', trust_score REAL DEFAULT 0.5,
                    retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
                INSERT INTO facts (content) VALUES ('Alice rides a red bicycle');
            """)
            c.close()
            store = MemoryStore(db, default_trust=0.5, hrr_dim=256)
            cols = {r[1] for r in store._conn.execute("PRAGMA table_info(facts)")}
            out["has_hrr_vector"], out["count"] = "hrr_vector" in cols, store.count_facts()
            store.close()
        elif case == "wal_kept":
            c = sqlite3.connect(db)  # a database another process left in WAL
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS keep (x)")
            c.close()
            store = MemoryStore(db, default_trust=0.5, hrr_dim=256)
            store.add_fact_status("Bob runs a bakery on the corner")
            store.close()
            out["mode"] = mode(db)
        elif case == "no_path":
            try:
                MemoryStore(None)
                out["error"] = None
            except Exception as exc:
                out["error"], out["message"] = type(exc).__name__, str(exc)
        elif case == "real_module_wins":
            store = MemoryStore(db, default_trust=0.5, hrr_dim=256)
            store.close()
            out["calls"] = getattr(sys.modules["hermes_state_wal"], "CALLS", None)
        out["sqlite"] = sqlite3.sqlite_version
    except Exception as exc:
        out["crash"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out))
''')


def _wal_reset_vulnerable(version: str) -> bool:
    """https://sqlite.org/wal.html#walresetbug: 3.7.0 up to 3.51.2, except the patched 3.50.7+ and 3.44.6+ lines."""
    v = tuple(int(p) for p in version.split(".")[:3])
    return not (v < (3, 7, 0) or v >= (3, 51, 3) or (3, 50, 7) <= v < (3, 51, 0) or (3, 44, 6) <= v < (3, 45, 0))


def _run(case: str, db: Path, blocked=_HERMES, extra: Path | None = None) -> dict:
    proc = subprocess.run([sys.executable, "-I", "-c", _RUNNER, str(ROOT), case, json.dumps(list(blocked)), str(db),
                           str(extra or "")], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_stores_finds_and_reopens(tmp_path):
    out = _run("add_search_reopen", tmp_path / "memory.db")
    assert "crash" not in out, out.get("crash")
    assert out["created"] and out["found"]
    assert out["count_after_reopen"] == 1
    if _wal_reset_vulnerable(out["sqlite"]):
        assert out["mode"] != "wal"  # a new file stays out of WAL on a SQLite with the WAL-reset bug


def test_upgrades_a_pre_hrr_database(tmp_path):
    out = _run("pre_hrr_db", tmp_path / "old.db")
    assert "crash" not in out, out.get("crash")
    assert out["has_hrr_vector"] and out["count"] == 1


def test_keeps_an_on_disk_wal_database_in_wal(tmp_path):
    out = _run("wal_kept", tmp_path / "wal.db")
    assert "crash" not in out, out.get("crash")
    assert out["mode"] == "wal"


def test_asks_for_a_path(tmp_path):
    out = _run("no_path", tmp_path / "unused.db")
    assert "crash" not in out, out.get("crash")
    assert out["error"] == "ValueError" and "db_path" in out["message"]


def test_a_real_hermes_helper_is_used_when_present(tmp_path):
    real = tmp_path / "site"
    real.mkdir()
    (real / "hermes_state_wal.py").write_text(
        "CALLS = []\n"
        "def apply_wal_with_fallback(conn, *, db_label='state.db', require_wal=False):\n"
        "    CALLS.append(db_label)\n"
        "    return 'delete'\n", encoding="utf-8")
    out = _run("real_module_wins", tmp_path / "m.db", blocked=[m for m in _HERMES if m != "hermes_state_wal"],
               extra=real)
    assert "crash" not in out, out.get("crash")
    assert out["calls"] == ["memory_store.db (holographic)"]


@pytest.mark.parametrize("version,vulnerable", [("3.49.1", True), ("3.51.2", True), ("3.51.3", False),
                                               ("3.50.7", False), ("3.44.6", False), ("3.6.23", False)])
def test_wal_reset_range_matches_sqlite_advisory(version, vulnerable):
    from holomem import compat

    assert compat.wal_reset_vulnerable(tuple(int(p) for p in version.split("."))) is vulnerable

"""The engine is this repo's own code and needs nothing outside it.

Requirement: ``holomem.engine`` imports and works in a bare interpreter where no agent framework is installed: it opens
a database at a path it is given, stores facts and finds them, and opens the same file again; it opens a database an
older version made (no ``hrr_vector`` column); it never switches an on-disk WAL database to DELETE, and never turns WAL
on where the linked SQLite has the WAL-reset bug; with no path it keeps ``memory_store.db`` in holomem's home
(``$HOLOMEM_HOME``, the same file the settings name by default). No module of another program is imported, by name or
by a stand-in put into ``sys.modules``: every module the engine and the bridge import is the standard library, the two
declared dependencies, or holomem's own.

Each case runs in a fresh interpreter (``-I``), as Claude Code or Codex would start a hook or the MCP server: nothing
of the test process (its sys.path, its imports) leaks in, and every import of a top-level module that is not the
standard library, numpy, snowballstemmer, holomem or already loaded at start is refused. Afterwards no such module may
be in ``sys.modules`` either (a stand-in put there by hand never passes an import hook).
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_OWN = ["holomem", "numpy", "snowballstemmer"]  # holomem and its two declared dependencies

_RUNNER = textwrap.dedent('''
    import importlib.abc, json, sqlite3, sys
    root, case, own, db = sys.argv[1], sys.argv[2], set(json.loads(sys.argv[3])), sys.argv[4]
    at_start = {m.split(".")[0] for m in sys.modules}
    allowed = set(sys.stdlib_module_names) | own | at_start

    class _OnlyOwn(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] not in allowed:  # name= as the import system sets it for a module that is not there
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _OnlyOwn())
    sys.path.insert(0, root)
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
            from holomem import config

            store = MemoryStore(None, hrr_dim=256)
            store.add_fact_status("Carol keeps the release notes")
            out["db"], out["count"] = str(store.db_path), store.count_facts()
            store.close()
            out["settings_db"] = str(config.load().db_path)
        out["foreign"] = sorted(m for m in sys.modules if m.split(".")[0] not in allowed)
        out["sqlite"] = sqlite3.sqlite_version
    except Exception as exc:
        out["crash"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out))
''')


def _wal_reset_vulnerable(version: str) -> bool:
    """https://sqlite.org/wal.html#walresetbug: 3.7.0 up to 3.51.2, except the patched 3.50.7+ and 3.44.6+ lines."""
    v = tuple(int(p) for p in version.split(".")[:3])
    return not (v < (3, 7, 0) or v >= (3, 51, 3) or (3, 50, 7) <= v < (3, 51, 0) or (3, 44, 6) <= v < (3, 45, 0))


def _run(case: str, db: Path, env: dict | None = None) -> dict:
    proc = subprocess.run([sys.executable, "-I", "-c", _RUNNER, str(ROOT), case, json.dumps(_OWN), str(db)],
                          capture_output=True, text=True, encoding="utf-8", timeout=120,
                          env={**os.environ, **(env or {})})
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "crash" not in out, out.get("crash")
    assert out["foreign"] == [], f"modules of another program were imported or stood in: {out['foreign']}"
    return out


def test_stores_finds_and_reopens(tmp_path):
    out = _run("add_search_reopen", tmp_path / "memory.db")
    assert out["created"] and out["found"]
    assert out["count_after_reopen"] == 1
    if _wal_reset_vulnerable(out["sqlite"]):
        assert out["mode"] != "wal"  # a new file stays out of WAL on a SQLite with the WAL-reset bug


def test_upgrades_a_pre_hrr_database(tmp_path):
    out = _run("pre_hrr_db", tmp_path / "old.db")
    assert out["has_hrr_vector"] and out["count"] == 1


def test_keeps_an_on_disk_wal_database_in_wal(tmp_path):
    out = _run("wal_kept", tmp_path / "wal.db")
    assert out["mode"] == "wal"


def test_without_a_path_the_store_lives_in_holomem_home(tmp_path):
    home = tmp_path / "holo"
    out = _run("no_path", tmp_path / "unused.db", env={"HOLOMEM_HOME": str(home)})
    assert Path(out["db"]) == home / "memory_store.db"
    assert (home / "memory_store.db").is_file() and out["count"] == 1
    assert Path(out["settings_db"]) == Path(out["db"])  # the engine's default is the file the settings name
    assert not (tmp_path / "unused.db").exists()


def _absolute_imports(path: Path) -> set[str]:
    """Top-level names of every absolute import in the file, at any depth, and of every import by name."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and not node.level:
            found.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            target = node.args[0].value
            if name in ("__import__", "import_module") and isinstance(target, str) and not target.startswith("."):
                found.add(target.split(".")[0])
    return found


@pytest.mark.parametrize("path", sorted((ROOT / "holomem").rglob("*.py")), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_module_imports_another_program(path):
    allowed = set(sys.stdlib_module_names) | set(_OWN) | {"_holographic_map"}  # memory_map run as a file
    foreign = _absolute_imports(path) - allowed
    assert not foreign, f"{path.relative_to(ROOT).as_posix()} imports {sorted(foreign)}"


@pytest.mark.parametrize("version,vulnerable", [("3.49.1", True), ("3.51.2", True), ("3.51.3", False),
                                               ("3.50.7", False), ("3.44.6", False), ("3.6.23", False)])
def test_wal_reset_range_matches_sqlite_advisory(version, vulnerable):
    from holomem.engine.sqlite_util import wal_reset_vulnerable

    assert wal_reset_vulnerable(tuple(int(p) for p in version.split("."))) is vulnerable

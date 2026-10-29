"""Stand-ins for the three Hermes core helpers the vendored engine's store calls when it opens a database.

``holomem.engine`` is the plugin's code unedited, and its ``store.py`` imports, lazily, ``hermes_state_wal``
(journal mode), ``hermes_cli.sqlite_util`` (add a column) and ``hermes_constants`` (where the database lives when no
path is given). Outside Hermes those modules are not there. ``install()`` puts a stand-in into ``sys.modules`` for
each one that is missing, and only for those: where a real one is importable it stays and is used.

The journal policy follows core's: a database already in WAL stays in WAL (switching it to DELETE under another open
connection loses that connection's uncheckpointed commits); a SQLite with the WAL-reset bug never turns WAL on for a
file not in WAL; a file another connection holds busy is left as it is; a filesystem that refuses WAL (network shares)
keeps its mode, with one warning. ``get_hermes_home`` has no answer here: the bridge always names its database.
"""
from __future__ import annotations

import importlib.util
import logging
import sqlite3
import sys
import threading
import types

logger = logging.getLogger("holomem")

_warned: set[str] = set()
_lock = threading.Lock()


def wal_reset_vulnerable(version: tuple[int, ...] | None = None) -> bool:
    """https://sqlite.org/wal.html#walresetbug: 3.7.0 to 3.51.2, except the patched 3.50.7+ and 3.44.6+ lines (the
    range core gates on)."""
    v = tuple(version or sqlite3.sqlite_version_info)[:3]
    return not (v < (3, 7, 0) or v >= (3, 51, 3) or (3, 50, 7) <= v < (3, 51, 0) or (3, 44, 6) <= v < (3, 45, 0))


def _warn_once(label: str, message: str, *args) -> None:
    with _lock:
        if label in _warned:
            return
        _warned.add(label)
    logger.warning(message, *args)


def apply_wal_with_fallback(conn: sqlite3.Connection, *, db_label: str = "memory_store.db",
                            require_wal: bool = False) -> str:
    """The journal mode the connection ends up in; WAL only where it is safe (see the module docstring)."""
    try:
        current = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    except sqlite3.OperationalError:  # busy: another connection holds the file; touching it could lose its commits
        if require_wal:
            raise
        _warn_once(f"probe:{db_label}", "holomem: journal mode of %s unreadable (busy); left as it is", db_label)
        return "unknown"
    if current == "wal":
        return current
    if wal_reset_vulnerable():
        if require_wal:
            raise sqlite3.OperationalError(f"WAL refused for {db_label}: SQLite {sqlite3.sqlite_version} has the "
                                           "WAL-reset bug")
        return current
    try:
        return str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
    except sqlite3.OperationalError as exc:
        if require_wal:
            raise
        _warn_once(f"wal:{db_label}", "holomem: WAL is not available for %s (%s); journal mode stays %s", db_label,
                   type(exc).__name__, current)
        return current


def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> bool:
    """``ALTER TABLE <table> ADD COLUMN <ddl>``: True when this call added it, False when a racing opener had."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
        return True
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return False
        raise


def get_hermes_home():
    raise ValueError("holomem: MemoryStore needs an explicit db_path; outside Hermes there is no HERMES_HOME to keep "
                     "memory_store.db in")


def _importable(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # a dotted name whose parent is not there
        return False


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name, f"holomem stand-in for Hermes core's {name}")
    module.__dict__.update(attrs)
    module.__holomem_standin__ = True
    return module


def install() -> list[str]:
    """Stand-ins for the missing helpers; returns the names put in place ([] inside Hermes)."""
    placed = []
    if not _importable("hermes_state_wal"):
        sys.modules["hermes_state_wal"] = _module("hermes_state_wal", apply_wal_with_fallback=apply_wal_with_fallback)
        placed.append("hermes_state_wal")
    if not _importable("hermes_constants"):
        sys.modules["hermes_constants"] = _module("hermes_constants", get_hermes_home=get_hermes_home)
        placed.append("hermes_constants")
    if not _importable("hermes_cli.sqlite_util"):
        util = _module("hermes_cli.sqlite_util", add_column_if_missing=add_column_if_missing)
        if _importable("hermes_cli"):  # a core without this helper: hang ours under the real package
            parent = importlib.import_module("hermes_cli")
        else:
            parent = sys.modules["hermes_cli"] = _module("hermes_cli")
            parent.__path__ = []  # a package, so the dotted import resolves through sys.modules
            placed.append("hermes_cli")
        sys.modules["hermes_cli.sqlite_util"] = util
        parent.sqlite_util = util
        placed.append("hermes_cli.sqlite_util")
    return placed

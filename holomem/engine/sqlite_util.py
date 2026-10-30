"""The SQLite helpers the store calls when it opens a database: the journal mode and adding a column.

The journal policy: a database already in WAL stays in WAL (switching it to DELETE under another open connection loses
that connection's uncheckpointed commits); a SQLite with the WAL-reset bug never turns WAL on for a file not in WAL; a
file another connection holds busy is left as it is; a filesystem that refuses WAL (network shares) keeps its mode,
with one warning.
"""
from __future__ import annotations

import logging
import sqlite3
import threading

logger = logging.getLogger(__name__)

_warned: set[str] = set()
_lock = threading.Lock()


def wal_reset_vulnerable(version: tuple[int, ...] | None = None) -> bool:
    """https://sqlite.org/wal.html#walresetbug: 3.7.0 to 3.51.2, except the patched 3.50.7+ and 3.44.6+ lines."""
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

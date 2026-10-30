"""The store's shared connections are released deterministically (from the regression tests for #44037, a leaked
SQLite connection on shutdown): ``MemoryStore.release_all_under`` closes exactly the connections under a directory,
and a stale holder's ``close()`` never evicts the fresh registry entry another holder opened on the same path."""

import sqlite3

import pytest


def test_release_all_under_closes_connections_inside_directory_only(tmp_path):
    """Regression test for #88347 — profile delete must break refcounted handles.

    ``close()`` alone can never free a database held by a live agent (refs >= 1),
    and on Windows an open SQLite handle makes rmtree of the profile directory
    fail with WinError 32. ``release_all_under`` force-closes exactly the shared
    connections under the doomed directory and leaves everything else alone.
    """
    from holomem.engine.store import MemoryStore

    profile_dir = tmp_path / "profiles" / "default-2"
    profile_dir.mkdir(parents=True)
    inside = MemoryStore(db_path=profile_dir / "memory_store.db", hrr_dim=64)
    outside = MemoryStore(db_path=tmp_path / "other" / "memory_store.db", hrr_dim=64)
    inside_conn = inside._conn
    outside_conn = outside._conn
    # Both live before the release; the inside one is held "forever" (refs=1,
    # like a long-running server — nobody calls close()).
    inside_conn.execute("SELECT 1").fetchone()
    outside_conn.execute("SELECT 1").fetchone()

    try:
        released = MemoryStore.release_all_under(profile_dir)
        assert released == 1

        # The doomed connection is really closed despite refs > 0 ...
        with pytest.raises(sqlite3.ProgrammingError):
            inside_conn.execute("SELECT 1")
        # ... and the registry entry is gone, so a second release finds
        # nothing (the CLI-process no-op case).
        assert str((profile_dir / "memory_store.db").resolve()) not in MemoryStore._shared
        assert MemoryStore.release_all_under(profile_dir) == 0
        # A new store on the same path reopens fresh instead of reusing
        # the dead connection.
        reopened = MemoryStore(db_path=profile_dir / "memory_store.db", hrr_dim=64)
        reopened._conn.execute("SELECT 1").fetchone()

        # The sibling outside the directory is untouched.
        outside_conn.execute("SELECT 1").fetchone()
    finally:
        inside.close()
        reopened.close()
        outside.close()


def test_stale_holder_close_does_not_evict_fresh_registry_entry(tmp_path):
    """Follow-up to #88347 — a stale holder's late ``close()`` must be inert.

    After ``release_all_under`` force-closes a profile's connection, a store
    re-created on the same path registers a FRESH shared entry under the same
    key. If the stale holder (whose entry was force-closed) then calls
    ``close()``, it must not pop the fresh entry out of the registry — that
    would let a third store open a SECOND connection to the same database and
    silently reintroduce the multi-writer contention the registry prevents.
    """
    from holomem.engine.store import MemoryStore

    profile_dir = tmp_path / "profiles" / "default-2"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "memory_store.db"

    stale = MemoryStore(db_path=db_path, hrr_dim=64)
    assert MemoryStore.release_all_under(profile_dir) == 1

    fresh = MemoryStore(db_path=db_path, hrr_dim=64)
    key = fresh._key
    fresh_entry = MemoryStore._shared[key]

    # The stale holder's late close must leave the fresh entry registered...
    stale.close()
    assert MemoryStore._shared.get(key) is fresh_entry
    # ...and a third store must attach to the SAME shared connection.
    third = MemoryStore(db_path=db_path, hrr_dim=64)
    try:
        assert third._conn is fresh._conn
    finally:
        third.close()
        fresh.close()
    # Normal last-holder close still evicts its own entry.
    assert key not in MemoryStore._shared

"""Bootstrap for the engine's own tests (``holomem.engine``).

The engine and these tests were ported from hermes-plugin-holographic v0.25.2 (MIT), now this repo's own: they are
edited here, together. Only the tests of the engine came over: the ones of the agent integration that wrapped it
there test code this repo does not have.
"""
import os
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test gets its own HOME; nothing touches the real home folder or a store kept there."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


# PRAGMA synchronous=OFF on every connection the suite opens: no test checks what survives a power cut. Where the
# linked SQLite has the WAL-reset bug every store runs in DELETE journal mode and each commit waits on two fsyncs; on a
# Windows disk that makes the suite many times slower. The journal mode, locking and every write stay as they are;
# only the wait for the disk goes. Patched at import, not in a fixture, so that stores the module fixtures open get it
# too.
_real_connect = sqlite3.connect


def _connect_without_fsync(*args, **kwargs):
    conn = _real_connect(*args, **kwargs)
    try:
        conn.execute("PRAGMA synchronous=OFF")
    except sqlite3.Error:
        pass  # a file a test made unreadable on purpose fails where that test expects it to, not here
    return conn


sqlite3.connect = _connect_without_fsync


def _symlinks_supported(tmp_dir: Path) -> bool:
    target, link = tmp_dir / "target", tmp_dir / "link"
    target.write_text("x", encoding="utf-8")
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        return False
    return True


def pytest_configure(config):
    config.addinivalue_line("markers", "require_symlinks: skip when symbolic links cannot be created here")


def pytest_runtest_setup(item):
    if item.get_closest_marker("require_symlinks"):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            if not _symlinks_supported(Path(tmp)):
                pytest.skip("symbolic links are not supported here (Windows needs developer mode)")

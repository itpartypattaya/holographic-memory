"""expecting() (issue #2) checks the versions in the transaction the write runs in: from the check to the end of
the block no other connection can write, so nothing changes a fact between the check and the write."""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("numpy")

from holomem.engine.store import MemoryStore


def test_no_other_writer_gets_in_between_the_check_and_the_write(tmp_path):
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=64)
    fact_id = store.add_fact("Lena lives in Lisbon")
    version = store.history(fact_id)["versions"][-1]["version_id"]
    other = sqlite3.connect(db, timeout=0)
    try:
        with store.expecting({fact_id: version}):
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("UPDATE facts SET content = 'Lena lives in Porto' WHERE fact_id = ?", (fact_id,))
            store.end_fact(fact_id)
    finally:
        other.close()
        store.close()

"""Opening state.db (every hook does) writes as little as it can. Seen on the Windows CI runner (2026-10-31): a hook
stood more than 5 s in State() and its deadline cut it. A new state.db made its tables one statement at a time, each
its own transaction: in rollback mode each creates and deletes a journal file, which on Windows can wait on the virus
scanner. Now the tables come in one transaction."""
from __future__ import annotations

import sqlite3

import pytest

import holomem.state as state_module
from holomem.state import State


@pytest.fixture
def statements(monkeypatch):
    """Each statement a State's connection runs, by its first word."""
    seen: list[str] = []
    connect = sqlite3.connect

    def traced(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(lambda sql: seen.append(sql.split()[0].upper().rstrip(";") if sql.strip() else ""))
        return conn

    monkeypatch.setattr(state_module.sqlite3, "connect", traced)
    return seen


def transactions(seen: list[str]) -> list[list[str]]:
    """The write statements run outside BEGIN … COMMIT each count as one transaction; a BEGIN … COMMIT as one."""
    found, open_one = [], None
    for word in seen:
        if word == "BEGIN":
            open_one = []
        elif word in ("COMMIT", "END"):
            found.append(open_one or [])
            open_one = None
        elif word in ("CREATE", "INSERT", "UPDATE", "DELETE", "DROP", "ALTER"):
            if open_one is None:
                found.append([word])
            else:
                open_one.append(word)
    return found


def test_a_new_state_makes_its_tables_in_one_transaction(tmp_path, statements):
    State(tmp_path / "state.db").close()
    made = transactions(statements)
    assert len(made) == 1 and made[0].count("CREATE") >= 7, statements


def test_a_state_db_an_older_release_made_is_brought_up_and_kept(tmp_path):
    """A state.db from before (tables, no version mark) keeps its rows; it is marked once and then only read."""
    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE turns (session TEXT PRIMARY KEY, turn INTEGER NOT NULL, touched REAL NOT NULL);"
                       "INSERT INTO turns VALUES ('s1', 4, 0);")
    conn.close()
    state = State(path)
    try:
        assert state.next_turn("s1") == 5
    finally:
        state.close()

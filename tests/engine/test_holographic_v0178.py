"""0.17.8: a wrong setting does not run quietly (second review against "10 principles of good code for AI agents",
2026-10-23).

- A setting with a fixed set of values that is none of them (scope, recall_source, embeddings, the on/off
  switches) turns memory off with the reason, as a setting that is no number does; a settings file that cannot be
  parsed is not taken as the defaults without a word (holomem.config, tests/test_config.py).
- A KeyError from a bug inside an action was answered as "no such fact / name" or "missing argument", and any other
  failure's text went to the agent, where SQLite may quote a fact. Now only NotFound is "not found", only a missing
  schema argument is "missing", and the rest is named by its kind and logged.
- Failures logged at debug only (stored vectors not deleted, search by meaning and fact vectors stopped, the memory
  mirror, a damaged dictionary, the timezone) are warnings."""
from __future__ import annotations

import logging
import sqlite3


from holomem.engine import text

SECRET = "Masha's PIN is 4821"


def _quoting(exc_type=sqlite3.OperationalError):
    exc = exc_type(f"Could not decode to UTF-8 column 'content' with text '{SECRET}'")
    exc.sqlite_errorname = "SQLITE_ERROR"
    return exc


# -- debug-only failures are warnings ---------------------------------------------------------------------------------


def test_a_damaged_dictionary_is_warned(tmp_path, monkeypatch, caplog):
    bad = tmp_path / "ja_variants.tsv.gz"
    bad.write_bytes(b"not gzip at all")
    monkeypatch.setattr(text, "_JA_FILE", bad)
    monkeypatch.setattr(text, "_JA_TABLE", None)
    with caplog.at_level(logging.WARNING):
        text._ja_table()
    assert "ja_variants.tsv.gz could not be read" in caplog.text
    monkeypatch.setattr(text, "_JA_TABLE", None)  # the next test loads the real one again


# -- after the Codex review of the commit ------------------------------------------------------------------------------


def test_a_new_failure_is_said_however_many_were_said_before(monkeypatch, caplog):
    from holomem.engine import retrieval
    monkeypatch.setattr(retrieval, "_WARNED", set())
    for n in range(75):  # 25 actions x 3 kinds of failure: more than the old cap of 64
        retrieval._warn_once(f"fact_store action{n} failed", KeyError("x"))
    with caplog.at_level(logging.WARNING):
        retrieval._warn_once("memory mirror failed", _quoting())
    assert "memory mirror failed (OperationalError (SQLITE_ERROR))" in caplog.text

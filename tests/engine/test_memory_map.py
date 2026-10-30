"""The memory map export (memory_map.py): a read-only snapshot of ONE store for a person to read.

Requirements, in the person's words, each a test below:
- the map of one store shows that store only: another scope's facts, names and counts never appear in it;
- "forget" means forgotten on the map too: a forgotten fact leaves only its number and the date, no text, no keyword,
  no name it gave;
- a correction is not a past: a corrected version shows no text; a fact the engine's screen would not let into an
  answer (a password, an instruction to the assistant) shows no text either, current or old;
- looking does not touch: the store file is byte for byte the same after an export;
- the night's questions (dreaming) are added when their files are there; missing or broken files leave that part
  empty with a reason, never an error, and a question about a forgotten fact or another store is not shown; the
  night's report and approval queue are the profile store's and never reach a chat's map; one record of another
  shape is left out with its reason, the rest stays; whose a night record is never depends on the folder the export
  runs from, and every text of a night record (numbers and negations included) passes the screen or stays out;
- the map is the store at one moment: a forget during the export is all in it or not at all, in WAL and in
  rollback-journal mode alike;
- "which fact search puts first" counts a fact's age as search does (from its current text, not its last touch);
- the output is JSON a dashboard can render as is: a schema version, the moment of the snapshot, the same order
  every time, from a function and from the command line alike.

The synthetic family below is invented (the names of the design mockup); no real data."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from holomem.engine.memory_map import export_map

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _dump(snapshot: dict) -> str:
    return json.dumps(snapshot, ensure_ascii=False)


# -- the JSON ------------------------------------------------------------------------------------------------------


def test_a_missing_store_is_an_error_and_no_file_is_made(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_map(tmp_path / "nothing.db", now=NOW)
    assert not (tmp_path / "nothing.db").exists()


# -- who is a person ------------------------------------------------------------------------------------------------
# Requirement: the map lists people apart from topics, and never calls a name a person because it is capitalized — a
# month, a brand and a pet are capitalized too. A person only by what the store holds (the name of someone who spoke
# in a chat, or a word for a relative or a role right beside the name in a stored fact, in any case form), a topic
# only by a form no person's name has; anything else is "not determined" (null), not a guess.


# -- what the agent leans on in a contradiction -----------------------------------------------------------------------
# Requirement: for each contradiction the map says which of the two facts search puts first now — by the same score
# search uses (trust, and the decay when the config turns it on; a fact below the trust floor is not found at all) —
# and says nothing (null, with the reason) when that score does not decide: equal, or the reranker reorders by text.


# -- the night's files belong to one store (Codex review, 30.10.2026, P1) ----------------------------------------------
# Requirement: the map of a chat's own store never shows what the night wrote about the profile: the report (MEMORY.md
# entries, its conflicts) and the approval queue are the profile store's, whatever else the files say; of the merge
# questions only the ones that name this very store. A question that names this store does not make the rest of the
# files its own, and a fact number in a file about another store is not this store's fact of that number.


# -- one snapshot (Codex review, 30.10.2026, P2) -------------------------------------------------------------------
# Requirement: the map is the store at one moment. A forget (or a change) that lands while the export reads is either
# all in it or not at all: never the fact's text and its tombstone side by side.

@pytest.mark.parametrize("journal", ["wal", "delete"])  # WAL: the writer goes on; DELETE: its COMMIT waits
def test_a_forget_during_the_export_is_all_in_or_all_out(tmp_path, monkeypatch, journal):
    from holomem.engine import memory_map
    from holomem.engine.store import MemoryStore

    db = tmp_path / "memory_store.db"
    writer = MemoryStore(db, hrr_dim=64)
    assert writer._conn.execute(f"PRAGMA journal_mode = {journal}").fetchone()[0].lower() == journal
    gone = writer.add_fact("Синие стулья клуба стоят в сарае.")
    writer.add_fact("Клуб собирается по пятницам.")
    original = memory_map._Snapshot.rows
    started, errors = [], []

    def forget():
        try:
            writer.forget_fact(gone)  # the store's own forget, in a thread of its own as in the agent
        except Exception as exc:  # noqa: BLE001 - reported by the assert below, not swallowed
            errors.append(exc)

    def rows(snapshot, sql, params=()):
        out = original(snapshot, sql, params)
        if not started and "FROM facts ORDER BY fact_id" in sql:
            # Only the moment is chosen: right after the facts were read, before anything else. The writer then
            # gets as far as the database lets it while the export goes on.
            started.append(threading.Thread(target=forget))
            started[0].start()
            started[0].join(timeout=5)
        return out

    monkeypatch.setattr(memory_map._Snapshot, "rows", rows)
    try:
        snap = export_map(db, now=NOW)
        started[0].join(timeout=60)
    finally:
        monkeypatch.undo()
        writer.close()
    assert started and not started[0].is_alive() and errors == []
    current = {f["id"] for f in snap["FACTS"]} | {h["id"] for h in snap["HIDDEN"]}
    tombs = {t["id"] for t in snap["TOMBS"]}
    assert not current & tombs, "one snapshot holds a live fact and its tombstone"
    after = export_map(db, now=NOW)   # the forget did happen: the next map has only the tombstone
    assert gone in {t["id"] for t in after["TOMBS"]} and gone not in {f["id"] for f in after["FACTS"]}


# -- a night file of another shape (Codex review, 30.10.2026, P2) -------------------------------------------------
# Requirement: one record of the night's files in a shape the map does not know (a list where a record was expected,
# a store that is not a path) is left out and named in the JSON with the reason; the rest of the map, the night's
# good records included, is there. Neither an error nor an empty map that looks fine.


# -- age as search counts it (merge of fix/trust-signals, 0.23.0) --------------------------------------------------
# Requirement: "which fact search puts first" on the map counts a fact's age the way search does: from when its
# current text was recorded. New keywords, tags or a vote on an old fact do not make it young on the map either.


# -- second Codex review of the map (30.10.2026) -------------------------------------------------------------------


def _put(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# P1. Requirement: no text of the night's report reaches the map of a store it is not about. The report is the
# profile's, but an item that names another store (dream names a chat's own database by its key, the file name
# without .db) is that store's: its MEMORY entry or its conflict, of any section, never shows on the profile's map.
# A `store` that is neither a name nor absent is a record of another shape: left out, with the reason.


# P1. Requirement: whose a merge question is never depends on the folder the export runs from. dream writes the
# store a question is about as an absolute path (the store's own `database`); a relative one proves nothing — it
# would name whatever store lies under the current folder — so the question is left out with the reason, and its
# words (extra_words) never reach a map.

def test_a_relative_store_in_a_merge_question_proves_nothing(tmp_path, monkeypatch):
    from holomem.engine.store import MemoryStore

    homes = {}
    for name in ("a", "b"):
        home = tmp_path / name
        home.mkdir()
        s = MemoryStore(home / "memory_store.db", hrr_dim=64)
        try:
            s.add_fact(f"Клуб {name} собирается по пятницам в библиотеке.")
            s.add_fact(f"Клуб {name} собирается по пятницам в библиотеке на втором этаже.")
        finally:
            s.close()
        homes[name] = home
    (homes["a"] / "cache").mkdir()
    day = (NOW - timedelta(days=1)).isoformat(timespec="seconds")
    _put(homes["a"] / "cache" / "dream-proposals.json", {"next_proposal": 2, "proposals": {
        "D-1": {"status": "open", "store": "memory_store.db", "created": day, "last_published": day,
                "keep": {"fact_id": 1, "version_id": 1}, "copy": {"fact_id": 2, "version_id": 2, "covered": False,
                                                                  "extra_words": ["Лиссабон"]}}}})
    bad = [{"file": "proposals", "id": "D-1", "why": "store_relative"}]
    monkeypatch.chdir(homes["b"])   # the scenario of the review: B's map, made from B's folder, with A's night
    of_b = export_map(homes["b"] / "memory_store.db", dreaming_home=homes["a"], now=NOW)
    assert "Лиссабон" not in _dump(of_b)
    assert [p for p in of_b["PENDING"] if p["kind"] == "merge"] == []
    assert of_b["SNAP"]["dreaming"]["invalid"] == bad
    monkeypatch.chdir(homes["a"])   # nor does A's own folder make it A's
    of_a = export_map(homes["a"] / "memory_store.db", dreaming_home=homes["a"], now=NOW)
    assert "Лиссабон" not in _dump(of_a) and [p for p in of_a["PENDING"] if p["kind"] == "merge"] == []
    assert of_a["SNAP"]["dreaming"]["invalid"] == bad


# P1, the other side. Requirement: one store named in two spellings is one store — an absolute path through "..",
# and on Windows (a file system that ignores case) the same path in other letters — so a question about this store
# is not lost because dream spelled the path another way.


# P2. Requirement: every text of a night record passes the engine's screen, not only its entry. The numbers and
# negations of a MEMORY conflict are what dream writes there — a number or a date as text ("5000", "12.03",
# "month:03", "weekday:2"), a negated phrase as a string — and pass the screen; anything else (a structure, a
# sentence among the numbers, a negation the screen refuses) leaves the whole record out, with the field named.
# The same for the other fields that reach the JSON as text: the MEMORY entry's index, the names of the night's
# stats. A number that is not finite never makes the JSON invalid.


# P2. Requirement: a date in the night's files that no calendar holds (1e100 seconds, NaN, a moment past the year
# 9999 once moved to UTC) is that record's problem, never the export's: the map comes out, the queue counts the
# record and names it, its oldest date comes from the records that have one; a question's unreadable date is null;
# a report whose own moment cannot be read (or is no moment at all: text after a date) shows no night and says why.


# Requirement: the order the map says search puts contradictions in comes from holomem's own settings file
# (config.toml, its top-level keys), the same keys and values recall reads: a file that holds them is "config", a
# missing one is "absent" (the defaults run), and one that does not parse says nothing about the order.

def test_the_ranking_is_read_from_holomems_config_toml(tmp_path):
    from holomem.engine.store import MemoryStore

    db = tmp_path / "memory_store.db"
    MemoryStore(db, hrr_dim=64).close()
    config = tmp_path / "config.toml"
    config.write_text('min_trust_threshold = 0.45\ntemporal_decay_half_life = 30\nrerank = true\n', encoding="utf-8")
    ranking = export_map(db, config=config, now=NOW)["SNAP"]["ranking"]
    assert ranking == {"source": "config", "min_trust": 0.45, "half_life": 30, "rerank": True, "meaning": False}
    assert export_map(db, config=tmp_path / "none.toml", now=NOW)["SNAP"]["ranking"]["source"] == "absent"
    config.write_text('min_trust_threshold = = 0.45\n', encoding="utf-8")
    broken = export_map(db, config=config, now=NOW)["SNAP"]["ranking"]
    assert broken["source"] == "unreadable" and broken["min_trust"] is None

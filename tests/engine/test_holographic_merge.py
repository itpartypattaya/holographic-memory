"""0.19.0: `fact_store merge` — two facts that say the same thing become one, without losing either history.
plans/2026-10-25-fact-merge.md (v3) is the contract these tests check.

Requirements, in the user's words:
- a duplicate merged into the fact that stays leaves search, recall and the list; its versions, sources and
  confirmations stay under its own id, and `history` of the keep shows them;
- the text of a merged copy does not come back as a new fact: `add`, auto-extraction, `change`, `update` and a resume
  answer with the keep;
- what held on a date before the merge is still answered, and the copy stops "holding" on the day it was merged;
- the keep keeps its own trust; the group's counts add up once; keywords are joined or replaced as asked;
- one person confirming the same claim twice (once on the copy) raises trust once;
- `forget` of the keep forgets the copies too, a tombstone each; `forget` of a copy leaves the keep as it is;
- an error anywhere in a merge leaves nothing written; the file keeps working after a failed commit;
- a scheduled job merges only what it stored in this session (contract addition of 2026-10-26);
- a file without merges keeps history schema 1; the first merge raises it to 2;
- what an older release did to a merged file (a forget of the keep or of a copy, a row removed by SQL, the copy's
  text added again as a fact of its own) is reconciled on the next writer open: the copies of a forgotten keep are
  erased with a tombstone each and their confirmations with them, a merge row without its copy is dropped, a fact of
  its own is not folded back.

The engine's merge is tested on MemoryStore; what lives in holomem's layer (the tool's keyword joining, refusals and
answers, memory_add's checks, auto-extraction, recall) through holomem's server in this process."""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

pytest.importorskip("numpy")

from holomem.engine.history import ChangedSince, HistoryMixin, NotFound, NotOwned
from holomem.engine.retrieval import FactRetriever
from holomem.engine.store import MemoryStore

DIM = 256
KEEP_TEXT = "Alice Smith lives in Paris near the Louvre"
COPY_TEXT = "Alice Smith is living in Paris by the Louvre zorblat"
THIRD_TEXT = "Alice Smith resides in Paris at the Louvre quxxle"
SETTINGS = f"hrr_dim = {DIM}\n"


class Clock:
    def __init__(self, day: str):
        self.day = day


@pytest.fixture
def clock(monkeypatch):
    c = Clock("2026-10-12")
    monkeypatch.setattr(HistoryMixin, "_now", staticmethod(lambda: f"{c.day} 12:00:00"))
    monkeypatch.setattr(HistoryMixin, "_today", lambda self: c.day)
    return c


def _snapshot(store) -> dict:
    """Everything a refused merge must leave as it was."""
    tables = ("facts", "fact_versions", "fact_entities", "entities", "fact_confirmations", "fact_merges", "meta",
              "fact_embeddings", "memory_banks", "entity_aliases", "forgotten")
    return {t: [tuple(r) for r in store.read(f"SELECT * FROM {t}")] for t in tables}


def _raw_bytes(db_path) -> bytes:
    data = b""
    for suffix in ("", "-wal"):
        try:
            with open(f"{db_path}{suffix}", "rb") as fh:
                data += fh.read()
        except FileNotFoundError:
            pass
    return data.lower()


# --- forget ------------------------------------------------------------------------------------------------------------------


def test_forget_of_the_keep_with_a_blocked_scrub_is_retried(tmp_path, clock):
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    assert store._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
    for i in range(30):
        store.add_fact(f"Person{i} Surname{i} lives in city number {i}")
    keep = store.add_fact("Alice Smith lives in Paris")
    copy = store.add_fact("Alice Smith is living in Paris zorblat")
    store.merge_facts(keep, [copy])
    other = sqlite3.connect(str(db), isolation_level=None)
    cur = other.execute("SELECT fact_id, content FROM facts")
    cur.fetchone()
    try:
        result = store.forget_fact(keep)
    finally:
        cur.close()
        other.close()
    assert result["forgotten"] is True and result["scrub"] == "pending" and result["merged_copies"] == [copy]
    assert store.read("SELECT COUNT(*) FROM forgotten")[0][0] == 2
    again = store.forget_fact(keep)
    assert again["already"] is True and again["scrub"] == "done"
    assert b"zorblat" not in _raw_bytes(db)
    store.close()


# --- a failure leaves nothing written ------------------------------------------------------------------------------------


def test_a_failed_commit_leaves_nothing_and_the_next_merge_works(tmp_path, clock):
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    assert store._conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0].lower() == "delete"
    for i in range(5):
        store.add_fact(f"Alice Smith lives in Paris, flat {i}")
    keep = store.add_fact("Alice Smith lives in Paris")
    copy = store.add_fact("Alice Smith is living in Paris zorblat")
    before = _snapshot(store)
    other = sqlite3.connect(str(db), isolation_level=None)
    cur = other.execute("SELECT fact_id, content FROM facts")
    cur.fetchone()
    store._conn.execute("PRAGMA busy_timeout = 200")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            store.merge_facts(keep, [copy])
        assert not store._conn.in_transaction
    finally:
        cur.close()
        other.close()
    assert _snapshot(store) == before
    assert sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM fact_merges").fetchone()[0] == 0
    assert store.merge_facts(keep, [copy])["merged"] == [copy]
    assert sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM fact_merges").fetchone()[0] == 1
    store.close()


# === the engine (MemoryStore.merge_facts and what reads its result) ======================================================


def _src(session: str = "s1", origin: str = "said", user: str = "") -> dict:
    return {"via": "tool", "platform": "claude-code", "session_id": session, "user_id": user, "origin": origin}


def _add(store, text: str, *, session: str = "s1", origin: str = "said", keywords=(), entities=(),
         valid_from=None, category: str = "general") -> int:
    """A fact as a session's add stores it: its origin's trust, keywords in the tags, the source of its version."""
    tags = "keywords: " + ", ".join(keywords) if keywords else ""
    fact_id, created = store.add_fact_status(text, category=category, tags=tags, entities=list(entities),
                                             valid_from=valid_from, trust=store.start_trust(origin),
                                             source=_src(session, origin))
    assert created, text
    return fact_id


def _pair(store, session: str = "s1") -> tuple[int, int]:
    """Two facts that say the same, with keywords of their own."""
    return (_add(store, KEEP_TEXT, session=session, keywords=["home", "residence"]),
            _add(store, COPY_TEXT, session=session, keywords=["residence", "flat", "квартира"]))


def _found(store, query: str, retriever=None) -> list[int]:
    return [f["fact_id"] for f in (retriever or FactRetriever(store, hrr_dim=DIM)).search(query, min_trust=0)]


def _as_of(store, query: str, day: str, **kw) -> list[dict]:
    return FactRetriever(store, hrr_dim=DIM).search_as_of(query, day, min_trust=0, **kw)["results"]


def _trust(store, fact_id: int) -> float:
    return store.read("SELECT trust_score FROM facts WHERE fact_id = ?", (fact_id,))[0][0]


def _names(store, fact_id: int) -> set[str]:
    return {r[0] for r in store.read("SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = "
                                     "fe.entity_id WHERE fe.fact_id = ?", (fact_id,))}


def _mark(store) -> str:
    return store.read("SELECT value FROM meta WHERE key = 'history_schema'")[0][0]


@pytest.fixture
def store(tmp_path, clock):
    s = MemoryStore(str(tmp_path / "memory_store.db"), hrr_dim=DIM)
    yield s
    s.close()


def _reopen(store) -> MemoryStore:
    """The next writer's open of the same file (the last holder closed: the open migrates and reconciles)."""
    path = store.db_path
    store.close()
    return MemoryStore(str(path), hrr_dim=DIM)


# --- the merge itself ------------------------------------------------------------------------------------------------


def test_merge_three_at_once_and_the_history_lists_them_in_order(store):
    keep, copy = _pair(store)
    third = _add(store, THIRD_TEXT)
    assert store.merge_facts(keep, [copy, third])["merged"] == [copy, third]
    assert [f["fact_id"] for f in store.list_facts()] == [keep]
    assert [m["fact_id"] for m in store.history(keep)["merged_from"]] == [copy, third]


def test_history_of_the_keep_shows_the_copies_with_their_sources_and_confirmations(store):
    keep, copy = _pair(store, session="s1")
    store.record_confirmation(copy, {"session_id": "s2", "user_id": "u2", "user_name": "Bob", "via": "feedback"})
    assert store.merge_facts(keep, [copy])["merged_confirmations"] == 1
    history = store.history(keep)
    assert history["current"] is True and "merged_into" not in history
    assert history["merged_confirmations"] == 1
    (merged,) = history["merged_from"]
    assert merged["fact_id"] == copy and merged["merged_at"] == "2026-10-12 12:00:00"
    assert [v["status"] for v in merged["versions"]] == ["merged"]
    assert merged["versions"][0]["content"] == COPY_TEXT
    assert merged["versions"][0]["source"]["session_id"] == "s1" and merged["versions"][0]["source"]["origin"] == "said"
    assert all("version_id" in v for v in history["versions"] + merged["versions"])
    of_copy = store.history(copy)
    assert of_copy["merged_into"] == keep and of_copy["current"] is False
    assert [v["status"] for v in of_copy["versions"]] == ["merged"]


def test_counts_add_up_once_trust_stays_the_keeps_and_the_same_merge_again_changes_nothing(store):
    keep, copy = _pair(store)
    for _ in range(3):
        store.record_retrievals([copy])
    store.record_retrievals([keep])
    for _ in range(2):
        store.record_feedback(copy, helpful=True)   # the copy: 0.6 -> 0.7, two helpful marks
        store.record_feedback(keep, helpful=False)  # the keep: 0.6 -> 0.4
    keep_trust = _trust(store, keep)
    assert keep_trust == pytest.approx(0.4) and _trust(store, copy) == pytest.approx(0.7)
    result = store.merge_facts(keep, [copy])
    assert result["counts_added"] == {"retrieval": 3, "helpful": 2}
    row = store.read("SELECT retrieval_count, helpful_count, trust_score FROM facts WHERE fact_id = ?", (keep,))[0]
    assert tuple(row) == (4, 2, keep_trust)  # the group's past added, the keep's own trust (not the copy's)
    again = store.merge_facts(keep, [copy])
    assert again["merged"] == [] and again["already"] == [copy]
    assert again["counts_added"] == {"retrieval": 0, "helpful": 0}
    assert tuple(store.read("SELECT retrieval_count, helpful_count FROM facts WHERE fact_id = ?", (keep,))[0]) == (4, 2)
    assert store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 1


# --- the copy's text does not come back ---------------------------------------------------------------------------------


def test_change_update_and_resume_into_the_copys_text_answer_with_the_keep(store):
    from holomem.engine.store import DuplicateContentError

    keep, copy = _pair(store)
    store.merge_facts(keep, [copy])
    other = _add(store, "Alice Smith works at the Louvre")
    assert store.change_fact(other, COPY_TEXT) == {"fact_id": keep, "status": "exists", "ended": other}
    assert store.count_facts() == 1
    third = _add(store, "Alice Smith paints at the Louvre")
    with pytest.raises(DuplicateContentError) as refused:
        store.update_fact(third, content=COPY_TEXT)
    assert refused.value.fact_id == keep
    resumed = store.change_fact(other, COPY_TEXT)  # the ended fact does not come back as a twin
    assert resumed == {"fact_id": keep, "status": "exists"}
    assert store.count_facts() == 2


def test_the_keep_may_take_its_copys_wording_back(store):
    keep, copy = _pair(store)
    store.merge_facts(keep, [copy])
    changed = store.change_fact(keep, COPY_TEXT)
    assert changed["status"] == "changed" and changed["fact_id"] == keep


# --- as_of -----------------------------------------------------------------------------------------------------------


def test_as_of_before_the_merge_shows_the_copy_under_its_own_id_and_after_only_the_keep(store, clock):
    keep = _add(store, "Alice Smith lives in Paris", valid_from="2026-01-01")
    copy = _add(store, "Alice Smith is living in Paris zorblat", valid_from="2026-03-01")
    store.merge_facts(keep, [copy])
    before = store.versions_on("2026-06-01")["facts"]
    assert {f["fact_id"]: f.get("merged_into") for f in before} == {keep: None, copy: keep}
    copy_entry = next(f for f in before if f["fact_id"] == copy)
    assert copy_entry["status"] == "merged" and copy_entry["valid_to"] == "2026-10-12"
    searched = _as_of(store, "Alice Paris", "2026-06-01")
    assert {f["fact_id"]: f.get("merged_into") for f in searched} == {keep: None, copy: keep}
    assert [f["fact_id"] for f in store.versions_on("2026-10-12")["facts"]] == [keep]
    assert [f["fact_id"] for f in _as_of(store, "Alice Paris", "2026-10-12")] == [keep]


def test_after_the_keep_changes_the_copy_does_not_hold_the_old_claim_on(store, clock):
    keep = _add(store, "Alice Smith lives in Paris", valid_from="2026-01-01")
    copy = _add(store, "Alice Smith is living in Paris zorblat", valid_from="2026-01-01")
    store.merge_facts(keep, [copy])
    clock.day = "2026-12-05"
    store.change_fact(keep, "Alice Smith lives in Berlin", valid_from="2026-12-01")
    after = store.versions_on("2026-12-03")
    assert [(f["fact_id"], f["content"]) for f in after["facts"]] == [(keep, "Alice Smith lives in Berlin")]
    assert after["undetermined"] == []
    searched = _as_of(store, "Alice Smith lives", "2026-12-03")
    assert [(f["fact_id"], f["content"]) for f in searched] == [(keep, "Alice Smith lives in Berlin")]


def test_after_the_keep_ends_the_copy_does_not_hold_on(store, clock):
    keep = _add(store, "Alice Smith lives in Paris", valid_from="2026-01-01")
    copy = _add(store, "Alice Smith is living in Paris zorblat", valid_from="2026-01-01")
    store.merge_facts(keep, [copy])
    clock.day = "2026-11-20"
    store.end_fact(keep, valid_to="2026-11-01")
    assert store.versions_on("2026-11-20")["facts"] == []
    assert _as_of(store, "Alice Paris", "2026-11-20") == []
    assert [f["fact_id"] for f in store.versions_on("2026-10-01")["facts"]] == [keep, copy]


def test_as_of_on_the_merge_day_is_not_crowded_out_by_the_merged_copies(store, clock):
    """The merge-day cut applies before ranking: 105 higher-trust copies of a lower-trust keep, merged today, take no
    place in the candidate pool of a search as of today (limit=1 was empty)."""
    keep = _add(store, "Alice Smith lives in Paris", valid_from="2026-01-01")
    copies = [_add(store, f"Alice Smith lives in Paris, note {n} zorblat", valid_from="2026-01-01") for n in range(105)]
    conn = sqlite3.connect(store.db_path)
    conn.execute(f"UPDATE facts SET trust_score = 0.9 WHERE fact_id IN ({','.join('?' * len(copies))})", copies)
    conn.execute("UPDATE facts SET trust_score = 0.4 WHERE fact_id = ?", (keep,))
    conn.commit()
    conn.close()
    for i in range(0, len(copies), 20):
        store.merge_facts(keep, copies[i:i + 20])
    assert [f["fact_id"] for f in _as_of(store, "Alice Smith Paris", clock.day, limit=1)] == [keep]


# --- confirmations -------------------------------------------------------------------------------------------------------


def test_a_person_who_confirmed_the_copy_does_not_raise_the_keeps_trust_again(store):
    keep, copy = _pair(store, session="s1")
    assert store.record_confirmation(copy, {"session_id": "s2", "user_id": "bob"})["trust_score"] == pytest.approx(0.65)
    store.merge_facts(keep, [copy])
    trust = _trust(store, keep)
    again = store.record_confirmation(keep, {"session_id": "s3", "user_id": "bob"})
    assert again["confirmed"] is True and again["trust_score"] == pytest.approx(trust)  # known: no second rise
    assert again["confirmations"] == 1  # the keep's own rows
    same_session = store.record_confirmation(keep, {"session_id": "s2", "user_id": "carol"})  # the copy's session
    assert same_session["confirmed"] is False


def test_the_copys_confirmations_do_not_make_the_keep_confirmed(store):
    keep = _add(store, "Alice Smith lives in Paris", origin="inferred")
    copy = _add(store, "Alice Smith is living in Paris zorblat", origin="inferred")
    store.record_confirmation(copy, {"session_id": "s2", "user_id": "bob"})
    store.merge_facts(keep, [copy])
    assert store.origins([keep])[keep] == {"origin": "inferred", "confirmations": 0}
    assert store.history(keep)["merged_confirmations"] == 1


def test_a_confirmation_of_the_copy_does_not_count_against_the_keeps_new_claim(store, clock):
    """Bob confirmed «Paris» (the copy); after the keep changed to «Berlin», his first confirmation of Berlin is a new
    one and raises trust: change cleared the old claim's feedback."""
    keep, copy = _pair(store, session="s1")
    store.record_confirmation(copy, {"session_id": "s2", "user_id": "bob"})
    store.merge_facts(keep, [copy])
    clock.day = "2026-12-05"
    store.change_fact(keep, "Alice Smith lives in Berlin", valid_from="2026-12-01")
    trust = _trust(store, keep)
    result = store.record_confirmation(keep, {"session_id": "s3", "user_id": "bob"})
    assert result["confirmed"] is True and result["trust_score"] > trust, result


# --- entities --------------------------------------------------------------------------------------------------------------


def test_the_copys_names_move_to_the_keep_and_its_text_aliases_go(store):
    keep = _add(store, "Alice Smith lives in Paris")
    copy = _add(store, "Alice Smith aka Ally Smith lives in Paris with Bob Brown", entities=["Masha"])
    assert store.read("SELECT source FROM entity_aliases WHERE alias = 'Ally Smith'")[0][0] == "aka"
    store.merge_facts(keep, [copy])
    assert {"Alice Smith", "Bob Brown", "Masha"} <= _names(store, keep)
    assert store.read("SELECT entity_type FROM entities WHERE name = 'Masha'")[0][0] == "declared"
    assert FactRetriever(store, hrr_dim=DIM).probe("Bob Brown", min_trust=0)[0]["fact_id"] == keep
    # an alias only the copy's text gave goes with its row; the text stays in the copy's history
    assert store.read("SELECT COUNT(*) FROM entity_aliases WHERE alias = 'Ally Smith'")[0][0] == 0
    assert "Ally Smith" in store.history(keep)["merged_from"][0]["versions"][0]["content"]
    assert store.read("SELECT COUNT(*) FROM fact_entities WHERE fact_id = ?", (copy,))[0][0] == 0


# --- counts, chains -------------------------------------------------------------------------------------------------------


def test_history_counts_tell_merged_facts_from_ended_ones(store):
    keep = _add(store, "Alice Smith lives in Paris")
    copy = _add(store, "Alice Smith is living in Paris zorblat")
    store.end_fact(copy)
    assert store.history_counts()["ended_facts"] == 1
    assert store.change_fact(copy, "Alice Smith is living in Paris zorblat")["status"] == "resumed"
    store.merge_facts(keep, [copy])
    counts = store.history_counts()
    assert counts["merged_facts"] == 1 and counts["ended_facts"] == 0 and counts["past_versions"] == 1


def test_a_chain_is_rebound_to_the_root(store):
    a = _add(store, "Alice Smith lives in Paris")
    b = _add(store, "Alice Smith is living in Paris zorblat")
    c = _add(store, "Alice Smith resides in Paris quxxle")
    store.merge_facts(a, [b])
    assert store.merge_facts(c, [a])["merged"] == [a]
    assert {tuple(r) for r in store.read("SELECT from_fact_id, into_fact_id FROM fact_merges")} == {(a, c), (b, c)}
    assert store.history(b)["merged_into"] == c
    assert [m["fact_id"] for m in store.history(c)["merged_from"]] == [a, b]
    assert store.merged_match("Alice Smith is living in Paris zorblat")["fact_id"] == c


def test_a_chain_through_a_changed_keep_leaves_the_old_copy_a_past_claim(store, clock):
    """A «Paris» <- B «Paris»; A changes to «Berlin»; then C «Berlin» <- A. B said Paris: after the second merge it is
    still a past claim, not C's current one, and Bob's confirmation of B does not count as one of Berlin. A copy
    merged into A's current (Berlin) version does follow A into C's current claim."""
    a = _add(store, "Alice Smith lives in Paris")
    b = _add(store, "Alice Smith is living in Paris zorblat")
    store.record_confirmation(b, {"session_id": "s2", "user_id": "bob"})
    store.merge_facts(a, [b])
    clock.day = "2026-12-05"
    store.change_fact(a, "Alice Smith lives in Berlin", valid_from="2026-12-01")
    d = _add(store, "Alice Smith is living in Berlin quxxle")
    store.merge_facts(a, [d])
    c = _add(store, "Alice Smith resides in Berlin now")
    assert store.merged_match("Alice Smith is living in Paris zorblat") is None
    assert store.merge_facts(c, [a])["merged"] == [a]
    assert store.merged_match("Alice Smith is living in Paris zorblat") is None
    past = store.past_version("Alice Smith is living in Paris zorblat")
    assert past is not None and past["fact_id"] == a and past["status"] == "changed"
    assert store.merged_match("Alice Smith is living in Berlin quxxle")["fact_id"] == c
    trust = _trust(store, c)
    result = store.record_confirmation(c, {"session_id": "s3", "user_id": "bob"})
    assert result["confirmed"] is True and result["trust_score"] > trust, result  # Paris was not a vote for Berlin


# --- refusals: nothing changes ---------------------------------------------------------------------------------------------


def test_the_keep_in_from_a_copy_as_keep_and_an_ended_fact_are_refused_without_changes(store):
    keep, copy = _pair(store)
    third = _add(store, THIRD_TEXT)
    store.merge_facts(keep, [copy])
    ended = _add(store, "Alice Smith works at the Louvre")
    store.end_fact(ended)
    other = _add(store, "Alice Smith lives in Paris, France")
    before = _snapshot(store)
    for args, said in (((keep, [keep, third]), "into itself"), ((copy, [third]), f"was merged into #{keep}"),
                       ((keep, [third, ended]), "has ended"), ((ended, [third]), "has ended"),
                       ((other, [copy]), f"was merged into #{keep}")):
        with pytest.raises(ValueError, match=said):
            store.merge_facts(*args)
    assert _snapshot(store) == before


def test_a_store_opened_without_migrating_refuses_to_merge(tmp_path, clock):
    db = tmp_path / "memory_store.db"
    writer = MemoryStore(str(db), hrr_dim=DIM)
    keep, copy = _pair(writer)
    writer.close()
    reader = MemoryStore(str(db), hrr_dim=DIM, migrate=False)
    try:
        with pytest.raises(ValueError, match="read-only"):
            reader.merge_facts(keep, [copy])
        assert reader.count_facts() == 2 and reader.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
    finally:
        reader.close()


def test_a_merge_limited_to_one_session_merges_only_what_that_session_stored(store):
    """A scheduled job merges only what it stored in this session: checked in the merge's own transaction."""
    keep, old = _pair(store, session="main")
    new = _add(store, THIRD_TEXT, session="cron1")
    before = _snapshot(store)
    with pytest.raises(NotOwned) as refused:
        store.merge_facts(keep, [old, new], session_only="cron1")
    assert refused.value.ids == [old]
    assert _snapshot(store) == before
    assert store.merge_facts(keep, [new], session_only="cron1")["merged"] == [new]


def test_a_copy_another_session_changed_is_no_longer_the_sessions_to_merge(store):
    keep = _add(store, KEEP_TEXT, session="main")
    new = _add(store, THIRD_TEXT, session="cron1")
    store.change_fact(new, "Alice Smith resides in Paris at the Louvre, main says", source=_src("main"))
    with pytest.raises(NotOwned):
        store.merge_facts(keep, [new], session_only="cron1")
    assert store.read("SELECT content FROM facts WHERE fact_id = ?", (new,))[0][0].endswith("main says")
    assert store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0


def test_expect_refuses_a_version_the_caller_did_not_see(store):
    keep, copy = _pair(store)
    seen = {keep: store._current_version(keep)["version_id"], copy: store._current_version(copy)["version_id"]}
    store.change_fact(copy, "Alice Smith is living in Paris by the Louvre, she said")
    before = _snapshot(store)
    with pytest.raises(ChangedSince) as refused:
        store.merge_facts(keep, [copy], expect=seen)
    assert refused.value.current == {copy: store._current_version(copy)["version_id"]}
    assert _snapshot(store) == before
    seen[copy] = store._current_version(copy)["version_id"]
    assert store.merge_facts(keep, [copy], expect=seen)["merged"] == [copy]
    assert store.merge_facts(keep, [copy], expect=seen)["merged"] == []  # the same again: the version it saw, merged


# --- forget ------------------------------------------------------------------------------------------------------------------


def test_forget_of_the_keep_forgets_the_copies_with_a_tombstone_each(store):
    keep, copy = _pair(store)
    third = _add(store, THIRD_TEXT)
    for fact_id, user in ((keep, "u1"), (copy, "u2"), (third, "u3")):  # each with a person's confirmation
        store.record_confirmation(fact_id, {"session_id": f"s-{user}", "user_id": user, "via": "feedback"})
    store.merge_facts(keep, [copy, third])
    assert store.read("SELECT COUNT(*) FROM fact_confirmations")[0][0] == 3
    assert b"zorblat" in _raw_bytes(store.db_path)
    result = store.forget_fact(keep)
    assert result["forgotten"] is True and result["versions"] == 3 and result["scrub"] == "done"
    assert result["merged_copies"] == [copy, third]
    for fact_id in (keep, copy, third):
        assert store.read("SELECT COUNT(*) FROM forgotten WHERE fact_id = ?", (fact_id,))[0][0] == 1
        assert store.history(fact_id)["forgotten"] is True
    for text in (COPY_TEXT, THIRD_TEXT, KEEP_TEXT):
        assert store.forgotten_match(text) is not None
    assert store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
    for table in ("facts", "fact_versions", "fact_entities", "fact_confirmations"):
        assert store.read(f"SELECT COUNT(*) FROM {table}")[0][0] == 0
    assert store.read("SELECT COUNT(*) FROM entities WHERE name = 'Alice Smith'")[0][0] == 0
    for word in (b"zorblat", b"quxxle", b"louvre"):
        assert word not in _raw_bytes(store.db_path)


# --- a failure leaves nothing written ------------------------------------------------------------------------------------


def test_a_failure_on_the_second_copy_rolls_the_whole_merge_back(store, monkeypatch):
    keep, copy = _pair(store)
    third = _add(store, THIRD_TEXT)
    retriever = FactRetriever(store, hrr_dim=DIM)
    assert _found(store, "quxxle", retriever) == [third]  # the retriever caches the texts
    before = _snapshot(store)
    original = MemoryStore._close_version
    third_version = store._current_version(third)["version_id"]

    def failing(self, version_id, status, **kw):  # the second copy: the first is closed and gone already
        if version_id == third_version and status == "merged":
            raise sqlite3.OperationalError("disk I/O error")
        return original(self, version_id, status, **kw)

    monkeypatch.setattr(MemoryStore, "_close_version", failing)
    with pytest.raises(sqlite3.OperationalError):
        store.merge_facts(keep, [copy, third])
    monkeypatch.setattr(MemoryStore, "_close_version", original)
    assert not store._conn.in_transaction
    assert _snapshot(store) == before
    assert _mark(store) == "1"
    assert sorted(_found(store, "Alice Louvre", retriever)) == [keep, copy, third]
    assert _found(store, "zorblat", retriever) == [copy]
    assert store.read("SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH 'zorblat'")[0][0] == 1
    reopened = _reopen(store)
    try:
        assert reopened.count_facts() == 3 and _snapshot(reopened)["fact_versions"] == before["fact_versions"]
        assert reopened.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
    finally:
        reopened.close()


def test_merge_is_serialised_with_the_other_operations(store):
    """Two threads: a merge and searches on the same store do not interleave inside it."""
    keep, copy = _pair(store)
    retriever = FactRetriever(store, hrr_dim=DIM)
    errors = []

    def run():
        try:
            for _ in range(5):
                retriever.search("Alice Louvre", min_trust=0)
        except Exception as exc:  # noqa: BLE001 - reported by the assert below
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.01)
    result = store.merge_facts(keep, [copy])
    thread.join()
    assert result["merged"] == [copy] and not errors, errors
    assert _found(store, "Alice Louvre", retriever) == [keep]
    with pytest.raises(NotFound):
        store.merge_facts(keep, [99999])


# --- the schema mark and what an older release left ----------------------------------------------------------------------


def test_the_schema_mark_rises_with_the_first_merge_not_on_open(store):
    assert _mark(store) == "1"
    keep, copy = _pair(store)
    store = _reopen(store)
    assert _mark(store) == "1"
    store.merge_facts(keep, [copy])
    assert _mark(store) == "2"
    store.forget_fact(keep)  # the mark stays: it says what the file may hold
    store = _reopen(store)
    try:
        assert _mark(store) == "2"
    finally:
        store.close()


def test_reconciliation_erases_the_copies_of_a_keep_an_older_release_removed(store):
    """A keep whose row a release without history deleted (facts only): the reconciliation erases it with a tombstone,
    and its merged copies the same way — at the latest on the next writer open."""
    keep, copy = _pair(store)
    store.merge_facts(keep, [copy])
    store._write("DELETE FROM fact_entities WHERE fact_id = ?", (keep,))
    store._write("DELETE FROM facts WHERE fact_id = ?", (keep,))
    orphan = _add(store, "Bob Brown lives in Rome")
    twin = _add(store, "Bob Brown is living in Rome quxxle")
    store.merge_facts(orphan, [twin])
    store._write("DELETE FROM fact_versions WHERE fact_id = ?", (twin,))  # an older forget(copy) left the merge row
    store = _reopen(store)
    try:
        assert {r[0] for r in store.read("SELECT fact_id FROM forgotten")} == {keep, copy}
        assert store.read("SELECT COUNT(*) FROM fact_versions WHERE fact_id IN (?, ?)", (keep, copy))[0][0] == 0
        assert [tuple(r) for r in store.read("SELECT from_fact_id, into_fact_id FROM fact_merges")] == []
        assert store.forgotten_match(COPY_TEXT) is not None
        assert b"zorblat" not in _raw_bytes(store.db_path)
        assert store.history(orphan)["current"] is True
    finally:
        store.close()


def _older_writer_forgets(db, fact_id: int, digest: str) -> None:
    """What a release before merges (0.8-0.18) does to forget a fact, on a file this release merged: the row, its
    links and its versions go, a tombstone is written; it knows nothing of fact_merges, so the copies stay."""
    conn = sqlite3.connect(str(db))
    with conn:
        conn.execute("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
        conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
        conn.execute("DELETE FROM fact_versions WHERE fact_id = ?", (fact_id,))
        conn.execute("INSERT INTO forgotten (fact_id, forgotten_at, digests) VALUES (?, ?, ?)",
                     (fact_id, "2026-10-12 12:00:00", f'["{digest}"]'))
    conn.close()


@pytest.mark.parametrize("also", ["nothing", "a confirmed copy", "a row removed by sql"])
def test_an_older_writer_that_forgot_the_keep_leaves_the_copies_to_the_next_open(store, also):
    """A writer of an older release, open before the merge raised the mark, forgets the keep: its copy's text stays
    in the file. This release's next writer open erases the copy too (a tombstone for it, its confirmations with it)
    and rewrites the file — also when the same open finds a fact whose row an older release removed by SQL."""
    keep, copy = _pair(store)
    if also == "a confirmed copy":
        store.record_confirmation(copy, {"session_id": "s9", "user_id": "u2", "via": "feedback"})
    other = _add(store, "Bob Brown lives in Rome") if also == "a row removed by sql" else None
    store.merge_facts(keep, [copy])
    digest, db = store.digest(KEEP_TEXT), store.db_path
    store.close()
    _older_writer_forgets(db, keep, digest)
    if other is not None:
        conn = sqlite3.connect(str(db))
        with conn:  # a release without history removes the row only (its remove)
            conn.execute("DELETE FROM fact_entities WHERE fact_id = ?", (other,))
            conn.execute("DELETE FROM facts WHERE fact_id = ?", (other,))
        conn.close()
    assert b"zorblat" in _raw_bytes(db)
    store = MemoryStore(str(db), hrr_dim=DIM)
    try:
        assert {r[0] for r in store.read("SELECT fact_id FROM forgotten")} == {keep, copy} | ({other} if other else set())
        assert store.read("SELECT COUNT(*) FROM fact_versions")[0][0] == 0
        assert store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
        assert store.read("SELECT COUNT(*) FROM fact_confirmations")[0][0] == 0
        assert store.forgotten_match(COPY_TEXT) is not None
        assert store.read("SELECT COUNT(*) FROM meta WHERE key = 'scrub_pending'")[0][0] == 0  # rewritten on open
        assert b"zorblat" not in _raw_bytes(db)
    finally:
        store.close()


def test_a_merge_row_whose_copy_an_older_writer_forgot_is_dropped_without_a_word(store):
    keep, copy = _pair(store)
    store.merge_facts(keep, [copy])
    digest, db = store.digest(COPY_TEXT), store.db_path
    store.close()
    _older_writer_forgets(db, copy, digest)  # its forget of the copy leaves the merge row
    store = MemoryStore(str(db), hrr_dim=DIM)
    try:
        assert store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
        assert store.count_facts() == 1 and store.history(keep)["current"] is True
        assert "merged_from" not in store.history(keep)
        assert store.merge_facts(keep, [_add(store, THIRD_TEXT)])["merged"]  # merging works on
    finally:
        store.close()


# === holomem's layer (its tools over the engine, recall, auto-extraction) ====================================================


@pytest.fixture
def memory(tmp_path, clock, holomem_server):
    return holomem_server(tmp_path / "holo", SETTINGS)


def _tool_add(server, content: str, **extra) -> int:
    answer, is_error = server.call("memory_add", content=content, **extra)
    assert not is_error and answer["status"] == "added", answer
    return answer["fact_id"]


def _tool_pair(server) -> tuple[int, int]:
    return (_tool_add(server, KEEP_TEXT, origin="said", keywords=["home", "residence"]),
            _tool_add(server, COPY_TEXT, origin="said", keywords=["residence", "flat", "квартира"]))


def _merge(server, keep: int, ids: list, **extra) -> dict:
    answer, is_error = server.call("memory_merge", fact_id=keep, **{"from": ids}, **extra)
    assert not is_error, answer
    return answer


def _search(server, query: str) -> list[int]:
    return [f["fact_id"] for f in server.call("memory_search", query=query, min_trust=0)[0]["results"]]


def test_merge_two_keeps_one_fact_and_the_copy_leaves_search_recall_and_the_list(memory):
    keep, copy = _tool_pair(memory)
    assert sorted(_search(memory, "Alice Louvre")) == [keep, copy]
    result = _merge(memory, keep, [copy])
    assert result["merged"] == [copy] and result["into"] == keep
    assert _search(memory, "Alice Louvre") == [keep]
    assert _search(memory, "zorblat") == []  # a word only the copy had finds nothing current
    assert [f["fact_id"] for f in memory.call("memory_list")[0]["results"]] == [keep]
    assert "zorblat" not in memory.recall("Where is Alice Smith living, by the Louvre zorblat?")
    assert memory.call("memory_probe", entity="Alice Smith")[0]["count"] == 1
    store = memory.store
    assert store.read("SELECT status FROM fact_versions WHERE fact_id = ?", (copy,))[0][0] == "merged"
    row = store.read("SELECT into_fact_id, into_version_id FROM fact_merges WHERE from_fact_id = ?", (copy,))[0]
    assert row[0] == keep and row[1] == store._current_version(keep)["version_id"]
    assert store.read("SELECT COUNT(*) FROM facts WHERE fact_id = ?", (copy,))[0][0] == 0


def test_keywords_are_joined_the_keeps_first_or_replaced_or_cleared(memory):
    keep, copy = _tool_pair(memory)
    result = _merge(memory, keep, [copy])
    assert result["keywords"] == ["home", "residence", "flat", "квартира"]  # the keep's first, no repeats
    assert memory.store.fact_tags(keep).endswith("keywords: home, residence, flat, квартира")
    assert _search(memory, "квартира") == [keep]  # the copy's keyword now finds the keep
    third = _tool_add(memory, "Alice Smith is a Paris resident at the Louvre")
    result = _merge(memory, keep, [third], keywords=["dwelling", "address"])
    assert result["keywords"] == ["dwelling", "address"]
    assert memory.store.fact_tags(keep).endswith("keywords: dwelling, address")
    fourth = _tool_add(memory, "Alice Smith has her home in Paris at the Louvre")
    result = _merge(memory, keep, [fourth], keywords=[])
    assert result["keywords"] == [] and memory.store.fact_tags(keep) == ""
    assert memory.store._current_version(keep)["tags"] == ""  # the version row follows (update does the same)


def test_keywords_past_the_limit_keep_the_keeps_first(memory):
    keep = _tool_add(memory, "Mom's birthday is May 3", keywords=[f"k{i}" for i in range(8)])
    copy = _tool_add(memory, "Mom was born on May 3", keywords=[f"c{i}" for i in range(8)])
    result = _merge(memory, keep, [copy])
    assert result["keywords"] == [f"k{i}" for i in range(8)] + ["c0", "c1", "c2", "c3"]
    assert memory.store.fact_tags(keep) == "keywords: " + ", ".join(result["keywords"])


def test_the_keep_keeps_its_origin_category_and_user_tags(memory):
    keep = _tool_add(memory, KEEP_TEXT, origin="inferred", category="user_pref")
    copy = _tool_add(memory, "Alice Smith is living in Paris by the Louvre", origin="said", category="general",
                     keywords=["flat"])
    conn = sqlite3.connect(memory.store.db_path)  # tags of its own another program gave the keep
    with conn:
        conn.execute("UPDATE facts SET tags = 'family' WHERE fact_id = ?", (keep,))
    conn.close()
    _merge(memory, keep, [copy])
    fact = memory.store.read("SELECT category, tags FROM facts WHERE fact_id = ?", (keep,))[0]
    assert fact[0] == "user_pref" and fact[1] == "family\nkeywords: flat"
    assert memory.store.origins([keep])[keep]["origin"] == "inferred"  # the copy's "said" does not rewrite it


def test_add_of_the_copys_text_answers_with_the_keep_and_its_names_count_for_the_keep(memory):
    keep, copy = _tool_pair(memory)
    _merge(memory, keep, [copy])
    answer, is_error = memory.call("memory_add", content=COPY_TEXT, origin="said", entities=["Nina Petrova"])
    assert not is_error and answer["status"] == "merged" and answer["fact_id"] == keep and answer["stored"] is False
    assert memory.store.count_facts() == 1
    # who the text is about still counts, as for a text a current fact already says (add) or one a change runs into
    probed = memory.call("memory_probe", entity="Nina Petrova")[0]
    assert [f["fact_id"] for f in probed["results"]] == [keep], probed


def test_an_earlier_changed_version_of_the_copy_is_a_past_version_and_a_corrected_one_is_nothing(memory):
    keep = _tool_add(memory, "Alice Smith lives in Paris")
    copy = _tool_add(memory, "Alice Smith lived in Lyon")
    assert not memory.call("memory_change", fact_id=copy, content="Alice Smith is living in Paris")[1]
    assert not memory.call("memory_correct", fact_id=copy, content="Alice Smith is living in Paris, France")[1]
    _merge(memory, keep, [copy])
    past = memory.call("memory_add", content="Alice Smith lived in Lyon")[0]
    assert past["status"] == "past_version" and past["fact_id"] == copy
    gone = memory.call("memory_add", content="Alice Smith is living in Paris")[0]  # corrected: it never held
    assert gone["status"] == "added"
    head = memory.call("memory_add", content="Alice Smith is living in Paris, France")[0]
    assert head["status"] == "merged" and head["fact_id"] == keep


def test_after_the_keep_changes_the_copys_text_is_a_past_claim_not_the_new_one(memory, clock):
    """«Paris» merged into a keep that later became «Berlin»: adding the Paris text is the keep's past claim, not its
    current one (which would confirm Berlin by the words of Paris), and changing another fact into that text does
    not end it."""
    keep, copy = _tool_pair(memory)
    _merge(memory, keep, [copy])
    clock.day = "2026-12-05"
    assert not memory.call("memory_change", fact_id=keep, content="Alice Smith lives in Berlin",
                           valid_from="2026-12-01")[1]
    other = _tool_add(memory, "Bob Jones lives in Rome")
    confirmations = memory.store.read("SELECT COUNT(*) FROM fact_confirmations WHERE fact_id = ?", (keep,))[0][0]
    result = memory.call("memory_add", content=COPY_TEXT, origin="said")[0]
    assert result["status"] == "past_version" and result["fact_id"] == keep, result
    assert memory.store.read("SELECT COUNT(*) FROM fact_confirmations WHERE fact_id = ?", (keep,))[0][0] == confirmations
    changed, _ = memory.call("memory_change", fact_id=other, content=COPY_TEXT)
    assert changed.get("status") != "exists", changed  # the Berlin fact does not hold that text
    assert memory.store.read("SELECT 1 FROM facts WHERE fact_id = ?", (other,)), "the other fact must not be ended"


def test_the_write_check_points_a_near_duplicate_at_the_keep(memory):
    keep, copy = _tool_pair(memory)
    _merge(memory, keep, [copy])
    result = memory.call("memory_add", content=COPY_TEXT + ", she says")[0]
    merged = [p for p in result.get("past_versions", []) if p["fact_id"] == copy]
    assert merged and merged[0]["status"] == "merged" and merged[0]["merged_into"] == keep, result
    assert "until" not in merged[0]


def test_add_with_a_similar_fact_points_to_merge(memory):
    """A new fact that says what a stored one says (a retelling) is what memory_merge is for: the answer that lists
    the similar fact names it, with the new fact as the one to merge."""
    old = _tool_add(memory, KEEP_TEXT, origin="said")
    result = memory.call("memory_add", content=KEEP_TEXT + " museum", origin="said")[0]
    assert result["similar"][0]["fact_id"] == old, result
    assert "memory_merge" in result.get("hint", ""), result["hint"]


@pytest.mark.parametrize("args, said", [
    ({"from": []}, "needs 'from'"),
    ({"from": ["x"]}, "needs 'from'"),
    ({"from": [1, 1]}, "twice"),
    ({"from": [999]}, "not found"),
    ({"from": list(range(100, 121))}, "at most 20"),
])
def test_a_bad_from_is_refused_without_changes(memory, args, said):
    keep, _ = _tool_pair(memory)
    before = _snapshot(memory.store)
    answer, is_error = memory.call("memory_merge", fact_id=keep, **args)
    assert is_error and said in answer["error"], answer
    assert _snapshot(memory.store) == before


def test_forget_of_a_copy_erases_its_history_only_and_its_text_may_be_said_again(memory):
    keep, copy = _tool_pair(memory)
    _merge(memory, keep, [copy])
    result = memory.call("memory_forget", fact_id=copy)[0]
    assert result["forgotten"] is True and result["versions"] == 1 and result["merged_into"] == keep
    store = memory.store
    assert store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
    assert store.read("SELECT COUNT(*) FROM fact_versions WHERE fact_id = ?", (copy,))[0][0] == 0
    history = memory.call("memory_history", fact_id=keep)[0]
    assert history["current"] is True and "merged_from" not in history
    assert _search(memory, "Alice Louvre") == [keep]
    assert b"zorblat" not in _raw_bytes(store.db_path)
    assert memory.call("memory_add", content=COPY_TEXT)[0]["status"] == "added"  # the person says it again


def test_merge_keeps_tags_and_keywords_another_writer_gave_the_keep_after_they_were_read(memory, monkeypatch):
    """The keep's tags are joined inside the merge's transaction, so a tag or keyword another writer gave the keep a
    moment earlier (no new version: expect cannot see it) is not overwritten."""
    keep = _tool_add(memory, KEEP_TEXT, keywords=["home"])
    copy = _tool_add(memory, COPY_TEXT, keywords=["flat"])
    original, fired = MemoryStore.merge_facts, []

    def merge_facts(self, *args, **kw):  # another writer lands between the tool's call and the store's transaction
        if not fired:
            fired.append(True)
            assert self.update_fact(keep, tags="newtag\nkeywords: home, residence", touch=False)
        return original(self, *args, **kw)

    monkeypatch.setattr(MemoryStore, "merge_facts", merge_facts)
    result = _merge(memory, keep, [copy])
    assert fired and result["merged"] == [copy], result
    assert memory.store.fact_tags(keep) == "newtag\nkeywords: home, residence, flat"
    assert memory.store._current_version(keep)["tags"] == "newtag\nkeywords: home, residence, flat"
    assert result["keywords"] == ["home", "residence", "flat"]


@pytest.mark.parametrize("how", ["merged", "change", "end", "correct", "forget"])
def test_auto_extraction_does_not_store_the_copys_text_again(memory, clock, how):
    """auto_extract stores a stated preference from the person's prompt. The text of a copy merged into a fact is
    that fact's claim (or, once the fact changed, ended or was corrected, its past claim; once forgotten, forgotten):
    it is never stored again as a new fact."""
    from holomem.extract import extract

    keep = _tool_add(memory, "I prefer green tea in the morning")
    copy = _tool_add(memory, "I prefer green tea every morning zorblat")
    _merge(memory, keep, [copy])
    clock.day = "2026-12-05"
    if how == "end":
        assert not memory.call("memory_end", fact_id=keep)[1]
    elif how == "forget":
        assert memory.call("memory_forget", fact_id=keep)[0]["merged_copies"] == [copy]
    elif how != "merged":
        assert not memory.call(f"memory_{how}", fact_id=keep, content="I prefer orange juice at lunch")[1]
    before = memory.store.count_facts()
    assert extract(memory.server.memory, memory.server.ctx, "I prefer green tea every morning zorblat") == []
    assert memory.store.count_facts() == before
    assert memory.store.read("SELECT COUNT(*) FROM facts WHERE content LIKE '%green tea every morning%'")[0][0] == 0


def test_a_fact_an_older_release_added_with_the_copys_text_is_not_folded_back(memory):
    """A release without history knows no merge: it adds the copy's text again as a second fact. That fact is not
    folded back (the boundary), and the current fact wins over the merged copy when the text comes again."""
    keep, copy = _tool_pair(memory)
    _merge(memory, keep, [copy])
    conn = sqlite3.connect(memory.store.db_path)
    with conn:  # its add: a row in facts (the FTS trigger is the file's), no version
        twin = conn.execute("INSERT INTO facts (content, category, tags, trust_score) VALUES (?, 'general', '', 0.5)",
                            (COPY_TEXT,)).lastrowid
    conn.close()
    assert twin > copy and memory.store.count_facts() == 2
    result = memory.call("memory_add", content=COPY_TEXT)[0]
    assert result["status"] == "exists" and result["fact_id"] == twin
    assert [v["status"] for v in memory.call("memory_history", fact_id=twin)[0]["versions"]] == ["current"]
    assert memory.call("memory_history", fact_id=copy)[0]["merged_into"] == keep

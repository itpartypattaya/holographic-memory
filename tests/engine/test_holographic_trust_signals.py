"""Two trust signals that said more than they knew (F1 research, plans/2026-10-29-f1-reliability-usefulness-research.md
§1 and §7, approved 2026-10-30).

Requirements, in the user's words:
- Feedback on a fact (helpful, unhelpful) is a vote on the fact, not a new fact: it does not make an old fact look
  recent. With temporal decay on, an old fact marked unhelpful must not jump above a recent one; a fact marked
  helpful rises only by its trust, not by its age. Decay counts from the date the current version was recorded, so a
  store whose facts an earlier release already made "young" by feedback ranks them by their real age again.
- A correction of the text (`correct content=`, `update content=`) is a new claim: it starts at its origin's trust,
  as `change` does, without the helpful marks, recall count or confirmations the wrong text earned. A correction of
  the date only keeps all of them. `revert` brings the old text back with the trust it had."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from holomem.engine.retrieval import FactRetriever
from holomem.engine.store import MemoryStore


def _ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


# -- feedback does not make a fact young -------------------------------------------------------------------------------


def test_without_history_decay_still_reads_the_last_change(tmp_path):
    """A store no writer of 0.8.0 has migrated (a reader opened it first) has no versions: decay counts from
    updated_at, which feedback no longer moves."""
    store = MemoryStore(db_path=str(tmp_path / "plain.db"), default_trust=0.5, hrr_dim=64, migrate=False)
    try:
        assert not store.history_ready()
        old = store.add_fact_status("Irma swims on Sundays at the city pool")[0]
        new = store.add_fact_status("Irma swims on Tuesdays at the school pool")[0]
        store._write("UPDATE facts SET created_at = ?, updated_at = ? WHERE fact_id = ?", (_ago(10), _ago(10), old))
        store._write("UPDATE facts SET created_at = ?, updated_at = ? WHERE fact_id = ?", (_ago(2), _ago(2), new))
        store._conn.commit()
        store.record_feedback(old, helpful=False)
        found = FactRetriever(store, temporal_decay_half_life=1, hrr_dim=64).search("Irma swims pool", min_trust=0.0)
        assert [f["fact_id"] for f in found] == [new, old]
    finally:
        store.close()


# -- a corrected text starts over --------------------------------------------------------------------------------------


@pytest.mark.parametrize("origin, start", [("inferred", 0.4), ("said", 0.6)])
def test_without_history_a_new_text_starts_at_the_origin_it_names(tmp_path, origin, start):
    """A store no writer of 0.8.0 has migrated keeps no versions, but the origin a correction names still sets where
    its new text starts (Codex review: it fell back to the default 0.5)."""
    store = MemoryStore(db_path=str(tmp_path / "plain.db"), default_trust=0.5, hrr_dim=64, migrate=False)
    try:
        assert not store.history_ready()
        fact = store.add_fact_status("Irma's school starts at 8")[0]
        store.update_fact(fact, content="Irma's school starts at 9", source={"origin": origin})
        trust = store._one("SELECT trust_score FROM facts WHERE fact_id = ?", (fact,))["trust_score"]
        assert trust == pytest.approx(start)
    finally:
        store.close()

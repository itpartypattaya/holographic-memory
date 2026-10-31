"""contradict compares every current fact, not only the 500 most recently changed, and facts that name no one too.

Seen 2026-10-31 (review of memory_contradict): a conflict between two facts older than the newest 500 was never found,
with no sign in the answer; and two facts about a thing nobody linked as a name ("The billing service deploys with
systemd" / "... with docker") were never compared. A fact with no name has no subject to take out of its claim, so
such a pair qualifies only when the two say nearly the same and differ in a detail (or in polarity)."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("numpy")

from holomem.engine.retrieval import FactRetriever
from holomem.engine.store import MemoryStore

DIM = 256


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def pairs(found) -> list[set[int]]:
    return [{c["fact_a"]["fact_id"], c["fact_b"]["fact_id"]} for c in found]


def test_a_contradiction_older_than_the_newest_500_facts_is_found(store):
    paris = store.add_fact("Alice Smith lives in Paris")
    london = store.add_fact("Alice Smith lives in London")
    store._write("UPDATE facts SET created_at = '2024-01-01 10:00:00', updated_at = '2024-01-01 10:00:00' "
                 "WHERE fact_id IN (?, ?)", (paris, london))
    for i in range(520):
        store.add_fact(f"Visitor{i} Number{i} signed the guest book on day {i}")
    assert pairs(FactRetriever(store, hrr_dim=DIM).contradict()) == [{paris, london}]


def test_facts_that_name_no_one_and_differ_in_one_detail_are_a_pair(store):
    systemd = store.add_fact("the billing service deploys with systemd")
    docker = store.add_fact("the billing service deploys with docker")
    store.add_fact("the billing service writes its logs to loki")  # same thing, another topic
    found = FactRetriever(store, hrr_dim=DIM).contradict()
    assert {systemd, docker} in pairs(found), found
    assert all(c["shared_entities"] == [] for c in found)


def test_facts_that_name_no_one_and_share_only_a_topic_are_no_pair(store):
    store.add_fact("the billing service deploys every monday morning")
    store.add_fact("the billing service writes its logs to loki")
    store.add_fact("we bought a new coffee machine for the office")
    assert FactRetriever(store, hrr_dim=DIM).contradict() == []


def test_a_flipped_claim_that_names_no_one_is_a_pair(store):
    yes = store.add_fact("the office kitchen has a dishwasher")
    no = store.add_fact("the office kitchen has no dishwasher")
    assert pairs(FactRetriever(store, hrr_dim=DIM).contradict()) == [{yes, no}]


def test_two_thousand_facts_on_one_name_are_compared_in_seconds(store):
    """The owner's own name is on many facts: the comparison must stay bounded and still find the pairs. Codex review
    2026-10-31: with contradict answering [] this test passed; now an old pair and a new one must be in the answer."""
    north = store.add_fact_status("Alice Smith parks the motorbike at the north gate", entities=["Alice Smith"])[0]
    south = store.add_fact_status("Alice Smith parks the motorbike at the south gate", entities=["Alice Smith"])[0]
    for i in range(2000):
        store.add_fact_status(f"Alice Smith noted item{i} in ledger{i % 37} on page{i % 101}",
                              entities=["Alice Smith"])
    systemd = store.add_fact_status("Alice Smith noted the deploy of billing with systemd", entities=["Alice Smith"])[0]
    docker = store.add_fact_status("Alice Smith noted the deploy of billing with docker", entities=["Alice Smith"])[0]
    started = time.monotonic()
    found = pairs(FactRetriever(store, hrr_dim=DIM).contradict(limit=10 ** 6))
    assert time.monotonic() - started < 10.0
    assert {north, south} in found and {systemd, docker} in found


def test_long_claims_that_share_only_words_most_facts_say_are_still_paired(store):
    """Codex review 2026-10-31: 201 facts "Alice Smith deploys billing service with X" — every word of the claim but
    X is said by over 200 of them, and claims longer than three words were then never paired: systemd / docker
    (overlap 0.667) was not found."""
    targets = ["systemd", "docker"] + [f"backend{i}" for i in range(199)]
    ids = {t: store.add_fact_status(f"Alice Smith deploys billing service with {t}", entities=["Alice Smith"])[0]
           for t in targets}
    found = pairs(FactRetriever(store, hrr_dim=DIM).contradict(limit=10 ** 6))
    assert {ids["systemd"], ids["docker"]} in found

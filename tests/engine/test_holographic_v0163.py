"""0.16.3: fact vectors after a turn on a slow endpoint (issue #5 by sprd-tmu).

A 4B embedding model on a local NPU server took 7.7 s for a batch of 32 facts; the after-turn budget left 7 s. Every
batch was cut, every cut paused the endpoint (60 s doubling to 15 min, recall by meaning off with it), and the same 32
facts went out again after each pause: the store never got a vector. Here a fake endpoint takes a fixed time per text,
and the budget is scaled down so a whole batch does not fit."""
from __future__ import annotations

import json
import threading
import time
import urllib.error


from holomem.engine import embeddings
from test_holographic_v0140_embeddings import ON, FakeEndpoint


def _embedder(**config) -> embeddings.Embedder:
    return embeddings.Embedder(embeddings.Settings({**ON, **config}), environ={"TEST_EMBEDDINGS_KEY": "sk-test-key"})


# -- what still is a failure ------------------------------------------------------------------------------------------


def test_an_error_that_comes_after_a_cut_still_pauses(monkeypatch):
    """Review 0.16.3 (Codex): a 503 that came after the caller stopped waiting was dropped, and the endpoint stayed
    ready while every batch was cut."""
    class LateError(FakeEndpoint):
        def open(self, request, timeout=None):
            time.sleep(0.7)
            raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)

    monkeypatch.setattr(embeddings, "_OPENER", LateError())
    e = _embedder()
    assert e.embed_facts(["one", "two"], 0.5) is None and e.status() == "ready"
    time.sleep(0.4)  # the 503 arrives
    assert e.embed_facts(["one", "two"], 0.5) is None
    assert e.status().startswith("paused after an error (HTTP 503)")


# -- what runs out of time ------------------------------------------------------------------------------------------


def test_a_query_waits_for_a_batch_cut_short_a_moment_ago(monkeypatch):
    release = threading.Event()

    class Held(FakeEndpoint):
        def open(self, request, timeout=None):
            if len(json.loads(request.data)["input"]) > 1:
                release.wait(5)
            return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", Held())
    e = _embedder(embedding_timeout=3)
    assert e.embed_facts(["a backup fact", "a birthday fact"], 0.6) is None  # cut, left running
    assert e.status() == "ready" and e._abandoned
    threading.Timer(0.3, release.set).start()
    t = time.monotonic()
    assert e.query("which machine copies our data while we sleep?") is not None  # 0.16.2: failed at once, paused
    assert time.monotonic() - t < 1.5 and e.status() == "ready"


def test_a_query_behind_our_batch_still_running_is_no_failure(monkeypatch):
    release = threading.Event()

    class Held(FakeEndpoint):
        def open(self, request, timeout=None):
            release.wait(5)
            return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", Held())
    e = _embedder(embedding_timeout=1)
    try:
        assert e.embed_facts(["one", "two"], 0.5) is None
        t = time.monotonic()
        assert e.query("which machine copies our data?") is None
        assert 0.4 < time.monotonic() - t < 0.9  # half of embedding_timeout
        assert e.status() == "ready"  # our own batch held the endpoint: not its failure
        assert e.calls == 1  # and nothing more was sent
    finally:
        release.set()


def test_a_message_behind_our_batch_on_a_one_slot_server_is_no_failure(monkeypatch):
    """Review 0.16.3 (Codex): a local server serving one request at a time made a message's vector wait behind a fact
    batch; it ran out of embedding_timeout and paused the endpoint."""
    slot = threading.Lock()

    class OneSlot(FakeEndpoint):
        def open(self, request, timeout=None):
            with slot:
                time.sleep(0.2 * len(json.loads(request.data)["input"]))
                return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", OneSlot())
    e = _embedder(embedding_url="http://127.0.0.1:8731/v1", embedding_timeout=0.5)
    batch = threading.Thread(target=e.embed_facts, args=(["one", "two", "three", "four", "five"], 5.0))
    batch.start()
    time.sleep(0.1)
    assert e.query("which machine copies our data while we sleep?") is None  # waited behind the batch
    assert e.status() == "ready"
    batch.join(5)
    time.sleep(0.3)
    assert e.query("which machine copies our data while we sleep?") is not None


# -- sizing -----------------------------------------------------------------------------------------------------------


def test_the_batch_follows_the_measured_speed():
    e = _embedder()
    assert e.fact_batch([100] * 40, 7.0) == embeddings._PROBE_BATCH  # nothing measured yet: a small batch
    e._fact_rate = 0.001  # seconds per character
    assert e.fact_batch([100] * 32, 2.0) == 10  # half of 2 s: 1,000 characters
    assert e.fact_batch([5000, 100], 2.0) == 1  # a probe (the turn's first request): one, to try
    assert e.fact_batch([100] * 32, 0.5) == 2
    # review 0.16.3 (Codex): what would not fit is not sent (a fact sent into the last half second paused a working
    # endpoint) ...
    assert e.fact_batch([5000, 100], 2.0, probe=False) == 0
    e._fact_rate = 1.0 / 570  # one fact: 1 s
    assert e.fact_batch([570], 0.5, probe=False) == 0
    # ... but one fact that fits goes out even when half the time left holds none (round 2: 1 fact a turn on a local
    # server whose one fact took longer than half of embedding_timeout)
    assert e.fact_batch([570, 570], 1.5, probe=False) == 1


# -- round 2: what holds the endpoint, and for how long --------------------------------------------------------------

class _Never(FakeEndpoint):
    """Answers nothing until released."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def open(self, request, timeout=None):
        self.requests.append(json.loads(request.data))
        self.release.wait(10)
        raise OSError("released")


def test_a_fact_request_that_never_ends_is_ours_only_for_a_while(monkeypatch):
    """Review 0.16.3 (round 2): a fact worker that never ended kept every later request "busy" and the endpoint
    "ready" for good."""
    never = _Never()
    monkeypatch.setattr(embeddings, "_OPENER", never)
    e = _embedder(embedding_timeout=0.5)
    try:
        assert e.embed_facts(["one", "two", "three", "four"], 0.5) is None  # cut, left running
        assert e.query("which machine copies our data?") is None and e.status() == "ready"  # ours, a moment ago
        time.sleep(e._grace())
        assert e.query("which machine copies our data?") is None
        assert "earlier request is still waiting" in e.status()  # a stalled server now
        assert len(never.requests) == 1  # and never a second worker
    finally:
        never.release.set()


def test_a_batch_that_ended_just_before_the_message_gave_up_still_explains_its_wait(monkeypatch):
    """Review 0.16.3 (round 2): on a one-slot server a message waited behind a batch that ended just before the
    message's deadline; nothing of ours was running at that moment, and the endpoint was paused."""
    slot = threading.Lock()

    class OneSlot(FakeEndpoint):
        def open(self, request, timeout=None):
            with slot:
                time.sleep(0.45 if len(json.loads(request.data)["input"]) > 1 else 0.15)
                return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", OneSlot())
    e = _embedder(embedding_url="http://127.0.0.1:8731/v1", embedding_timeout=0.5)
    batch = threading.Thread(target=e.embed_facts, args=(["one", "two"], 5.0))
    batch.start()
    time.sleep(0.02)
    assert e.query("which machine copies our data while we sleep?") is None
    batch.join(5)
    assert e.status() == "ready"


def test_a_batch_a_socket_timeout_ended_still_explains_a_message_s_wait(monkeypatch):
    """Review 0.16.3 (round 3, Antigravity): a batch ended by urllib's own timeout (its thread over before the join)
    left no trace, and a message queued behind it on a one-slot server paused the endpoint."""
    slot = threading.Lock()

    class OneSlot(FakeEndpoint):
        def open(self, request, timeout=None):
            with slot:
                if len(json.loads(request.data)["input"]) > 1:
                    time.sleep(0.4)
                    raise TimeoutError("read timed out")
                time.sleep(0.2)
                return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", OneSlot())
    e = _embedder(embedding_url="http://127.0.0.1:8731/v1", embedding_timeout=0.5)
    batch = threading.Thread(target=e.embed_facts, args=(["one", "two"], 1.0))
    batch.start()
    time.sleep(0.02)
    assert e.query("which machine copies our data while we sleep?") is None  # 0.4 s behind the batch, then 0.2 s
    batch.join(5)
    assert e.status() == "ready" and "not done" in e.last_batch


def test_the_status_tells_a_stalled_server_without_waiting_for_a_request(monkeypatch):
    """Review 0.16.3 (round 3, Antigravity): a request that never ended showed "ready" until the next request."""
    never = _Never()
    monkeypatch.setattr(embeddings, "_OPENER", never)
    e = _embedder(embedding_timeout=0.5)
    try:
        assert e.embed_facts(["one", "two"], 0.5) is None
        assert e.status() == "ready"  # ours, a moment ago
        time.sleep(e._grace() + 0.1)
        assert "earlier request is still waiting" in e.status()
        pause = e._pause
        assert "earlier request is still waiting" in e.status() and e._pause == pause  # counted once
    finally:
        never.release.set()


def test_an_answer_does_not_end_a_pause_that_came_after_it_was_sent(monkeypatch):
    """Review 0.16.3 (round 2): a batch still running ended well and cleared the pause a message's 503 had set
    meanwhile."""
    release = threading.Event()

    class Held(FakeEndpoint):
        def open(self, request, timeout=None):
            release.wait(5)
            return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", Held())
    e = _embedder()
    out = {}
    batch = threading.Thread(target=lambda: out.setdefault("v", e.embed_facts(["one", "two"], 3.0)))
    batch.start()
    time.sleep(0.1)
    e._failed("HTTP 503")  # what a message's request met meanwhile
    release.set()
    batch.join(5)
    assert out["v"] is not None  # the vectors are kept
    assert e.status().startswith("paused after an error (HTTP 503)")


def test_one_request_is_one_step_of_the_pause(monkeypatch):
    """Review 0.16.3 (round 2): a fact cut short paused the endpoint, and the 503 it brought later paused it again —
    120 s for one request."""
    release = threading.Event()

    class LateError(FakeEndpoint):
        def open(self, request, timeout=None):
            release.wait(5)
            raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)

    monkeypatch.setattr(embeddings, "_OPENER", LateError())
    e = _embedder()
    assert e.embed_facts(["one"], 0.5, decisive=True) is None
    assert e._pause == 60.0
    release.set()
    for given in list(e._abandoned):
        given.worker.join(5)
    e._paused_until = 0.0  # the pause is over
    with e._lock:
        assert e._reap_locked() is None  # already counted
    assert e._pause == 60.0


def test_late_errors_of_several_requests_pause_once_and_say_so(monkeypatch):
    e = _embedder()
    for code in (503, 429):
        box = {"error": urllib.error.HTTPError("https://x", code, "busy", {}, None), "ended": time.monotonic()}
        worker = threading.Thread(target=lambda: None)
        worker.start()
        worker.join()
        e._abandoned.append(embeddings._GivenUp(worker, box, fact=True, busy=False, counted=False))
    with e._lock:
        why, _, pause = e._reap_locked()
    assert why == "HTTP 503 (and 1 more)" and pause == 60.0 and e._abandoned == []


def test_a_batch_a_message_queued_with_is_not_measured(monkeypatch):
    """Review 0.16.3 (round 2): a message queued with a batch on a one-slot server made the batch look twice as slow,
    and the next batch held none."""
    release = threading.Event()

    class Held(FakeEndpoint):
        def open(self, request, timeout=None):
            if len(json.loads(request.data)["input"]) > 1:
                release.wait(5)
            return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", Held())
    e = _embedder()
    e._fact_rate = 1e-4
    batch = threading.Thread(target=e.embed_facts, args=(["one fact", "two facts"], 3.0))
    batch.start()
    time.sleep(0.05)
    assert e.query("which machine copies our data while we sleep?") is not None  # sent meanwhile
    time.sleep(0.3)
    release.set()
    batch.join(5)
    assert e._fact_rate == 1e-4

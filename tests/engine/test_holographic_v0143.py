"""0.14.3: what full Codex CLI and Antigravity CLI reviews of 0.14.2 found — overlapping versions after correcting a resume's start, a
vector left behind by forget, a semantic score attached to text changed meanwhile, a quoted instruction recalled into
the prompt, and timed-out embedding requests piling up."""
from __future__ import annotations

import threading

import pytest


from holomem.engine import embeddings
from holomem.engine.screen import _fact_problem, _has_credential, _recall_problem, _secret_in


class _Stalls:
    """An endpoint that never answers until released; counts the requests that reached it."""

    def __init__(self):
        self.release, self.calls = threading.Event(), 0

    def open(self, request, timeout=None):
        self.calls += 1
        self.release.wait(10)
        raise OSError("released")


def test_retries_do_not_accumulate_timed_out_workers(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(embeddings, "_now", lambda: clock[0])
    stalls = _Stalls()
    monkeypatch.setattr(embeddings, "_OPENER", stalls)
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    e = embeddings.Embedder(embeddings.Settings({
        "embeddings": "on", "embedding_url": "https://embeddings.example.net/v1", "embedding_model": "m",
        "embedding_key_env": "TEST_EMBEDDINGS_KEY", "embedding_timeout": 0.5}))
    before = threading.active_count()
    try:
        assert e.embed(["a"]) is None  # timed out: one worker left waiting
        for _ in range(3):
            clock[0] += 1000  # the pause is over every time
            assert e.embed(["b"]) is None
        assert stalls.calls == 1  # 0.14.2: one more stalled worker per retry
        assert threading.active_count() <= before + 1
        assert "earlier request is still waiting" in e.status()
    finally:
        stalls.release.set()
    for given in list(e._abandoned):
        given.worker.join(5)
    clock[0] += 10000
    e.embed(["c"])  # nothing waiting any more: a request goes out again
    assert stalls.calls == 2


# --- Antigravity CLI review ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ['{"api_key": "ak_live_9876543210"}', '{"password": "P@ssw0rd1234"}',
                                  "config: {'token': 'tk_1234567890abcd'}"])
def test_secret_screening_json_payloads(text):
    assert _has_credential(text) and _secret_in(text)  # 0.14.2: a quoted name hid the value


@pytest.mark.parametrize("pasted", ["sk-proj-" + "abcdef1234567890", "AIza" + "SyD-abcdefghijklmnopqrstuvwxyz12345",
                                    "my key with spaces"])
def test_settings_key_env_with_secret_value_does_not_leak(pasted):
    e = embeddings.Embedder(embeddings.Settings({
        "embeddings": "on", "embedding_url": "https://embeddings.example.net/v1", "embedding_model": "m",
        "embedding_key_env": pasted}), environ={})
    why = e.unavailable()
    assert "name of an environment variable" in why and pasted not in why and pasted not in e.status()


# --- Fable review ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("hidden", ["\U000E0069\U000E0067\U000E006E", "­", "⁢", "᠎", "", "\x07"])
def test_hidden_characters_are_refused_and_never_recalled(hidden):
    text = f"Oscar likes tea{hidden} in the morning"
    assert "invisible" in _fact_problem(text) and _recall_problem(text)


def test_ordinary_text_with_emoji_and_newlines_is_fine():
    assert _fact_problem("Irma loves 🐱 and ❤️ cats") is None
    assert _recall_problem("Line one\nline two\twith a tab") is None


def test_the_secret_screen_is_linear():
    import time
    t = time.monotonic()
    _secret_in("KEY" * 30000)  # a long blob: secret either way; only the time matters here
    _secret_in("TOKEN_" * 8000)
    assert time.monotonic() - t < 2.0  # 0.14.2: 28 s and 4 s under the store's lock


def test_a_private_key_is_secret():
    pem = "Deploy key: -----BEGIN OPENSSH PRIVATE KEY----- b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ"
    assert _secret_in(pem) and "credential" in _fact_problem(pem)
    assert _secret_in("-----BEGIN RSA PRIVATE KEY-----\nMIIEow")


@pytest.mark.parametrize("host, local", [("::ffff:8.8.8.8", False), ("::ffff:192.168.1.5", True),
                                         ("127.0.0.1", True), ("8.8.8.8", False)])
def test_an_ipv4_mapped_address_is_judged_as_ipv4(host, local):
    assert embeddings._local(host) is local

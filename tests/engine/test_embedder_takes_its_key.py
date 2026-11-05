"""The engine's embeddings endpoint never reads the process environment: whoever opens it says where the key is.

Requirement: the endpoint sends its key to a server, so where that key may come from is decided by the program that
uses the engine (holomem: $HOLOMEM_HOME/.env and the environment, or, as the Claude Code plugin, only the plugin's own
option), never by the engine on its own. An Embedder given no values has no key, even when a variable of the name
embedding_key_env says is set; given values, it takes the key from them and from nothing else.
"""
from __future__ import annotations

from holomem.engine import embeddings

SETTINGS = {"embeddings": True, "embedding_url": "https://embeddings.example.com/v1",
            "embedding_key_env": "TEST_EMBEDDINGS_KEY"}


def test_given_no_values_it_has_no_key(monkeypatch):
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-from-the-environment")
    e = embeddings.Embedder(embeddings.Settings(SETTINGS))
    assert e._key() == ""
    assert "TEST_EMBEDDINGS_KEY is not set" in e.unavailable() or e.unavailable().startswith("numpy")


def test_given_values_it_takes_the_key_from_them_only(monkeypatch):
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-from-the-environment")
    e = embeddings.Embedder(embeddings.Settings(SETTINGS), values={"TEST_EMBEDDINGS_KEY": "sk-given"})
    assert e._key() == "sk-given"

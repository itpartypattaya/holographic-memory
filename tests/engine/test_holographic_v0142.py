"""0.14.3: a column another program left in the facts table never reaches a tool result (0.14.2: binary ones only);
a 400 to `dimensions` is named (0.14.2).

Both reported on a live store that moved to 0.14.1 from a hand-rolled embedding deployment (issue #3)."""
from __future__ import annotations

import urllib.error

import pytest

pytest.importorskip("numpy")

from holomem.engine import embeddings


class _Refuses400:
    def open(self, request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, None)


ON = {"embeddings": "on", "embedding_url": "https://embeddings.example.net/v1", "embedding_model": "bge-m3",
      "embedding_key_env": "TEST_EMBEDDINGS_KEY"}


@pytest.mark.parametrize("dimensions, hinted", [(1024, True), (None, False)])
def test_a_400_to_dimensions_names_the_setting(monkeypatch, dimensions, hinted):
    monkeypatch.setattr(embeddings, "_OPENER", _Refuses400())
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    config = {**ON, **({"embedding_dimensions": dimensions} if dimensions else {})}
    e = embeddings.Embedder(embeddings.Settings(config))
    assert e.embed(["a"]) is None
    status = e.status()
    assert status.startswith("paused after an error (HTTP 400")
    assert ("embedding_dimensions is set" in status) is hinted
    assert "sk-test" not in status

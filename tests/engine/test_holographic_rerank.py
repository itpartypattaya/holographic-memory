"""Rescoring by a cross-encoder (rerank.py, RFC #9) — optional, offline-safe, off by default.

A fake endpoint stands in for the /v1/rerank server: it answers with a fixed or reversed order of the documents
sent, or fails as told.

One mode since the 2026-10-22 review (item 6): the cross-encoder reorders the pool, admission stays with coverage.
`rerank_window` (δ) optionally cuts the tail below δ of the best score before the top five (coverage ran
earlier, on the lexical stream);
`rerank_weight` > 0 blends the score into the blend. rerank_mode / rerank_margin are gone — a test covers that
they are ignored with a warning. Any failure keeps the current order and pauses the endpoint."""
from __future__ import annotations

import logging

import pytest

from holomem.engine import rerank


# -- the endpoint's own API ---------------------------------------------------------------------------------

def test_settings_validation():
    assert rerank.settings_from({}) is None
    assert rerank.settings_from({"rerank": "off"}) is None
    s = rerank.Settings({"rerank": "on", "rerank_top_k": "15", "rerank_window": "0.1"})
    assert (s.top_k, s.window) == (15, 0.1)
    assert rerank.Settings({"rerank": "on"}).window == rerank.DEFAULT_WINDOW  # the window default

    # item 4 (maintainer 24:10) + item 2 (19:39): the recall budget — a turn's recall gets 8 s, the sum of
    # embedding_timeout + rerank_timeout must fit inside it; emb counts ONLY
    # with embeddings: on (irrelevant tuning must not cut rerank), and the default rerank_timeout
    # also passes the cap (_number returns the default WITHOUT clamping — their finding).
    assert rerank.Settings({"rerank": "on", "embeddings": "on",
                             "embedding_timeout": "6", "rerank_timeout": "10"}).timeout == 1.5
    assert rerank.Settings({"rerank": "on", "embeddings": "on",
                             "embedding_timeout": "0.5", "rerank_timeout": "10"}).timeout == 7.0
    assert rerank.Settings({"rerank": "on", "embeddings": "on",
                             "embedding_timeout": "6"}).timeout == 1.5  # the default 3 is capped too
    assert rerank.Settings({"rerank": "on",
                             "embedding_timeout": "6", "rerank_timeout": "10"}).timeout == 7.5  # emb off
    assert rerank.Settings({"rerank": "on"}).timeout == 3.0  # emb off: the default 3 fits into 7.5


def test_removed_keys_are_ignored_with_a_warning(caplog):
    """rerank_mode / rerank_margin are gone (item 6): an old configuration breaks nothing, the key is ignored."""
    with caplog.at_level(logging.WARNING):
        s = rerank.Settings({"rerank": "on", "rerank_mode": "gate_reads_score", "rerank_margin": "1.5"})
    assert s.window == rerank.DEFAULT_WINDOW
    assert "rerank_mode was removed" in caplog.text and "rerank_margin was removed" in caplog.text


def test_ranked_shape():
    full = {"results": [{"index": 2, "relevance_score": -0.239},
                        {"index": 1, "relevance_score": -1.5},
                        {"index": 0, "relevance_score": -2.877}]}
    assert rerank._ranked(full, 3) == [(2, -0.239), (1, -1.5), (0, -2.877)]
    # the answer order is not trusted: a shuffled server list is sorted by score
    shuffled = {"results": [{"index": 1, "relevance_score": -1.5},
                            {"index": 2, "relevance_score": -0.239},
                            {"index": 0, "relevance_score": -2.877}]}
    assert rerank._ranked(shuffled, 3) == [(2, -0.239), (1, -1.5), (0, -2.877)]
    # item 4(3): a partial answer (fewer indices than documents) is an error, not «a smaller pool»:
    # the request always ships with top_n >= the number of documents, the server must answer them all.
    with pytest.raises(ValueError):
        rerank._ranked({"results": [{"index": 2, "relevance_score": -0.239},
                                     {"index": 0, "relevance_score": -2.877}]}, 5)  # 2 out of 5
    with pytest.raises(ValueError):
        rerank._ranked({"results": [{"index": 9, "relevance_score": 1.0}]}, 5)  # out of range
    with pytest.raises(ValueError):
        rerank._ranked({"results": [{"index": 0, "relevance_score": "x"}]}, 5)  # not a number
    with pytest.raises(ValueError):
        # 10**400 — valid JSON; math.isfinite(int) raises OverflowError, bypassing the except
        # ValueError in _request — before the fix such an answer was not counted as failed (item 1 of the checklist).
        rerank._ranked({"results": [{"index": 0, "relevance_score": 10 ** 400}]}, 5)  # isfinite overflow
    with pytest.raises(ValueError):
        rerank._ranked({"results": []}, 5)  # empty


def test_endpoint_validation():
    local = rerank._endpoint("http://127.0.0.1:8080/v1/rerank")
    assert local[1] is True  # local
    assert rerank._endpoint("http://example.com/v1/rerank")[0] is None  # public http refused
    assert rerank._endpoint("ftp://x")[0] is None  # wrong scheme


# -- fallback = off, the endpoint scale (item 3 of the 13:06 review), blend on one scale ------------------------------


    # The old behaviour left 2-4 on the raw 0.9/0.8/0.5 — they would have ranked ahead of 0.7.


# -- a query or a fact too long for the model (a live agent, 2026-10-26) ---------------------------------------------------
# llama-server answered 500 for the whole request when a query-document pair ran past its batch (643-732 tokens at 512)
# and 400 past the model's 1024-token context: 36 failed calls in a week on a live agent's night jobs, whose prompts are long.
# A message over RERANK_MAX_QUERY_CHARS is not rescored (its order stays the rerank-off order, not a failure) and each
# document is cut to what the message leaves of RERANK_PAIR_CHARS, a bound that holds in tokens (rerank.py).

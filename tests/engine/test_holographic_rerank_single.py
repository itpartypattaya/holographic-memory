# -*- coding: utf-8 -*-
"""PR #10, item 9: one sendable candidate — no call, the output matches
the reranked path (maintainer, 2026-10-23):

    "No call for a single candidate: with one sendable document there is nothing
    to reorder, and the window cannot drop the only one. Please add it with a test
    that the output matches the reranked path."
"""
from __future__ import annotations

import json

import pytest

from holomem.engine import rerank

RANK = {"rerank": "on",
        "rerank_url": "https://rerank.example.net/v1/rerank", "rerank_model": "fake-reranker"}
FACT = "The nightly backup runs from the OES box at 2 am."
QUERY = "when does the backup run?"


# -- fake endpoint (pattern from tests/..._rerank_review; conftest — a follow-up after the merge) --------


class _Response:
    def __init__(self, body: bytes):
        self.body, self.at = body, 0

    def read(self, n=-1):
        piece = self.body[self.at:] if n is None or n < 0 else self.body[self.at:self.at + n]
        self.at += len(piece)
        return piece

    read1 = read  # what urllib's HTTPResponse offers: a chunk, then b""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeRerank:
    """Records every request; answers identity (or fails as told)."""

    def __init__(self):
        self.requests: list[dict] = []
        self.fail = None
        self.score = None  # set -> the server answers with this logit (for a single doc)

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests.append({"url": request.full_url, "body": body, "timeout": timeout})
        if isinstance(self.fail, BaseException):
            raise self.fail
        n = len(body["documents"])
        if self.score is not None and n == 1:
            chosen = [{"index": 0, "relevance_score": float(self.score)}]
        else:
            chosen = [{"index": i, "relevance_score": float(n - i)}
                       for i in range(min(body.get("top_n", n), n))]
        return _Response(json.dumps({"results": chosen}).encode())


@pytest.fixture
def endpoint(monkeypatch):
    fake = FakeRerank()
    monkeypatch.setattr(rerank, "_OPENER", fake)
    return fake


def _rescorer():
    settings = rerank.settings_from(RANK)
    assert settings is not None
    return rerank.Rescorer(settings)


# -- a single candidate: no call -------------------------------------------------------


def test_single_candidate_scores_make_no_call(endpoint):
    rescorer = _rescorer()
    assert rescorer.scores(QUERY, [FACT]) == [(0, 1.0)]  # the only caller of order() is gone: scores() is the API
    assert rescorer.calls == 0
    assert endpoint.requests == [], "one document must not reach the endpoint"


def test_two_candidates_still_call_the_endpoint(endpoint):
    rescorer = _rescorer()
    ranked = rescorer.scores(QUERY, [FACT, "Another note about tapes."])
    assert ranked is not None and len(ranked) == 2
    assert len(endpoint.requests) == 1, "the shortcut must not swallow real pools"

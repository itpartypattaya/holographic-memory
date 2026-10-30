"""0.14.0: search by meaning (embeddings.py) — optional, online, off by default.

A fake endpoint stands in for the model: a text's vector is the sum of the concepts it names (a concept is a set of
words, so a paraphrase sharing no word with a fact still lands on its concepts) plus a little hashed noise per word.
One test runs a real HTTP server on 127.0.0.1."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

np = pytest.importorskip("numpy")

from holomem.engine import embeddings  # noqa: E402

DIM = 64
CONCEPTS = [
    ("backup", "copies", "archive", "data"),
    ("nightly", "night", "sleep"),
    ("box", "machine", "server"),
    ("birthday", "special day", "celebrate"),
    ("violin", "instrument"),
    ("allergic", "peanut", "satay"),
]
FILLERS = ["apple", "brick", "cloud", "daisy", "eagle", "fjord", "grape", "harbor", "igloo", "jungle", "kettle", "lemon",
           "maple", "nectar", "orbit", "pepper", "quartz", "raven", "saddle", "tulip", "umber", "velvet", "walnut",
           "yarrow", "zephyr"]


def fake_vector(text: str) -> list[float]:
    low = text.lower()
    v = np.zeros(DIM)
    for i, words in enumerate(CONCEPTS):
        if any(w in low for w in words):
            v[i] = 1.0
    for j, word in enumerate(FILLERS):  # each filler fact has a concept of its own
        if word in low:
            v[len(CONCEPTS) + j] = 1.0
    for token in "".join(c if c.isalnum() else " " for c in low).split():
        v[40 + zlib.crc32(token.encode()) % 24] += 0.35
    return [float(x) for x in v]


class _Response:
    def __init__(self, body: bytes):
        self.body, self.at = body, 0

    def read(self, n=-1):
        piece = self.body[self.at:] if n is None or n < 0 else self.body[self.at:self.at + n]
        self.at += len(piece)
        return piece

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeEndpoint:
    """Stands in for embeddings._OPENER: records every request, answers with fake_vector, or fails as told."""

    def __init__(self):
        self.requests: list[dict] = []
        self.fail = None  # an exception to raise, or bytes to answer with
        self.size = None  # answer with vectors cut to this many dimensions
        self.during = None  # called while the "network" request is in flight

    @property
    def inputs(self) -> list[str]:
        return [t for r in self.requests for t in r["body"]["input"]]

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body,
                              "timeout": timeout})
        if self.during:
            self.during()
        if isinstance(self.fail, BaseException):
            raise self.fail
        if isinstance(self.fail, bytes):
            return _Response(self.fail)
        data = [{"object": "embedding", "index": i, "embedding": fake_vector(t)[:self.size]}
                for i, t in enumerate(body["input"])]
        return _Response(json.dumps({"data": data, "model": body["model"]}).encode())


@pytest.fixture
def endpoint(monkeypatch):
    fake = FakeEndpoint()
    monkeypatch.setattr(embeddings, "_OPENER", fake)
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    return fake


ON = {"embeddings": "on", "embedding_url": "https://embeddings.example.net/v1", "embedding_model": "fake-1",
      "embedding_key_env": "TEST_EMBEDDINGS_KEY"}


def _vectors(p) -> dict:
    return {r[0]: r[1] for r in p._store.read("SELECT fact_id, model FROM fact_embeddings")}


# -- failures ---------------------------------------------------------------------------------------------------------


def test_the_pause_grows_and_ends_with_a_success(tmp_path, endpoint, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(embeddings, "_now", lambda: clock[0])
    e = embeddings.Embedder(embeddings.Settings(ON))
    endpoint.fail = urllib.error.URLError("refused")
    assert e.embed(["a"]) is None
    clock[0] += 61
    assert e.embed(["a"]) is None and e.status().endswith("tried again in 121 s")
    endpoint.fail = None
    clock[0] += 121
    assert e.embed(["a"]) is not None and e.status() == "ready"


# -- the endpoint -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("url, refused", [
    ("https://api.openai.com/v1", ""),
    ("http://127.0.0.1:11434/v1", ""),
    ("http://ollama:11434/v1", ""),
    ("http://192.168.1.20:8080", ""),
    ("http://[::1]:11434/v1", ""),
    ("http://[fd00::5]/v1", ""),
    ("http://[2606:4700:4700::1111]/v1", "plain http to a public host"),  # no dot, but a public address
    ("http://api.example.com/v1", "plain http to a public host"),
    ("ftp://example.com", "not an http(s) address"),
])
def test_plain_http_only_to_local_hosts(url, refused):
    address, _, why = embeddings._endpoint(url)
    assert refused in why if refused else (why == "" and address.endswith("/embeddings"))


class _Handler(BaseHTTPRequestHandler):
    seen: list = []
    redirect = False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, self.headers.get("Authorization"), body))
        if type(self).redirect:
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/elsewhere")
            self.end_headers()
            return
        data = [{"index": i, "embedding": fake_vector(t)} for i, t in reversed(list(enumerate(body["input"])))]
        out = json.dumps({"data": data}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_a_real_local_server_round_trip_and_no_redirects(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        _Handler.seen, _Handler.redirect = [], False
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        keyless = embeddings.Embedder(embeddings.Settings({"embedding_url": url, "embedding_model": "local",
                                                          "embedding_key_env": "NO_SUCH_KEY_VAR"}))
        assert keyless.status() == "ready"  # a local server needs no key
        vectors = keyless.embed(["backup at night", "violin"])
        assert _Handler.seen[0][0] == "/v1/embeddings" and _Handler.seen[0][1] is None
        assert _Handler.seen[0][2] == {"model": "local", "input": ["backup at night", "violin"]}
        assert abs(float(np.linalg.norm(vectors[0])) - 1.0) < 1e-5
        assert float(vectors[0] @ np.asarray(fake_vector("backup at night")) / np.linalg.norm(fake_vector("backup at night"))) > 0.999
        monkeypatch.setenv("LOCAL_KEY", "sk-local")
        keyed = embeddings.Embedder(embeddings.Settings({"embedding_url": url, "embedding_key_env": "LOCAL_KEY"}))
        keyed.embed(["x"])
        assert _Handler.seen[-1][1] == "Bearer sk-local"
        _Handler.redirect = True  # the key must not follow a redirect anywhere
        assert keyed.embed(["y"]) is None and keyed.status().startswith("paused after an error (HTTP 302)")
    finally:
        server.shutdown()


def test_settings_are_clamped_and_off_unless_on():
    assert embeddings.settings_from({}) is None
    assert embeddings.settings_from({"embeddings": "off"}) is None
    s = embeddings.settings_from({"embeddings": True, "embedding_timeout": 60, "embedding_prefetch_facts": 9,
                                  "embedding_weight": "x", "embedding_dimensions": "512"})
    assert (s.timeout, s.recall_facts, s.weight, s.dimensions, s.tag) == (6.0, 3, 0.3, 512, "https://api.openai.com/v1/text-embedding-3-small@512")


def test_the_bar_rises_with_the_store():
    assert embeddings.z_bar(20, 1.0) < embeddings.z_bar(100, 1.0) < embeddings.z_bar(20_000, 1.0)
    m = embeddings.Meaning(list(range(10)), np.linspace(0.1, 0.9, 10), 0.3)
    assert not m.standout(9, 1.0, None)  # too few vectors to judge a spread: only a floor admits
    assert m.standout(9, 1.0, 0.85) and not m.standout(8, 1.0, 0.85)


# -- after the review of 2026-10-14 ---------------------------------------------------------------------------------


def test_the_address_tag_never_holds_a_password_or_a_query():
    s = embeddings.Settings({"embedding_url": "https://user:secret@host.example:8443/v1?key=abc", "embedding_model": "m"})
    assert s.tag.startswith("https://host.example:8443/v1?") and s.tag.endswith("/m")  # the query by a hash only
    assert "secret" not in s.tag and "user" not in s.tag and "abc" not in s.tag
    upper = embeddings.Settings({"embedding_url": "https://host.example/A", "embedding_model": "m"})
    lower = embeddings.Settings({"embedding_url": "https://host.example/a", "embedding_model": "m"})
    assert upper.tag != lower.tag  # a path is case-sensitive: two routes, two spaces


@pytest.mark.parametrize("payload, count, dims", [
    ({"data": [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [0, 1]}]}, 2, None),  # a duplicate
    ({"data": [{"index": 0, "embedding": [1, 0]}, {"index": 2, "embedding": [0, 1]}]}, 2, None),  # out of range
    ({"data": [{"index": 0, "embedding": [1, 0]}, {"embedding": [0, 1]}]}, 2, None),  # one without an index
    ({"data": [{"index": 0, "embedding": [1, 0]}]}, 1, 3),  # not the size asked for
    ({"data": [{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [0, 1, 0]}]}, 2, None),  # sizes differ
])
def test_a_malformed_answer_is_refused(payload, count, dims):
    with pytest.raises(ValueError):
        embeddings._vectors(payload, count, dims)


def test_gemini_leaves_index_zero_out_and_is_read_in_order():
    vectors = embeddings._vectors({"data": [{"index": 1, "embedding": [0, 2]}, {"embedding": [1, 0]}]}, 2)
    assert [v.tolist() for v in vectors] == [[1.0, 0.0], [0.0, 1.0]]  # the one without an index is the first text


@pytest.mark.parametrize("values, expected", [
    ([1e30, 1e30], [0.7071, 0.7071]),  # squares overflow float32
    ([1e300, 1.0], [1.0, 0.0]),
    ([1e-40, 0.0], [1.0, 0.0]),  # squares underflow
])
def test_extreme_values_still_make_a_unit_vector(values, expected):
    (vector,) = embeddings._vectors({"data": [{"index": 0, "embedding": values}]}, 1)
    assert [round(float(x), 4) for x in vector] == expected


# -- after the second review of 2026-10-14 --------------------------------------------------------------------------


class _SlowHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "100000")
        self.end_headers()
        try:
            for _ in range(60):  # a byte every 0.1 s: never quiet long enough for the socket timeout
                self.wfile.write(b" ")
                self.wfile.flush()
                time.sleep(0.1)
        except OSError:
            pass

    def log_message(self, *a):
        pass


def test_a_server_that_drips_its_answer_is_cut_off_at_the_timeout():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        e = embeddings.Embedder(embeddings.Settings({"embedding_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                                                     "embedding_timeout": 0.5}))
        t = time.monotonic()
        assert e.embed(["x"]) is None
        assert time.monotonic() - t < 0.5 + 0.4
        assert e.status().startswith("paused after an error (no answer")
    finally:
        server.shutdown()


# -- after the third review of 2026-10-14 ---------------------------------------------------------------------------

class _DripHandler(BaseHTTPRequestHandler):
    """Never quiet long enough for a socket timeout: headers a byte at a time, or a chunked body whose chunk-size line
    never ends."""
    mode = "headers"

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if type(self).mode == "headers":
            head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nX-Slow: " + b"a" * 200
        else:
            head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n0" + b"0" * 200
        try:
            for byte in head:
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
                time.sleep(0.03)
        except OSError:
            pass

    def log_message(self, *a):
        pass


@pytest.mark.parametrize("mode", ["headers", "chunked"])
def test_a_request_never_outlives_its_timeout(mode):
    _DripHandler.mode = mode
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DripHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        e = embeddings.Embedder(embeddings.Settings({"embedding_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                                                     "embedding_timeout": 0.5}))
        t = time.monotonic()
        assert e.embed(["x"]) is None
        assert time.monotonic() - t < 0.5 + 0.3
        assert e.status().startswith("paused after an error (no answer")
        # issue #26: the request the caller gave up on must actually end — without the caller's shutdown the
        # header-drip mode leaves its worker reading for ~7 s, past the join(3) below. The shutdown ends the read
        # with ConnectionAbortedError / RemoteDisconnected / BadStatusLine; the endpoint already paused above.
        for given in e._abandoned:
            given.worker.join(3)
            assert not given.worker.is_alive(), \
                "a request given up on must end with the caller's deadline, not read on (issue #26)"
    finally:
        server.shutdown()


def test_a_query_string_and_the_space_label_are_part_of_the_identity():
    a = embeddings.Settings({"embedding_url": "https://h.example/v1?deployment=A&api-key=secret", "embedding_model": "m"})
    b = embeddings.Settings({"embedding_url": "https://h.example/v1?deployment=B&api-key=secret", "embedding_model": "m"})
    labelled = embeddings.Settings({"embedding_url": "https://h.example/v1?deployment=A&api-key=secret",
                                    "embedding_model": "m", "embedding_space": "2026-10"})
    assert len({a.tag, b.tag, labelled.tag}) == 3
    assert "secret" not in a.tag and "deployment" not in a.tag

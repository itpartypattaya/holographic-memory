"""Rescoring (rerank = true), against a local Cohere-shaped /v1/rerank endpoint started by the test.

Requirements (rescoring, with the bridge's state across processes):
- with rerank on, the facts recalled before a turn follow the endpoint's scores (a fact third by words can come
  first); with it off the block is exactly today's and nothing is sent;
- rerank_window (weight 0) cuts the low tail of the scored facts, never below two; a fact recall shows but must not
  send (a credential's shape) stays in the block behind the scored ones; rerank_weight > 0 blends every fact of the
  pool, an unscored one with the scored scale's bottom, and no window applies; equal scores keep the endpoint's
  order; a pool larger than rerank_top_k sends top_k and keeps the rest in their order behind the scored;
  prefetch_max_facts is the number of lines, the window's two the window's own;
- any failure — HTTP 500, silence, a body trickled byte by byte, an index out of range or repeated — gives exactly
  the block of rerank off (a store seeded alike), within the hook's deadline; the endpoint is then left alone by every
  process (hooks and the MCP server, both ways) for 60 s, doubling to 900 s, and asked again once that passes;
  a skipped call does not double it; success clears it;
- the speed the endpoint showed (seconds per character of the pairs) is kept between processes: after a cold
  timeout the next call sends about half; an endpoint that cannot be reached leaves no speed;
- the hook's deadline bounds the call: with a 2.5 s deadline and rerank_timeout 10 a silent endpoint still gives the
  off block in time, and the endpoint sees its socket closed before the deadline;
- memory_search reorders the facts its words found, keeps what was not scored behind them, adds facts by meaning
  after them; a search "as_of" a day sends nothing and records no attempt;
- rerank_scale outside logit/unit and a rerank_url with credentials or a query turn memory off naming the key, never
  the value (not in the session block, status, state.db or a tool's answer); a plain-http public host is
  "unavailable" and nothing is sent;
- holomem status and memory_stats say whether rescoring is on and the endpoint's state, never its address;
- the filter of one ordinary shared word holds in every layout: such a fact is never sent and never recalled;
- a fact changed by another process during the call never shows its old text under a receipt (the attempt is not
  recorded); a candidate changed that was not shown leaves the receipt with the change counted;
- a state.db held by another process while the call fails: the hook still answers, the pause is simply not written.
"""
from __future__ import annotations

import json
import os
import re
import select
import socket
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from holomem.recall import HEADER
from test_hooks import context, hook, prompt, seed
from test_mcp_server import Client
from test_meaning import CITY, FILLER, KEY, LONE_WORD, Endpoint
from test_meaning import cli as meaning_cli
from test_meaning import config as meaning_config

ROOT = Path(__file__).resolve().parent.parent
_LAUNCH = "import sys; sys.path.insert(0, sys.argv.pop(1)); from holomem.__main__ import main; sys.exit(main())"
ASK = "How does the billing service deploy?"
DEPLOYS = ["The billing service deploys to the staging server with systemd",
           "The billing service deploy uses a blue green switch",
           "The billing service deploy runs the database migrations first",
           "The billing service deploy posts a note to the ops channel",
           "The billing service deploy waits for the health check to pass",
           "The billing service deploy keeps the last three releases on disk"]
# recall shows it (no credential with its value), the endpoint never gets it (a bearer token's shape)
UNSENDABLE = "The billing service deploy sends Bearer abcdef123456789xyz to the registry"
ATTEMPT = re.compile(r"\nMemory attempt: ([0-9a-f]{32})\Z")


class Reranker:
    """POST /v1/rerank in the Cohere shape llama-server answers. Records each request's query and documents.

    mode: ok (results by descending score, ties in the request's order), reverse (equal scores, listed last document
    first), fail (HTTP 500), silent (no answer: waits until the client closes its socket, notes when), slow (sleeps
    *per_char* seconds per character of the pairs), trickle (a valid answer one byte per 0.1 s), out_of_range, duplicate.
    *prefer*: {substring: score}; a document holding none scores *default*. *on_request*: called with the body before
    answering.
    """

    def __init__(self):
        self.mode, self.prefer, self.default, self.per_char, self.on_request = "ok", {}, 0.0, 0.0, None
        self.requests: list[dict] = []
        self.closed_after: list[float] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                received = time.monotonic()
                outer.requests.append(body)
                if outer.on_request is not None:
                    outer.on_request(body)
                docs, query = body.get("documents") or [], body.get("query") or ""
                if outer.mode == "fail":
                    self._send(500, b'{"error":"boom"}')
                    return
                if outer.mode == "silent":
                    deadline = received + 30
                    while time.monotonic() < deadline:
                        readable, _, _ = select.select([self.connection], [], [], 0.05)
                        if readable:
                            try:
                                if not self.connection.recv(1, socket.MSG_PEEK):
                                    break
                            except OSError:
                                break
                    outer.closed_after.append(time.monotonic() - received)
                    self.close_connection = True
                    return
                if outer.mode == "slow":
                    time.sleep(outer.per_char * sum(len(query) + len(d) for d in docs))
                scores = [next((s for k, s in outer.prefer.items() if k in d), outer.default) for d in docs]
                order = sorted(range(len(docs)), key=lambda i: -scores[i])
                if outer.mode == "reverse":
                    order, scores = list(reversed(range(len(docs)))), [1.0] * len(docs)
                results = [{"index": i, "relevance_score": scores[i]} for i in order]
                if outer.mode == "out_of_range":
                    results[0]["index"] = len(docs)
                if outer.mode == "duplicate":
                    results[-1]["index"] = results[0]["index"]
                data = json.dumps({"model": body.get("model"), "results": results}).encode()
                if outer.mode == "trickle":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    try:
                        for byte in data:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            time.sleep(0.1)
                    except OSError:
                        pass
                    return
                self._send(200, data)

            def _send(self, code: int, data: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1/rerank"

    @property
    def documents(self) -> list[str]:
        return [d for body in self.requests for d in body.get("documents") or []]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def ranker():
    ep = Reranker()
    yield ep
    ep.close()


@pytest.fixture
def home(tmp_path) -> Path:
    return tmp_path / "holo"


def settings(ep: Reranker | None, **keys) -> str:
    """config.toml with rerank on at *ep* (or off with ep None) and *keys* (TOML values as Python literals)."""
    lines = [f'rerank = {"true" if ep is not None else "false"}']
    if ep is not None:
        lines.append(f'rerank_url = "{ep.url}"')
    for key, value in keys.items():
        lines.append(f"{key} = {json.dumps(value) if not isinstance(value, bool) else str(value).lower()}")
    return "\n".join(lines) + "\n"


def rows(path: Path, query: str, params=()) -> list:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def write(path: Path, query: str, params=()) -> None:
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute(query, params)
    finally:
        conn.close()


def backoff(home: Path) -> list[tuple]:
    """(pause, error) of the rerank endpoint's backoff, [] when none."""
    try:
        return rows(home / "state.db", "SELECT pause, error FROM backoff WHERE name LIKE 'rerank:%'")
    except sqlite3.OperationalError:
        return []


def expire(home: Path) -> None:
    """The pause has passed (as if 60 s went by); its length stays for the next doubling."""
    write(home / "state.db", "UPDATE backoff SET until = 0 WHERE name LIKE 'rerank:%'")


def speeds(home: Path) -> list[float]:
    try:
        return [r[0] for r in rows(home / "state.db", "SELECT sample FROM rerank_speed ORDER BY at")]
    except sqlite3.OperationalError:
        return []


def holomem(home: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOLOMEM_HOME": str(home), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, "-I", "-c", _LAUNCH, str(ROOT), *args], capture_output=True, env=env,
                          timeout=60)


def lines(block: str) -> list[str]:
    """The fact texts of a block, in order."""
    return [re.sub(r"^- #\d+ \[[0-9.]+\] ", "", line) for line in block.split("\n") if line.startswith("- #")]


def deploy_store(home: Path, config_text: str, extra: list[str] | None = None) -> list[int]:
    return seed(home, [(text, []) for text in FILLER[:8] + DEPLOYS[:4] + (extra or [])], config_text)


def recall_in(home: Path, text: str = ASK, session: str = "s1") -> str:
    return context(hook(home, "user-prompt", prompt(text, session=session)))


# -- 1. the order follows the endpoint; off is today's block --------------------------------------------------------

def test_the_recalled_order_follows_the_endpoint_and_off_sends_nothing(tmp_path, ranker):
    on, off = tmp_path / "on", tmp_path / "off"
    deploy_store(off, settings(None))
    words = lines(recall_in(off))
    assert ranker.requests == [] and len(words) == 4  # off: nothing sent; the four deploy facts by their words
    deploy_store(on, settings(ranker, rerank_window=0.0))
    ranker.prefer = {words[2]: 5.0, words[0]: 3.0, words[1]: 2.0, words[3]: 1.0}
    block = recall_in(on)
    assert ranker.requests and ranker.requests[0]["query"] == ASK
    assert ranker.requests[0]["documents"] == words  # the facts the words admitted, in their order
    assert lines(block) == [words[2], words[0], words[1], words[3]]
    assert block.startswith(HEADER + "\n")


def test_off_is_byte_for_byte_the_block_of_a_store_without_the_keys(tmp_path, ranker):
    plain, off = tmp_path / "plain", tmp_path / "off"
    deploy_store(plain, "")
    deploy_store(off, settings(None, rerank_top_k=3, rerank_window=0.5) + f'rerank_url = "{ranker.url}"\n')
    assert recall_in(off) == recall_in(plain)
    assert ranker.requests == []


# -- 2. the window ----------------------------------------------------------------------------------------------------

def test_the_window_keeps_two_and_an_unsendable_fact_stays_behind(tmp_path, ranker):
    off = tmp_path / "off"
    deploy_store(off, settings(None), [UNSENDABLE])
    words = lines(recall_in(off))
    assert UNSENDABLE in words and len(words) == 5
    scored = [w for w in words if w != UNSENDABLE]
    narrow = tmp_path / "narrow"
    deploy_store(narrow, settings(ranker, rerank_window=0.5), [UNSENDABLE])
    ranker.prefer, ranker.default = {scored[3]: 6.0}, -3.0  # one high, the rest far below: outside a 0.5 window
    block = lines(recall_in(narrow))
    assert UNSENDABLE not in ranker.documents  # never sent
    assert block == [scored[3], scored[0], UNSENDABLE]  # the window's two, then the fact that was not scored
    wide = tmp_path / "wide"
    deploy_store(wide, settings(ranker, rerank_window=0.0), [UNSENDABLE])
    assert lines(recall_in(wide)) == [scored[3], scored[0], scored[1], scored[2], UNSENDABLE]


def test_a_floor_drops_what_the_endpoint_scores_as_unrelated_even_the_last_fact(tmp_path, ranker):
    """The window is relative to the best candidate and keeps two: a lone stray fact (one candidate, or all of them
    unrelated) always went in. Live (2026-10-31, jina-reranker-v2): facts a question wanted scored −1.78…+1.54, the
    strays that went in ≤ −2.8. rerank_floor (on the 0..1 scale) drops a scored fact below it, the window's two too."""
    off = tmp_path / "off"
    deploy_store(off, settings(None), [UNSENDABLE])
    scored = [w for w in lines(recall_in(off)) if w != UNSENDABLE]
    floor = tmp_path / "floor"
    deploy_store(floor, settings(ranker, rerank_window=0.5, rerank_floor=0.075), [UNSENDABLE])  # logit −2.5
    ranker.prefer, ranker.default = {scored[1]: 2.0, scored[3]: -1.8}, -4.0
    assert lines(recall_in(floor)) == [scored[1], scored[3], UNSENDABLE]  # the window's two are above the floor
    ranker.prefer = {scored[1]: 2.0}
    assert lines(recall_in(floor, session="s2")) == [scored[1], UNSENDABLE]  # no second one above it
    lone = tmp_path / "lone"
    seed(lone, [(text, []) for text in FILLER[:8] + DEPLOYS[:1]], settings(ranker, rerank_floor=0.075))
    ranker.prefer = {}
    assert recall_in(lone) == ""  # the one candidate, unrelated
    ranker.prefer = {DEPLOYS[0]: -1.8}
    assert lines(recall_in(lone, session="s2")) == DEPLOYS[:1]  # wanted, if weakly
    unset = tmp_path / "unset"
    seed(unset, [(text, []) for text in FILLER[:8] + DEPLOYS[:1]], settings(ranker))
    ranker.prefer = {}
    assert lines(recall_in(unset)) == DEPLOYS[:1]  # no floor: the window keeps it, as before


# -- 3. failures give the off block, and the pause --------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["fail", "silent", "trickle", "out_of_range", "duplicate"])
def test_a_failing_endpoint_gives_exactly_the_off_block_in_time(tmp_path, ranker, mode):
    off, on = tmp_path / "off", tmp_path / "on"
    deploy_store(off, settings(None))
    expected = recall_in(off)
    deploy_store(on, settings(ranker, rerank_timeout=1.0))
    ranker.mode, ranker.prefer = mode, {DEPLOYS[3]: 9.0}
    started = time.monotonic()
    proc = hook(on, "user-prompt", prompt(ASK))
    elapsed = time.monotonic() - started
    assert context(proc) == expected
    assert elapsed < 8.0, elapsed
    assert len(ranker.requests) == 1
    (pause, error), = backoff(on)
    assert pause == 60
    assert error == {"fail": "HTTP 500", "silent": "no answer (timeout)", "trickle": "no answer (timeout)",
                     "out_of_range": "malformed answer", "duplicate": "malformed answer"}[mode]
    # the next process does not ask again while the pause lasts, and recalls as rerank off
    assert len(lines(context(hook(on, "user-prompt", prompt(ASK, session="s2"))))) == 4
    assert len(ranker.requests) == 1


def test_after_the_pause_a_success_clears_it(tmp_path, ranker):
    deploy_store(home := tmp_path / "on", settings(ranker, rerank_window=0.0))
    ranker.mode = "fail"
    recall_in(home)
    assert backoff(home) == [(60, "HTTP 500")]
    expire(home)
    ranker.mode, ranker.prefer = "ok", {DEPLOYS[3]: 9.0}
    block = lines(recall_in(home, session="s2"))
    assert len(ranker.requests) == 2 and block[0] == DEPLOYS[3]
    assert backoff(home) == []


# -- 4. one authority over the pause: hooks and the server ------------------------------------------------------------

def test_hooks_and_the_server_share_the_pause_both_ways(tmp_path, ranker):
    home = tmp_path / "on"
    deploy_store(home, settings(ranker, rerank_window=0.0))
    ranker.mode = "fail"
    recall_in(home)  # a hook fails: 60 s
    assert backoff(home) == [(60, "HTTP 500")]
    client = Client(home).start()
    try:
        found, is_error = client.call("memory_search", query=ASK)
        assert not is_error and found["count"] == 4
        assert len(ranker.requests) == 1  # the server saw the hook's pause
        recall_in(home, session="s2")
        assert len(ranker.requests) == 1 and backoff(home) == [(60, "HTTP 500")]  # a skip does not double
        expire(home)
        client.call("memory_search", query=ASK)  # the server fails now: doubled
        assert len(ranker.requests) == 2 and backoff(home) == [(120, "HTTP 500")]
        recall_in(home, session="s3")
        assert len(ranker.requests) == 2  # the hook saw the server's pause
        expire(home)
        ranker.mode, ranker.prefer = "ok", {DEPLOYS[2]: 9.0}
        found, _ = client.call("memory_search", query=ASK)  # the same warm server asks again, and is answered
        assert len(ranker.requests) == 3
        assert found["results"][0]["content"] == DEPLOYS[2]
        assert backoff(home) == []
    finally:
        client.close()


# -- 5. the endpoint's speed is kept between processes ----------------------------------------------------------------

def test_after_a_cold_timeout_the_next_process_sends_about_half(tmp_path, ranker):
    home = tmp_path / "on"
    seed(home, [(text, []) for text in FILLER[:8] + DEPLOYS], settings(ranker, rerank_timeout=1.0, rerank_window=0.0,
                                                                         prefetch_max_facts=8))
    chars = len(ASK) * 6 + sum(len(d) for d in DEPLOYS)
    ranker.mode, ranker.per_char = "slow", 1.5 / chars  # all six: 1.5 s, over the 1.0 s timeout
    recall_in(home)
    assert len(ranker.requests[0]["documents"]) == 6
    assert backoff(home)[0][1] == "no answer (timeout)" and len(speeds(home)) == 1
    expire(home)
    recall_in(home, session="s2")
    second = len(ranker.requests[1]["documents"])
    assert 2 <= second <= 3, second  # half of what one second held: about three of six
    assert backoff(home) == [] and len(speeds(home)) == 2  # answered in time: its speed is a sample too
    recall_in(home, session="s3")
    third = len(ranker.requests[2]["documents"])
    assert 2 <= third <= second


def test_an_endpoint_that_cannot_be_reached_leaves_no_speed(tmp_path):
    home = tmp_path / "on"
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # nothing listens there now
    deploy_store(home, f'rerank = true\nrerank_url = "http://127.0.0.1:{port}/v1/rerank"\n')
    assert len(lines(recall_in(home))) == 4  # the off block
    assert backoff(home)[0][1].startswith("cannot reach")
    assert speeds(home) == []


# -- 6. memory_search --------------------------------------------------------------------------------------------------

def test_search_reorders_the_words_keeps_the_tail_and_adds_meaning_after(tmp_path, ranker):
    emb = Endpoint()
    try:
        home = tmp_path / "on"
        target = "Mark gets around town on a blue Vespa scooter"
        ids = seed(home, [(text, []) for text in FILLER + DEPLOYS[:4] + [target]],
                   meaning_config(emb) + settings(ranker, rerank_top_k=2, rerank_window=0.0))
        (home / ".env").write_text(f"HOLO_TEST_EMBED_KEY={KEY}\n", encoding="utf-8")
        assert meaning_cli(home, "embed").returncode == 0
        query = "billing service deploy ride"
        client = Client(home).start()
        try:
            ranker.mode = "fail"
            plain, _ = client.call("memory_search", query=query)  # the endpoint fails: the order of rerank off
            expire(home)
            ranker.mode = "ok"
            words = [r["content"] for r in plain["results"] if r.get("match") != "meaning"]
            ranker.prefer = {words[1]: 4.0}
            found, is_error = client.call("memory_search", query=query)
        finally:
            client.close()
        assert not is_error and found["count"] == plain["count"]
        assert ranker.requests[-1]["documents"] == words[:2]  # top_k of what the words found
        contents = [r["content"] for r in found["results"]]
        assert contents[:len(words)] == [words[1], words[0], *words[2:]]  # the tail beyond top_k keeps its place
        meaning = [r for r in found["results"] if r.get("match") == "meaning"]
        assert meaning and meaning[0]["fact_id"] == ids[-1]
        assert all(r.get("match") == "meaning" for r in found["results"][len(words):])
    finally:
        emb.close()


def test_a_search_as_of_a_day_sends_nothing_and_records_no_attempt(tmp_path, ranker):
    home = tmp_path / "on"
    deploy_store(home, settings(ranker) + "experience = true\n")
    from holomem.engine.store import MemoryStore

    store = MemoryStore(home / "memory_store.db")
    try:
        today = store.today()  # the store's own day: the facts hold on it, a later day is refused
    finally:
        store.close()
    client = Client(home).start()
    try:
        found, is_error = client.call("memory_search", query=ASK, as_of=today)
    finally:
        client.close()
    assert not is_error and found["count"] == 4 and "memory_attempt" not in found
    assert ranker.requests == []


# -- 7. settings -------------------------------------------------------------------------------------------------------

def test_a_scale_outside_its_set_turns_memory_off_naming_the_key(home):
    deploy_store(home, 'rerank = true\nrerank_scale = "sigmoid-x7"\n')
    session = context(hook(home, "session-start", {"session_id": "s1", "cwd": ""}), "SessionStart")
    assert "rerank_scale" in session and "sigmoid-x7" not in session
    assert recall_in(home) == ""


def test_a_url_with_credentials_or_a_query_is_refused_and_never_echoed(home):
    url = "http://user:s3cr3t-token@127.0.0.1:1/v1/rerank?key=abc"
    deploy_store(home, f'rerank = true\nrerank_url = "{url}"\n')
    session = context(hook(home, "session-start", {"session_id": "s1", "cwd": ""}), "SessionStart")
    assert "rerank_url" in session
    status = holomem(home, "status")
    out = status.stdout.decode("utf-8") + status.stderr.decode("utf-8", errors="replace")
    assert "rerank_url" in out
    client = Client(home).start()
    try:
        answer = client.request("tools/call", {"name": "memory_search", "arguments": {"query": ASK}})
    finally:
        client.close()
    said = json.dumps(answer)
    assert "rerank_url" in said
    recall_in(home)
    state = (home / "state.db").read_bytes() if (home / "state.db").exists() else b""
    for text in (session, out, said):
        assert "s3cr3t" not in text and "key=abc" not in text
    assert b"s3cr3t" not in state and b"key=abc" not in state


def test_plain_http_to_a_public_host_is_unavailable_and_sends_nothing(home):
    deploy_store(home, 'rerank = true\nrerank_url = "http://example.com/v1/rerank"\n')
    assert len(lines(recall_in(home))) == 4  # the off block
    out = holomem(home, "status").stdout.decode("utf-8")
    assert "rescoring: on (local-reranker); endpoint: unavailable" in out
    assert "example.com" not in out
    assert backoff(home) == []


# -- 8. the hook's deadline bounds the call ---------------------------------------------------------------------------

def test_a_short_deadline_cuts_the_call_and_still_gives_the_off_block(tmp_path, ranker):
    off, on = tmp_path / "off", tmp_path / "on"
    deploy_store(off, settings(None))
    expected = recall_in(off)
    deploy_store(on, settings(ranker, rerank_timeout=10.0, hook_deadline_seconds=2.5))
    ranker.mode = "silent"
    started = time.monotonic()
    proc = hook(on, "user-prompt", prompt(ASK))
    elapsed = time.monotonic() - started
    assert context(proc) == expected  # not empty: the deadline did not take the ready block with it
    assert elapsed < 2.5 + 1.0, elapsed  # the deadline counts from the worker's start, after Python's own
    assert len(ranker.requests) == 1
    deadline = time.monotonic() + 5
    while not ranker.closed_after and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ranker.closed_after and ranker.closed_after[0] < 2.5


# -- 9. memory_recall through the server ------------------------------------------------------------------------------

def test_memory_recall_in_the_server_rescores_and_respects_the_pause(tmp_path, ranker):
    home = tmp_path / "on"
    deploy_store(home, settings(ranker, rerank_window=0.0))
    ranker.prefer = {DEPLOYS[3]: 9.0, DEPLOYS[2]: 8.0}
    client = Client(home).start()

    def recalled(session: str) -> str:
        reply = client.request("tools/call", {"name": "memory_recall", "arguments": {"prompt": ASK,
                                                                                     "session_id": session}})
        text = reply["result"]["content"][0]["text"]
        return json.loads(text)["hookSpecificOutput"]["additionalContext"] if text else ""

    try:
        assert lines(recalled("a"))[:2] == [DEPLOYS[3], DEPLOYS[2]]
        ranker.mode = "fail"
        assert len(lines(recalled("b"))) == 4
        assert len(lines(recalled("c"))) == 4
    finally:
        client.close()
    assert len(ranker.requests) == 2  # the failed call once, then the pause


# -- 10. status and stats ---------------------------------------------------------------------------------------------

def test_status_and_stats_say_the_state_never_the_address(tmp_path, ranker):
    home = tmp_path / "on"
    deploy_store(home, settings(ranker))
    out = holomem(home, "status").stdout.decode("utf-8")
    assert "rescoring: on (local-reranker); endpoint: ready" in out
    ranker.mode = "fail"
    recall_in(home)
    out = holomem(home, "status").stdout.decode("utf-8")
    assert re.search(r"rescoring: on \(local-reranker\); endpoint: paused \(HTTP 500\)", out), out
    assert "127.0.0.1" not in out and "/v1/rerank" not in out
    client = Client(home).start()
    try:
        stats, _ = client.call("memory_stats")
    finally:
        client.close()
    assert stats["rescoring"]["model"] == "local-reranker"
    assert stats["rescoring"]["endpoint"].startswith("paused (HTTP 500)")
    assert "127.0.0.1" not in json.dumps(stats["rescoring"])
    off = tmp_path / "off"
    deploy_store(off, "")
    assert "rescoring: off" in holomem(off, "status").stdout.decode("utf-8")


# -- 11. the filter of one ordinary word in every layout --------------------------------------------------------------

AGE_A = "Сколько лет мне было, когда я переехал, сейчас уже не вспомнить."
AGE_B = "Мне сейчас тридцать четыре полных лет."


def _lone_word_store(home: Path, emb: Endpoint, ranker: Reranker | None) -> list[int]:
    text = meaning_config(emb) + "embedding_prefetch_facts = 0\n" + (settings(ranker, rerank_window=0.0)
                                                                     if ranker else "")
    ids = seed(home, [(t, []) for t in FILLER + [LONE_WORD, CITY, AGE_A, AGE_B]], text)
    (home / ".env").write_text(f"HOLO_TEST_EMBED_KEY={KEY}\n", encoding="utf-8")
    assert meaning_cli(home, "embed").returncode == 0
    return ids


def test_one_ordinary_word_is_filtered_in_every_layout(tmp_path, ranker):
    emb = Endpoint()
    try:
        off = tmp_path / "off"
        _lone_word_store(off, emb, None)
        spurious_off = recall_in(off, "Сколько мне сейчас лет?")
        named_off = recall_in(off, "Чему будет учиться ребёнок?", session="b")
        assert LONE_WORD not in spurious_off and CITY in named_off  # (a) off, as before
        on = tmp_path / "on"
        _lone_word_store(on, emb, ranker)
        spurious = recall_in(on, "Сколько мне сейчас лет?")
        assert ranker.requests, "the pool held two facts: a call was made"
        assert LONE_WORD not in ranker.documents and LONE_WORD not in spurious  # (b) unified: never sent
        assert CITY in recall_in(on, "Чему будет учиться ребёнок?", session="b")
        broken = tmp_path / "broken"
        _lone_word_store(broken, emb, ranker)
        ranker.mode, before = "fail", len(ranker.requests)
        assert recall_in(broken, "Сколько мне сейчас лет?") == spurious_off  # (c) fallback = off
        assert len(ranker.requests) == before + 1
    finally:
        emb.close()


# -- 12. rerank_weight > 0 --------------------------------------------------------------------------------------------

def test_a_weight_blends_every_fact_of_the_pool_and_no_window_cuts(tmp_path, ranker):
    off = tmp_path / "off"
    deploy_store(off, settings(None), [UNSENDABLE])
    words = lines(recall_in(off))
    scored = [w for w in words if w != UNSENDABLE]
    assert words.index(UNSENDABLE) < words.index(scored[3])  # the fixture: its words rank it above the last scored
    on = tmp_path / "on"
    deploy_store(on, settings(ranker, rerank_weight=0.5, rerank_window=0.5), [UNSENDABLE])
    # scored[3] gets the scale's bottom (-3: normalised 0), as the fact the endpoint never saw
    ranker.prefer, ranker.default = {scored[2]: 6.0, scored[0]: 0.5, scored[1]: 0.2}, -3.0
    blended = lines(recall_in(on))
    assert len(blended) == 5  # a weight leaves the window out (with weight 0 a 0.5 window keeps two)
    assert blended[0] == scored[2]  # the top of the endpoint's scale: half the order is the endpoint's
    # the unscored fact and the lowest scored one both stand at the bottom of the endpoint's scale: their words then
    # decide, so the unscored one is not simply put behind every scored fact as with weight 0
    assert blended.index(UNSENDABLE) < blended.index(scored[3])
    plain = tmp_path / "plain"
    deploy_store(plain, settings(ranker, rerank_window=0.0), [UNSENDABLE])
    assert lines(recall_in(plain))[-1] == UNSENDABLE  # weight 0: behind every scored fact


# -- 13. ties ---------------------------------------------------------------------------------------------------------

def test_equal_scores_keep_the_endpoints_order(tmp_path, ranker):
    home = tmp_path / "on"
    deploy_store(home, settings(ranker, rerank_window=0.0))
    ranker.mode = "reverse"
    block = lines(recall_in(home))
    assert block == list(reversed(ranker.requests[0]["documents"]))


# -- 14. a pool larger than rerank_top_k -------------------------------------------------------------------------------

def test_top_k_of_the_pool_are_scored_and_first_the_rest_keep_their_order(tmp_path, ranker):
    off, on = tmp_path / "off", tmp_path / "on"
    seed(off, [(text, []) for text in FILLER[:8] + DEPLOYS], settings(None, prefetch_max_facts=8))
    words = lines(recall_in(off))
    assert len(words) == 6
    seed(on, [(text, []) for text in FILLER[:8] + DEPLOYS], settings(ranker, rerank_top_k=3, rerank_window=0.0,
                                                                     prefetch_max_facts=8))
    ranker.prefer = {words[2]: 3.0, words[1]: 2.0, words[0]: 1.0, words[5]: 99.0}
    block = lines(recall_in(on))
    assert ranker.requests[0]["documents"] == words[:3]
    assert block == [words[2], words[1], words[0], words[3], words[4], words[5]]


# -- 15. prefetch_max_facts, not a fixed five -------------------------------------------------------------------------

@pytest.mark.parametrize("most", [2, 8])
def test_prefetch_max_facts_is_the_number_of_lines(tmp_path, ranker, most):
    home = tmp_path / "on"
    seed(home, [(text, []) for text in FILLER[:8] + DEPLOYS], settings(ranker, rerank_window=0.0,
                                                                       prefetch_max_facts=most))
    ranker.prefer, ranker.default = {DEPLOYS[5]: 9.0}, -3.0
    block = lines(recall_in(home))
    assert len(block) == min(most, 6) and block[0] == DEPLOYS[5]
    narrow = tmp_path / "narrow"
    seed(narrow, [(text, []) for text in FILLER[:8] + DEPLOYS], settings(ranker, rerank_window=0.5,
                                                                         prefetch_max_facts=most))
    assert len(lines(recall_in(narrow))) == 2  # the window's two, whatever the setting


# -- 16. a fact changed during the call -------------------------------------------------------------------------------

_CHANGE = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    from holomem.engine.store import MemoryStore
    store = MemoryStore(sys.argv[2])
    store.change_fact(int(sys.argv[3]), sys.argv[4])
    store.close()
""")


def _change(home: Path, fact_id: int, text: str) -> None:
    proc = subprocess.run([sys.executable, "-I", "-c", _CHANGE, str(ROOT), str(home / "memory_store.db"),
                           str(fact_id), text], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-1500:]


def test_a_fact_changed_during_the_call_never_shows_its_old_text_under_a_receipt(tmp_path, ranker):
    home = tmp_path / "on"
    ids = deploy_store(home, settings(ranker, rerank_window=0.0, rerank_timeout=6.0) + "experience = true\n")
    shown = ids[-4]  # DEPLOYS[0]
    ranker.prefer = {DEPLOYS[0]: 9.0}
    ranker.on_request = lambda body: _change(home, shown, "The billing service deploys to production with podman")
    block = recall_in(home)
    assert block and not ATTEMPT.search(block)  # the receipt would pair the old text with the new version: none
    assert len(lines(block)) == 4


def test_a_candidate_changed_that_was_not_shown_leaves_the_receipt(tmp_path, ranker):
    home = tmp_path / "on"
    ids = deploy_store(home, settings(ranker, rerank_window=0.0, rerank_timeout=6.0, prefetch_max_facts=1)
                       + "experience = true\nexperience_trace = true\n")
    ranker.prefer = {DEPLOYS[0]: 9.0}
    hidden = ids[-1]  # DEPLOYS[3]: scored low, not among the one line
    ranker.on_request = lambda body: _change(home, hidden, "The billing service deploy posts to the release channel")
    block = recall_in(home)
    assert lines(block) == [DEPLOYS[0]]
    attempt = ATTEMPT.search(block).group(1)
    client = Client(home).start()
    try:
        shown, _ = client.call("memory_experience", action="explain", attempt_id=attempt)
    finally:
        client.close()
    assert any(stage.get("changed") == 1 for stage in shown["stages"]), shown["stages"]


# -- dirty: state.db held while the call fails ------------------------------------------------------------------------

def test_a_held_state_db_while_the_call_fails_leaves_the_hook_answering(tmp_path, ranker):
    off, home = tmp_path / "off", tmp_path / "on"
    deploy_store(off, settings(None))
    expected = recall_in(off)
    deploy_store(home, settings(ranker))
    holder: list[sqlite3.Connection] = []

    def hold(body) -> None:  # another hook takes state.db just as this call is answered
        conn = sqlite3.connect(home / "state.db", isolation_level=None, check_same_thread=False)
        conn.execute("BEGIN IMMEDIATE")
        holder.append(conn)
        threading.Timer(1.0, lambda: (conn.execute("ROLLBACK"), conn.close())).start()

    ranker.mode, ranker.on_request = "fail", hold
    started = time.monotonic()
    proc = hook(home, "user-prompt", prompt(ASK))
    assert time.monotonic() - started < 8.0
    assert len(ranker.requests) == 1 and holder  # the call was made, and failed while state.db was held
    assert context(proc) == expected
    assert backoff(home) == []  # the pause could not be written: the next call simply asks again
    log = (home / "holomem.log").read_text(encoding="utf-8") if (home / "holomem.log").exists() else ""
    assert "billing" not in log


# -- bookkeeping never costs a ready block; a connection that never came up is no speed --------------------------------

def test_a_state_db_held_after_a_slow_call_still_gives_the_ranked_block(tmp_path, ranker):
    """The call takes 2.5 s and, meanwhile, another process takes state.db and keeps it past the deadline: what is
    left to write (the pause, the speed, the repeat window) is bookkeeping, and the ranked block still goes out."""
    home = tmp_path / "on"
    deploy_store(home, settings(ranker, rerank_window=0.0, rerank_timeout=4.0))
    holders: list[sqlite3.Connection] = []

    def hold_then_answer(body) -> None:
        conn = sqlite3.connect(home / "state.db", isolation_level=None, check_same_thread=False)
        conn.execute("BEGIN IMMEDIATE")  # held until after the hook's deadline
        holders.append(conn)
        time.sleep(2.5)

    ranker.prefer, ranker.on_request = {DEPLOYS[3]: 9.0}, hold_then_answer
    started = time.monotonic()
    try:
        proc = hook(home, "user-prompt", prompt(ASK))
        elapsed = time.monotonic() - started
    finally:
        for conn in holders:
            conn.execute("ROLLBACK")
            conn.close()
    block = lines(context(proc))
    assert len(ranker.requests) == 1 and holders
    assert block and block[0] == DEPLOYS[3]  # the endpoint's order, not the fallback
    assert elapsed < 8.0 + 1.0, elapsed  # the deadline counts from the worker's start, after Python's own


def test_a_connection_that_never_comes_up_leaves_no_speed(tmp_path):
    """The endpoint's address accepts no connection in time (a listener whose queue is full: Linux drops the
    handshake, Windows refuses only after its retries, both well past the call's 1 s): a timeout, a pause, but no
    speed sample — the endpoint never saw the documents, so nothing is known of how fast it scores them."""
    home = tmp_path / "on"
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(0)
    port = listener.getsockname()[1]
    queued = []
    for _ in range(16):  # fill the accept queue; nothing ever accepts
        sock = socket.socket()
        sock.setblocking(False)
        sock.connect_ex(("127.0.0.1", port))
        queued.append(sock)
    try:
        deploy_store(home, f'rerank = true\nrerank_url = "http://127.0.0.1:{port}/v1/rerank"\nrerank_timeout = 1.0\n')
        assert len(lines(recall_in(home))) == 4  # the off block
    finally:
        for sock in queued:
            sock.close()
        listener.close()
    assert backoff(home) == [(60, "no answer (timeout)")]
    assert speeds(home) == []

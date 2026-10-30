"""Search by meaning (embeddings on), against a local OpenAI-shaped endpoint started by the test.

Requirements:
- a fact that shares no word with the message is recalled before the turn when it stands out in meaning, marked
  "(by meaning)"; with embeddings off the same message recalls nothing and nothing is sent anywhere;
- memory_search returns such a fact with match "meaning";
- the key is read from $HOLOMEM_HOME/.env (not needed in the environment) and sent as a Bearer token; holomem status
  says where it came from and never prints it;
- a stored fact that looks like a credential is never sent to the endpoint;
- holomem embed makes the missing vectors; the MCP server makes them in the background after memory_add;
- an endpoint that fails costs one request: the next hooks (new processes) do not wait for it again, and still answer;
- turning embeddings off deletes the stored vectors (nothing derived by an outside model stays in the file);
- a fact that shares one ordinary word with the message ("сейчас"), unique in a small store, and is not close in
  meaning is not recalled; a fact that shares one word and is about the message (a city named in it) still is.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from test_hooks import context, hook, prompt, seed
from test_mcp_server import Client

ROOT = Path(__file__).resolve().parent.parent
_LAUNCH = "import sys; sys.path.insert(0, sys.argv.pop(1)); from holomem.__main__ import main; sys.exit(main())"
KEY = "test-key-7f3a9c"
DIM = 48

# Words that share a meaning: a text holding any of them leans on that axis (a paraphrase shares no word with its fact)
CONCEPTS = [("scooter", "ride", "motorbike", "commute"), ("bread", "bakery", "loaf"), ("python", "script", "code"),
            ("учиться", "школ")]

FILLER = ["The quarterly report goes to the board in March", "The office plant needs water on Mondays",
          "The team retro happens every second Friday", "The printer on the third floor jams often",
          "Invoices are paid within thirty days", "The wifi password changes every quarter in the lobby notice",
          "The cat sleeps on the windowsill in the afternoon", "Laundry day is Sunday",
          "The gym closes at ten in the evening", "Grandma calls on Saturday mornings",
          "The dentist appointment is in December", "Tea is better without sugar",
          "The balcony faces the sea", "The library returns are due on Thursdays",
          "The neighbours have a loud parrot", "Winter coats live in the hall cupboard",
          "Chess club meets on Wednesday nights", "The blue umbrella is broken",
          "Rent is transferred on the first of the month", "Concert tickets are kept in the drawer",
          "The garden hose leaks near the tap", "The old laptop is kept for spare parts"]
TARGET = "Mark gets around town on a blue Vespa scooter"
QUESTION = "What do I ride to the beach every morning?"


def vector(text: str) -> list[float]:
    words = text.lower()
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest() * 3
    # per-text noise well below one concept: a paraphrase then stands out as it does with a real model (cos ~0.9
    # against ~0.1 for unrelated facts), while unrelated texts still differ from each other
    v = [(b - 127.5) / 127.5 * 0.08 for b in seed_bytes[:DIM]]
    for axis, group in enumerate(CONCEPTS):
        if any(w in words for w in group):
            v[axis] += 1.0
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


class Endpoint:
    """POST /v1/embeddings; mode 'ok' or 'fail' (HTTP 500). Records every text sent and the Authorization header."""

    def __init__(self):
        self.mode, self.texts, self.auth, self.requests = "ok", [], [], 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                outer.requests += 1
                outer.auth.append(self.headers.get("Authorization"))
                texts = body.get("input") or []
                texts = [texts] if isinstance(texts, str) else texts
                outer.texts += texts
                if outer.mode == "fail":
                    self.send_response(500)
                    self.end_headers()
                    return
                data = json.dumps({"data": [{"index": i, "embedding": vector(t)} for i, t in enumerate(texts)],
                                   "model": body.get("model")}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def endpoint():
    ep = Endpoint()
    yield ep
    ep.close()


@pytest.fixture
def home(tmp_path) -> Path:
    return tmp_path / "holo"


def config(ep: Endpoint, on: bool = True) -> str:
    return (f"embeddings = {'true' if on else 'false'}\nembedding_url = \"{ep.url}\"\n"
            "embedding_model = \"Fake-Embed-1\"\nembedding_key_env = \"HOLO_TEST_EMBED_KEY\"\n")


def cli(home: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "HOLO_TEST_EMBED_KEY"}
    env.update({"HOLOMEM_HOME": str(home), "PYTHONIOENCODING": "utf-8"})
    return subprocess.run([sys.executable, "-I", "-c", _LAUNCH, str(ROOT), *args], capture_output=True, env=env,
                          timeout=120)


def store_with_vectors(home: Path, ep: Endpoint) -> int:
    ids = seed(home, [(text, []) for text in FILLER + [TARGET]], config(ep))
    (home / ".env").write_text(f'# key for search by meaning\nHOLO_TEST_EMBED_KEY="{KEY}"\n', encoding="utf-8")
    done = cli(home, "embed")
    assert done.returncode == 0, done.stdout.decode(errors="replace") + done.stderr.decode(errors="replace")[-800:]
    return ids[-1]


def vectors(home: Path) -> int:
    conn = sqlite3.connect(home / "memory_store.db")
    try:
        return conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0]
    finally:
        conn.close()


def test_a_fact_sharing_no_word_is_recalled_by_meaning(home, endpoint):
    target = store_with_vectors(home, endpoint)
    assert vectors(home) == len(FILLER) + 1
    text = context(hook(home, "user-prompt", prompt(QUESTION)))
    assert f"- #{target} [0.5] {TARGET} (by meaning)" in text
    assert QUESTION in endpoint.texts  # the message itself was compared, not guessed from words


def test_with_embeddings_off_nothing_is_recalled_or_sent(home, endpoint):
    seed(home, [(text, []) for text in FILLER + [TARGET]], config(endpoint, on=False))
    assert context(hook(home, "user-prompt", prompt(QUESTION))) == ""
    assert endpoint.requests == 0


def test_search_returns_the_fact_by_meaning(home, endpoint):
    target = store_with_vectors(home, endpoint)
    client = Client(home).start()
    try:
        found, is_error = client.call("memory_search", query=QUESTION)
    finally:
        client.close()
    assert not is_error
    hits = [r for r in found["results"] if r["fact_id"] == target]
    assert hits and hits[0]["match"] == "meaning"


def test_the_key_comes_from_the_env_file_and_is_never_printed(home, endpoint):
    store_with_vectors(home, endpoint)
    assert endpoint.auth and set(endpoint.auth) == {f"Bearer {KEY}"}
    status = cli(home, "status")
    out = status.stdout.decode("utf-8")
    assert "search by meaning: on (Fake-Embed-1)" in out  # the model name keeps its case
    assert f"HOLO_TEST_EMBED_KEY: in {home / '.env'}" in out
    assert KEY not in out and KEY not in status.stderr.decode(errors="replace")


def test_a_fact_that_looks_secret_is_never_sent(home, endpoint):
    secret = "The staging deploy token is ghp_0123456789abcdefABCDEF0123456789abcd"
    seed(home, [(text, []) for text in FILLER + [secret]], config(endpoint))
    (home / ".env").write_text(f"HOLO_TEST_EMBED_KEY={KEY}\n", encoding="utf-8")
    assert cli(home, "embed").returncode == 0
    assert endpoint.texts and not any("ghp_" in t for t in endpoint.texts)
    assert vectors(home) == len(FILLER)


def test_the_server_makes_vectors_after_a_new_fact(home, endpoint):
    home.mkdir(parents=True)
    (home / "config.toml").write_text(config(endpoint), encoding="utf-8")
    (home / ".env").write_text(f"HOLO_TEST_EMBED_KEY={KEY}\n", encoding="utf-8")
    def wait_for(count: int) -> None:
        deadline = time.monotonic() + 15  # far below the idle pass (300 s): only the wake after a write is this fast
        while time.monotonic() < deadline and vectors(home) < count:
            time.sleep(0.2)

    client = Client(home).start()
    try:
        added, is_error = client.call("memory_add", content=FILLER[0], origin="said")
        assert not is_error and added["status"] == "added"
        wait_for(1)
        assert vectors(home) == 1  # the start pass (or the first wake) is over: the thread now sleeps
        added, is_error = client.call("memory_add", content=TARGET, origin="said")
        assert not is_error and added["status"] == "added"
        wait_for(2)
        stats, _ = client.call("memory_stats")
    finally:
        client.close()
    assert vectors(home) == 2
    assert stats["search_by_meaning"]["facts_with_vector"] == 2 and TARGET in endpoint.texts


def test_a_failing_endpoint_is_asked_once_and_hooks_still_answer(home, endpoint):
    target = store_with_vectors(home, endpoint)
    endpoint.mode = "fail"
    before = endpoint.requests
    first = hook(home, "user-prompt", prompt("Which bakery bread do I buy on weekends?", session="a"))
    second = hook(home, "user-prompt", prompt(QUESTION, session="b"))
    assert first.returncode == 0 and second.returncode == 0
    assert endpoint.requests - before == 1  # the second process did not try again
    assert f"#{target}" not in context(second)  # no meaning, and the words alone do not reach the scooter fact


def test_turning_embeddings_off_deletes_the_vectors(home, endpoint):
    store_with_vectors(home, endpoint)
    assert vectors(home) > 0
    (home / "config.toml").write_text(config(endpoint, on=False), encoding="utf-8")
    assert cli(home, "status").returncode == 0
    assert vectors(home) == 0


def test_a_credential_anywhere_in_a_long_message_keeps_it_from_the_endpoint(home, endpoint):
    """The whole message is screened, not the part recall reads (4000 characters) or sends (2000)."""
    store_with_vectors(home, endpoint)
    before = endpoint.requests
    long = QUESTION + " " + "and some more words about the trip " * 130 + "token ghp_0123456789abcdefABCDEF0123456789abcd"
    assert len(long) > 4100
    proc = hook(home, "user-prompt", prompt(long))
    assert proc.returncode == 0
    assert endpoint.requests == before


def test_status_never_prints_a_key_pasted_as_the_variable_name(home, endpoint):
    pasted = "sk-proj-AbCdEf0123456789AbCdEf0123456789"
    home.mkdir(parents=True)
    (home / "config.toml").write_text(config(endpoint).replace("HOLO_TEST_EMBED_KEY", pasted), encoding="utf-8")
    status = cli(home, "status")
    out = status.stdout.decode("utf-8") + status.stderr.decode("utf-8", errors="replace")
    assert "search by meaning: on" in out
    assert pasted not in out and "AbCdEf0123" not in out


def test_vectors_are_made_at_session_start_and_the_server_still_serves(home, endpoint):
    """With embeddings on, a new session's server makes the missing vectors while it only waits for the agent (no tool
    call; on Windows a DLL loaded by its thread used to wait for the next one), then recall and tools work on the
    main thread with what that thread opened, and the server exits cleanly."""
    target = seed(home, [(text, []) for text in FILLER + [TARGET]], config(endpoint))[-1]
    (home / ".env").write_text(f"HOLO_TEST_EMBED_KEY={KEY}\n", encoding="utf-8")
    client = Client(home).start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and vectors(home) < len(FILLER) + 1:
        time.sleep(0.2)
    assert vectors(home) == len(FILLER) + 1  # made while the server only waited on stdin
    reply = client.request("tools/call", {"name": "memory_recall", "arguments": {"prompt": QUESTION, "session_id": "x"}})
    recalled = reply["result"]["content"][0]["text"]
    forgot, is_error = client.call("memory_forget", fact_id=target)
    client.proc.stdin.close()
    code = client.proc.wait(timeout=30)
    err = client.proc.stderr.read().decode(errors="replace")
    client.proc.stdout.close()
    client.proc.stderr.close()
    assert f"#{target}" in recalled
    assert not is_error and forgot.get("forgotten") is True, forgot
    assert code == 0 and "Traceback" not in err and "ProgrammingError" not in err, err[-1500:]


LONE_WORD = "В тексте статей блога не писать о прошлых версиях: статья звучит как написанная сейчас."
CITY = "После переезда ребёнок пойдёт в хорошую школу в Лиссабоне."


def test_one_ordinary_shared_word_without_meaning_recalls_nothing(home, endpoint):
    """Seen live (2026-10-30, 35 facts): «Сколько мне сейчас лет?» recalled a rule about blog articles, the only fact
    holding «сейчас»; in a small store an everyday word looks as rare as a name."""
    # no fact on meaning alone: the city's fact can only come through the words, by its one shared word
    ids = seed(home, [(text, []) for text in FILLER + [LONE_WORD, CITY]],
               config(endpoint) + "embedding_prefetch_facts = 0\n")
    lone, city = ids[-2], ids[-1]
    (home / ".env").write_text(f"HOLO_TEST_EMBED_KEY={KEY}\n", encoding="utf-8")
    assert cli(home, "embed").returncode == 0
    spurious = context(hook(home, "user-prompt", prompt("Сколько мне сейчас лет?", session="a")))
    assert f"#{lone}" not in spurious
    named = context(hook(home, "user-prompt", prompt("Чему будет учиться ребёнок?", session="b")))
    assert f"#{city}" in named  # one ordinary shared word too («ребёнок»), but meaning backs it


def test_one_ordinary_shared_word_recalls_nothing_while_meaning_is_down(home, endpoint):
    """Seen live (2026-10-30): the embeddings endpoint timed out, search by meaning paused for every hook, and for those
    minutes «Сколько мне сейчас лет?» brought the blog rule again: the filter needed meaning to judge one shared word.
    With search by meaning on but not answering, one ordinary shared word is not enough; two still are."""
    ids = seed(home, [(text, []) for text in FILLER + [LONE_WORD]], config(endpoint))
    lone = ids[-1]
    (home / ".env").write_text(f"HOLO_TEST_EMBED_KEY={KEY}\n", encoding="utf-8")
    assert cli(home, "embed").returncode == 0
    endpoint.mode = "fail"
    first = context(hook(home, "user-prompt", prompt("Сколько мне сейчас лет?", session="a")))  # the call fails
    paused = context(hook(home, "user-prompt", prompt("Что сейчас нового?", session="b")))  # paused: no call
    assert f"#{lone}" not in first and f"#{lone}" not in paused
    two = context(hook(home, "user-prompt", prompt("Как писать текст статей для блога?", session="c")))
    assert f"#{lone}" in two


def test_one_ordinary_shared_word_is_left_to_the_words_before_any_vector_exists(home, endpoint):
    """Codex review 2026-10-31: search by meaning just turned on (or a new model), the endpoint answers, no fact has a
    vector yet: that is not an outage, nothing is judged by meaning yet, and the words decide as before."""
    ids = seed(home, [(text, []) for text in FILLER + [LONE_WORD]], config(endpoint))
    (home / ".env").write_text(f"HOLO_TEST_EMBED_KEY={KEY}\n", encoding="utf-8")
    assert f"#{ids[-1]}" in context(hook(home, "user-prompt", prompt("Сколько мне сейчас лет?")))


SCHOOL = "Лена ходит в хорошую школу у моря."


def test_a_name_in_passing_brings_no_fact_meaning_does_not_back(home, endpoint):
    """A long message that names someone in passing ("Лена прислала счёт…") brought that person's top facts by the
    name alone, about anything; with meaning on, such a fact must stand out for the message (z ≥ 1). A message about
    the same person and the fact's subject still brings it."""
    ids = seed(home, [(text, []) for text in FILLER] + [(SCHOOL, ["Лена"])], config(endpoint))
    school = ids[-1]
    (home / ".env").write_text(f"HOLO_TEST_EMBED_KEY={KEY}\n", encoding="utf-8")
    assert cli(home, "embed").returncode == 0
    passing = "Лена прислала счёт за электричество, интернет и воду за прошлый месяц, оплати его до пятницы"
    assert f"#{school}" not in context(hook(home, "user-prompt", prompt(passing, session="a")))
    about = "Лена спрашивает, где и чему она будет учиться в следующем году после переезда в другой город"
    assert f"#{school}" in context(hook(home, "user-prompt", prompt(about, session="b")))

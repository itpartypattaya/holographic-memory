"""A Latin word or identifier written into Chinese or Thai text with no space around it is found by itself (ported
2026-11-02 from the Hermes plugin's #65, NousResearch/hermes-agent#83593, as holographic-memory's own change).

FTS5 indexes "用户用keyring保存密钥" as one token, letters and characters together, and the n-gram path runs only for a
query with that script in it, so `keyring` alone found nothing while `密钥` did, and an English question about such a
fact recalled nothing before the turn. The requirement: memory_search finds the fact by the word, only as a whole word
(`key` is not in `keyring`, `pid` not in `ppid`), and the recall before a message counts the fact as holding the word.
"""
from __future__ import annotations

import pytest

from test_hooks import context, hook, prompt

pytest.importorskip("numpy")

WATCHDOG = "桌面后端秒退根因:web_server的HERMES_PARENT_PID父进程watchdog单层ppid比较会误杀健康后端。"
KEYRING = "用户用keyring保存密钥"


@pytest.fixture
def tools(tmp_path, monkeypatch):
    from holomem import config
    from holomem.memory import Context, open_memory
    from holomem.state import State
    from holomem.tools import Tools

    monkeypatch.setenv("HOLOMEM_HOME", str(tmp_path / "holo"))
    cfg = config.load()
    ctx = Context(client="claude-code", session_id="s1")
    memory = open_memory(cfg, ctx)
    yield Tools(memory, ctx, State(cfg.state_path))
    memory.store.close()


def add(tools, text: str) -> int:
    answer, is_error = tools.call("memory_add", {"content": text, "origin": "said"})
    assert not is_error and answer["status"] == "added", answer
    return answer["fact_id"]


def found(tools, query: str) -> list[int]:
    answer, is_error = tools.call("memory_search", {"query": query})
    assert not is_error, answer
    return [r["fact_id"] for r in answer.get("results", [])]


@pytest.mark.parametrize("query", ["watchdog", "ppid", "HERMES_PARENT_PID", "web_server", "Watchdog"])
def test_a_word_glued_to_chinese_is_found_by_itself(tools, query):
    fact = add(tools, WATCHDOG)
    assert found(tools, query) == [fact]


def test_the_issue_keyring_example(tools):
    fact = add(tools, KEYRING)
    add(tools, "用户的猫叫小白")
    assert found(tools, "keyring") == [fact]
    assert found(tools, "keyring 密钥") == [fact]  # unchanged: a query with Chinese found it before


@pytest.mark.parametrize("query", ["key", "ring", "dog", "pid", "server", "web"])
def test_only_the_whole_word(tools, query):
    add(tools, KEYRING)
    add(tools, WATCHDOG)
    assert found(tools, query) == []


def test_a_letter_of_another_script_beside_the_word_is_no_edge(tools):
    add(tools, "яkeyring保存")
    assert found(tools, "keyring") == []


def test_a_word_glued_to_thai_is_found_too(tools):
    fact = add(tools, "เซิร์ฟเวอร์ใช้systemdในการรัน")
    assert found(tools, "systemd") == [fact]


def test_both_spellings_are_found(tools):
    glued = add(tools, KEYRING)
    apart = add(tools, "The keyring backend on the laptop is KWallet")
    assert sorted(found(tools, "keyring")) == sorted([glued, apart])


def test_an_english_question_recalls_the_chinese_fact(tmp_path):
    """The hook, as a client runs it: before, the word counted as held by no fact, so nothing was recalled."""
    from test_hooks import seed

    home = tmp_path / "holo"
    seed(home, [(WATCHDOG, []), ("The production server hr-vps runs Debian 12", [])])
    block = context(hook(home, "user-prompt", prompt("How does the watchdog compare ppid?")))
    assert "watchdog单层ppid" in block and "Debian" not in block, block

"""As the Claude Code plugin, the embeddings key comes from the plugin's own option, never from the person's machine.

Requirement: the Claude plugin directory holds a plugin that reads a credential from the user's environment or files
and could send it to a server ("Uses a credential from the user's machine", 2026-11-05), and asks for the value through
a userConfig option marked sensitive instead. So, run as the plugin (--plugin), search by meaning takes its key only
from the option embedding_api_key: Claude Code keeps it in the system's secure storage and hands it to the hooks and
to the MCP server as CLAUDE_PLUGIN_OPTION_EMBEDDING_API_KEY. The variable embedding_key_env names is read neither from
the environment nor from $HOLOMEM_HOME/.env. An option left empty, or a reference Claude Code left as written
("${user_config.embedding_api_key}"), is no key. Installed with `holomem install` (not the plugin), the key is found as
before: $HOLOMEM_HOME/.env first, then the environment.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from _launch import LAUNCH, child_env
from test_hooks import context, prompt, seed
from test_meaning import FILLER, QUESTION, ROOT, TARGET, Endpoint, cli, config, vectors

OPTION = "CLAUDE_PLUGIN_OPTION_EMBEDDING_API_KEY"


@pytest.fixture
def endpoint():
    ep = Endpoint()
    yield ep
    ep.close()


@pytest.fixture
def home(tmp_path):
    return tmp_path / "holo"

ENV_KEY, FILE_KEY, OPTION_KEY = "env-key-1111", "file-key-2222", "option-key-3333"


def _hook(home, *, plugin: bool, option: str | None) -> subprocess.CompletedProcess:
    # an empty Claude config: the plugin stands aside where `holomem install claude-code` is in place
    claude_dir = home.parent / "claude-config"
    claude_dir.mkdir(exist_ok=True)
    env = child_env(drop=("HOLO_TEST_EMBED_KEY", OPTION), HOLOMEM_HOME=str(home), HOLO_TEST_EMBED_KEY=ENV_KEY,
                    CLAUDE_CONFIG_DIR=str(claude_dir))
    if option is not None:
        env[OPTION] = option
    argv = [sys.executable, "-I", "-c", LAUNCH, str(ROOT), "hook", "user-prompt", "--client", "claude-code"]
    return subprocess.run(argv + (["--plugin"] if plugin else []), input=json.dumps(prompt(QUESTION)).encode(),
                          capture_output=True, env=env, timeout=60)


def _store(home, ep) -> None:
    """Facts with their vectors (made by `holomem embed`, not the plugin), then a clean record of what was sent: a
    message is compared by meaning only where facts have vectors."""
    seed(home, [(text, []) for text in FILLER + [TARGET]], config(ep))  # embedding_key_env = HOLO_TEST_EMBED_KEY
    (home / ".env").write_text(f'HOLO_TEST_EMBED_KEY="{FILE_KEY}"\n', encoding="utf-8")
    done = cli(home, "embed")
    assert done.returncode == 0, done.stderr.decode(errors="replace")[-800:]
    assert ep.requests and vectors(home) == len(FILLER) + 1
    ep.auth.clear()
    ep.requests = 0


def test_the_plugin_sends_the_key_of_its_option(home, endpoint):
    _store(home, endpoint)
    context(_hook(home, plugin=True, option=OPTION_KEY))
    assert endpoint.requests and set(endpoint.auth) == {f"Bearer {OPTION_KEY}"}


@pytest.mark.parametrize("option", [None, "", "${user_config.embedding_api_key}"])
def test_the_plugin_never_sends_a_key_from_the_machine(home, endpoint, option):
    _store(home, endpoint)
    context(_hook(home, plugin=True, option=option))
    assert endpoint.requests  # the message was compared, without a key (a local endpoint may be asked without one)
    sent = [a for a in endpoint.auth if a]
    assert not any(ENV_KEY in a or FILE_KEY in a for a in sent)
    assert sent == []


def test_installed_not_as_the_plugin_the_key_is_found_as_before(home, endpoint):
    _store(home, endpoint)
    context(_hook(home, plugin=False, option=OPTION_KEY))
    assert endpoint.requests and set(endpoint.auth) == {f"Bearer {FILE_KEY}"}


def test_the_plugins_mcp_server_takes_the_key_of_its_option(home, endpoint, monkeypatch):
    from holomem import config as holo_config
    from holomem.memory import Context, open_memory

    _store(home, endpoint)
    monkeypatch.setenv("HOLOMEM_HOME", str(home))
    monkeypatch.setenv("HOLO_TEST_EMBED_KEY", ENV_KEY)
    monkeypatch.setenv(OPTION, OPTION_KEY)
    for plugin, expected in ((True, OPTION_KEY), (False, FILE_KEY)):
        memory = open_memory(holo_config.load(), Context(client="claude-code", plugin=plugin))
        try:
            assert memory.embedder._key() == expected
        finally:
            memory.close()
    monkeypatch.delenv(OPTION)
    memory = open_memory(holo_config.load(), Context(client="claude-code", plugin=True))
    try:
        assert memory.embedder._key() == ""  # not the environment's, not the .env file's
    finally:
        memory.close()

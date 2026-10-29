"""Settings: ~/.holographic/config.toml.

Requirement: with no file the defaults hold and memory works; a value is read from the file; a number that is not
one, a value outside a key's set, or a file that does not parse turns memory off with a reason that names the key
and never quotes the value (a key pasted into the wrong line must not reach a prompt or a log); a key the bridge does
not know is reported as a warning, not silently dropped; HOLOMEM_HOME moves the whole directory; ``~`` in db_path is
the home directory; scope "project" keeps one database per project.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from holomem import config


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "holo"
    monkeypatch.setenv("HOLOMEM_HOME", str(path))
    return path


def _write(home: Path, text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(text, encoding="utf-8")


def test_no_file_means_defaults_and_memory_on(home):
    cfg = config.load()
    assert cfg.problems == [] and cfg.warnings == []
    assert cfg.db_path == home / "memory_store.db"
    assert (cfg.min_trust_threshold, cfg.prefetch_min_coverage, cfg.prefetch_repeat_turns) == (0.3, 0.3, 6)
    assert cfg.auto_extract is False and cfg.scope == "global" and cfg.store_policy == "ask"


def test_values_are_read(home):
    _write(home, 'prefetch_repeat_turns = 3\nmin_trust_threshold = 0.4\nauto_extract = true\nstore_policy = "proactive"\n')
    cfg = config.load()
    assert cfg.problems == []
    assert (cfg.prefetch_repeat_turns, cfg.min_trust_threshold, cfg.auto_extract, cfg.store_policy) == (3, 0.4, True,
                                                                                                    "proactive")


def test_a_bad_number_names_the_key_not_the_value(home):
    _write(home, 'min_trust_threshold = "sk-live-0123456789abcdef"\n')
    cfg = config.load()
    assert cfg.problems and "min_trust_threshold" in cfg.problems[0]
    assert "sk-live" not in " ".join(cfg.problems)


def test_a_value_outside_the_set_turns_memory_off(home):
    _write(home, 'scope = "everyone"\n')
    cfg = config.load()
    assert any("scope" in p for p in cfg.problems) and "everyone" not in " ".join(cfg.problems)


def test_a_file_that_does_not_parse_turns_memory_off(home):
    _write(home, 'prefetch_repeat_turns = 3\nkey = "unterminated\n')
    cfg = config.load()
    assert cfg.problems and "config.toml" in cfg.problems[0]
    assert "unterminated" not in " ".join(cfg.problems)


def test_an_unknown_key_is_a_warning(home):
    _write(home, 'prefetch_repeat_turn = 3\n')  # a typo of prefetch_repeat_turns
    cfg = config.load()
    assert cfg.problems == [] and cfg.warnings and "prefetch_repeat_turn" in cfg.warnings[0]


def test_tilde_in_db_path_is_the_home_directory(home, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "u"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "u"))
    _write(home, 'db_path = "~/mem/facts.db"\n')
    assert config.load().db_path == tmp_path / "u" / "mem" / "facts.db"


def test_project_scope_keeps_one_database_per_project(home, tmp_path):
    _write(home, 'scope = "project"\n')
    cfg = config.load()
    a, b = tmp_path / "repo-a", tmp_path / "repo-b"
    for repo in (a, b):
        (repo / ".git").mkdir(parents=True)
        (repo / "src").mkdir()
    assert cfg.database_for(a / "src") == cfg.database_for(a)
    assert cfg.database_for(a) != cfg.database_for(b)
    assert cfg.database_for(a).parent == home / "projects"
    assert cfg.database_for(None) is None  # no project: nothing to recall or write in project scope


def test_global_scope_ignores_the_project(home, tmp_path):
    cfg = config.load()
    assert cfg.database_for(tmp_path) == cfg.database_for(None) == home / "memory_store.db"


def test_a_negative_budget_is_refused(home):
    _write(home, 'prefetch_max_tokens = -5\n')
    assert any("prefetch_max_tokens" in p for p in config.load().problems)

"""Failures said, not swallowed (review against "10 principles of good code for AI agents", 2026-10-21).

A declared name that cannot link, a setting that is no number, a recall before a turn that fails, a store that cannot be
counted, an extraction that cannot store, an embeddings setting replaced or clamped, a database whose owner-only mode
cannot be set, a settings file that cannot be read: each used to be swallowed (or logged at debug) and is now said where
someone looks. Plus the credential screen's time on its longest inputs."""
from __future__ import annotations

import logging
import os
import time

import pytest

from holomem.engine import embeddings
from holomem.engine.screen import _fact_problem, _recall_problem, _secret_in
from holomem.engine.history import NotFound
from holomem.engine.store import MemoryStore, unusable_names

DIM = 64


# -- declared names that cannot link ----------------------------------------------------------------------------------


def test_unusable_names_takes_what_a_declaration_takes():
    assert unusable_names(["Masha", "Jean-Luc", ".NET", "3M"]) == []
    assert [e["index"] for e in unusable_names(["a\nb", "Masha", "https://example.com", "x", None])] == [0, 2, 3, 4]


# -- not found ---------------------------------------------------------------------------------------------------------

def test_not_found_is_a_key_error_of_its_own(tmp_path):
    s = MemoryStore(str(tmp_path / "s.db"), hrr_dim=DIM)
    assert issubclass(NotFound, KeyError)
    with pytest.raises(NotFound):
        s.forget_fact(12345)
    with pytest.raises(NotFound):
        s.delete_entity("Nobody Here")
    s.close()


# -- search by meaning's settings --------------------------------------------------------------------------------------

def test_an_embeddings_setting_replaced_or_clamped_is_said(caplog):
    with caplog.at_level(logging.WARNING):
        s = embeddings.Settings({"embedding_timeout": "fast", "embedding_weight": 2, "embedding_prefetch_facts": "2"})
    assert (s.timeout, s.weight, s.recall_facts) == (3.0, 0.8, 2)
    assert "embedding_timeout must be a number; using 3.0" in caplog.text and "fast" not in caplog.text
    assert "embedding_weight = 2.0 is outside 0.0..0.8; using 0.8" in caplog.text
    assert "embedding_prefetch_facts" not in caplog.text


def test_unset_embeddings_settings_say_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        s = embeddings.Settings({})
    assert s.timeout == 3.0 and s.dimensions is None and caplog.text == ""


# -- the file, the config ---------------------------------------------------------------------------------------------

@pytest.mark.skipif(os.name != "posix", reason="owner-only mode is set on POSIX only")
def test_an_owner_only_mode_that_cannot_be_set_is_said_once(tmp_path, monkeypatch, caplog):
    from holomem.engine import store as store_module
    path = tmp_path / "p.db"
    path.write_bytes(b"")
    os.chmod(path, 0o644)

    def refuse(*args, **kwargs):
        raise PermissionError("not the owner")

    monkeypatch.setattr(store_module.os, "chmod", refuse)
    with caplog.at_level(logging.WARNING):
        store_module._restrict_permissions(path)
        store_module._restrict_permissions(path)
    assert caplog.text.count("readable by its owner only") == 1


# -- the screen's time on its longest inputs ---------------------------------------------------------------------------

LONGEST = [
    "password " * 230, "отправь пароль " * 140, "token: " * 300, "пароль от wifi: " * 130, "a-" * 1000,
    "x" * 2000, "密码是" * 700, "=" * 2000, "OPENAI_API_KEY=" * 140, "send the " * 230, "«" * 2000,
    "ignore all previous " * 100, "pins card " * 200, "password-" * 230, "~/.atlas/token-" * 130,
]


@pytest.mark.parametrize("text", LONGEST, ids=range(len(LONGEST)))
def test_the_screen_takes_little_time_on_its_longest_inputs(text):
    """A fact is screened at up to 1,000 characters, a message for search by meaning at up to 2,000; a stored fact
    an older version wrote may be longer, and recall screens it whole. 0.5 s is far above what any of these takes
    (0.17.3: 28 ms on the worst 2,000 characters), so a regex that turns quadratic fails here first."""
    text = text[:2000]
    start = time.perf_counter()
    _fact_problem(text[:1000])
    _recall_problem(text)
    _secret_in(text)
    assert time.perf_counter() - start < 0.5

"""Names given with an add of a text merged into another fact go to that fact only while it still says that text.

Codex review 2026-10-31: memory_add found the merged text (merged_match) and then declared the names in a second
step; another process that changed the kept fact in between ("lives in Paris" -> "lives in Berlin") got the names
on its new text. Recognising the merged text and declaring the names are one transaction now."""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")


def linked(store, fact_id: int, name: str) -> bool:
    return bool(store.read("SELECT 1 FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
                           "WHERE fe.fact_id = ? AND e.name = ?", (fact_id, name)))


def test_a_keep_changed_meanwhile_does_not_get_the_names_of_its_old_text(tmp_path, monkeypatch):
    from holomem import config
    from holomem.memory import Context, open_memory
    from holomem.state import State
    from holomem.tools import Tools

    monkeypatch.setenv("HOLOMEM_HOME", str(tmp_path / "holo"))
    cfg = config.load()
    ctx = Context(client="claude-code", session_id="s1")
    memory = open_memory(cfg, ctx)
    tools = Tools(memory, ctx, State(cfg.state_path))
    store = memory.store
    keep = tools.call("memory_add", {"content": "Alice Smith lives in Paris", "origin": "said"})[0]["fact_id"]
    copy = tools.call("memory_add", {"content": "Alice Smith is living in Paris", "origin": "said"})[0]["fact_id"]
    answer, is_error = tools.call("memory_merge", {"fact_id": keep, "from": [copy]})
    assert not is_error, answer

    found = store.merged_match
    calls = []

    def another_process_changes_the_keep(text):
        match = found(text)
        first = not calls
        calls.append(match)
        if first:  # once, right after the merged text was recognised (the change itself asks merged_match too)
            store.change_fact(keep, "Alice Smith lives in Berlin")
        return match

    monkeypatch.setattr(store, "merged_match", another_process_changes_the_keep)
    answer, is_error = tools.call("memory_add", {"content": "Alice Smith is living in Paris", "entities": ["Mila"],
                                                 "origin": "said"})
    assert not is_error, answer
    assert calls and calls[0] is not None  # the race happened: the text was recognised before the change
    assert store.read("SELECT content FROM facts WHERE fact_id = ?", (keep,))[0][0] == "Alice Smith lives in Berlin"
    assert not linked(store, keep, "Mila"), answer
    memory.close() if hasattr(memory, "close") else store.close()

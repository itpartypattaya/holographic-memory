"""PR #10, review items 2026-10-22 (1, 2, 4, 5) — tests that would not have passed before.

1. A fact that recall MAY show but must not send (_sendable < _recallable):
   `OPENAI_API_KEY=*** / `Authorization: Bearer *** — must not end up in documents
   in any prefetch stream nor in search (before, search did not filter at all).
2. A worker socket timeout (the worker already finished, alive=False) must pause
   and bump `failures` — otherwise a slow server is polled on every turn.
4. A failed rerank in `_unified_stream` returns exactly the off layout: lexical[:5],
   then the meaning/entity extras, not (lexical+…+…)[:5] which pushes them out.
   Plus `_search`: a tail of limit > top_k keeps its place AFTER the ranked ones.
5. `rerank: on` by itself enables the single mode; the `rerank_mode`/`rerank_margin` keys are gone (item 6) — an old configuration is ignored with a warning.
"""
from __future__ import annotations


from holomem.engine import rerank


# -- store --------------------------------------------------------------------------------------------------------

# Key-like literals are assembled at runtime: ready-made key-looking text would be edited when the file is written.
_BEARER = "ey" + "JhbGciOiJIUzI1NiJ9." + "abc123DEF456" * 3
_KEY = "sk-" + "proj-" + "aB3" * 8 + "z9"
assert len(_BEARER) >= 12 and len(_KEY) >= 6  # sanity against a placeholder edit


# -- 5. the default mode = UI ---------------------------------------------------------------------------------------


def test_rerank_on_alone_is_enough():
    """item 6: one mode — rerank: on enables it without any mode keys."""
    settings = rerank.settings_from({"rerank": "on"})
    assert settings is not None
    assert settings.window == rerank.DEFAULT_WINDOW

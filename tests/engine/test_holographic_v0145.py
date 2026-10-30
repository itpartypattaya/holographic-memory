"""0.14.5: parallel searches, the secret shapes a fact must not hold, and loanwords written in Latin or Cyrillic."""
from __future__ import annotations


import pytest


# -- a token whose shape says it is one --------------------------------------------------------------------------------

from holomem.engine.screen import _fact_problem, _recall_problem, _secret_in

# Made-up tokens, put together at run time so that secret scanners do not flag the test file (GitHub raised an alert
# on the literal bot token of 0.14.5).
BOT_TOKEN = "7123456789" + ":" + "AA" + "x" * 33
JWT = "eyJ" + "hbGciOiJIUzI1NiJ9" + ".eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0" + "." + "s" * 43
TOKENS = [f"The family bot runs on {BOT_TOKEN}", f"Session cookie {JWT}", "Deploy uses " + "ghp_" + "a" * 24,
          "The OpenAI one is " + "sk-proj-" + "b" * 20, "Key file: -----BEGIN OPENSSH PRIVATE " + "KEY----- b3BlbnNzaC1r"]


@pytest.mark.parametrize("text", TOKENS)
def test_a_bare_token_is_never_stored_recalled_or_sent(text):
    assert _fact_problem(text) and _recall_problem(text) and _secret_in(text)  # bot token and JWT: none of them before


@pytest.mark.parametrize("text", ["The meeting moved to 12:30 on Friday", "Nala's bot id is 7123456789",
                                  "Nala runs with MAX_TOKENS=128000", "Order 123456:7 shipped",
                                  "The token expires in 30 days", "eyJ is how a JWT begins"])
def test_ordinary_text_is_still_a_fact(text):
    assert _fact_problem(text) is None and _recall_problem(text) is None


# -- one word in Latin and Cyrillic ------------------------------------------------------------------------------------

from holomem.engine import text as T  # noqa: E402


def test_the_mark_never_reaches_an_fts_query():
    expr = T.query_terms("бэкап")[0].fts(T.FTS_OPERATORS)
    assert "loan" not in expr and '"backup"*' in expr and '"бекап"*' in expr

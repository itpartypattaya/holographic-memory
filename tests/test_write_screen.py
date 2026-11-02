"""What is no fact is not stored, though nothing in it is unsafe (ported 2026-11-02 from the Hermes plugin's #66, #67
and #68, as holographic-memory's own change).

- A prompt to an assistant (hermes-agent#66380): "You are Henry, … You received an email …" stored as a fact came back
  before every message as a live task. It is refused by memory_add, change, correct and a revert to it, and one stored
  before is not recalled (memory_search still shows it, to be cleaned up). A rule for the assistant ("You are allowed
  to push to main") and a second-person fact ("You received a parcel from DHL") stay facts.
- The agent's report of its own steps (hermes-agent#117790): "I just read the file and updated the config as
  requested." is refused unless origin says the person said or confirmed it; "I moved to Bangkok" stays a fact.
- auto_extract (hermes-agent#22907): a reply to the assistant ("I like that, sounds good") is no preference; "I like
  that, I always use tabs" keeps its second half; a prompt pasted whole is not taken sentence by sentence.
"""
from __future__ import annotations

import pytest

from test_hooks import context, hook, prompt, seed

pytest.importorskip("numpy")

ROLE = ("You are Henry, the deal-flow assistant. You received an email at the AgentMail inbox and should handle it "
        "like a normal Henry request.\nEMAIL METADATA\nFrom: founder@example.com")
STEPS = "I just read the file and updated the config as requested."


@pytest.fixture
def memory(tmp_path, monkeypatch):
    from holomem import config
    from holomem.memory import Context, open_memory

    monkeypatch.setenv("HOLOMEM_HOME", str(tmp_path / "holo"))
    (tmp_path / "holo").mkdir()
    (tmp_path / "holo" / "config.toml").write_text("auto_extract = true\n", encoding="utf-8")
    m = open_memory(config.load(), Context(client="claude-code", session_id="s1"))
    yield m
    m.store.close()


@pytest.fixture
def tools(memory):
    from holomem.memory import Context
    from holomem.state import State
    from holomem.tools import Tools

    return Tools(memory, Context(client="claude-code", session_id="s1"), State(memory.cfg.state_path))


def call(tools, name: str, **args) -> tuple[dict, bool]:
    return tools.call(name, args)


def test_a_prompt_to_an_assistant_is_refused_at_every_write(tools):
    answer, is_error = call(tools, "memory_add", content=ROLE, origin="inferred")
    assert is_error and "prompt to an assistant" in answer["error"], answer
    fact = call(tools, "memory_add", content="Henry handles the deal-flow inbox", origin="said")[0]["fact_id"]
    for tool in ("memory_change", "memory_correct"):
        answer, is_error = call(tools, tool, fact_id=fact, content=ROLE)
        assert is_error and "prompt to an assistant" in answer["error"], (tool, answer)


@pytest.mark.parametrize("text", ["You are allowed to push to main without asking.",
                                  "You received a parcel from DHL on Monday.",
                                  "When I forward a message, you should handle it like my own request."])
def test_a_rule_or_a_second_person_fact_stays_a_fact(tools, text):
    answer, is_error = call(tools, "memory_add", content=text, origin="said")
    assert not is_error and answer["status"] == "added", answer


def test_a_prompt_stored_before_is_not_recalled(tmp_path):
    home = tmp_path / "holo"
    (role,) = seed(home, [(ROLE, [])])  # written by an older version, before the screen knew it
    block = context(hook(home, "user-prompt", prompt("Did Henry get the AgentMail email from the founder?")))
    assert f"#{role} " not in block, block


def test_a_revert_to_a_prompt_is_refused(tools, memory):
    fact = memory.store.add_fact(ROLE)  # an old version, stored before the screen knew it
    answer, is_error = call(tools, "memory_change", fact_id=fact, content="Henry handles the deal-flow inbox")
    assert not is_error, answer
    answer, is_error = call(tools, "memory_revert", fact_id=fact)
    assert is_error and "prompt to an assistant" in answer["error"], answer
    assert memory.store.read("SELECT content FROM facts WHERE fact_id = ?", (fact,))[0][0] == \
        "Henry handles the deal-flow inbox"


def test_the_agents_report_of_its_steps_is_refused_unless_the_person_said_it(tools):
    answer, is_error = call(tools, "memory_add", content=STEPS, origin="inferred")
    assert is_error and "report of steps" in answer["error"], answer
    answer, is_error = call(tools, "memory_add", content=STEPS)  # no origin: the agent's own words
    assert is_error, answer
    answer, is_error = call(tools, "memory_add", content="I updated my passport in March as you asked.", origin="said")
    assert not is_error and answer["status"] == "added", answer


@pytest.mark.parametrize("text", ["I moved to Bangkok in 2024.", "I switched the billing service to Postgres.",
                                  "The course was updated as requested by the dean."])
def test_a_persons_own_past_stays_a_fact(tools, text):
    answer, is_error = call(tools, "memory_add", content=text, origin="inferred")
    assert not is_error and answer["status"] == "added", answer


def extracted(memory, text: str) -> list[str]:
    from holomem.extract import extract
    from holomem.memory import Context

    ids = extract(memory, Context(client="claude-code", session_id="s1"), text)
    return [memory.store.read("SELECT content FROM facts WHERE fact_id = ?", (i,))[0][0] for i in ids]


@pytest.mark.parametrize("text", ["I like that, sounds good.", "Ok, I love it!", "I prefer this one.",
                                  "Я предпочитаю этот вариант."])
def test_auto_extract_skips_a_reply_to_the_assistant(memory, text):
    assert extracted(memory, text) == []


def test_auto_extract_keeps_a_preference_beside_a_reply(memory):
    assert extracted(memory, "I like that, I always use tabs in Python.") == ["I like that, I always use tabs in Python."]


def test_auto_extract_does_not_take_a_pasted_prompt_apart(memory):
    text = ("You are Mia, the family assistant. You received a new message from the school.\n"
            "I prefer short answers in Russian. I always use tabs in Python.")
    assert extracted(memory, text) == []


# Codex review 2026-11-02 (all new code): four misses of the screen above.

FOLLOWING = ("You are Henry, the deal-flow assistant. You received the following email from the founder. Handle it like "
             "a normal Henry request.")


def test_a_prompt_that_received_the_following_email_is_refused_and_not_recalled(tools, tmp_path):
    answer, is_error = call(tools, "memory_add", content=FOLLOWING, origin="inferred")
    assert is_error and "prompt to an assistant" in answer["error"], answer
    home = tmp_path / "older"
    (role,) = seed(home, [(FOLLOWING, [])])
    block = context(hook(home, "user-prompt", prompt("What email did Henry receive from the founder?")))
    assert f"#{role} " not in block, block


CONDITIONAL = ["You are allowed to deploy once you have received a ticket from the owner.",
               "You are required to answer within a day when you have received a request from a client."]


@pytest.mark.parametrize("text", CONDITIONAL)
def test_a_rule_with_a_condition_stays_a_fact(tools, text):
    answer, is_error = call(tools, "memory_add", content=text, origin="said")
    assert not is_error and answer["status"] == "added", answer


def test_a_rule_with_a_condition_stored_before_is_recalled(tmp_path):
    home = tmp_path / "holo"
    (rule,) = seed(home, [(CONDITIONAL[0], [])])
    block = context(hook(home, "user-prompt", prompt("Am I allowed to deploy after a ticket from the owner?")))
    assert f"#{rule} " in block, block


PASSPORT = "I updated my passport in March as you asked."


def test_the_persons_own_words_can_be_corrected_changed_and_reverted(tools):
    fact = call(tools, "memory_add", content=PASSPORT, origin="said")[0]["fact_id"]
    answer, is_error = call(tools, "memory_correct", fact_id=fact, content="I updated my passport in April as you asked.")
    assert not is_error and answer["status"] == "corrected", answer  # a correction keeps the fact's origin
    answer, is_error = call(tools, "memory_change", fact_id=fact, origin="said",
                            content="I renewed my passport in May as you asked.")
    assert not is_error and answer["status"] == "changed", answer
    answer, is_error = call(tools, "memory_change", fact_id=fact, content="I keep my passport in the safe.")
    assert not is_error, answer
    answer, is_error = call(tools, "memory_revert", fact_id=fact)  # back to the person's own words
    assert not is_error and answer["status"] == "reverted", answer


def test_the_agents_report_stays_refused_on_change_correct_and_revert(tools, memory):
    fact = call(tools, "memory_add", content="The config uses port 8080", origin="inferred")[0]["fact_id"]
    for tool in ("memory_change", "memory_correct"):
        answer, is_error = call(tools, tool, fact_id=fact, content=STEPS)
        assert is_error and "report of steps" in answer["error"], (tool, answer)
    old = memory.store.add_fact(STEPS)  # stored before the screen knew it, with no origin
    assert not call(tools, "memory_change", fact_id=old, content="The config uses port 8080 now")[1]
    answer, is_error = call(tools, "memory_revert", fact_id=old)
    assert is_error and "report of steps" in answer["error"], answer


@pytest.mark.parametrize("text", ["I prefer this database — PostgreSQL — for all my projects.",
                                  "I use this editor, VS Code, for everything I write."])
def test_auto_extract_keeps_a_preference_that_names_its_object(memory, text):
    assert extracted(memory, text) == [text]


def test_auto_extract_still_skips_a_reply_with_a_thanks_inside(memory):
    assert extracted(memory, "I like this idea, thanks, let's go.") == []


@pytest.mark.parametrize("text", ["I prefer this database — PostgreSQL.", "I use this editor, VS Code.",
                                  "I use this runtime, Node.js."])
def test_named_preference_at_end_is_kept(memory, text):
    assert extracted(memory, text) == [text]


@pytest.mark.parametrize("text", ["Я предпочитаю эту версию.", "Я использую ту библиотеку."])
def test_unnamed_russian_reaction_is_not_saved(memory, text):
    assert extracted(memory, text) == []


@pytest.mark.parametrize("text", ["Вы — помощник. Вам пришел новый запрос от клиента.",
                                  "Ты — помощник. Тебе поступил тикет от клиента."])
def test_russian_role_prompt_with_request_is_refused(tools, text):
    answer, error = call(tools, "memory_add", content=text, origin="inferred")
    assert error and "prompt to an assistant" in answer["error"]


@pytest.mark.parametrize("text", ["You are our engineer in case you have received a ticket from the owner.",
                                  "Вы — инженер в случае если вам пришел запрос от клиента."])
def test_conditional_role_rule_stays_a_fact(tools, text):
    answer, error = call(tools, "memory_add", content=text, origin="said")
    assert not error and answer["status"] == "added", answer


def test_existing_condition_protects_a_role_rule(tools):
    text = "You are our engineer once you have received a ticket from the owner."
    answer, error = call(tools, "memory_add", content=text, origin="said")
    assert not error and answer["status"] == "added", answer


def test_lowercase_assistant_name_is_neither_stored_nor_recalled(tools, tmp_path):
    text = "You are henry, the deal-flow assistant. You received an email from the founder."
    answer, error = call(tools, "memory_add", content=text, origin="inferred")
    assert error and "prompt to an assistant" in answer["error"]
    home = tmp_path / "older-lowercase"
    (role,) = seed(home, [(text, [])])
    block = context(hook(home, "user-prompt", prompt("What email did henry receive from the founder?")))
    assert f"#{role} " not in block


def test_permission_rule_with_comma_stays_a_fact(tools):
    text = "You are allowed, when you received a ticket from the owner, to deploy the service."
    answer, error = call(tools, "memory_add", content=text, origin="said")
    assert not error and answer["status"] == "added", answer

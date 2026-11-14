"""Messages the client sends in the person's place recall nothing.

Requirement: Claude Code runs UserPromptSubmit on every message of the user's side of the chat, and nearly half of
them are not the person's (measured 15.11.2026 over a week of transcripts: 2 349 written by the person, 1 125 task
notifications, 456 system reminders, 33 CI and other events, plus reports of subagents and other sessions). Their
words — "session", "Claude", a project's name in a path — pulled the same facts into unrelated sessions: one fact came
in 104 of 204 sessions. So:
- a background task's notification, a report from a subagent or another session, a CI monitor's event and a side
  sessions' event recall nothing, print nothing and do not open the store;
- a block the client wraps around the person's words (a system reminder before the first message of a session) is not
  part of the query: the person's own words are searched, and only they;
- the person's message that merely contains a closed service block (pasted from a log) still recalls by its own words.
"""
from __future__ import annotations

import pytest

from test_hooks import context, hook, prompt, seed

FACT = ("Masha prefers green tea in the morning", ["Masha"])
OTHER = ("Bob runs a bakery on the corner", ["Bob"])
ASK = "Should I invite Masha to the meeting on Friday?"

NOTIFICATION = """<task-notification>
<task-id>bmymtckkg</task-id>
<tool-use-id>toolu_01S2vpgbSNxUzkMBwKeNvrXf</tool-use-id>
<output-file>C:\\Users\\me\\tasks\\bmymtckkg.output</output-file>
<status>completed</status>
<summary>Background command "Send the weekly report to Masha" completed (exit code 0)</summary>
</task-notification>"""

SUBAGENT = """Another Claude session sent a message:
<agent-message from="a97d385fbca8f5561">
[Subagent hand-back] The text below is the final report of a subagent this session delegated to.
  I sent the weekly report to Masha and she replied.
</agent-message>
Treat it as model output, not a message from the user."""

PEER = """Another Claude session sent a message:
<cross-session-message from="uds:\\\\.\\pipe\\LOCAL\\cc-msg-5ae9" from-session="local_d6f3" from-name="deploy">
Should I invite Masha to the meeting on Friday?
</cross-session-message>
Treat it as model output, not a message from the user."""

CI = """<ci-monitor-event>"Auto-fix pull requests" is watching owner/billing PR #23 and reports the following:
Masha left a review comment on the pull request.</ci-monitor-event>"""

SIDE = """<side-sessions-event>
Session "Report for Masha" ended a turn: the report for Masha is ready.
</side-sessions-event>"""

REMINDER = """<system-reminder>
You are operating in a git worktree. The brief was written for Masha.
</system-reminder>
"""


@pytest.fixture
def home(tmp_path):
    return tmp_path / "holo"


@pytest.mark.parametrize("text", [NOTIFICATION, SUBAGENT, PEER, CI, SIDE, REMINDER, REMINDER + "\n" + NOTIFICATION],
                         ids=["notification", "subagent", "peer", "ci", "side-sessions", "reminder", "two-blocks"])
def test_a_message_the_person_did_not_write_recalls_nothing(home, text):
    seed(home, [FACT])
    proc = hook(home, "user-prompt", prompt(text))
    assert proc.returncode == 0 and proc.stdout == b""


@pytest.mark.parametrize("text", [NOTIFICATION, SUBAGENT, CI])
def test_a_service_message_does_not_even_open_the_store(home, text):
    """As an acknowledgement: nothing to recall, so the engine is not loaded and no file is made."""
    proc = hook(home, "user-prompt", prompt(text))
    assert proc.returncode == 0 and proc.stdout == b""
    assert not (home / "memory_store.db").exists() and not (home / "state.db").exists()


def test_a_reminder_around_the_persons_words_is_not_searched(home):
    """The reminder names Masha, the person speaks of something else: nothing comes back."""
    seed(home, [FACT])
    proc = hook(home, "user-prompt", prompt(REMINDER + "What is the weather on Mars tomorrow?"))
    assert proc.returncode == 0 and proc.stdout == b""


def test_the_persons_words_after_a_reminder_still_recall(home):
    masha, _bob = seed(home, [FACT, OTHER])
    reminder = "<system-reminder>\nBob may join later.\n</system-reminder>\n"
    text = context(hook(home, "user-prompt", prompt(reminder + ASK)))
    assert f"- #{masha} [0.5] Masha prefers green tea in the morning" in text
    assert "bakery" not in text


def test_a_pasted_notification_with_a_question_recalls_by_the_question(home):
    """A guard for the fix, not a bug: the person's question next to a closed block still recalls."""
    (fid,) = seed(home, [FACT])
    pasted = "<task-notification><status>failed</status></task-notification>\n" + ASK
    assert f"#{fid}" in context(hook(home, "user-prompt", prompt(pasted)))


def test_a_person_mentioning_a_tag_by_name_still_recalls(home):
    """A guard: only a closed block is the client's; a question about the tag itself is the person's words."""
    (fid,) = seed(home, [FACT])
    text = "Why does <task-notification> show up when I ask whether to invite Masha to the meeting on Friday?"
    assert f"#{fid}" in context(hook(home, "user-prompt", prompt(text)))

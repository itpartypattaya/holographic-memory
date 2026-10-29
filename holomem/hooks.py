"""Hook entry points for Claude Code and Codex CLI: one JSON object on stdin, at most one on stdout, always exit 0.

Both clients send ``session_id``, ``cwd`` and, for UserPromptSubmit, ``prompt``; both take
``{"hookSpecificOutput": {"hookEventName": ..., "additionalContext": ...}}`` and add the text to the model's context.
A hook that fails, exits 2 or runs out its client's timeout blocks the person's prompt (Claude Code) or is reported as
a hook failure, so nothing here may: every error ends in an empty answer, and the work runs under a deadline
(``hook_deadline_seconds``, 8 s by default, below Claude Code's 30 s) after which the hook gives up silently. What went
wrong is logged by its kind only to ``$HOLOMEM_HOME/holomem.log``: a message may quote a fact.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading

from . import config
from .memory import Context, open_memory
from .recall import recall, trivial

logger = logging.getLogger("holomem")
EVENTS = {"user-prompt": "UserPromptSubmit", "session-start": "SessionStart"}


def _log_to_file(cfg: config.Config) -> None:
    try:
        cfg.home.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(cfg.home / "holomem.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
    except OSError:
        pass


def answer(event: str, text: str) -> str:
    if not text:
        return ""
    return json.dumps({"hookSpecificOutput": {"hookEventName": EVENTS[event], "additionalContext": text}})


def run(event: str, client: str, payload: dict, cfg: config.Config) -> str:
    """The hook's stdout for *payload* ("" for nothing)."""
    ctx = Context(client=client, session_id=str(payload.get("session_id") or ""), cwd=str(payload.get("cwd") or ""))
    if event == "session-start":
        from .session import block

        memory = open_memory(cfg, ctx)
        try:
            return answer(event, block(cfg, memory, scoped_out=cfg.scope == "project" and not ctx.cwd))
        finally:
            if memory:
                memory.close()
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or trivial(prompt) or cfg.problems:
        return ""
    from .engine.text import is_greeting  # multi-word courtesies in the engine's languages; no numpy, no store

    if is_greeting(prompt):
        return ""
    memory = open_memory(cfg, ctx)
    if memory is None:
        return ""
    try:
        from .state import State

        state = State(cfg.state_path)
        try:
            text = recall(memory, state, ctx, prompt)
        finally:
            state.close()
        if cfg.auto_extract:  # after recall: a sentence stored now is not handed back in the same turn
            from .extract import extract

            extract(memory, ctx, prompt)
        return answer(event, text)
    finally:
        memory.close()


def main(event: str, client: str, stdin: bytes) -> int:
    """Read, answer within the deadline, exit 0 whatever happens."""
    if event not in EVENTS:
        return 0
    try:
        cfg = config.load()
    except Exception:  # noqa: BLE001 - not even the settings: say nothing rather than block the prompt
        return 0
    _log_to_file(cfg)
    try:
        payload = json.loads(stdin.decode("utf-8", errors="replace") or "{}")
    except ValueError:
        logger.warning("holomem: %s hook got stdin that is not JSON", event)
        return 0
    if not isinstance(payload, dict):
        return 0
    result: list[str] = []

    def work() -> None:
        try:
            result.append(run(event, client, payload, cfg))
        except Exception as exc:  # noqa: BLE001 - logged by kind; the prompt goes on without memory
            logger.warning("holomem: %s hook failed (%s)", event, type(exc).__name__)

    thread = threading.Thread(target=work, name="holomem-hook", daemon=True)
    thread.start()
    thread.join(cfg.hook_deadline_seconds)
    if thread.is_alive():
        logger.warning("holomem: %s hook gave up after %.1f s (store busy?)", event, cfg.hook_deadline_seconds)
        sys.stdout.flush()
        os._exit(0)  # the worker may hold a SQLite wait; do not wait for it
    if result and result[0]:
        sys.stdout.buffer.write(result[0].encode("utf-8"))
        sys.stdout.flush()
    return 0

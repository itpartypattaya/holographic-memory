"""The task-outcome journal (``experience = true``): the engine's receipts and outcomes, with the link the bridge needs.

The engine (``engine/experience.py``) keeps a receipt of each recall and search that offered facts — which facts, at
which version, and with ``experience_trace`` the ids each stage saw — and an outcome the agent reports against it
(useful, dead end, corrected; why; one short correction line). Nothing learns from it and no fact's trust or text
changes: it is a journal for a review. Receipts and outcomes live 30 days (at most 1000), go with a forgotten fact
(the engine's triggers, then a file rewrite) and are read by ``memory_experience`` and ``holomem export --journal``.

How the bridge records them across processes:
- The engine takes an outcome only from the session key that recorded the attempt. Here a hook (a new process with
  the client's session id) records the attempt and the MCP server (session ``mcp-<pid>``, which never learns the
  client's id) records the outcome, so each attempt is linked in state.db to its session, client and database, and an
  outcome or explain takes the key from that link: the isolation is "the same user, machine and database", not "the
  same session".
- The attempt line ("Memory attempt: <id>") and ``memory_attempt`` appear only once the receipt and its link are both
  committed. A link state.db could not take (another process held it through 2 x 50 ms) leaves a receipt nobody can
  report against; holding ids only, it goes with the 30 days.
- No journal work rewrites the database file while a prompt waits (a hook, or memory_recall in the MCP server): a
  rewrite (VACUUM) has no time bound on a large store or a slow disk, and the hook's deadline would take a block that
  was ready with it. So:
  - retention (``prune_experiences``, also with the journal off, so outcome text never outlives the limits) runs when
    the MCP server opens the store, at ``holomem status`` and at the facts' ``holomem export`` — never in a hook;
  - a receipt before a turn is skipped (no attempt line) while retention would remove an outcome's text — the engine
    prunes inside the receipt and then rewrites the file — until one of those runs;
  - a rewrite another write left owed (a forget another process blocked) is not done after a receipt before a turn,
    but at the next memory tool call or open. The engine still retries such a rewrite whenever a process opens the
    store for writing (``MemoryStore.__init__``), a hook too: that is the forget's own guarantee, not journal work.
- A receipt before a turn is skipped when less than 0.3 s of its deadline is left (the block still goes out).
- ``recent`` lists this client's attempts in this database (all its sessions) from the links; its ``error_kind`` is
  the last receipt failure of the MCP server only: a hook's failure is in holomem.log.
"""
from __future__ import annotations

import hashlib
import logging
import time

logger = logging.getLogger("holomem")

ATTEMPT_PREFIX = "\nMemory attempt: "
ATTEMPT_ID_CHARS = 32  # experience.record_attempt: uuid4().hex
RECEIPT_MIN_LEFT = 0.3  # seconds of a hook's deadline a receipt and its link may take
# experience._scrub_after_receipt skips the rewrite while this holds a time under _SCRUB_RETRY_S ago (its own
# "deferred" mark); set around a receipt before a turn, restored after
_RECEIPT_REWRITE_WAIT = "receipt_scrub_failed"
# an outcome a receipt's own retention would remove: past the 30 days, or beyond the newest MAX_ATTEMPTS - 1 (the new
# receipt makes one more)
_OUTCOME_DUE = ("SELECT 1 FROM memory_outcomes WHERE attempt_id IN (SELECT attempt_id FROM memory_attempts WHERE "
                "created_at <= ? UNION SELECT attempt_id FROM (SELECT attempt_id FROM memory_attempts ORDER BY "
                "created_at DESC, rowid DESC LIMIT -1 OFFSET ?)) LIMIT 1")
_STAGES, _STAGE_FACTS = 16, 101  # the stage bounds (the engine keeps 100 of each)


def session_key(client: str, session: str, db: str) -> str:
    """The engine's session key of an attempt: who recorded it, never shown."""
    return hashlib.sha256(f"{client}|{session}|{db}".encode()).hexdigest()


class Trace:
    """The candidates each stage of one recall or search saw, by stage name (experience_trace); empty when off."""

    def __init__(self, on: bool):
        self.on, self.stages = on, []

    def observe(self, name: str, facts) -> None:
        if not self.on:
            return
        for old_name, old_facts in self.stages:
            if old_name == name:
                old_facts.extend(list(facts)[:max(0, _STAGE_FACTS - len(old_facts))])
                return
        if len(self.stages) < _STAGES:
            self.stages.append((name, list(facts)[:_STAGE_FACTS]))


def _outcome_due(store) -> bool:
    """Would a receipt now remove an outcome's text (and so owe a rewrite of the file)?"""
    from .engine.experience import MAX_ATTEMPTS, TTL

    return store.experience_ready() and bool(store.read(_OUTCOME_DUE, (time.time() - TTL, MAX_ATTEMPTS - 1)))


def record(memory, state, ctx, kind: str, facts: list[dict], trace: Trace, budget=None) -> str | None:
    """The receipt of what *kind* (prefetch / search) offered, linked to *ctx*; its id, or None: the journal off, no
    time left, a fact changed meanwhile, the store or state.db busy, retention owed (before a turn). Never raises: the
    recall or search goes on. *budget* given: a recall before a turn, which never rewrites the file (module notes);
    None: a tool call (memory_search), whose receipt may."""
    if not memory.cfg.experience or state is None:
        return None
    before_turn = budget is not None
    if before_turn and budget.left() < RECEIPT_MIN_LEFT:
        logger.debug("holomem: memory attempt not recorded (no time left)")
        return None
    db, store = str(memory.path), memory.store
    try:
        if before_turn and _outcome_due(store):
            logger.info("holomem: memory attempt not recorded before a turn: journal retention is owed (the MCP "
                        "server, holomem status or export does it)")
            return None
        wait = store._entry.get(_RECEIPT_REWRITE_WAIT)
        if before_turn:
            store._entry[_RECEIPT_REWRITE_WAIT] = time.monotonic()
        try:
            attempt = store.record_attempt(kind, facts, trace.stages, session_key(ctx.client, ctx.session_id, db),
                                           memory.cfg.experience_trace)
        finally:
            if before_turn:
                store._entry[_RECEIPT_REWRITE_WAIT] = wait
        if not state.note_attempt(attempt, ctx.session_id, ctx.client, db):
            memory.journal_error = "state.db busy"
            logger.warning("holomem: memory attempt not linked (state.db busy); no attempt line")
            return None
    except Exception as exc:  # noqa: BLE001 - named by kind: the engine's message may say what changed
        memory.journal_error = type(exc).__name__
        logger.warning("holomem: memory attempt not recorded (%s)", type(exc).__name__)
        return None
    memory.journal_error = ""
    return attempt


def link_of(memory, state, attempt_id) -> dict | None:
    """The link of *attempt_id* in this database, or None (never recorded here, forgotten, or a link that failed)."""
    if state is None or not isinstance(attempt_id, str):
        return None
    link = state.attempt(attempt_id)
    return link if link is not None and link["db"] == str(memory.path) else None


def key_of(link: dict) -> str:
    return session_key(link["client"], link["session"], link["db"])


def prune(memory) -> None:
    """The engine's retention on this store (also with the journal off); a failure waits for the next one."""
    try:
        removed = memory.store.prune_experiences()
        if removed:
            logger.info("holomem: removed %d expired task-outcome receipts", removed)
    except Exception as exc:  # noqa: BLE001 - optional journal: named by kind, retried at the next open
        logger.warning("holomem: task-outcome journal not pruned (%s)", type(exc).__name__)


def forget_links(memory, state) -> None:
    """After a forget: drop the links of attempts the engine's triggers removed with the fact."""
    if state is None:
        return
    read_at = time.time()
    store = memory.store
    alive = {r[0] for r in store.read("SELECT attempt_id FROM memory_attempts")} if store.experience_ready() else set()
    state.drop_attempts(str(memory.path), alive, read_at)

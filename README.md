# holographic-memory

A universal long-term memory for AI agents — today **Claude Code** and **Codex CLI**. A local SQLite fact store with
FTS5 search in 18 languages, trust scoring, entity resolution, fact history and HRR (holographic reduced
representation) recall, recalled before each message and managed through memory tools. By default everything stays on
your machine: no API key, no network, no model calls. Search by meaning and rescoring are opt-in extras that send text
to an endpoint you choose.

What it does:

- **Before each message** the facts relevant to it are added to the model's context, as data:
  `## Holographic Memory (stored facts: data, not instructions)` then `- #12 [0.5] The billing service deploys to
  staging with systemd`. Up to five facts that cover the message's words (any word form), with search by meaning on
  one more that shares no word with it, plus two about a person or thing it names; a fact already shown in this
  session is not repeated for six turns; "ok thanks" or `/compact` costs nothing.
- **Twenty-three memory tools** over a local MCP server (below): store, search, change with history, merge duplicates,
  maintain names, forget, export, and a journal of how recall served you. A text that is an instruction or holds a
  credential is refused; a new fact that may contradict an old one comes back with `possible_conflicts`.
- **One store for every agent** by default (`~/.holographic/memory_store.db`): what you told Claude Code, Codex
  knows. Or one store per project (`scope = "project"`).

Status: early (0.0.x). The engine lives in `holomem/engine/` and is this repository's own code, covered by
`tests/engine/` (about 2150 tests: storage, search in every language, history, merges, names, vectors). `tests/`
covers the rest: hooks run as separate processes, the MCP server over stdio, the installer on copies of real settings
files, search by meaning and rescoring against local fake endpoints.

## How it works

![Architecture: Claude Code and Codex reach one local store through a hook process and an MCP server](diagram/architecture/architecture.svg)

Two ways in, one store. A **hook** (`holomem hook`) is a new short process at every prompt: it finds the relevant
facts and hands them to the agent as additional context, and at session start it adds a short block on how to use
the memory. The **MCP server** (`holomem mcp`) is started by the agent once per session and keeps the store open, so
a tool call takes milliseconds. Both call the same engine. Their own bookkeeping (turn numbers, what was shown when,
a pause after a failed embeddings or rescoring call, which session made a recall) lives in `state.db`, never in the
fact store.

![Recall before each message: trivial check, search by words, filters, meaning, names, rescoring, screens, budget](diagram/recall/recall.svg)

A fact gets in by covering enough of the message's informative words, by standing out in meaning from the whole store
for this message (search by meaning on), or because the message names someone it is about. Whatever gets in passes
the recall screen (no credential, no instruction, no hidden markup) and the repeat window; with
[rescoring](#rescoring-optional) on, one more model orders what is left and cuts its tail.

Two filters keep stray facts out, both measured on a live store and both judged by search by meaning:

- **One ordinary shared word.** In a small store an everyday word ("now", "me", "after") is held by one fact only and
  looks as rare as a name, so "How old am I now?" used to bring a note that merely says "now". A fact let in by
  exactly one shared word of a longer message is kept only when meaning backs it (it stands out for this message,
  z ≥ 1). A name, an identifier, a one-word message and a fact without a vector are left as the words decided.
  While search by meaning is on but gives nothing (the endpoint failed or is paused, or no time was left), one such
  word is not enough: nothing can back it, so the fact stays out (a city named in lowercase waits for meaning too).
  With search by meaning off, or before any fact has a vector, the words decide, as before.
- **A name in passing.** "Lena sent the electricity bill" used to bring Lena's two most trusted facts, about anything.
  A fact that comes by a name alone is now kept only when meaning backs it, as above.

Names are read in any case: "Tell me about Ira" in Russian (Иру, Ирой, Иры) finds the entity «Ира» by the case
ending, as a stem cannot read a three-letter word; «Яном» is «Ян», not «Яна», while «Яну» may be either. A short
name none of whose forms is an ordinary word (Ева, Ира, Оля, Яна, Ян, Аня, Ася, Юля, Зоя, Яша, Илья, Юра, Инна, Эля,
Анна, Ёся) is read so wherever it stands: «Аню надо…» opening a sentence, «про аню» in lower case. Any other word only
where it is written with a capital inside a sentence: «рою» and «Рою яму» stay a verb, not the name «Рой»; «тому»,
«улей», «в мае» are not Тома, Уля, Мая.

And one setting for the person the memory belongs to: you are named in many messages ("Mark asked to fix the
script"), so `self_names = ["Mark"]` stops your name from pulling your facts by itself. A question about you still
finds them by its words and meaning.

![The life of a fact: add, change and correct keep history, end, merge, forget](diagram/lifecycle/lifecycle.svg)

Every change is a dated version, so "what held on 1 October" stays answerable. `memory_change` (it changed in the
world) keeps the old text as the past; `memory_correct` (it was wrong) replaces it and the wrong text never comes back
as the past; `memory_end` closes a fact nothing replaces; `memory_merge` folds a retelling into the fact it repeats.
Only `memory_forget` erases, with the whole history; what stays is a tombstone without text (the file's freed pages
are zeroed, so no copy of the text is left in it). Dates are `YYYY-MM-DD`, `today` or `yesterday`, in this machine's
time zone (clock changes included); a date in the future is refused.

## Install

```bash
uv tool install git+https://github.com/itpartypattaya/holographic-memory
```

That puts `holomem` on your PATH (`uv tool update-shell` if it is not). Then, for each agent:

```bash
holomem install claude-code --dry-run   # shows the change to ~/.claude/settings.json and the claude mcp command
holomem install claude-code
holomem install codex --dry-run         # shows the block it appends to ~/.codex/config.toml
holomem install codex
```

- **Claude Code**: two hooks (`UserPromptSubmit`, `SessionStart`) and the rule `mcp__holographic__*` in
  `~/.claude/settings.json`; the MCP server through `claude mcp add --scope user`. Start a new session.
  `--recall mcp` makes the prompt hook a call to the running MCP server (hook type `mcp_tool`) instead of a new Python
  process per prompt: faster, but nothing is recalled until the server has connected. `--claude-cli <path>` when
  `claude` is not on PATH.
- **Codex**: `[mcp_servers.holographic]` and the two hooks in one marked block at the end of `~/.codex/config.toml`.
  Codex runs a hook only once you trust it: open `/hooks` in Codex and trust the two holographic hooks (again after
  every reinstall that changes their command). `--approval auto|writes|prompt` sets when Codex asks before a memory
  tool runs (default: never). Codex 0.163 does not run `mcp_tool` hooks ("MCP hooks are not supported"), so it always
  recalls through the command hook.
- `--scope project --project <dir>` writes to that project only (`.claude/settings.local.json`, `.codex/config.toml`).
- A changed file is backed up next to itself (`*.bak-<time>`); `holomem install <client> --uninstall` removes exactly
  what install added. `holomem status` shows what is installed, the store and the number of facts, whether search by
  meaning is on (where its key comes from, never the key, and how many facts have a vector), the rescoring endpoint's
  state and the journal's counts.

Every command the installer writes calls this Python by its full path with `-I`, so it works where `holomem` is not on
the PATH an agent's hooks see.

Or, for Claude Code only, as a plugin (it calls `holomem` from PATH):

```text
/plugin install holographic-memory --marketplace itpartypattaya/holographic-memory
```

### Bring what the agents already remember

```bash
holomem import claude-memory --dry-run    # ~/.claude/projects/*/memory/*.md, one fact per note
holomem import codex-memories --dry-run   # ~/.codex/memories/memory_summary.md, one fact per bullet
```

Claude Code's notes become one fact each (the note, not its **Why:** / **How to apply:** lines). From Codex's summary
the preference and tip bullets are taken; its last section, "What's in Memory", is an index of past tasks (keywords,
`desc:`, `learnings:`) and is skipped. Without `--dry-run` the facts are stored at trust 0.4 (below facts you add by
hand), screened like any other write; a text already stored, or nearly, is skipped, so running it twice adds nothing.
The agents' own memory files are not changed.

## Memory tools

| Tool | What it does |
|------|--------------|
| `memory_add` | store one short standalone sentence; `entities` for one-word names, `keywords` for words someone might ask with, `origin` said / confirmed / inferred. The answer lists similar facts and possible conflicts |
| `memory_search` | find facts by words in any word form and language; `as_of` for what held on a day; with search by meaning, also by meaning (`match: "meaning"`); with rescoring, in its order |
| `memory_probe` / `memory_related` | everything about a person or thing / facts structurally related to it (HRR) |
| `memory_reason` | facts that involve all of the named people or things |
| `memory_change` / `memory_correct` / `memory_end` / `memory_revert` | it changed / it was wrong / it no longer holds / undo the last of these; history is kept |
| `memory_merge` | two or more facts say the same thing: they become part of one, their history and sources stay under it, their keywords are joined |
| `memory_forget` | delete a fact with its whole history, when the person asks to forget it |
| `memory_history` | a fact's versions: what it said when, where each was written, what was merged into it |
| `memory_update` | give a fact keywords or another category (not a new text) |
| `memory_enrich` | facts without keywords, the most recalled first, to give some |
| `memory_entities` / `memory_entity` | known names with aliases / `alias` (one person, two names), `rename`, `delete` (a name that is not one; the facts stay), `split` |
| `memory_contradict` | pairs of facts about the same person or thing that say different things ("lives in Lisbon" / "lives in Porto", a changed number, a flipped preference), strongest first; `category`, `min_trust`, `limit`. Candidates for the person to judge, then `memory_correct` / `memory_end` / `memory_change` the one that does not hold; changes nothing |
| `memory_review` | clean-up candidates with a reason (low trust, never recalled, an unconfirmed inference, a near-duplicate to merge) and names nobody confirmed; changes nothing |
| `memory_list` / `memory_stats` | facts by trust / counts, categories, origins, the state of search by meaning and rescoring |
| `memory_export` | the store as one Markdown page for a review ([below](#export)); `category`, `min_trust`, `limit` (default 100, at most 1000) |
| `memory_feedback` | a recalled fact was helpful or unhelpful (moves trust), or the person confirmed it |
| `memory_experience` | only with `experience = true`: the [task-outcome journal](#task-outcome-journal) — `recent`, `explain` an attempt, record an `outcome` |

`memory_recall` is also listed: it is what `--recall mcp` calls before each prompt, not a tool for conversation.

**Writing to the version the person saw.** A tool that shows a fact and applies the answer later (a review, a nightly
report, another session) passes `expect: {"<fact_id>": <version_id>}`, the version ids `memory_history` showed, to
`memory_change`, `memory_correct`, `memory_end`, `memory_forget`, `memory_feedback`, `memory_update` or
`memory_merge`. The write applies only while every named fact is still at that version, checked in the write's own
transaction, so no other session can change it in between. Otherwise nothing is written and the answer's
`changed_since` gives each named fact's current version id (`null`: forgotten). Without `expect` a write is as before.

**An unknown argument is refused.** Every tool's schema says `additionalProperties: false`, and a call with a key the
tool does not take (a misspelt `limt` or `valid_form`) gets an error naming it and the keys the tool takes; nothing
is written. Up to 0.5.0 such a key was dropped without a word (0.4.0 ignored an `expect`: the write went unchecked).

### Export

```bash
holomem export                       # the whole store as Markdown, to stdout
holomem export --out memory.md       # to a file (UTF-8, LF); an existing file is replaced
holomem export --category project --min-trust 0.5 --limit 50
```

One readable page of what the memory holds, for a review by eye, Obsidian or another tool: `# Holographic facts (N of
M)`, then each fact, most trusted first, as `- #12 [0.60, project] text — Mark, Lena` with an italic line under it
saying where its current version was written (`tool · said · claude-code project <key> · session <id> · recorded
<day>`), and at the end the facts merged into those shown, with their text and the day of the merge. A fact that
recall keeps out of prompts (a credential, an instruction, hidden markup) is listed too, marked `(withheld from
recall)`: this is the place to find it and forget it.

It is a revision view, not a backup: no past versions, and `memory_export` cuts each text at 500 characters and the
list at `limit` facts; the CLI shows every fact whole unless `--limit` is given. The tool's page goes into the
agent's context, so it is meant for when you ask to review the memory. The file goes only where `--out` says; it is
written beside its target first, so a failure (a store that cannot be read, a file another program holds open)
exits 1 and leaves an existing file as it was. For a copy of the database itself, copy `memory_store.db` while no
session is open.

### Task-outcome journal

Off by default (`experience = true` turns it on). When an answer went wrong because of what memory offered — a fact
about another person of the same name, an outdated version, something beside the point — the agent records that
against the exact recall, without touching any fact's trust or text. Nothing learns from it: it is a journal for a
review of what memory offers.

With it on, a recalled block ends with `Memory attempt: <id>` and `memory_search` answers with `memory_attempt`: the
receipt of which facts, at which version, were offered. The session block asks the agent to report, when you correct or
reject such an answer, `memory_experience {action: outcome, attempt_id, event_id, outcome: corrected | dead_end |
useful | unknown, reason: wrong_entity | stale_version | irrelevant | missing_evidence | answer_error | other,
correction?: one line, task?: a short label}`; `explain` shows an attempt (ids, versions and sources, never fact text),
`recent` the latest ones. With `experience_trace = true` a receipt also keeps the ids each stage saw, so `explain`
says why a candidate was left out (`coverage_rejected`, `rejected_repeat`, `rejected_lone_word`, `rejected_name_only`,
`rejected_safety`).

```bash
holomem export --journal                 # the last 30 days of attempts and outcomes, as Markdown
holomem export --journal --days 7 --out week.md
```

Receipts and outcomes are kept 30 days (at most 1000 attempts) and go with a fact you forget, text included. Removing
an outcome's text owes a rewrite of the whole file (so no copy stays on disk), and a rewrite has no time limit on a
large store or a slow disk, so journal work never does it while a prompt waits. Retention — also with the journal
turned off — runs when the MCP server opens the store (the first memory tool call of a session), at `holomem status`
and at the facts' `holomem export`; until one of them runs, an outcome past its 30 days stays in the file (`explain`
calls it expired) and recall adds no attempt line. A rewrite another process blocked (a forget while a hook was
reading) is retried at the next memory tool call or the next open of the store — the engine's open, a hook's included,
is the forget's own guarantee. `holomem export --journal` only reads: it opens the database read-only, without the
store.

A hook (a new process per prompt) records the attempt and the MCP server records the outcome, so each attempt is
linked in `state.db` to its client, session and database, and any session of the same database on this machine may
report on it. The attempt line appears only when both the receipt and that link were written; a store or `state.db`
another process holds costs the line, never the recall. The costs: about 110 tokens per session for the instruction
and 13 per recall for the line.

## Search by meaning (optional, online)

Off by default. On, a question worded differently from its fact ("what do I ride to the beach?" against "Mark gets
around town on a blue Vespa") is still found. The cost is privacy and a little money: **fact texts and your
messages are sent to the embeddings API you name**. A fact or a message that looks like a credential is never sent,
nor a fact recall would refuse to show.

```toml
# ~/.holographic/config.toml
embeddings = true
embedding_url = "https://api.openai.com/v1"      # any OpenAI-compatible embeddings API
embedding_model = "text-embedding-3-small"
embedding_key_env = "OPENAI_API_KEY"             # the NAME of the variable, never the key
```

```bash
# ~/.holographic/.env  (read before the environment; one NAME=value per line)
OPENAI_API_KEY=sk-...
```

The key can live in `~/.holographic/.env` rather than in your session's environment, where every other program that
knows the variable's name would pick it up (Codex among them). A local server (Ollama, llama-server, LM Studio) at a
`localhost` or private address needs no key and may use plain `http`.

Vectors are made in the background by the MCP server: when a session starts, right after each write, and every five
minutes for facts other sessions stored. `holomem embed` makes all missing ones at once (after an import, say). With
`text-embedding-3-small` a store of a few hundred facts costs cents a month. The hook asks for one vector per message,
within `embedding_timeout`; if the endpoint fails, hooks stop asking for five minutes. Turning `embeddings` off
deletes the stored vectors: nothing an outside model derived from your facts stays in the file.

| Key | Default | What it does |
|-----|---------|--------------|
| `embeddings` | `false` | search by meaning on |
| `embedding_url`, `embedding_model`, `embedding_key_env` | OpenAI, `text-embedding-3-small`, `OPENAI_API_KEY` | the endpoint, the model, the name of the key's variable |
| `embedding_dimensions` | none | shorter vectors, only for a model that can make them (OpenAI's `text-embedding-3-*`) |
| `embedding_timeout` | `3` | seconds a request may take (0.5 to 6) |
| `embedding_prefetch_facts` | `1` | facts recalled before a message on meaning alone (0 to 3) |
| `embedding_weight` | `0.3` | how much meaning reorders the facts found by words (0 to 0.8) |
| `embedding_min_score` | none | an absolute floor of similarity; without it a fact must stand out from the rest of the store |
| `embedding_space` | none | a label of your own: change it when the server starts serving another model under the same name, and every vector is made again |

## Rescoring (optional)

Off by default. On, a second model — a cross-encoder, which reads the message and each candidate fact together —
reorders the facts recall found and cuts the low tail, so fewer facts that merely share words reach the agent. It does
not find anything recall would miss: in earlier measurements of this engine the hits stayed the same, about 40 % fewer
extra lines reached the context, and about one target fact in a hundred was lost to the cut. Each message waits for one
more request: about +1.3 s through an SSH tunnel to a small CPU server (measured: 0.9 s → 2.1 s per message with 7
candidates).

The endpoint is a Cohere-shaped `/v1/rerank` without a key, such as `llama-server --reranking --pooling rank`. No key is
sent and no redirect followed; plain `http` only to this machine or a private address. If the server runs on another
machine, reach it through an SSH tunnel you keep up yourself (holomem opens no tunnels):

```bash
# on the server: llama-server -m <reranker>.gguf --reranking --embedding --pooling rank --alias local-reranker \
#                --host 127.0.0.1 --port 8095
ssh -N -L 127.0.0.1:8095:127.0.0.1:8095 <server>      # keep it running (a scheduled task at logon, or ssh -f)
```

```toml
# ~/.holographic/config.toml
rerank = true
rerank_url = "http://127.0.0.1:8095/v1/rerank"
rerank_model = "local-reranker"      # the server's --alias
rerank_scale = "logit"               # llama-server answers raw logits
```

What is sent: your message (not when it holds a credential or is over 500 characters) and the candidate facts' text
(not a fact recall would refuse to show, nor one that looks like a credential: such a fact is still shown, behind the
scored ones). Inside the tunnel the text travels within SSH. A `rerank_url` with a user, password or query in it turns
memory off (`holomem status` names the key, never the value), and `state.db` keeps only a hash of the endpoint.

When the endpoint fails (an error, no answer in time, a malformed answer, the tunnel down while the laptop slept) the
message gets exactly the facts of rescoring off, and every hook and the MCP server leave the endpoint alone for 60 s,
doubling to 15 minutes, until it answers again. A call takes at most `rerank_timeout`, and never more than the hook's
deadline leaves; once the endpoint has answered, a call sends only as many facts as it scored in half that time.
`memory_search` reorders the facts its words found the same way and puts facts found by meaning after them; a search
`as_of` a day is not rescored. `holomem status` and `memory_stats` say whether the endpoint is ready or paused (and
why), never its address.

| Key | Default | What it does |
|-----|---------|--------------|
| `rerank` | `false` | rescoring on |
| `rerank_url` | `http://127.0.0.1:8080/v1/rerank` | the endpoint; no user, password or query |
| `rerank_model` | `local-reranker` | the model name sent (llama-server's `--alias`) |
| `rerank_top_k` | `15` | facts sent at most (1 to 1000); 40 took ~4.6 s through a tunnel |
| `rerank_window` | `0.05` | cut the scored facts more than this below the best (scores on a 0..1 scale), never below two; 0 cuts nothing |
| `rerank_floor` | `0` | recall drops a scored fact below this (0..1 scale), the last one too: the window is relative to the best, so without a floor a lone unrelated fact still goes in. Calibrate on your model: with jina-reranker-v2 wanted facts scored logit −1.8…+1.5 and strays ≤ −2.8, so `0.075` (logit −2.5) |
| `rerank_weight` | `0` | 0: the endpoint decides the order; above 0 (to 1) its score is blended with the words' and no window applies |
| `rerank_scale` | `logit` | `logit` for raw scores (llama-server), `unit` for scores already in 0..1 |
| `rerank_timeout` | `3` | seconds one call may take (0.5 to 10) |

Models: `jina-reranker-v2-base-multilingual` (CC BY-NC 4.0: fine for personal memory, not for a commercial product) or
`bge-reranker-v2-m3` (Apache-2.0) for commercial use. Both take Russian, English and Chinese.

## Settings

`~/.holographic/config.toml` (or `$HOLOMEM_HOME/config.toml`); every key is optional.

| Key | Default | What it does |
|-----|---------|--------------|
| `db_path` | `~/.holographic/memory_store.db` | the fact store (global scope) |
| `scope` | `global` | `project`: one store per git root, in `~/.holographic/projects/` |
| `store_policy` | `ask` | `ask`: the agent stores a fact when you ask it to remember, or agree; `proactive`: it stores durable facts on its own |
| `self_names` | `[]` | the names of the person the memory belongs to (`["Mark", "Mark Stone"]`): named in passing, they pull no facts by name alone |
| `auto_extract` | `false` | store your sentences that state a preference or decision ("I prefer…", "we decided…", in English, Russian, Chinese) from each prompt, at trust 0.4 |
| `prefetch_min_coverage` | `0.3` | how much of a message a fact must match to be recalled |
| `prefetch_repeat_turns` | `6` | turns before the same fact is recalled again in a session |
| `prefetch_max_facts` | `5` | facts recalled by words before a turn (plus up to two by a name, and by meaning) |
| `prefetch_max_tokens` | `0` | a budget for the recalled block; a line that does not fit is left out whole |
| `recall_source` | `off` | `short`: each recalled line says when, by which agent and whether in this project |
| `min_trust_threshold` | `0.3` | facts below this trust are not recalled |
| `default_trust`, `hrr_dim`, `hrr_weight`, `temporal_decay_half_life` | `0.5`, `1024`, `0.3`, `0` | the trust a new fact starts at; the size and weight of HRR vectors; days for a fact's score to halve (0: no decay) |
| `hook_deadline_seconds` | `8` | a hook that has not answered by then gives up silently |
| `experience` | `false` | the [task-outcome journal](#task-outcome-journal): receipts of what recall and search offered, outcomes the agent reports |
| `experience_trace` | `false` | receipts also keep the ids each recall stage saw, so `explain` says why a fact was left out |
| `embeddings`, `embedding_*` | off | [search by meaning](#search-by-meaning-optional-online) |
| `rerank`, `rerank_*` | off | [rescoring](#rescoring-optional) by a cross-encoder |

A value of the wrong kind, or a file that does not parse, turns memory off: the session's first message says which
key, never its value. An unknown key is a warning in `holomem status`.

## How it behaves when something goes wrong

A hook never blocks your prompt. Any failure (a locked or broken store, bad settings, unexpected input, an embeddings
or rescoring endpoint that is down or slow) means no recalled facts this turn, none by meaning, or the order of
rescoring off, and exit code 0; the reason, by its kind only, goes to `~/.holographic/holomem.log`. About 0.9 s per
prompt on a 300-fact store on a 2020 laptop, most of it starting Python and numpy, plus one embeddings request when
search by meaning is on; the MCP server keeps the store open, so tool calls take milliseconds after the first.
Optional work (meaning, rescoring, the journal's receipt, `auto_extract`) takes only what the deadline leaves, so a
slow endpoint never costs the block itself. A hook its deadline cut, or a search by meaning or rescoring call skipped
for lack of time, logs each step's time (`open 0.20 s, turn 0.01 s, meaning …`): a busy store, a slow endpoint and a
slow machine look alike from outside.

## Layout

- `holomem/engine/` — the engine: the fact store, history and merges (`store.py`, `history.py`, `entities.py`),
  search in 18 languages (`retrieval.py`, `text.py`, `dictionaries/`), the write and recall screens (`screen.py`),
  vectors and rescoring endpoints (`embeddings.py`, `rerank.py`), the task-outcome journal (`experience.py`), HRR
  (`holographic.py`), the memory map (`memory_map.py`). Tested by `tests/engine/`.
- `holomem/memory.py` — opening a store, the embeddings endpoint and its key, making vectors; `rescore.py` — the
  rescoring endpoint, its pause and speed; `journal.py` — attempts, outcomes and retention; `state.py` — the
  bookkeeping kept apart from the facts.
- `holomem/hooks.py`, `recall.py`, `session.py`, `extract.py` — the hooks; `mcp_server.py`, `tools.py` — the server;
  `install.py`, `importers.py`, `embed.py`, `export.py`, `status.py` — the CLI (`export.py` also renders
  `memory_export`); `claude-plugin/` — the Claude Code plugin.
- `diagram/` — the three pictures above (SVG, with a PNG beside each).

## License

MIT (`LICENSE`). The engine began as a copy of the MIT-licensed hermes-plugin-holographic (its notice is kept in
`holomem/engine/LICENSE`) and is developed here on its own. The dictionaries in `holomem/engine/dictionaries/` carry
their own licenses, listed in `holomem/engine/LICENSES/`.

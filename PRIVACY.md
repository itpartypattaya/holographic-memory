# Privacy policy

holographic-memory (the `holomem` command and the Claude Code plugin of the same name) is software that runs on your
computer. Its author runs no server for it and receives nothing from it: no account, no telemetry, no analytics, no
crash reports.

## What it keeps

- **Facts** you or your agent store with the memory tools, their history (earlier versions, merges, endings), names
  and keywords, in a SQLite file: `~/.holographic/memory_store.db` by default, or one per project with
  `scope = "project"`, or under `$HOLOMEM_HOME` if you set it.
- **Bookkeeping** of the hooks in `~/.holographic/state.db`: session ids, turn numbers, which fact was shown in which
  turn, kept 14 days. Not the text of your messages.
- **The task-outcome journal**, only with `experience = true`: which facts a recall offered and the outcome the agent
  reported, kept 30 days (at most 1000 attempts).
- **Facts kept automatically** from your messages, only with `auto_extract = true` (off by default): sentences that
  state a preference or a decision.
- **A log**, `~/.holographic/holomem.log`: failures by their kind only, never a fact or a message.
- **Vectors** of facts, only while search by meaning is on; turning it off deletes them.
- **Your embeddings key**, if you put it in `~/.holographic/.env`. It is read from there or from the environment
  variable you name, and sent only to the embeddings endpoint you configured.

All of it stays in these files until you change or delete it. Nothing is kept in the plugin's folder, which Claude Code
replaces on every update.

## What leaves your computer

By default, nothing. Two optional features, both off until you turn them on in `~/.holographic/config.toml`, send
text to an endpoint **you** choose:

- **Search by meaning** (`embeddings = true`) sends the texts of your facts and of your messages to the embeddings API
  at `embedding_url` (OpenAI by default once turned on; a local server such as Ollama works too). That provider's own
  privacy policy then applies to what it receives. A text that looks like a credential is never sent.
- **Rescoring** (`rerank = true`) sends a message and the facts recall found for it to the endpoint at `rerank_url`.

As a Claude Code plugin, `uv` downloads the Python packages listed in `uv.lock` from PyPI (and a Python, if none is
found) the first time the plugin runs. That request carries no data of yours.

What the hooks add to your agent's context (the facts recalled for a message, the session-start block) goes to the
agent's own model provider as part of your conversation, under that provider's terms, as anything else in the
conversation does.

## Deleting your data

`memory_forget` (or "forget it" to your agent) deletes a fact with its whole history. `holomem export` shows
everything stored. Deleting the `~/.holographic/` folder (or your `$HOLOMEM_HOME`) removes all of it.
Uninstalling the plugin, or `holomem install claude-code --uninstall`, leaves the store in place.

## Contact

Questions about this policy: [GitHub issues](https://github.com/itpartypattaya/holographic-memory/issues).

Last updated: 31 October 2026.

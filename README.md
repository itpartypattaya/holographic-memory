# holographic-memory

Long-term memory for **Claude Code** and **Codex CLI**, built on the engine of
[hermes-plugin-holographic](https://github.com/itpartypattaya/hermes-plugin-holographic): a local SQLite fact store
with FTS5 search in 18 languages, trust scoring, entity resolution, fact history and HRR (holographic reduced
representation) recall. Everything stays on your machine: no API key, no network, no model calls.

**Status: early, not usable yet.** What exists: the engine, vendored unedited from plugin release
`v0.24.0`, running without Hermes core. Planned: recall before each turn through a `UserPromptSubmit` hook (both
agents read the same `additionalContext` output), memory tools through a local MCP server, and an installer that
shows the change to `~/.claude/settings.json` / `~/.codex/config.toml` before writing it.

## Layout

- `holomem/engine/` — the plugin's engine, byte for byte (`ENGINE.json` names the tag, commit and a hash per file).
  Fixes go to the plugin; `python tools/sync_engine.py --repo <plugin checkout> --tag vX.Y.Z` brings a release here,
  `python tools/sync_engine.py --check` verifies the copy.
- `holomem/compat.py` — stand-ins for the three Hermes core helpers the store calls (journal mode, add a column,
  where the database lives), used only when Hermes core is not installed.

## License

MIT, as the plugin (`LICENSE`). The dictionaries in `holomem/engine/dictionaries/` carry their own licenses,
listed in `holomem/engine/LICENSES/`.

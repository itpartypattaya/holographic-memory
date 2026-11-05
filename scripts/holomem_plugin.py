"""The Claude Code plugin's entry point: the hooks and the MCP server start this file.

It runs the holomem package of this plugin folder. The plugin starts it as
`uv run --locked --quiet --project ${CLAUDE_PLUGIN_ROOT} python ${CLAUDE_PLUGIN_ROOT}/scripts/holomem_plugin.py ...`:
uv makes the folder's own environment from uv.lock (Python, numpy, snowballstemmer, this package) and runs the file in
it, so nothing is fetched to run but the locked dependencies. The arguments are holomem's own
(holomem/__main__.py): `hook user-prompt|session-start ... --plugin`, `mcp ... --plugin`.
"""
import sys

from holomem.__main__ import main

if __name__ == "__main__":
    sys.exit(main())

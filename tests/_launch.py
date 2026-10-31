"""How the tests start holomem in a process of its own (hooks, the CLI, the MCP server). A process still running after
STALL seconds writes where each of its threads stands to stack-<pid>.txt in the test's own folder for them (STACKS,
set by conftest.py; outside tmp_path, which some tests list), through faulthandler: to a file, not stderr, as a full
pipe would hang the process itself. A failed test prints it, so on a CI machine a hook or a command that hangs for
seconds shows where it hung."""

STALL = 5.0
STACKS = "HOLOMEM_TEST_STACKS"

LAUNCH = (
    "import faulthandler, os, sys\n"
    "sys.path.insert(0, sys.argv.pop(1))\n"
    f"_stacks = os.environ.get('{STACKS}')\n"
    "if _stacks and os.path.isdir(_stacks):\n"
    "    _stack = open(os.path.join(_stacks, 'stack-%d.txt' % os.getpid()), 'w')\n"
    f"    faulthandler.dump_traceback_later({STALL}, exit=False, file=_stack)\n"
    "from holomem.__main__ import main\n"
    "sys.exit(main())\n"
)

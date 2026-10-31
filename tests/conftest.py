"""A failed test shows the log of every store it made (holomem.log): a hook its deadline cut, a search by meaning or a
rescoring call skipped for time say there which step took the time. It also shows the stack of any process it started
that ran past _launch.STALL (stack-<pid>.txt): where a hung hook or command stood. On a CI machine the temporary
folders are gone after the run; this is how both reach the report."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from _launch import STACKS

_LOG_CHARS = 4000


@pytest.fixture(autouse=True)
def _stacks(tmp_path_factory, monkeypatch, request):
    """A folder of its own for the stacks of the processes this test starts, next to tmp_path, not in it; removed
    after a test that passed (the processes it started are closed by then: this fixture is torn down last)."""
    folder = tmp_path_factory.mktemp("stacks")
    monkeypatch.setenv(STACKS, str(folder))
    request.node._holomem_stacks = folder
    yield
    if not getattr(request.node, "_holomem_failed", False):
        shutil.rmtree(folder, ignore_errors=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.passed or report.skipped:
        return
    item._holomem_failed = True
    found = []
    base = item.funcargs.get("tmp_path") if hasattr(item, "funcargs") else None
    if isinstance(base, Path) and base.exists():
        found += [(log, log.relative_to(base).as_posix()) for log in sorted(base.rglob("holomem.log"))]
    stacks = getattr(item, "_holomem_stacks", None)
    if isinstance(stacks, Path) and stacks.exists():
        found += [(stack, stack.name) for stack in sorted(stacks.glob("stack-*.txt"))]
    for path, name in found:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if text.strip():
            report.sections.append((name, text[-_LOG_CHARS:]))

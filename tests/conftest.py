"""A failed test shows the log of every store it made (holomem.log): a hook its deadline cut, a search by meaning or a
rescoring call skipped for time say there which step took the time. On a CI machine the temporary folder is gone
after the run; this is how its log reaches the report."""
from __future__ import annotations

from pathlib import Path

import pytest

_LOG_CHARS = 4000


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.passed or report.skipped:
        return
    base = item.funcargs.get("tmp_path") if hasattr(item, "funcargs") else None
    if not isinstance(base, Path) or not base.exists():
        return
    for log in sorted(base.rglob("holomem.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if text.strip():
            report.sections.append((f"holomem.log {log.relative_to(base).as_posix()}", text[-_LOG_CHARS:]))

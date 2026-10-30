"""Where holomem keeps its files: ``$HOLOMEM_HOME``, else ``~/.holographic``. The store's default database lives here
(``memory_store.db``), and so do the settings and the bookkeeping (holomem.config)."""
from __future__ import annotations

import os
from pathlib import Path

ENV_HOME = "HOLOMEM_HOME"


def home() -> Path:
    raw = os.environ.get(ENV_HOME, "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".holographic"

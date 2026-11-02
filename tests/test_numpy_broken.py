"""A numpy that is installed but does not load leaves memory working by words (ported 2026-11-02 from the Hermes
plugin's #69, as holographic-memory's own change).

numpy's own import raises RuntimeError when its BLAS sanity check fails (a wrong OpenBLAS, mixed package managers),
and its Windows wheel can raise OSError from os.add_dll_directory. The loader caught only ImportError, so such a numpy
raised out of opening the store: the hook recalled nothing and the memory tools failed. The requirement: it is taken
as missing — the store opens, recall and tools go on by words, structural recall and search by meaning stay off, and
the log says why once.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from _launch import LAUNCH
from test_hooks import context, prompt, seed

pytest.importorskip("numpy")  # the facts are seeded with a working numpy, as on the machine before it broke

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def broken(tmp_path) -> Path:
    """A folder with a numpy that Python finds and that raises on import, put first on sys.path."""
    folder = tmp_path / "broken"
    (folder / "numpy").mkdir(parents=True)
    (folder / "numpy" / "__init__.py").write_text(
        "raise RuntimeError('The current Numpy installation fails to pass a sanity check due to a bug in the "
        "windows runtime.')\n", encoding="utf-8")
    return folder


def run(home: Path, broken: Path, args: list[str], payload: dict | None = None) -> subprocess.CompletedProcess:
    code = f"import sys; sys.path.insert(0, {str(broken)!r})\n" + LAUNCH
    env = {**os.environ, "HOLOMEM_HOME": str(home)}
    data = json.dumps(payload).encode("utf-8") if payload is not None else b""
    return subprocess.run([sys.executable, "-I", "-c", code, str(ROOT), *args], input=data, capture_output=True,
                          env=env, timeout=60)


def test_a_hook_with_a_broken_numpy_still_recalls_by_words(tmp_path, broken):
    home = tmp_path / "holo"
    seed(home, [("The billing service deploys to the staging server with systemd", [])])
    proc = run(home, broken, ["hook", "user-prompt", "--client", "claude-code"],
               prompt("How does the billing service deploy to staging?"))
    assert "deploys to the staging server with systemd" in context(proc), proc.stdout
    log = (home / "holomem.log").read_text(encoding="utf-8") if (home / "holomem.log").exists() else ""
    assert "numpy" in log and "did not load" in log, log  # said, not swallowed


def test_the_engine_with_a_broken_numpy_opens_stores_and_finds(tmp_path, broken):
    script = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(broken)!r}); sys.path.insert(1, {str(ROOT)!r})
        from holomem.engine import holographic
        from holomem.engine.store import MemoryStore
        from holomem.engine.retrieval import FactRetriever
        s = MemoryStore({str(tmp_path / 'facts.db')!r})
        fid = s.add_fact("The billing service deploys to the staging server with systemd")
        hits = [f["fact_id"] for f in FactRetriever(s).search("billing service staging deploy")]
        s.close()
        print(json.dumps({{"fid": fid, "hits": hits, "numpy": holographic.has_numpy(),
                          "error": holographic.numpy_error()}}))
    """)
    proc = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-1500:]
    out = json.loads(proc.stdout.decode().strip().splitlines()[-1])
    assert out["fid"] in out["hits"] and out["numpy"] is False, out
    assert out["error"].startswith("RuntimeError"), out

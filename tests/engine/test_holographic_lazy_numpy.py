"""Issue #43: loading the engine must not import numpy.

The engine's modules can be imported from inside a loader's lock while a managed worker starts its reader thread,
and a top-level numpy import there deadlocked a worker inside numpy._core's create_module (py-spy stack in the
issue). numpy is imported on first use instead. Checked in a fresh process, importing the two modules that used to
import it at module level."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The engine's two modules that used to import numpy at module level, imported in a fresh interpreter.
_PROBE = f"""
import importlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, {str(REPO_ROOT)!r})
hrr = importlib.import_module("holomem.engine.holographic")
emb = importlib.import_module("holomem.engine.embeddings")
out = {{"numpy_imported_at_load": "numpy" in sys.modules, "has_numpy": hrr.has_numpy()}}
if out["has_numpy"]:
    out["atom"] = len(hrr.encode_atom("probe", 8))
    out["blob"] = len(emb.to_blob([0.0] * 4))
out["numpy_imported_after_use"] = "numpy" in sys.modules
print(json.dumps(out))
"""


def test_loading_the_modules_does_not_import_numpy():
    """The loader's lock never covers a numpy import: the modules load with numpy untouched."""
    result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert state["numpy_imported_at_load"] is False, "numpy was imported while the engine loaded"
    # and the first use still gets it (or reports it missing when it is, exactly as before)
    assert state["has_numpy"] == (importlib.util.find_spec("numpy") is not None)
    if state["has_numpy"]:
        assert state["atom"] == 8 and state["blob"] == 16 and state["numpy_imported_after_use"] is True
    else:  # pragma: no cover - the CI images install numpy
        pytest.skip("numpy is not installed here")


def test_concurrent_first_use_waits_for_numpy_instead_of_reporting_it_missing():
    """A second first use cannot permanently disable HRR while another thread imports numpy."""
    probe = _PROBE.split("out = ")[0] + r'''
import builtins
import queue
import threading

events = queue.Queue()
release = threading.Event()
real_import = builtins.__import__

def gated_import(name, *args, **kwargs):
    if name == "numpy" and args and args[0] is hrr.__dict__:
        events.put(("enter", threading.current_thread().name))
        if not release.wait(10):
            raise RuntimeError("test did not release numpy import")
    return real_import(name, *args, **kwargs)

# Only delay this module's import statement; the real numpy import still runs.
builtins.__import__ = gated_import

def use():
    try:
        events.put(("done", threading.current_thread().name, hrr.has_numpy()))
    except Exception as exc:
        events.put(("error", threading.current_thread().name, repr(exc)))

first = threading.Thread(target=use, name="first")
second = threading.Thread(target=use, name="second")
first.start()
try:
    initial = events.get(timeout=10)
    second.start()
    observed = events.get(timeout=10)
finally:
    release.set()
    first.join(10)
    if second.ident is not None:
        second.join(10)
    builtins.__import__ = real_import
records = []
while not events.empty():
    records.append(events.get_nowait())
print(json.dumps({"initial": initial, "observed": observed, "records": records,
                  "workers_alive": first.is_alive() or second.is_alive(),
                  "available_after": hrr.has_numpy()}))
'''
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert state["initial"] == ["enter", "first"]
    assert state["observed"] == ["enter", "second"], "a concurrent first use reported numpy missing"
    assert sorted(state["records"]) == [["done", "first", True], ["done", "second", True]]
    assert state["available_after"] is True and state["workers_alive"] is False

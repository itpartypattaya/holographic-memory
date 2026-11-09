"""A holomem process keeps numpy's OpenBLAS on one thread (2026-11-09).

numpy's OpenBLAS commits a buffer of ~32 MB per CPU thread when numpy is imported: on an 8-thread laptop the MCP server
of every Claude Code session held ~260 MB, ~225 MB of it those buffers, so ten sessions took ~2.8 GB of the commit
charge of an 8 GB machine. holomem's math is a matrix-vector product over the facts and FFTs (not BLAS): one thread does
it as fast. The requirement: any process that imports holomem and then numpy runs OpenBLAS on one thread, unless the
person set OPENBLAS_NUM_THREADS themselves, which is kept.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from _launch import child_env

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("numpy")

# The child imports holomem as each entry point does (the package first), then numpy, and asks the OpenBLAS that
# numpy loaded how many threads it runs. numpy's wheels (Windows, Linux) bundle it as scipy-openblas in numpy.libs; a
# numpy built against another BLAS has nothing to limit, but one that says it uses OpenBLAS and whose library is not
# found fails the test (the check would be blind, not passed).
CHILD = textwrap.dedent("""
    import ctypes, glob, json, os, sys
    sys.path.insert(0, sys.argv[1])
    import holomem.__main__
    import numpy
    blas = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"].get("name", "")
    libs = glob.glob(os.path.join(os.path.dirname(numpy.__file__) + ".libs", "*openblas*"))
    libs = [p for p in libs if p.endswith((".dll", ".so", ".dylib")) or ".so." in p]
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    threads = None
    if libs:
        lib = ctypes.CDLL(libs[0])
        get = getattr(lib, "scipy_openblas_get_num_threads64_", None) or getattr(lib, "openblas_get_num_threads")
        threads = get()
    print(json.dumps({"blas": blas, "libs": libs, "cpus": cpus, "threads": threads,
                      "env": os.environ.get("OPENBLAS_NUM_THREADS")}))
""")


def _child(**env: str) -> dict:
    out = subprocess.run([sys.executable, "-I", "-c", CHILD, str(ROOT)], capture_output=True, text=True, timeout=60,
                         env=child_env(drop=("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "GOTO_NUM_THREADS"), **env))
    assert out.returncode == 0, out.stderr
    found = json.loads(out.stdout.strip().splitlines()[-1])
    if "openblas" not in found["blas"].lower():
        pytest.skip(f"this numpy is built against {found['blas'] or 'an unnamed BLAS'}, not OpenBLAS: nothing to limit")
    assert found["threads"] is not None, f"numpy says it uses {found['blas']} but its library is not in numpy.libs"
    return found


def test_holomem_runs_openblas_on_one_thread():
    found = _child()
    assert found["env"] == "1"
    assert found["threads"] == 1


def test_a_count_the_person_set_is_kept():
    found = _child(OPENBLAS_NUM_THREADS="2")
    assert found["env"] == "2"  # not replaced by holomem's 1
    assert found["threads"] == min(2, found["cpus"])  # OpenBLAS runs no more threads than there are CPUs

"""The vendored engine is the plugin's release, byte for byte.

Requirement: vendoring takes from a tag exactly the engine (top-level modules but the provider ``__init__.py``,
``dictionaries/``, ``LICENSE``, ``LICENSES/``) and nothing else (tests, tools, caches); it records the tag, its commit
and a hash per file; a later check catches a hand edit, a missing file, a file added by hand, and a tag that moved
since the copy was made. The copy committed in this repository passes that check.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("sync_engine", ROOT / "tools" / "sync_engine.py")
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


_PATTERNS = "_EXTRACT_CATEGORIES = (\n    ([re.compile(r'\\bI\\s+prefer\\s+\\S')], \"user_pref\"),\n)"


def _plugin(tmp_path: Path) -> Path:
    """A plugin-shaped repository with one tag, its provider, tests and stray caches."""
    repo = tmp_path / "plugin"
    for name, data in {
        "__init__.py": ("import re\nfrom agent.memory_provider import MemoryProvider\n" + _PATTERNS
                        + "\nOTHER = 1\n").encode("utf-8"),
        "store.py": b"STORE = 1\n",
        "text.py": b"TEXT = 1\n",
        "LICENSE": b"MIT\n",
        "LICENSES/Wikidata-CC0.txt": b"CC0\n",
        "dictionaries/__init__.py": b"",
        "dictionaries/ru.tsv.gz": bytes(range(256)),
        "tests/test_store.py": b"def test(): pass\n",
        "tools/build.py": b"print(1)\n",
        "README.md": b"# plugin\n",
    }.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    _git(tmp_path, "init", "-q", str(repo))
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", "commit", "-qm", "release")
    _git(repo, "tag", "v1.0.0")
    return repo


def test_vendoring_takes_the_engine_and_nothing_else(tmp_path):
    repo, engine = _plugin(tmp_path), tmp_path / "engine"
    manifest = sync.vendor(repo, "v1.0.0", engine)
    assert sorted(manifest["files"]) == ["LICENSE", "LICENSES/Wikidata-CC0.txt", "dictionaries/__init__.py",
                                         "dictionaries/ru.tsv.gz", "extract_patterns.py", "store.py", "text.py"]
    assert manifest["commit"] == _git(repo, "rev-list", "-n", "1", "v1.0.0")
    assert (engine / "dictionaries" / "ru.tsv.gz").read_bytes() == bytes(range(256))  # binary untouched
    assert "MemoryProvider" not in (engine / "__init__.py").read_text(encoding="utf-8")  # ours, not the provider
    assert sync.check(engine, repo) == []


def test_vendoring_takes_the_extraction_patterns_out_of_the_provider(tmp_path):
    """The auto-extraction patterns live in the provider, which is not vendored: the one assignment is copied out of
    the tag's __init__.py, exactly, and a hand edit of that copy is caught like any other."""
    repo, engine = _plugin(tmp_path), tmp_path / "engine"
    sync.vendor(repo, "v1.0.0", engine)
    text = (engine / "extract_patterns.py").read_text(encoding="utf-8")
    assert _PATTERNS in text
    assert "MemoryProvider" not in text and "OTHER" not in text
    namespace: dict = {}
    exec(compile(text, "extract_patterns.py", "exec"), namespace)
    assert namespace["_EXTRACT_CATEGORIES"][0][1] == "user_pref"
    (engine / "extract_patterns.py").write_text(text.replace("prefer", "hate"), encoding="utf-8")
    assert sync.check(engine) == ["changed: extract_patterns.py"]


def test_check_catches_edits_removals_additions_and_a_moved_tag(tmp_path):
    repo, engine = _plugin(tmp_path), tmp_path / "engine"
    sync.vendor(repo, "v1.0.0", engine)
    (engine / "store.py").write_bytes(b"STORE = 2\n")
    (engine / "text.py").unlink()
    (engine / "patch.py").write_bytes(b"\n")
    assert sorted(sync.check(engine)) == ["changed: store.py", "missing: text.py", "not from the plugin: patch.py"]

    sync.vendor(repo, "v1.0.0", engine)
    (repo / "store.py").write_bytes(b"STORE = 3\n")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "fix")
    _git(repo, "tag", "-f", "v1.0.0")
    problems = sync.check(engine, repo)
    assert any("tag v1.0.0 is now" in p for p in problems)


def test_the_committed_copy_is_exact():
    assert sync.check() == []


def test_the_copy_is_a_release_of_the_plugin():
    manifest = (ROOT / "holomem" / "engine" / "ENGINE.json").read_text(encoding="utf-8")
    assert '"source": "itpartypattaya/hermes-plugin-holographic"' in manifest
    assert '"tag": "v' in manifest


def test_a_copy_of_a_copy_keeps_cache_dirs_out_of_the_check(tmp_path):
    engine = tmp_path / "engine"
    shutil.copytree(ROOT / "holomem" / "engine", engine, ignore=shutil.ignore_patterns("__pycache__"))
    (engine / "__pycache__").mkdir()
    (engine / "__pycache__" / "store.cpython-312.pyc").write_bytes(b"\0")
    assert sync.check(engine) == []

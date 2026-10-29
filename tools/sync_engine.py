"""Vendor the holographic engine from a release tag of hermes-plugin-holographic, byte for byte.

    python tools/sync_engine.py --repo ../hermes-plugin-holographic --tag v0.24.0   # replace holomem/engine
    python tools/sync_engine.py --check                                            # the copy matches ENGINE.json
    python tools/sync_engine.py --check --repo ../hermes-plugin-holographic        # ... and the tag it names

The engine is every top-level module of the plugin but its provider (``__init__.py``, which needs Hermes core), the
``dictionaries/`` it reads, and the licenses that cover them. Nothing in the copy is edited: the three Hermes helpers
the store calls are supplied from outside (``holomem.compat``), so a fix to the engine goes to the plugin and arrives
here with the next tag. ``holomem/engine/__init__.py`` is the one file written here; it is not part of the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "holomem" / "engine"
MANIFEST = "ENGINE.json"
SOURCE = "itpartypattaya/hermes-plugin-holographic"
GENERATED = ("__init__.py", MANIFEST)  # written here, not copied

INIT = '''"""The holographic engine, vendored from {source} {tag} ({commit}) by tools/sync_engine.py.

Do not edit the modules here: change the plugin and vendor its next tag. This file is the only one written by the
sync; it supplies the Hermes core helpers the store calls when Hermes core is not installed (holomem.compat)."""

from .. import compat as _compat

_compat.install()
'''


def wanted(name: str) -> bool:
    """A path of the plugin's tree that belongs to the engine."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "__pycache__" in path.parts:
        return False
    if len(path.parts) == 1:
        return (path.suffix == ".py" and path.name != "__init__.py") or path.name == "LICENSE"
    return path.parts[0] in ("dictionaries", "LICENSES")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def tag_files(repo: Path, tag: str) -> tuple[str, dict[str, bytes]]:
    """(commit, {path: bytes}) of the engine files at *tag*."""
    commit = _git(repo, "rev-list", "-n", "1", tag).decode().strip()
    files = {}
    with tarfile.open(fileobj=io.BytesIO(_git(repo, "archive", "--format=tar", commit)), mode="r:") as tar:
        for member in tar.getmembers():
            if member.isfile() and wanted(member.name):
                files[member.name] = tar.extractfile(member).read()
    if not any(name.endswith(".py") for name in files) or "store.py" not in files:
        raise SystemExit(f"{tag} holds no engine (store.py missing): is {repo} the plugin repository?")
    return commit, files


def vendor(repo: Path, tag: str, engine: Path = ENGINE) -> dict:
    commit, files = tag_files(repo, tag)
    if engine.exists():
        shutil.rmtree(engine)
    engine.mkdir(parents=True)
    for name, data in sorted(files.items()):
        target = engine / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = {"source": SOURCE, "tag": tag, "commit": commit,
                "files": {name: sha256(data) for name, data in sorted(files.items())}}
    (engine / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    (engine / "__init__.py").write_text(INIT.format(source=SOURCE, tag=tag, commit=commit), encoding="utf-8",
                                        newline="\n")
    return manifest


def check(engine: Path = ENGINE, repo: Path | None = None) -> list[str]:
    """What differs from the manifest (and, with *repo*, from the tag it names); [] = the copy is exact."""
    try:
        manifest = json.loads((engine / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{MANIFEST} unreadable ({type(exc).__name__})"]
    problems = []
    listed = manifest.get("files", {})
    for name, digest in listed.items():
        path = engine / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif sha256(path.read_bytes()) != digest:
            problems.append(f"changed: {name}")
    for path in engine.rglob("*"):
        rel = path.relative_to(engine).as_posix()
        if path.is_file() and rel not in listed and rel not in GENERATED and "__pycache__" not in path.parts:
            problems.append(f"not from the plugin: {rel}")
    if repo is not None:
        commit, files = tag_files(repo, manifest.get("tag", ""))
        if commit != manifest.get("commit"):
            problems.append(f"tag {manifest.get('tag')} is now {commit}, the copy was made from {manifest.get('commit')}")
        if {n: sha256(d) for n, d in files.items()} != listed:
            problems.append(f"the manifest does not match the files of {manifest.get('tag')}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, help="a checkout of hermes-plugin-holographic")
    parser.add_argument("--tag", help="the release tag to vendor (with --repo)")
    parser.add_argument("--check", action="store_true", help="verify the vendored copy instead of replacing it")
    parser.add_argument("--engine", type=Path, default=ENGINE, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.check:
        problems = check(args.engine, args.repo)
        for line in problems:
            print(line)
        print("engine copy exact" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    if not (args.repo and args.tag):
        parser.error("vendoring needs --repo and --tag")
    manifest = vendor(args.repo, args.tag, args.engine)
    print(f"vendored {manifest['tag']} ({manifest['commit'][:12]}): {len(manifest['files'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

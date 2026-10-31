"""Text files are committed with LF line endings.

Requirement: the repository keeps LF in every committed text file without a .gitattributes rule that rewrites file
contents (the Claude plugin directory refuses to validate a repository with one), so a file staged with CRLF must be
caught here, by what the index holds, whatever a checkout on Windows shows in the working tree. Files marked `-text`
are kept byte for byte as they came (the vendored engine, its license texts among them) and are not checked.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def crlf_in_index(repo: Path) -> list[str]:
    """Paths whose staged content has CRLF line endings (git's own detection: `i/crlf` or `i/mixed`), except those
    marked `-text`."""
    out = subprocess.run(["git", "-c", "core.quotepath=off", "ls-files", "--eol"], cwd=repo, check=True,
                         capture_output=True, text=True, encoding="utf-8").stdout
    found = []
    for line in out.splitlines():
        info, _, path = line.partition("\t")
        index, _, attr = info.split(None, 2)  # the attribute column may hold spaces ("attr/text=auto eol=lf")
        if index in ("i/crlf", "i/mixed") and attr.strip() != "attr/-text":
            found.append(path)
    return found


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=repo, check=True, capture_output=True)


def test_a_staged_crlf_file_is_found(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "lf.md").write_bytes(b"one\ntwo\n")
    (tmp_path / "crlf.md").write_bytes(b"one\r\ntwo\r\n")
    (tmp_path / "mixed.py").write_bytes(b"one\r\ntwo\n")
    (tmp_path / "data.gz").write_bytes(b"\x1f\x8b\x08\x00\r\n\x00\x00")
    (tmp_path / "vendored").mkdir()
    (tmp_path / "vendored" / "LICENSE.txt").write_bytes(b"as it came\r\n")
    (tmp_path / ".gitattributes").write_bytes(b"vendored/** -text\n")
    _git(tmp_path, "add", ".")
    assert sorted(crlf_in_index(tmp_path)) == ["crlf.md", "mixed.py"]


def test_repository_has_no_crlf_text_files():
    assert crlf_in_index(ROOT) == []

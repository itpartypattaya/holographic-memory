"""What the Claude plugin directory can read of the plugin.

Requirement: the plugin folder is the repository root, and the directory's validator reads every file of it as text.
It holds a version for a person to look at when a file is binary and not an image (the gzipped dictionaries,
2026-11-05: "Files or downloads the validator couldn't inspect"), when a file is 256 KiB or larger, or when the
plugin has 512 files or more ("anything past either limit is not read and is held for review"). So everything the
plugin ships, except its images, is UTF-8 text under 256 KiB, and the plugin stays under 512 files. Files not yet
committed count too: a new file is caught before it is pushed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMIT = 256 * 1024
IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}  # the validator checks images as images, not as code


def _shipped() -> list[Path]:
    out = subprocess.run(["git", "-c", "core.quotepath=off", "ls-files", "--cached", "--others", "--exclude-standard",
                          "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    return [ROOT / p for p in out.decode("utf-8").split("\0") if p and (ROOT / p).is_file()]


def test_every_file_but_an_image_is_text_under_the_limit():
    bad = []
    for path in _shipped():
        if path.suffix.lower() in IMAGES:
            continue
        data = path.read_bytes()
        try:
            data.decode("utf-8")
            binary = b"\0" in data
        except UnicodeDecodeError:
            binary = True
        if binary or len(data) >= LIMIT:
            bad.append(f"{path.relative_to(ROOT).as_posix()} ({len(data)} bytes{', binary' if binary else ''})")
    assert bad == []


def test_the_plugin_has_fewer_files_than_the_directory_reads():
    assert len(_shipped()) < 512

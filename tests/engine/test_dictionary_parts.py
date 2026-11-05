"""The language dictionaries ship as plain text, cut into parts.

Requirement: the Claude plugin directory reads every file of a plugin as text and holds a version for a person when a
file is one it cannot read (a .gz, 2026-11-05: "Files or downloads the validator couldn't inspect") or larger than
256 KiB. So a table is a folder of UTF-8 parts (01.tsv, 02.tsv, …), each opening with "# part i/n" and closing with
"# end i/n". Damage a .gz used to show (a file cut short, a missing one) must still show: a part missing, cut short,
out of place or not UTF-8 makes the whole table unreadable, which is said once, and the language goes on without it,
never half a table read as if whole. A part checked out with Windows line ends reads the same.
"""
from __future__ import annotations

import logging

import pytest

from holomem.engine import text


def _parts(folder, bodies, ext="tsv"):
    folder.mkdir()
    n = len(bodies)
    for i, body in enumerate(bodies, 1):
        (folder / f"{i:02d}.{ext}").write_bytes(f"# part {i}/{n}\n{body}# end {i}/{n}\n".encode("utf-8"))
    return folder


def test_a_table_is_its_parts_joined(tmp_path):
    folder = _parts(tmp_path / "t", ["a\tb\n", "c\td\n", "e\tf\n"])
    assert text._read_table(folder) == "a\tb\nc\td\ne\tf\n"


def test_windows_line_ends_read_the_same(tmp_path):
    folder = _parts(tmp_path / "t", ["a\tb\n", "c\td\n"])
    for part in folder.iterdir():
        part.write_bytes(part.read_bytes().replace(b"\n", b"\r\n"))
    assert text._read_table(folder) == "a\tb\nc\td\n"


def test_a_missing_part_is_damage(tmp_path):
    folder = _parts(tmp_path / "t", ["a\tb\n", "c\td\n", "e\tf\n"])
    (folder / "02.tsv").unlink()
    with pytest.raises(ValueError):
        text._read_table(folder)


def test_a_part_cut_short_is_damage(tmp_path):
    folder = _parts(tmp_path / "t", ["a\tb\n", "c\td\n" * 50])
    last = folder / "02.tsv"
    last.write_bytes(last.read_bytes()[:-30])
    with pytest.raises(ValueError):
        text._read_table(folder)


def test_a_part_out_of_place_is_damage(tmp_path):
    folder = _parts(tmp_path / "t", ["a\tb\n", "c\td\n"])
    one, two = (folder / "01.tsv").read_bytes(), (folder / "02.tsv").read_bytes()
    (folder / "01.tsv").write_bytes(two)
    (folder / "02.tsv").write_bytes(one)
    with pytest.raises(ValueError):
        text._read_table(folder)


def test_a_part_that_is_not_utf8_is_damage(tmp_path):
    folder = _parts(tmp_path / "t", ["a\tb\n"])
    (folder / "01.tsv").write_bytes("# part 1/1\nпривет\tмир\n# end 1/1\n".encode("cp1251"))
    with pytest.raises(ValueError):
        text._read_table(folder)


def test_a_missing_table_is_missing(tmp_path):
    with pytest.raises(OSError):
        text._read_table(tmp_path / "nothing")


def test_a_damaged_table_is_said_once_and_not_used(tmp_path, monkeypatch, caplog):
    folder = _parts(tmp_path / "ja_variants", ["".join(f"子ども{i}\t子供{i}\n" for i in range(200)),
                                              "".join(f"子ども{i}\t子供{i}\n" for i in range(200, 400))])
    (folder / "02.tsv").write_bytes((folder / "02.tsv").read_bytes()[:-100])  # cut short: half a table
    monkeypatch.setattr(text, "_JA_FILE", folder)
    monkeypatch.setattr(text, "_JA_TABLE", None)
    with caplog.at_level(logging.WARNING):
        assert text._ja_table() == ({}, {})
    assert "ja_variants could not be read" in caplog.text
    monkeypatch.setattr(text, "_JA_TABLE", None)  # the next test loads the real one again


@pytest.mark.parametrize("name", ["de_compounds", "de_es_forms", "en_derivation", "ja_variants", "ru_derivation",
                                  "zh_words"])
def test_each_shipped_table_reads_whole(name):
    body = text._read_table(text._DICTIONARIES / name)
    assert body.count("\n") > 1000 and "\r" not in body
    assert "# part " not in body and "# end " not in body  # the marks are taken off, never read as rows

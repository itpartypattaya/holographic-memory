"""Export: the whole store as one readable Markdown page, for a review by eye (memory_export and holomem export).

Requirements (a revision view, not a backup):
- memory_export answers {markdown, count}: a header "# Holographic facts (N of M)" (N shown, M every fact in the
  store), then one line per fact, most trusted first: "- #id [trust, category] text — names", and under it, in
  italics, where its current version was written (via · origin · client project <key> · session · recorded <day>);
- a fact merged into a shown fact is listed under "## Merged into the facts above" with its text and the day of the
  merge, not in the main list; the answer's "merged" names each pair;
- the tool cuts a text at 500 characters ("…") and shows at most `limit` facts (default 100, at most 1000); the CLI
  shows every fact whole unless --limit is given;
- category and min_trust narrow the list; an empty store is "(0 of 0)", not an error;
- a fact recall refuses to inject (a credential) is in the export, marked "(withheld from recall)": the export is the
  place to see it and clean it up;
- holomem export writes UTF-8 with LF line ends whatever the console's code page: to stdout, or with --out to that
  file through a temporary file and a replace, so a failure (a store that cannot be read, a target that cannot be
  replaced) exits 1 and leaves an existing file as it was, with no temporary file left beside it.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from _launch import LAUNCH as _LAUNCH
from holomem.config import project_key, project_root
from test_hooks import context, hook, prompt, seed
from test_mcp_server import Client

ROOT = Path(__file__).resolve().parent.parent
LONG = " ".join(f"step{i} of the Lisbon office move" for i in range(1, 200))[:699] + "."  # one line, 700 characters


@pytest.fixture
def home(tmp_path) -> Path:
    return tmp_path / "holo"


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "bakery-site"
    (path / ".git").mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(path))  # the server's project, as Claude Code sets it
    return path


def cli(home: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """holomem export as a person runs it: a new process, the console's own code page (no UTF-8 override)."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    env["HOLOMEM_HOME"] = str(home)
    return subprocess.run([sys.executable, "-I", "-c", _LAUNCH, str(ROOT), "export", *args], capture_output=True,
                          env=env, cwd=str(cwd or home.parent), timeout=60)


def export(home: Path, **arguments) -> dict:
    client = Client(home).start()
    try:
        answer, is_error = client.call("memory_export", **arguments)
    finally:
        client.close()
    assert not is_error, answer
    return answer


def test_a_fact_is_exported_with_trust_category_names_and_source(home, project):
    client = Client(home).start()
    try:
        added, _ = client.call("memory_add", content="Lena moved to Lisbon in March", entities=["Lena"],
                               category="user_pref", origin="said")
        listed, _ = client.call("memory_list")
        answer, is_error = client.call("memory_export")
    finally:
        client.close()
    assert not is_error, answer
    trust = listed["results"][0]["trust_score"]
    md = answer["markdown"]
    assert answer["count"] == 1
    assert md.startswith("# Holographic facts (1 of 1)\n")
    assert f"- #{added['fact_id']} [{trust:.2f}, user_pref] Lena moved to Lisbon in March — Lena\n" in md
    source = md.split("\n")[-1]
    assert source.startswith("  _tool · said · claude-code project ") and source.endswith("_")
    assert project_key(project_root(project)) in source
    assert re.search(r"session mcp-\d+ · recorded \d{4}-\d{2}-\d{2}_$", source), source


def test_a_merged_fact_is_listed_under_the_fact_it_became_part_of(home):
    client = Client(home).start()
    try:
        keep, _ = client.call("memory_add", content="The bakery on Third Street opens at seven", origin="said")
        copy, _ = client.call("memory_add", content="Third Street's bakery starts selling bread at 7 am", origin="said")
        merged, is_error = client.call("memory_merge", fact_id=keep["fact_id"], **{"from": [copy["fact_id"]]})
        assert not is_error, merged
        answer, is_error = client.call("memory_export")
    finally:
        client.close()
    assert not is_error, answer
    md = answer["markdown"]
    main, _, tail = md.partition("\n## Merged into the facts above\n")
    assert tail, md
    assert f"- #{keep['fact_id']} " in main and f"- #{copy['fact_id']} " not in main
    assert re.search(rf"- #{copy['fact_id']} → #{keep['fact_id']} \(\d{{4}}-\d{{2}}-\d{{2}}\): "
                     r"Third Street's bakery starts selling bread at 7 am", tail), tail
    assert [(m["fact_id"], m["into"]) for m in answer["merged"]] == [(copy["fact_id"], keep["fact_id"])]
    assert md.startswith("# Holographic facts (1 of 1)\n")


def test_cli_writes_utf8_with_lf_to_a_file_and_the_same_bytes_to_stdout(home, tmp_path):
    seed(home, [("Лена переехала в Лиссабон в марте", ["Лена"]), ("罗伯特每天早上喝绿茶", ["罗伯特"]),
                ("Mark keeps the bakery keys in the blue box", ["Mark"])])
    out = tmp_path / "review" / "memory.md"
    out.parent.mkdir()
    out.write_bytes(b"an older export\r\n")  # an existing file is replaced, not appended to
    written = cli(home, "--out", str(out))
    assert written.returncode == 0, written.stderr.decode(errors="replace")[-1500:]
    data = out.read_bytes()
    text = data.decode("utf-8")
    assert b"\r\n" not in data and text.endswith("\n") and "an older export" not in text
    assert text.startswith("# Holographic facts (3 of 3)\n")
    assert "Лена переехала в Лиссабон в марте — Лена\n" in text and "罗伯特每天早上喝绿茶 — 罗伯特\n" in text
    assert [p.name for p in out.parent.iterdir()] == ["memory.md"]  # no temporary file left
    printed = cli(home)
    assert printed.returncode == 0, printed.stderr.decode(errors="replace")[-1500:]
    assert printed.stdout == data


def test_the_tool_cuts_long_texts_and_limits_the_list_the_cli_does_not(home):
    assert len(LONG) == 700
    seed(home, [(LONG, []), ("Robert waters the office plants on Mondays", []),
                ("Mark orders the office coffee beans", [])])
    answer = export(home)
    line = next(x for x in answer["markdown"].split("\n") if "step1 of" in x)
    text = line.split("] ", 1)[1]
    assert text == LONG[:499] + "…"
    one = export(home, limit=1)
    assert one["count"] == 1 and one["markdown"].startswith("# Holographic facts (1 of 3)\n")
    whole = cli(home)
    assert whole.returncode == 0
    assert f"] {LONG}\n" in whole.stdout.decode("utf-8")
    first = cli(home, "--limit", "1")
    assert first.returncode == 0 and first.stdout.decode("utf-8").startswith("# Holographic facts (1 of 3)\n")


def test_an_empty_store_and_the_filters(home):
    empty = export(home)
    assert empty["markdown"].startswith("# Holographic facts (0 of 0)")
    assert empty["count"] == 0
    assert cli(home).stdout.decode("utf-8").startswith("# Holographic facts (0 of 0)")
    client = Client(home).start()
    try:
        sure, _ = client.call("memory_add", content="The bakery site is built with Astro", category="project",
                              origin="said")
        guess, _ = client.call("memory_add", content="Robert probably prefers green tea", category="user_pref",
                               origin="inferred")
        listed, _ = client.call("memory_list")
    finally:
        client.close()
    trust = {r["fact_id"]: r["trust_score"] for r in listed["results"]}
    assert trust[sure["fact_id"]] > trust[guess["fact_id"]]
    by_category = export(home, category="project")
    assert by_category["count"] == 1 and by_category["markdown"].startswith("# Holographic facts (1 of 2)\n")
    assert "Astro" in by_category["markdown"] and "green tea" not in by_category["markdown"]
    trusted = export(home, min_trust=trust[sure["fact_id"]])
    assert trusted["count"] == 1 and "Astro" in trusted["markdown"]
    narrowed = cli(home, "--category", "user_pref", "--min-trust", "0")
    assert narrowed.returncode == 0
    text = narrowed.stdout.decode("utf-8")
    assert "green tea" in text and "Astro" not in text


def test_a_fact_recall_withholds_is_exported_and_marked(home):
    secret = "The staging deploy token for Robert is ghp_0123456789abcdefABCDEF0123456789abcd"
    seed(home, [(secret, ["Robert"]), ("The staging deploy for Robert runs every Friday", ["Robert"])])
    recalled = context(hook(home, "user-prompt", prompt("What is the staging deploy token for Robert?")))
    assert "runs every Friday" in recalled and "ghp_" not in recalled  # recall really hides it
    md = export(home)["markdown"]
    lines = md.split("\n")
    hidden = next(x for x in lines if "ghp_" in x)
    shown = next(x for x in lines if "runs every Friday" in x)
    assert hidden.endswith(f"{secret} (withheld from recall) — Robert")
    assert "withheld" not in shown


def test_a_store_that_cannot_be_read_leaves_the_old_file(home, tmp_path):
    home.mkdir(parents=True)
    (home / "memory_store.db").write_bytes(b"this is not a database\n" * 200)
    out = tmp_path / "memory.md"
    out.write_bytes(b"# last week's export\n")
    proc = cli(home, "--out", str(out))
    assert proc.returncode == 1
    message = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
    assert "export failed" in message and "DatabaseError" in message
    assert out.read_bytes() == b"# last week's export\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["holo", "memory.md"]


def test_a_target_that_cannot_be_replaced_leaves_the_old_file(home, tmp_path):
    seed(home, [("Mark keeps the bakery keys in the blue box", ["Mark"])])
    folder = tmp_path / "locked"
    folder.mkdir()
    out = folder / "memory.md"
    out.write_bytes(b"# last week's export\n")
    if sys.platform == "win32":
        held = open(out, "rb")  # held open: Windows refuses to replace a file another handle has open
        try:
            proc = cli(home, "--out", str(out))
        finally:
            held.close()
    else:
        folder.chmod(0o555)  # the folder cannot take a new file
        try:
            proc = cli(home, "--out", str(out))
        finally:
            folder.chmod(0o755)
    assert proc.returncode == 1, proc.stdout.decode(errors="replace")
    message = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
    assert "export failed" in message and "blue box" not in message
    assert out.read_bytes() == b"# last week's export\n"
    assert [p.name for p in folder.iterdir()] == ["memory.md"]


def test_the_tool_is_listed_read_only(home):
    client = Client(home).start()
    try:
        tools = {t["name"]: t for t in client.request("tools/list")["result"]["tools"]}
    finally:
        client.close()
    schema = tools["memory_export"]
    assert schema["annotations"]["readOnlyHint"] is True
    assert schema["inputSchema"]["properties"]["limit"]["maximum"] == 1000
    assert "context" in schema["description"]  # it says the page goes into the conversation
    assert "withheld from recall" in schema["description"]

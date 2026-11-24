"""D5 intake follow-ups, wave 11 (eval M5). Pillar docs/evals/D5-KIT-STEP-2026-11-24.md section 7.

  (1) package check at intake: on a closed task that names a package hash, the intake rebuilds the package hash from
      the submitted package folder and refuses a hash that differs, or no folder, with a clear message
  (2) the task fields the builder still takes from the caller are listed (CALLER_FIELDS) and printed (from_caller);
      every spec.Task field is pinned, from the package table, or listed
  (3) a closed kit form entry (no package hash field) is refused (ruling 57 c, rule book amendment 2)

Synthetic only: every site id, hash and earthquake id is made up. No em dashes.
"""
import dataclasses
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoreboard.evalkit import intake  # noqa: E402
from scoreboard.evalkit import package as P  # noqa: E402
from scoreboard.evalkit import register as reg  # noqa: E402
from scoreboard.evalkit import task_builder as TB  # noqa: E402
from scoreboard.evalkit.spec import Task, site_list_sha256  # noqa: E402

IDS = ("ex:0001", "ex:0002", "ex:0003")
CSV = "site_id,p\nex:0001,0.412\nex:0002,0.03\nex:0003,1\n"
B3 = {"rule_book_version": "1.0", "entrant_id": "E0007", "task_sha256": "0" * 64, "model": "example-model",
      "model_version": "2.1", "primary": True, "output_kind": "target_probability", "conversion_sha256": None,
      "division": "closed", "package_sha256": "1" * 64, "inputs": ["package", "shakemap"],
      "exposure": {"us7000abcd": "unseen"}, "training_cases_sha256": "2" * 64, "choices_after_outcomes": "none",
      "text_read": "none about this earthquake",
      "statement": "Probability of minor or worse ejecta at the site after this earthquake.", "house": False}
KIT = {"model": "m", "output_kind": "score", "statement": "s", "house": False, "exposure": {"us7000abcd": "unseen"}}
T0 = datetime(2026, 11, 25, 12, 0, 0, tzinfo=timezone.utc)


def clock():
    state = {"t": T0}

    def now():
        t = state["t"]
        state["t"] = t + timedelta(seconds=1)
        return t
    return now


def make_package(d: Path) -> Path:
    pd = d / "pkg"
    (pd / "shakemap").mkdir(parents=True)
    (pd / "cpt_list.csv").write_text("site_id,lat,lon\nex:0001,-43.5,172.6\n", encoding="utf-8")
    (pd / "shakemap" / "grid.xml").write_text("<grid synthetic='1'/>\n", encoding="utf-8")
    return pd


@pytest.fixture
def closed(tmp_path):
    """A package folder, its table (outside the folder), a closed task built from the table, and a matching entry."""
    pd = make_package(tmp_path)
    table = tmp_path / "table.csv"
    pkg = P.write_table(pd, table)["package_sha256"]
    fields = {"event_id": "us7000abcd", "target": "minor or worse ejecta", "label_source": "synthetic",
              "label_rule_sha256": "4" * 64, "site_list_sha256": site_list_sha256(IDS), "track": "record",
              "division": "closed", "label_caveats": "synthetic"}
    tp = tmp_path / "task.json"
    TB.write_task(fields, tp, package_table=table)
    t = TB.load_task(tp)
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(dict(B3, task_sha256=t.sha256(), package_sha256=pkg)), encoding="utf-8")
    (tmp_path / "entry.csv").write_text(CSV, encoding="utf-8")
    return {"dir": pd, "table": table, "pkg": pkg, "task": t, "task_path": tp, "entry": mp}


# ---- (1) package check at intake --------------------------------------------------------------------------------

def test_the_rebuilt_package_hash_equals_the_written_table_hash(tmp_path):
    pd = make_package(tmp_path)
    (pd / ".DS_Store").write_bytes(b"\x00finder")                  # hidden: left out, as `package write` does
    got, n = P.rebuild_sha256(pd)
    assert n == 2 and got == P.write_table(pd, tmp_path / "t.csv")["package_sha256"] == P.table_sha256(tmp_path / "t.csv")


def test_a_closed_entry_with_its_package_passes_and_is_registered(closed, tmp_path):
    it = intake.read_entry(closed["entry"], tmp_path / "entry.csv")
    res = intake.check_against_task(it, closed["task"], closed["dir"])
    assert res["package"] == res["package_files"] == "PASS" and intake.task_failures(res) == []
    r = tmp_path / "r.jsonl"
    assert reg.submit_entry(r, closed["entry"], task=closed["task_path"], package=closed["dir"],
                            now=clock())["seq"] == 1


@pytest.mark.parametrize("damage", ["changed", "missing", "extra", "empty"])
def test_a_package_that_rebuilds_to_another_hash_is_refused_with_both_hashes(closed, tmp_path, damage):
    pd = closed["dir"]
    if damage == "changed":
        (pd / "cpt_list.csv").write_text("site_id,lat,lon\nex:0001,-43.5,172.7\n", encoding="utf-8")
    elif damage == "missing":
        (pd / "shakemap" / "grid.xml").unlink()
    elif damage == "extra":
        (pd / "notes.txt").write_text("not an input\n", encoding="utf-8")
    else:                                                           # review 1 F3 (b): an empty folder never passes
        (pd / "shakemap" / "grid.xml").unlink()
        (pd / "cpt_list.csv").unlink()
    it = intake.read_entry(closed["entry"], tmp_path / "entry.csv")
    msg = intake.check_against_task(it, closed["task"], pd)["package_files"]
    assert msg.startswith("FAIL") and closed["pkg"] in msg
    if damage != "empty":
        got = P.rebuild_sha256(pd)[0]
        assert got in msg and "package check" in msg and "not_in_table" in msg     # F2: extra files are refused
    r = tmp_path / "r.jsonl"
    with pytest.raises(reg.RegisterError, match="package_files: FAIL"):
        reg.submit_entry(r, closed["entry"], task=closed["task_path"], package=pd, now=clock())
    assert not r.exists() or r.read_text() == ""                   # refused: no line


def test_a_closed_entry_against_a_package_task_without_a_folder_is_refused(closed, tmp_path):
    it = intake.read_entry(closed["entry"], tmp_path / "entry.csv")
    msg = intake.check_against_task(it, closed["task"])["package_files"]
    assert msg.startswith("FAIL") and "--package DIR" in msg and closed["pkg"] in msg
    r = tmp_path / "r.jsonl"
    with pytest.raises(reg.RegisterError, match="no package folder was given"):
        reg.submit_entry(r, closed["entry"], task=closed["task_path"], now=clock())
    with pytest.raises(reg.RegisterError, match="not a folder"):
        reg.submit_entry(r, closed["entry"], task=closed["task_path"], package=tmp_path / "nope", now=clock())
    assert not r.exists() or r.read_text() == ""


def test_an_open_task_refuses_a_folder_and_a_no_hash_task_says_the_folder_was_not_used(closed):
    open_t = dataclasses.replace(closed["task"], division="open", package_sha256=None)
    assert intake.package_files_check(open_t, None) == "PASS"
    assert intake.package_files_check(open_t, closed["dir"]).startswith("FAIL")
    assert intake.package_files_check(open_t, None, closed["table"]).startswith("FAIL")
    no_pkg = dataclasses.replace(closed["task"], package_sha256=None)
    msg = intake.package_files_check(no_pkg, closed["dir"])
    assert msg.startswith(intake.NOT_CHECKED) and "was not used" in msg               # review 1 F6
    assert "not used" not in intake.package_files_check(no_pkg, None)


# ---- review 1 F4: a closed entry without --task ----------------------------------------------------------------

def test_without_task_package_is_checked_against_the_entrys_own_hash(closed, tmp_path):
    r = tmp_path / "r.jsonl"
    out = {}
    assert reg.submit_entry(r, closed["entry"], package=closed["dir"], checks_out=out, now=clock())["seq"] == 1
    assert out["package_files"] == "PASS" and out["package"].startswith(intake.NOT_CHECKED)
    (closed["dir"] / "cpt_list.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(reg.RegisterError, match="package_files: FAIL.*the entry's own package hash"):
        reg.submit_entry(r, closed["entry"], package=closed["dir"], now=clock())
    out = {}
    assert reg.submit_entry(r, closed["entry"], checks_out=out, now=clock())["seq"] == 2   # no --package: said
    assert out["package_files"].startswith(intake.NOT_CHECKED) and "--package DIR" in out["package_files"]


def test_without_task_an_entry_with_no_package_hash_refuses_a_folder(closed, tmp_path):
    op = tmp_path / "open.json"
    op.write_text(json.dumps(dict(B3, division="open", package_sha256=None)), encoding="utf-8")
    (tmp_path / "open.csv").write_text(CSV, encoding="utf-8")
    r = tmp_path / "r.jsonl"
    with pytest.raises(reg.RegisterError, match="names no package_sha256"):
        reg.submit_entry(r, op, package=closed["dir"], now=clock())
    out = {}
    assert reg.submit_entry(r, op, checks_out=out, now=clock())["seq"] == 1
    assert "package_files" not in out                               # nothing to check, nothing said


# ---- review 1 F1: the task's table, for a table with a hidden input row ----------------------------------------

@pytest.fixture
def hidden_row(tmp_path):
    """A closed task whose table has a hidden input row (amendment 2, section 4.3 point 5: by hand)."""
    pd = make_package(tmp_path)
    (pd / ".params").write_text("k=1\n", encoding="utf-8")
    rows = {fid: P._sha_file(p) for fid, p in P._files(pd).items()}
    assert ".params" in rows
    table = tmp_path / "table.csv"
    table.write_text(P.table_text(rows), encoding="utf-8", newline="")
    pkg = P.table_sha256(table)
    fields = {"event_id": "us7000abcd", "target": "minor or worse ejecta", "label_source": "synthetic",
              "label_rule_sha256": "4" * 64, "site_list_sha256": site_list_sha256(IDS), "track": "record",
              "division": "closed", "label_caveats": "synthetic"}
    tp = tmp_path / "task.json"
    TB.write_task(fields, tp, package_table=table)                  # the kit's own builder accepts the table
    t = TB.load_task(tp)
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(dict(B3, task_sha256=t.sha256(), package_sha256=pkg)), encoding="utf-8")
    (tmp_path / "entry.csv").write_text(CSV, encoding="utf-8")
    return {"dir": pd, "table": table, "pkg": pkg, "task": t, "task_path": tp, "entry": mp}


def test_a_table_with_a_hidden_row_passes_through_the_task_table(hidden_row, tmp_path):
    h = hidden_row
    assert P.check_files(h["table"], h["dir"])["status"] == "PASS"
    assert intake.package_files_check(h["task"], h["dir"]).startswith("FAIL")       # no rebuild can give it
    assert "--package-table" in intake.package_files_check(h["task"], h["dir"])
    assert intake.package_files_check(h["task"], h["dir"], h["table"]) == "PASS"
    (h["dir"] / ".DS_Store").write_bytes(b"\x00")                  # a hidden extra file is not refused
    assert intake.package_files_check(h["task"], h["dir"], h["table"]) == "PASS"
    r = tmp_path / "r.jsonl"
    assert reg.submit_entry(r, h["entry"], task=h["task_path"], package=h["dir"], package_table=h["table"],
                            now=clock())["seq"] == 1


@pytest.mark.parametrize("damage", ["changed", "missing", "hidden_missing", "extra", "other_table"])
def test_the_task_table_path_refuses_each_damage(hidden_row, tmp_path, damage):
    h = hidden_row
    table = h["table"]
    if damage == "changed":
        (h["dir"] / "cpt_list.csv").write_text("x\n", encoding="utf-8")
        want = "cpt_list.csv FAIL"
    elif damage == "missing":
        (h["dir"] / "shakemap" / "grid.xml").unlink()
        want = "shakemap/grid.xml MISSING"
    elif damage == "hidden_missing":
        (h["dir"] / ".params").unlink()
        want = ".params MISSING"
    elif damage == "extra":
        (h["dir"] / "README.txt").write_text("not an input\n", encoding="utf-8")
        want = "files the table does not name: README.txt"
    else:
        table = tmp_path / "other.csv"
        P.write_table(h["dir"], table)                              # a valid table, not the task's
        want = h["pkg"]
    msg = intake.package_files_check(h["task"], h["dir"], table)
    assert msg.startswith("FAIL") and want in msg
    r = tmp_path / "r.jsonl"
    with pytest.raises(reg.RegisterError, match="package_files: FAIL"):
        reg.submit_entry(r, h["entry"], task=h["task_path"], package=h["dir"], package_table=table, now=clock())
    assert not r.exists() or r.read_text() == ""


def test_the_cli_hashes_the_package_once_and_prints_the_checks_it_ran(closed, tmp_path, capsys, monkeypatch):
    calls = []
    real = P.rebuild_sha256
    monkeypatch.setattr(P, "rebuild_sha256", lambda d: calls.append(d) or real(d))     # review 1 F5
    r = str(tmp_path / "r.jsonl")
    assert reg.main(["submit", "--register", r, str(closed["entry"]), "--task", str(closed["task_path"]),
                     "--package", str(closed["dir"])]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(calls) == 1 and out["intake"]["package_files"] == "PASS"
    assert out["intake"]["task_checks"]["package_files"] == "PASS"


def test_the_cli_takes_package_and_prints_package_files(closed, tmp_path, capsys):
    r = str(tmp_path / "r.jsonl")
    base = ["submit", "--register", r, str(closed["entry"]), "--task", str(closed["task_path"])]
    assert reg.main(base) == 2                                      # no --package: refused
    assert "no package folder was given" in capsys.readouterr().err
    assert reg.main(base + ["--package", str(closed["dir"])]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["intake"]["package"] == out["intake"]["package_files"] == "PASS"
    assert reg.check_register(r).n_lines == 1


# ---- (3) a closed kit form entry is refused ---------------------------------------------------------------------

@pytest.mark.parametrize("division", ["closed", None])
def test_a_closed_kit_form_entry_is_refused(closed, tmp_path, division):
    kit = tmp_path / "kit.json"
    kit.write_text(json.dumps(dict(KIT, division=division) if division else KIT), encoding="utf-8")
    (tmp_path / "kit.csv").write_text(CSV, encoding="utf-8")
    it = intake.read_entry(kit, tmp_path / "kit.csv")
    res = intake.check_against_task(it, closed["task"], closed["dir"])
    assert res["package"] == "FAIL: " + intake.CLOSED_KIT_FORM and "rule book form manifest" in res["package"]
    no_hash = dataclasses.replace(closed["task"], package_sha256=None)             # review 1 F3 (a)
    assert intake.package_check(it, no_hash) == "FAIL: " + intake.CLOSED_KIT_FORM
    assert intake.task_failures(intake.check_against_task(it, no_hash, None))
    r = tmp_path / "r.jsonl"
    with pytest.raises(reg.RegisterError, match="closed kit form entry is refused"):
        reg.submit_entry(r, kit, task=closed["task_path"], package=closed["dir"], now=clock())
    if division == "closed":                                        # refused even without --task
        with pytest.raises(reg.RegisterError, match="closed kit form entry is refused"):
            reg.submit_entry(r, kit, now=clock())
    assert not r.exists() or r.read_text() == ""


def test_an_open_or_unstated_kit_form_entry_without_a_task_is_still_registered(tmp_path):
    now = clock()
    for seq, (name, d) in enumerate((("open", dict(KIT, division="open")), ("none", KIT)), start=1):
        mp = tmp_path / f"{name}.json"
        mp.write_text(json.dumps(d), encoding="utf-8")
        (tmp_path / f"{name}.csv").write_text(CSV, encoding="utf-8")
        assert reg.submit_entry(tmp_path / "r.jsonl", mp, entrant="E1", now=now)["seq"] == seq


# ---- (2) the caller fields are listed and printed ---------------------------------------------------------------

def test_every_task_field_is_pinned_from_the_table_or_listed_as_from_caller():
    pinned = set(json.loads(TB.PINS_FILE.read_text(encoding="utf-8"))["pinned_fields"])
    names = {f.name for f in dataclasses.fields(Task)}
    assert names == pinned | {"package_sha256"} | set(TB.CALLER_FIELDS)
    assert not pinned & set(TB.CALLER_FIELDS)
    for k, src in TB.CALLER_FIELDS.items():
        assert src and ("R2" in src or "R7" in src or "R27" in src or "R8" in src or "R9" in src), k


def test_write_task_prints_the_fields_it_took_from_the_caller(closed, tmp_path):
    fields = json.loads(closed["task_path"].read_text(encoding="utf-8"))
    for k in ("clip", "tile_m", "seed", "tile_rule", "package_sha256"):
        fields.pop(k)
    res = TB.write_task(fields, tmp_path / "again.json", package_table=closed["table"])
    assert res["task_sha256"] == closed["task"].sha256()           # the same task: nothing moved
    assert res["from_caller"] == sorted(fields) and set(res["from_caller"]) <= set(TB.CALLER_FIELDS)

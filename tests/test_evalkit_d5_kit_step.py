"""D5 kit step (eval M5): dry run 2 points 7, 9, 10 and 11, sorted (i+k) in pillar docs/evals/D5B-REVIEW-1-2026-11-24.md
(section 3).

  point 7   site list hash recipe (sorted ids, one per line, final newline) said where an outsider meets it:
            the FAIL text, the CLI help, `register hash`, the README task table
  point 9   B3 file names (manifest.json + entry.csv) work without --csv; a refusal names the file it wants
  point 10  a closed entry's package hash is checked against the task's (R15); NOT CHECKED, said beside the
            receipt, when the task names none; no older task hash moves
  point 11  files in hand that differ from the receipt print ENTRY MISMATCH, not REGISTER FAULT

Synthetic only: every site id, hash and earthquake id is made up (the rule book B3 example). No em dashes.
"""
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoreboard.evalkit import intake  # noqa: E402
from scoreboard.evalkit import register as reg  # noqa: E402
from scoreboard.evalkit.spec import (SITE_LIST_RECIPE, SpecError, Task, canonical_json,  # noqa: E402
                                     load_task, site_list_sha256)

IDS = ("ex:0001", "ex:0002", "ex:0003")
B3_CSV = "site_id,p\nex:0001,0.412\nex:0002,0.03\nex:0003,1\n"
B3_REVERSED = "site_id,p\nex:0003,1\nex:0002,0.03\nex:0001,0.412\n"
B3 = {"rule_book_version": "1.0", "entrant_id": "E0007", "task_sha256": "0" * 64, "model": "example-model",
      "model_version": "2.1", "primary": True, "output_kind": "target_probability", "conversion_sha256": None,
      "division": "closed", "package_sha256": "1" * 64, "inputs": ["package", "shakemap"],
      "exposure": {"us7000abcd": "unseen"}, "training_cases_sha256": "2" * 64, "choices_after_outcomes": "none",
      "text_read": "none about this earthquake",
      "statement": "Probability of minor or worse ejecta at the site after this earthquake.", "house": False}
B3_HASH = "913254e4277b730bf805c5f41ab2b5fa32cc7cdba13819d75b026fe531385cc9"
README = ROOT / "scoreboard" / "evalkit" / "README.md"
TASK_HASH_ON_MAIN = "54913af3a92e70c133bff68fc055df80327fd211420aafbd8a0b987722e2028b"   # task() at 3073ef6
T0 = datetime(2026, 11, 24, 12, 0, 0, tzinfo=timezone.utc)


def clock():
    state = {"t": T0}

    def now():
        t = state["t"]
        state["t"] = t + timedelta(seconds=1)
        return t
    return now


def task(division="closed", ids=IDS, **extra) -> Task:
    return Task(event_id="us7000abcd", target="minor or worse ejecta", label_source="synthetic",
                label_rule_sha256="4" * 64, site_list_sha256=site_list_sha256(ids), track="record",
                division=division, tile_m=2000.0, clip=1e-6, seed=1, label_caveats="synthetic", **extra)


def package_dir(d: Path) -> tuple[Path, str]:
    """A synthetic package folder and the package hash the intake rebuilds from it (D5 intake, wave 11)."""
    from scoreboard.evalkit import package as P
    pd = d / "pkg"
    (pd / "shakemap").mkdir(parents=True, exist_ok=True)
    (pd / "sites.csv").write_text("site_id\n" + "".join(i + "\n" for i in IDS), encoding="utf-8")
    (pd / "shakemap" / "grid.xml").write_text("<grid synthetic='1'/>\n", encoding="utf-8")
    return pd, P.rebuild_sha256(pd)[0]


def write_task(path: Path, t: Task) -> Path:
    path.write_text(json.dumps(t.to_dict()), encoding="utf-8")
    return path


def entry(d: Path, name: str = "entry", csv_text: str = B3_CSV, **change) -> Path:
    mp = d / f"{name}.json"
    mp.write_text(json.dumps(dict(B3, **change), indent=1), encoding="utf-8")
    (d / f"{name}.csv").write_text(csv_text, encoding="utf-8")
    return mp


# ---- point 7: the site list hash recipe -------------------------------------------------------------------------

def test_site_list_recipe_is_the_kit_hash_and_ignores_row_order():
    text = "".join(s + "\n" for s in sorted(IDS))                 # the recipe as the README and the FAIL text say it
    assert site_list_sha256(IDS) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert site_list_sha256(reversed(IDS)) == site_list_sha256(IDS)
    assert site_list_sha256(IDS) != hashlib.sha256("\n".join(IDS).encode()).hexdigest()   # no final newline: other
    for words in ("SORTED", "followed by a newline", "ends with a newline", "Row order"):
        assert words in SITE_LIST_RECIPE


def test_site_list_fail_prints_both_hashes_the_row_count_and_the_recipe(tmp_path):
    t = task()
    mp = entry(tmp_path, csv_text=B3_CSV + "ex:0004,0.5\n", task_sha256=t.sha256())
    res = intake.check_against_task(intake.read_entry(mp, mp.with_suffix(".csv")), t)
    got = site_list_sha256(IDS + ("ex:0004",))
    msg = res["site_list"]
    assert msg.startswith("FAIL") and got in msg and t.site_list_sha256 in msg and "4 site ids" in msg
    assert SITE_LIST_RECIPE in msg
    with pytest.raises(reg.RegisterError, match="SORTED site ids"):
        reg.submit_entry(tmp_path / "r.jsonl", mp, task=write_task(tmp_path / "task.json", t), now=clock())


def test_reverse_row_order_passes_the_site_list_but_changes_the_entry_hash(tmp_path):
    pd, pkg = package_dir(tmp_path)
    t = task(package_sha256=pkg)
    tp = write_task(tmp_path / "task.json", t)
    fwd = entry(tmp_path, "fwd", task_sha256=t.sha256(), package_sha256=pkg)
    rev = entry(tmp_path, "rev", csv_text=B3_REVERSED, task_sha256=t.sha256(), package_sha256=pkg)
    a, b = (intake.read_entry(m, m.with_suffix(".csv")) for m in (fwd, rev))
    assert intake.check_against_task(b, t)["site_list"] == "PASS"
    assert a.entry_sha256 != b.entry_sha256                         # D3 hashes rows in file order
    r = tmp_path / "r.jsonl"
    assert [reg.submit_entry(r, m, task=tp, package=pd, now=clock())["seq"] for m in (fwd, rev)] == [1, 2]


def test_hash_cli_prints_the_site_list_hash_and_help_and_readme_say_the_recipe(tmp_path, capsys):
    mp = entry(tmp_path, csv_text=B3_REVERSED)
    assert reg.main(["hash", str(mp)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["site_list_sha256"] == site_list_sha256(IDS) and out["site_list_recipe"] == SITE_LIST_RECIPE
    assert out["predictions_file"].endswith("entry.csv")
    with pytest.raises(SystemExit):
        reg.main(["submit", "--help"])
    helptext = " ".join(capsys.readouterr().out.split())
    assert "SORTED site ids" in helptext and "row order does not matter" in helptext and "package_sha256" in helptext
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert "the sha256 of the SORTED site ids, each followed by `\\n`" in readme
    assert "Row order in the entry CSV does not change this hash" in readme


# ---- point 9: B3 file names and the refusal text ----------------------------------------------------------------

def test_rule_book_file_names_work_without_csv(tmp_path, capsys):
    (tmp_path / "manifest.json").write_text(json.dumps(B3, indent=1), encoding="utf-8")
    (tmp_path / "entry.csv").write_text(B3_CSV, encoding="utf-8")
    assert reg.csv_for(tmp_path / "manifest.json") == tmp_path / "entry.csv"
    assert reg.entry_file_sha256(tmp_path / "manifest.json") == B3_HASH
    assert reg.main(["submit", "--register", str(tmp_path / "r.jsonl"), str(tmp_path / "manifest.json")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["receipt"]["entry_sha256"] == B3_HASH and out["intake"]["predictions_file"].endswith("entry.csv")
    (tmp_path / "manifest.csv").write_text(B3_CSV, encoding="utf-8")   # the kit's own name still comes first
    assert reg.csv_for(tmp_path / "manifest.json") == tmp_path / "manifest.csv"


def test_a_missing_predictions_file_is_named_with_how_to_name_it(tmp_path, capsys):
    (tmp_path / "manifest.json").write_text(json.dumps(B3), encoding="utf-8")
    with pytest.raises(SpecError) as e:
        reg.read_entry_files(tmp_path / "manifest.json")
    msg = str(e.value)
    assert "no matching predictions file manifest.csv" not in msg
    for words in ("manifest.csv or entry.csv", "NAME.csv beside NAME.json", "--csv PATH", "site_id,p"):
        assert words in msg
    (tmp_path / "x.json").write_text(json.dumps(B3), encoding="utf-8")
    with pytest.raises(SpecError, match="looked for x.csv in"):
        reg.read_entry_files(tmp_path / "x.json")
    with pytest.raises(SpecError, match="--csv .*nope.csv: no such predictions file"):
        reg.read_entry_files(tmp_path / "x.json", tmp_path / "nope.csv")
    assert reg.main(["hash", str(tmp_path / "manifest.json")]) == 2
    assert "Name the predictions file manifest.csv or entry.csv" in capsys.readouterr().err


# ---- point 10: the package hash (R15) ---------------------------------------------------------------------------

def test_package_field_moves_no_older_task_hash():
    t = task()
    d = t.to_dict()
    assert "package_sha256" not in d and len(d) == 20
    assert t.sha256() == hashlib.sha256(canonical_json(d).encode("utf-8")).hexdigest()
    assert t.sha256() == TASK_HASH_ON_MAIN                         # the same task hashed by kit main 3073ef6
    with_pkg = task(package_sha256="1" * 64)
    assert with_pkg.to_dict()["package_sha256"] == "1" * 64 and with_pkg.sha256() != t.sha256()
    assert Task.from_dict(with_pkg.to_dict()) == with_pkg and Task.from_dict(d) == t


def test_package_field_is_checked_in_the_task(tmp_path):
    with pytest.raises(SpecError, match="only a closed division task"):
        task(division="open", package_sha256="1" * 64)
    with pytest.raises(SpecError, match="task.package_sha256"):
        task(package_sha256="ONES")
    tp = write_task(tmp_path / "t.json", task(package_sha256="3" * 64))
    assert load_task(tp).package_sha256 == "3" * 64


def test_a_closed_entry_with_another_package_hash_is_refused(tmp_path):
    pd, pkg = package_dir(tmp_path)
    t = task(package_sha256=pkg)
    tp = write_task(tmp_path / "task.json", t)
    mp = entry(tmp_path, task_sha256=t.sha256())                    # package_sha256 all 1s, as the B3 example
    res = intake.check_against_task(intake.read_entry(mp, mp.with_suffix(".csv")), t)
    assert res["package"].startswith("FAIL") and "1" * 64 in res["package"] and pkg in res["package"]
    r = tmp_path / "r.jsonl"
    with pytest.raises(reg.RegisterError, match="package"):
        reg.submit_entry(r, mp, task=tp, package=pd, now=clock())
    assert not r.exists() or r.read_text() == ""                   # refused: no line
    ok = entry(tmp_path, "ok", task_sha256=t.sha256(), package_sha256=pkg)
    assert intake.check_against_task(intake.read_entry(ok, ok.with_suffix(".csv")), t)["package"] == "PASS"
    assert reg.submit_entry(r, ok, task=tp, package=pd, now=clock())["seq"] == 1


def test_an_unchecked_package_hash_is_said_beside_the_receipt(tmp_path, capsys):
    t = task()                                                     # a closed task that names no package hash
    tp = write_task(tmp_path / "task.json", t)
    mp = entry(tmp_path, task_sha256=t.sha256())
    r = str(tmp_path / "r.jsonl")
    assert reg.main(["submit", "--register", r, str(mp), "--task", str(tp)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["intake"]["package"].startswith("NOT CHECKED") and "1" * 64 in out["intake"]["package"]
    assert out["intake"]["task_checks"]["site_list"] == "PASS"
    assert reg.main(["submit", "--register", r, str(mp)]) == 0      # no --task: not checked, and said so
    assert json.loads(capsys.readouterr().out)["intake"]["package"].startswith("NOT CHECKED: no --task")
    kit = tmp_path / "kit.json"
    kit.write_text(json.dumps({"model": "m", "output_kind": "score", "statement": "s", "house": False,
                               "exposure": {"us7000abcd": "unseen"}, "division": "closed"}), encoding="utf-8")
    (tmp_path / "kit.csv").write_text(B3_CSV, encoding="utf-8")
    res = intake.check_against_task(intake.read_entry(kit, tmp_path / "kit.csv"), task(package_sha256="3" * 64))
    assert res["package"] == "FAIL: " + intake.CLOSED_KIT_FORM                # D5 intake, wave 11: refused
    assert any(f.startswith("package: FAIL") for f in intake.task_failures(res))
    open_t = task(division="open")
    om = entry(tmp_path, "open", task_sha256=open_t.sha256(), division="open", package_sha256=None)
    assert intake.check_against_task(intake.read_entry(om, om.with_suffix(".csv")), open_t)["package"] == "PASS"


# ---- point 11: ENTRY MISMATCH -----------------------------------------------------------------------------------

def test_changed_files_in_hand_print_entry_mismatch_not_register_fault(tmp_path, capsys):
    r, rp = tmp_path / "r.jsonl", tmp_path / "receipt.json"
    mp = entry(tmp_path)
    assert reg.main(["submit", "--register", str(r), str(mp), "--receipt-out", str(rp)]) == 0
    capsys.readouterr()
    (tmp_path / "entry.csv").write_text(B3_CSV.replace("0.412", "0.413"), encoding="utf-8")
    assert reg.main(["verify", "--register", str(r), "--receipt", str(rp), "--entry", str(mp)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == reg.ENTRY_MISMATCH == "ENTRY MISMATCH"
    assert len(out["reasons"]) == 1 and B3_HASH in out["reasons"][0]
    in_hand = intake.read_entry(mp, tmp_path / "entry.csv").entry_sha256
    assert in_hand != B3_HASH and in_hand in out["reasons"][0]       # the reason prints both hashes
    with pytest.raises(SystemExit):
        reg.main(["verify", "--help"])                              # the --entry help names the status
    assert "(ENTRY MISMATCH if not)" in " ".join(capsys.readouterr().out.split())
    with pytest.raises(SystemExit):
        reg.main(["--help"])                                        # the verify line in the command list
    assert "ENTRY MISMATCH when only the files in hand differ" in " ".join(capsys.readouterr().out.split())
    assert reg.main(["check", "--register", str(r)]) == 0           # the register itself is intact
    assert json.loads(capsys.readouterr().out)["status"] == reg.VERIFIED
    r.write_text(r.read_text().replace('"E0007"', '"E0008"'))       # now the register changes too
    assert reg.main(["verify", "--register", str(r), "--receipt", str(rp), "--entry", str(mp)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == reg.REGISTER_FAULT and any("in hand" in x for x in out["reasons"])
    assert reg.verify_receipt(r, json.loads(rp.read_text()))["status"] == reg.REGISTER_FAULT

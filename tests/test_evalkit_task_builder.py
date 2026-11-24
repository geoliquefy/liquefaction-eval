"""Public task builder, first slice (D5 kit review 1 N4, eval M5): every closed task it writes names package_sha256.

Pillar docs/evals/D5-KIT-REVIEW-1-2026-11-24.md, N4. Synthetic only: every id and hash is made up. No em dashes.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoreboard.evalkit import intake  # noqa: E402
from scoreboard.evalkit import task_builder as TB  # noqa: E402
from scoreboard.evalkit.spec import SpecError, Task, load_task, site_list_sha256  # noqa: E402

IDS = ("ex:0001", "ex:0002", "ex:0003")
PKG = "3" * 64
FIELDS = {"event_id": "us7000abcd", "target": "minor or worse ejecta", "label_source": "synthetic",
          "label_rule_sha256": "4" * 64, "site_list_sha256": site_list_sha256(IDS), "track": "record",
          "division": "closed", "tile_m": 2000, "clip": 1e-6, "seed": 1, "label_caveats": "synthetic"}


def closed(**change):
    return dict(FIELDS, package_sha256=PKG, **change)


def test_a_closed_task_without_a_package_hash_is_refused_and_nothing_is_written(tmp_path, capsys):
    for fields in (FIELDS, dict(FIELDS, package_sha256=None), dict(FIELDS, track="forward"),
                   dict(FIELDS, track="forward", package_sha256=None)):      # every closed task, both tracks (N1)
        with pytest.raises(SpecError, match="package_sha256") as e:
            TB.build_task(fields)
        assert "R15" in str(e.value) and "closed" in str(e.value)
        with pytest.raises(SpecError, match="package_sha256"):
            TB.write_task(fields, tmp_path / "t.json")
        assert not (tmp_path / "t.json").exists()
    fp = tmp_path / "fields.json"
    fp.write_text(json.dumps(FIELDS), encoding="utf-8")
    assert TB.main([str(fp), "--out", str(tmp_path / "t.json")]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "REFUSED" and "package_sha256" in out["reason"]
    assert not (tmp_path / "t.json").exists()
    Task.from_dict(dict(FIELDS))                     # spec.Task still reads it: older task files keep their hash


def test_every_closed_task_written_names_its_package_hash(tmp_path, capsys):
    res = TB.write_task(closed(), tmp_path / "t.json")
    on_disk = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert on_disk["package_sha256"] == PKG and on_disk["division"] == "closed"
    t = load_task(tmp_path / "t.json")
    assert res == {"task_sha256": t.sha256(), "package_sha256": PKG, "division": "closed",
                   "path": str(tmp_path / "t.json")}
    assert t.sha256() == Task.from_dict(closed()).sha256()          # the builder adds nothing to the task hash
    fp = tmp_path / "fields.json"
    fp.write_text(json.dumps(closed()), encoding="utf-8")
    assert TB.main([str(fp), "--out", str(tmp_path / "cli.json")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "WRITTEN" and out["package_sha256"] == PKG and out["task_sha256"] == t.sha256()
    assert json.loads((tmp_path / "cli.json").read_text(encoding="utf-8"))["package_sha256"] == PKG


def test_a_builder_task_checks_the_package_at_intake_not_not_checked(tmp_path):
    t = TB.build_task(closed())
    b3 = {"rule_book_version": "1.0", "entrant_id": "E0007", "task_sha256": t.sha256(), "model": "example-model",
          "model_version": "2.1", "primary": True, "output_kind": "target_probability", "conversion_sha256": None,
          "division": "closed", "package_sha256": PKG, "inputs": ["package", "shakemap"],
          "exposure": {"us7000abcd": "unseen"}, "training_cases_sha256": "2" * 64, "choices_after_outcomes": "none",
          "text_read": "none", "statement": "Probability of minor or worse ejecta at the site.", "house": False}
    (tmp_path / "e.csv").write_text("site_id,p\nex:0001,0.4\nex:0002,0.03\nex:0003,1\n", encoding="utf-8")
    for pkg, word in ((PKG, "PASS"), ("1" * 64, "FAIL")):
        (tmp_path / "e.json").write_text(json.dumps(dict(b3, package_sha256=pkg)), encoding="utf-8")
        got = intake.check_against_task(intake.read_entry(tmp_path / "e.json", tmp_path / "e.csv"), t)["package"]
        assert got.split(":")[0] == word


def test_an_open_task_is_written_without_a_package_hash(tmp_path):
    res = TB.write_task(dict(FIELDS, division="open"), tmp_path / "o.json")
    assert res["package_sha256"] is None and res["division"] == "open"
    assert "package_sha256" not in json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    with pytest.raises(SpecError, match="only a closed division task names a package hash"):
        TB.build_task(dict(FIELDS, division="open", package_sha256=PKG))
    with pytest.raises(SpecError, match="JSON object"):
        TB.build_task([FIELDS])


def test_a_write_that_drops_the_package_hash_is_refused_and_deleted(tmp_path, monkeypatch):
    def lossy(task, path):                     # a writer that loses the field, as an older save_task would
        d = task.to_dict()
        d.pop("package_sha256", None)
        Path(path).write_text(json.dumps(d), encoding="utf-8")
    monkeypatch.setattr(TB, "save_task", lossy)
    with pytest.raises(SpecError, match="read back"):
        TB.write_task(closed(), tmp_path / "t.json")
    assert not (tmp_path / "t.json").exists()
    assert list(tmp_path.iterdir()) == []                  # the temporary file is gone too


def test_an_existing_out_path_is_refused_and_kept(tmp_path, capsys, monkeypatch):
    old = tmp_path / "t.json"                  # task builder review 1 N2: a task file is a public item
    old.write_text("an older task file\n", encoding="utf-8")
    with pytest.raises(SpecError, match="exists"):
        TB.write_task(closed(), old)
    fp = tmp_path / "fields.json"
    fp.write_text(json.dumps(closed()), encoding="utf-8")
    assert TB.main([str(fp), "--out", str(old)]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "REFUSED" and "exists" in out["reason"]
    real = TB.save_task

    def racer(task, path):                     # --out appears after the check, before the link
        real(task, path)
        old.write_text("an older task file\n", encoding="utf-8")
    old.unlink()
    monkeypatch.setattr(TB, "save_task", racer)
    with pytest.raises(SpecError, match="exists"):
        TB.write_task(closed(), old)
    assert old.read_text(encoding="utf-8") == "an older task file\n"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["fields.json", "t.json"]

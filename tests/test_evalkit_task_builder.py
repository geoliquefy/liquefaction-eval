"""Public task builder (D5 kit review 1 N4; D5 follow-ups, wave 10; eval M5).

Every closed task it writes names package_sha256, taken from a package table (R15); clip, tile_m, seed and tile_rule
come from the committed pins (rule book v1 amendment 1); a caller value that differs is refused (R2).
Pillar docs/evals/D5-KIT-REVIEW-1-2026-11-24.md, N4, and docs/evals/D5-KIT-STEP-2026-11-24.md. Synthetic only:
every id and hash is made up. No em dashes.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoreboard.evalkit import intake  # noqa: E402
from scoreboard.evalkit import package as P  # noqa: E402
from scoreboard.evalkit import task_builder as TB  # noqa: E402
from scoreboard.evalkit.spec import SpecError, Task, load_task, site_list_sha256  # noqa: E402

IDS = ("ex:0001", "ex:0002", "ex:0003")
FIELDS = {"event_id": "us7000abcd", "target": "minor or worse ejecta", "label_source": "synthetic",
          "label_rule_sha256": "4" * 64, "site_list_sha256": site_list_sha256(IDS), "track": "record",
          "division": "closed", "label_caveats": "synthetic"}
PINS = {"clip": 1e-6, "tile_m": 4000.0, "seed": 20261106, "tile_rule": "fixed"}


@pytest.fixture
def table(tmp_path_factory):
    """A package folder and its table, outside tmp_path (the tests below count the files in tmp_path)."""
    d = tmp_path_factory.mktemp("pkg")
    (d / "files").mkdir()
    (d / "files" / "cpt_list.csv").write_text("site_id,lat,lon\nex:0001,-43.5,172.6\n", encoding="utf-8")
    (d / "files" / "shakemap").mkdir()
    (d / "files" / "shakemap" / "grid.xml").write_text("<grid/>\n", encoding="utf-8")
    res = P.write_table(d / "files", d / "table.csv")
    return d / "table.csv", res["package_sha256"]


def test_the_pins_file_holds_amendment_1_section_4():
    doc = json.loads(TB.PINS_FILE.read_text(encoding="utf-8"))
    assert sorted(doc["card_versions"]) == ["later", "v2", "v3", "v3.1"]
    for v in doc["card_versions"]:
        assert TB.load_pins(v) == PINS
    assert "RULEBOOK-V1-AMENDMENT-1-2026-11-24.md" in doc["source"]
    with pytest.raises(SpecError, match="no pinned values"):
        TB.load_pins("v9")


@pytest.mark.parametrize("change", ["drop", "extra"])
def test_a_pins_file_that_does_not_pin_exactly_the_four_fields_is_refused(tmp_path, monkeypatch, table, change):
    """D5 follow-ups review 1 (Q5): each card version pins exactly the pinned_fields, no fewer and no more."""
    doc = json.loads(TB.PINS_FILE.read_text(encoding="utf-8"))
    v = doc["card_versions"]["later"]
    if change == "drop":
        v.pop("seed")
    else:
        v["n_boot"] = 1000
    bad = tmp_path / "task_pins.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(TB, "PINS_FILE", bad)
    with pytest.raises(SpecError, match="does not pin exactly"):
        TB.load_pins("later")
    with pytest.raises(SpecError, match="does not pin exactly"):
        TB.build_task(dict(FIELDS, seed=1), package_table=table[0])


def test_pinned_fields_come_from_the_pins_and_a_caller_value_that_differs_is_refused(tmp_path, table, capsys):
    tp, pkg = table
    t = TB.build_task(FIELDS, package_table=tp)                    # caller leaves the pins out
    assert (t.clip, t.tile_m, t.seed, t.tile_rule) == (1e-6, 4000.0, 20261106, "fixed")
    same = dict(FIELDS, clip=1e-06, tile_m=4000, seed=20261106, tile_rule="fixed")   # equal values pass
    assert TB.build_task(same, package_table=tp).sha256() == t.sha256()
    for k, bad in (("clip", 1e-3), ("tile_m", 2000), ("seed", 1), ("seed", 20261106.0), ("seed", True),
                   ("tile_rule", "variogram"), ("clip", "1e-6")):
        with pytest.raises(SpecError, match=f"{k} is pinned") as e:
            TB.build_task(dict(FIELDS, **{k: bad}), package_table=tp)
        assert "amendment 1" in str(e.value)
        with pytest.raises(SpecError, match="pinned"):
            TB.write_task(dict(FIELDS, **{k: bad}), tmp_path / "t.json", package_table=tp)
        assert not (tmp_path / "t.json").exists()
    fp = tmp_path / "fields.json"
    fp.write_text(json.dumps(dict(FIELDS, seed=7)), encoding="utf-8")
    assert TB.main([str(fp), "--out", str(tmp_path / "t.json"), "--package-table", str(tp)]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "REFUSED" and "seed is pinned" in out["reason"]
    assert not (tmp_path / "t.json").exists()


def test_a_closed_task_without_a_package_table_is_refused_and_nothing_is_written(tmp_path, capsys):
    for fields in (FIELDS, dict(FIELDS, package_sha256=None), dict(FIELDS, package_sha256="3" * 64),
                   dict(FIELDS, track="forward"), dict(FIELDS, track="forward", package_sha256="3" * 64)):
        with pytest.raises(SpecError, match="package_sha256") as e:      # every closed task, both tracks
            TB.build_task(fields)
        assert "R15" in str(e.value) and "package table" in str(e.value)
        with pytest.raises(SpecError, match="package_sha256"):
            TB.write_task(fields, tmp_path / "t.json")
        assert not (tmp_path / "t.json").exists()
    fp = tmp_path / "fields.json"
    fp.write_text(json.dumps(FIELDS), encoding="utf-8")
    assert TB.main([str(fp), "--out", str(tmp_path / "t.json")]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "REFUSED" and "package_sha256" in out["reason"]
    assert not (tmp_path / "t.json").exists()
    Task.from_dict(dict(FIELDS, **PINS))             # spec.Task still reads it: older task files keep their hash


def test_the_package_hash_comes_from_the_table_and_a_caller_hash_that_differs_is_refused(tmp_path, table):
    tp, pkg = table
    assert TB.build_task(FIELDS, package_table=tp).package_sha256 == pkg == P.table_sha256(tp)
    assert TB.build_task(dict(FIELDS, package_sha256=pkg), package_table=tp).package_sha256 == pkg
    with pytest.raises(SpecError, match="hashes to"):
        TB.build_task(dict(FIELDS, package_sha256="1" * 64), package_table=tp)
    with pytest.raises(SpecError, match="only a closed division task has a package table"):
        TB.build_task(dict(FIELDS, division="open"), package_table=tp)
    bad = tmp_path / "bad.csv"                       # a table not in the kit form is refused
    bad.write_text(tp.read_text(encoding="utf-8").replace("\n", "\r\n"), encoding="utf-8", newline="")
    with pytest.raises(SpecError, match="kit form"):
        TB.build_task(FIELDS, package_table=bad)


def test_every_closed_task_written_names_its_package_hash(tmp_path, table, capsys):
    tp, pkg = table
    res = TB.write_task(FIELDS, tmp_path / "t.json", package_table=tp)
    on_disk = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert on_disk["package_sha256"] == pkg and on_disk["division"] == "closed"
    assert {k: on_disk[k] for k in PINS} == PINS
    t = load_task(tmp_path / "t.json")
    assert res == {"task_sha256": t.sha256(), "package_sha256": pkg, "division": "closed",
                   "card_version": "later", "path": str(tmp_path / "t.json")}
    assert t.sha256() == Task.from_dict(dict(FIELDS, package_sha256=pkg, **PINS)).sha256()   # nothing added
    fp = tmp_path / "fields.json"
    fp.write_text(json.dumps(FIELDS), encoding="utf-8")
    assert TB.main([str(fp), "--out", str(tmp_path / "cli.json"), "--package-table", str(tp),
                    "--card-version", "v3.1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "WRITTEN" and out["package_sha256"] == pkg and out["task_sha256"] == t.sha256()
    assert out["card_version"] == "v3.1"
    assert json.loads((tmp_path / "cli.json").read_text(encoding="utf-8"))["package_sha256"] == pkg


def test_a_builder_task_checks_the_package_at_intake_not_not_checked(tmp_path, table):
    tp, pkg = table
    t = TB.build_task(FIELDS, package_table=tp)
    b3 = {"rule_book_version": "1.0", "entrant_id": "E0007", "task_sha256": t.sha256(), "model": "example-model",
          "model_version": "2.1", "primary": True, "output_kind": "target_probability", "conversion_sha256": None,
          "division": "closed", "package_sha256": pkg, "inputs": ["package", "shakemap"],
          "exposure": {"us7000abcd": "unseen"}, "training_cases_sha256": "2" * 64, "choices_after_outcomes": "none",
          "text_read": "none", "statement": "Probability of minor or worse ejecta at the site.", "house": False}
    (tmp_path / "e.csv").write_text("site_id,p\nex:0001,0.4\nex:0002,0.03\nex:0003,1\n", encoding="utf-8")
    for p, word in ((pkg, "PASS"), ("1" * 64, "FAIL")):
        (tmp_path / "e.json").write_text(json.dumps(dict(b3, package_sha256=p)), encoding="utf-8")
        got = intake.check_against_task(intake.read_entry(tmp_path / "e.json", tmp_path / "e.csv"), t)["package"]
        assert got.split(":")[0] == word


def test_an_open_task_is_written_without_a_package_hash(tmp_path):
    res = TB.write_task(dict(FIELDS, division="open"), tmp_path / "o.json")
    assert res["package_sha256"] is None and res["division"] == "open"
    assert "package_sha256" not in json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    with pytest.raises(SpecError, match="only a closed division task names a package hash"):
        TB.build_task(dict(FIELDS, division="open", package_sha256="3" * 64))
    with pytest.raises(SpecError, match="JSON object"):
        TB.build_task([FIELDS])


def test_a_write_that_drops_the_package_hash_is_refused_and_deleted(tmp_path, monkeypatch, table):
    tp, _ = table

    def lossy(task, path):                     # a writer that loses the field, as an older save_task would
        d = task.to_dict()
        d.pop("package_sha256", None)
        Path(path).write_text(json.dumps(d), encoding="utf-8")
    monkeypatch.setattr(TB, "save_task", lossy)
    with pytest.raises(SpecError, match="read back"):
        TB.write_task(FIELDS, tmp_path / "t.json", package_table=tp)
    assert not (tmp_path / "t.json").exists()
    assert list(tmp_path.iterdir()) == []                  # the temporary file is gone too


def test_an_existing_out_path_is_refused_and_kept(tmp_path, capsys, monkeypatch, table):
    tp, _ = table
    old = tmp_path / "t.json"                  # task builder review 1 N2: a task file is a public item
    old.write_text("an older task file\n", encoding="utf-8")
    with pytest.raises(SpecError, match="exists"):
        TB.write_task(FIELDS, old, package_table=tp)
    fp = tmp_path / "fields.json"
    fp.write_text(json.dumps(FIELDS), encoding="utf-8")
    assert TB.main([str(fp), "--out", str(old), "--package-table", str(tp)]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "REFUSED" and "exists" in out["reason"]
    real = TB.save_task

    def racer(task, path):                     # --out appears after the check, before the link
        real(task, path)
        old.write_text("an older task file\n", encoding="utf-8")
    old.unlink()
    monkeypatch.setattr(TB, "save_task", racer)
    with pytest.raises(SpecError, match="exists"):
        TB.write_task(FIELDS, old, package_table=tp)
    assert old.read_text(encoding="utf-8") == "an older task file\n"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["fields.json", "t.json"]

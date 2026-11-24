"""evalkit intake in both manifest forms (D5, eval M5): the stuck points of the outsider dry run. Synthetic only.

Every site id, hash and earthquake id here is made up (the rule book B3 example and its three example sites).
"""
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoreboard.evalkit import intake  # noqa: E402
from scoreboard.evalkit import register as reg  # noqa: E402
from scoreboard.evalkit.spec import (SpecError, Task, entry_sha256, load_entry,  # noqa: E402
                                     site_list_sha256)

# Rule book v1.0 B3, printed example files and the hash the rule book gives for them.
B3_CSV = "site_id,p\nex:0001,0.412\nex:0002,0.03\nex:0003,1\n"
B3_MANIFEST = """{
 "rule_book_version": "1.0",
 "entrant_id": "E0007",
 "task_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
 "model": "example-model",
 "model_version": "2.1",
 "primary": true,
 "output_kind": "target_probability",
 "conversion_sha256": null,
 "division": "closed",
 "package_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
 "inputs": ["package", "shakemap"],
 "exposure": {"us7000abcd": "unseen"},
 "training_cases_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
 "choices_after_outcomes": "none",
 "text_read": "none about this earthquake",
 "statement": "Probability of minor or worse ejecta at the site after this earthquake.",
 "house": false
}
"""
B3_HASH = "913254e4277b730bf805c5f41ab2b5fa32cc7cdba13819d75b026fe531385cc9"
T0 = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)


def clock():
    state = {"t": T0}

    def now():
        t = state["t"]
        state["t"] = t + timedelta(seconds=1)
        return t
    return now


def b3_recipe(manifest_path: Path, csv_path: Path) -> str:
    """The rule book B3 hash recipe, copied as printed (standard library only)."""
    def cell(s):
        x = float(s)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        t = format(x, ".9f")
        return format(0.0, ".9f") if float(t) == 0.0 else t
    rows = list(csv.reader(open(csv_path, newline="", encoding="utf-8")))
    text = "\n".join([",".join(rows[0])] + [r[0] + "," + cell(r[1]) for r in rows[1:]]) + "\n"
    m = json.load(open(manifest_path, encoding="utf-8"))
    mj = json.dumps(m, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256((mj + "\n" + text).encode("utf-8")).hexdigest()


def write(d: Path, name: str, manifest: dict | str, csv_text: str = B3_CSV) -> Path:
    mp = d / f"{name}.json"
    mp.write_text(manifest if isinstance(manifest, str) else json.dumps(manifest, indent=2), encoding="utf-8")
    (d / f"{name}.csv").write_text(csv_text, encoding="utf-8")
    return mp


def b3(**change) -> dict:
    d = json.loads(B3_MANIFEST)
    d.update(change)
    return d


def test_the_rule_book_example_is_accepted_and_hashes_as_printed(tmp_path):
    mp = write(tmp_path, "entry", B3_MANIFEST)
    assert b3_recipe(mp, tmp_path / "entry.csv") == B3_HASH          # our copy of the recipe matches the book
    it = intake.read_entry(mp, tmp_path / "entry.csv")
    assert it.form == intake.RULEBOOK_FORM and it.entry_sha256 == B3_HASH
    assert it.entrant == "E0007" and it.primary is True
    assert it.entry.manifest.division == "closed" and it.entry.manifest.conversion_id is None
    rec = reg.submit_entry(tmp_path / "r.jsonl", mp, now=clock())
    assert rec["entry_sha256"] == B3_HASH and rec["entrant"] == "E0007"
    assert reg.verify_receipt(tmp_path / "r.jsonl", rec, entry_sha=reg.entry_file_sha256(mp))["status"] == reg.VERIFIED


def test_extra_fields_are_allowed_and_hashed(tmp_path):
    a = write(tmp_path, "a", b3())
    b = write(tmp_path, "b", b3(notes="an extra field"))
    ha, hb = (intake.read_entry(p, p.with_suffix(".csv")).entry_sha256 for p in (a, b))
    assert ha == B3_HASH and hb != ha
    assert hb == b3_recipe(b, b.with_suffix(".csv"))


@pytest.mark.parametrize("change, words", [
    ({"entrant_id": None}, "may be null"),
    ({"primary": "yes"}, "true or false"),
    ({"task_sha256": "abc"}, "sha256"),
    ({"division": "open"}, "null on the open division"),
    ({"package_sha256": None}, "closed division names"),
    ({"inputs": "package"}, "list of non-empty strings"),
    ({"output_kind": "target_probability", "conversion_sha256": "3" * 64}, r"manifest\.conversion_sha256: must be null"),
])
def test_rule_book_form_breaches_are_refused(tmp_path, change, words):
    mp = write(tmp_path, "e", b3(**change))
    with pytest.raises(SpecError, match=words):
        intake.read_entry(mp, mp.with_suffix(".csv"))


def test_a_missing_rule_book_field_is_named(tmp_path):
    d = b3()
    del d["training_cases_sha256"]
    mp = write(tmp_path, "e", d)
    with pytest.raises(SpecError, match="training_cases_sha256"):
        intake.read_entry(mp, mp.with_suffix(".csv"))


def test_rule_book_form_needs_unit_values_even_for_a_score(tmp_path):
    mp = write(tmp_path, "e", b3(output_kind="score"), "site_id,p\nex:0001,1.5\nex:0002,0.03\nex:0003,1\n")
    with pytest.raises(SpecError, match=r"\[0, 1\]"):
        intake.read_entry(mp, mp.with_suffix(".csv"))


def test_entrant_comes_from_the_manifest_in_rule_book_form(tmp_path):
    mp = write(tmp_path, "e", B3_MANIFEST)
    r = tmp_path / "r.jsonl"
    with pytest.raises(reg.RegisterError, match="differs from the manifest"):
        reg.submit_entry(r, mp, entrant="someone-else", now=clock())
    assert not r.exists() or r.read_text() == ""                     # nothing registered
    assert reg.submit_entry(r, mp, entrant="E0007", now=clock())["entrant"] == "E0007"


def test_kit_form_hash_is_unchanged_and_its_hashed_text_is_printed(tmp_path):
    kit = {"model": "example-model", "output_kind": "score", "division": "open",
           "exposure": {"us7000abcd": "unseen"}, "statement": "made up", "house": False}
    mp = write(tmp_path, "k", kit)
    it = intake.read_entry(mp, mp.with_suffix(".csv"))
    assert it.form == intake.KIT_FORM and it.entrant is None
    assert it.entry_sha256 == entry_sha256(load_entry(mp, mp.with_suffix(".csv")))   # as before D5
    assert json.loads(it.manifest_text)["conversion_id"] is None                      # the field the kit fills
    filled = write(tmp_path, "f", dict(kit, conversion_id=None))
    assert b3_recipe(filled, filled.with_suffix(".csv")) == it.entry_sha256           # recipe agrees once filled
    no_div = write(tmp_path, "n", {k: v for k, v in kit.items() if k != "division"})
    assert json.loads(intake.read_entry(no_div, no_div.with_suffix(".csv")).manifest_text)["division"] is None


def test_a_kit_form_with_some_rule_book_fields_names_both_forms(tmp_path):
    kit = {"model": "m", "output_kind": "score", "exposure": {"us7000abcd": "unseen"}, "statement": "s",
           "house": False, "entrant_id": "E0007"}
    mp = write(tmp_path, "k", kit)
    with pytest.raises(SpecError, match="rule book form"):
        intake.read_entry(mp, mp.with_suffix(".csv"))


def make_task(path: Path, event="us7000abcd", division="closed", ids=("ex:0001", "ex:0002", "ex:0003")) -> Task:
    t = Task(event_id=event, target="minor or worse ejecta", label_source="synthetic", label_rule_sha256="4" * 64,
             site_list_sha256=site_list_sha256(ids), track="record", division=division, tile_m=2000.0, clip=1e-6,
             seed=1, label_caveats="synthetic")
    path.write_text(json.dumps(t.to_dict()), encoding="utf-8")
    return t


def test_task_checks_site_list_exposure_division_and_task_hash(tmp_path):
    tp = tmp_path / "task.json"
    t = make_task(tp)
    good = write(tmp_path, "g", b3(task_sha256=t.sha256()))
    res = intake.check_against_task(intake.read_entry(good, good.with_suffix(".csv")), t)
    assert {k: v for k, v in res.items() if k != "package"} == dict.fromkeys(res.keys() - {"package"}, "PASS")
    assert res["package"].startswith(intake.NOT_CHECKED)              # this task names no package hash
    r = tmp_path / "r.jsonl"
    assert reg.submit_entry(r, good, task=tp, now=clock())["seq"] == 1
    cases = {
        "site_list": write(tmp_path, "s", b3(task_sha256=t.sha256()), B3_CSV + "ex:0004,0.5\n"),
        "exposure_key": write(tmp_path, "x", b3(task_sha256=t.sha256(), exposure={"other": "unseen"})),
        "task_sha256": write(tmp_path, "h", b3()),
    }
    for what, mp in cases.items():
        res = intake.check_against_task(intake.read_entry(mp, mp.with_suffix(".csv")), t)
        assert res[what].startswith("FAIL"), (what, res)
        with pytest.raises(reg.RegisterError, match=what):
            reg.submit_entry(r, mp, task=tp, now=clock())
    open_task = make_task(tmp_path / "open.json", division="open")
    res = intake.check_against_task(intake.read_entry(good, good.with_suffix(".csv")), open_task)
    assert res["division"].startswith("FAIL")
    assert reg.check_register(r).n_lines == 1                         # refused entries left no line


def test_cli_hash_and_submit_print_the_form_and_the_hashed_text(tmp_path, capsys):
    mp = write(tmp_path, "entry", B3_MANIFEST)
    assert reg.main(["hash", str(mp)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["entry_sha256"] == B3_HASH and out["form"] == intake.RULEBOOK_FORM
    canon = "site_id,p\nex:0001,0.412000000\nex:0002,0.030000000\nex:0003,1.000000000\n"   # D3 predictions text
    assert hashlib.sha256((out["hashed_manifest"] + "\n" + canon).encode()).hexdigest() == B3_HASH
    assert reg.main(["submit", "--register", str(tmp_path / "r.jsonl"), str(mp)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["receipt"]["entrant"] == "E0007"
    assert out["intake"]["form"] == intake.RULEBOOK_FORM and out["intake"]["register_public"] is False
    assert out["intake"]["receipt_signed"] is False and out["intake"]["task"].startswith("not checked")


# ---- D5 review 1 fixes ------------------------------------------------------------------------------------------

def test_task_hash_is_defined_printed_and_named_on_a_refusal(tmp_path, capsys):
    """A task file with only the 11 required fields: the task hash is the kit's, not the file bytes or the file's
    canonical JSON; `register hash --task` prints it; a refusal prints the hash given and the hash needed."""
    t = make_task(tmp_path / "full.json")
    short = {k: v for k, v in t.to_dict().items() if k in Task.__dataclass_fields__ and
             list(Task.__dataclass_fields__).index(k) < 11}
    short["tile_m"] = 2000                                            # an int in the file, a float in the kit
    tp = tmp_path / "task.json"
    tp.write_text(json.dumps(short, indent=1), encoding="utf-8")
    sha, text = intake.task_hash(tp)
    assert sha == t.sha256() == hashlib.sha256(text.encode()).hexdigest()
    assert len(json.loads(text)) == 20 and '"tile_m":2000.0' in text
    file_canon = hashlib.sha256(json.dumps(short, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert len({sha, file_canon, hashlib.sha256(tp.read_bytes()).hexdigest()}) == 3
    assert reg.main(["hash", "--task", str(tp)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"task_sha256": sha, "hashed_task": text}
    mp = write(tmp_path, "e", b3(task_sha256=file_canon))
    res = intake.check_against_task(intake.read_entry(mp, mp.with_suffix(".csv")), t)
    assert res["task_sha256"].startswith("FAIL") and file_canon in res["task_sha256"] and sha in res["task_sha256"]
    ok = write(tmp_path, "ok", b3(task_sha256=sha))
    assert reg.submit_entry(tmp_path / "r.jsonl", ok, task=tp, now=clock())["seq"] == 1
    assert reg.main(["hash"]) == 2 and "REFUSED" in capsys.readouterr().err


def test_submit_says_so_when_the_files_change_after_the_append(tmp_path, capsys, monkeypatch):
    mp = write(tmp_path, "entry", B3_MANIFEST)
    real = reg._finish_submit

    def change_then_finish(a, rec):
        (tmp_path / "entry.csv").write_text(B3_CSV.replace("0.412", "0.5"), encoding="utf-8")
        return real(a, rec)
    monkeypatch.setattr(reg, "_finish_submit", change_then_finish)
    assert reg.main(["submit", "--register", str(tmp_path / "r.jsonl"), str(mp)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["receipt"]["entry_sha256"] == B3_HASH
    assert "hashed_manifest" not in out["intake"] and "changed after the append" in out["intake"]["error"]
    assert B3_HASH in out["intake"]["error"]

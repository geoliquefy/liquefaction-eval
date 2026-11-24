"""D3 + D4 joint test (eval M6): the real register (R20) and the real forward runner (R21) together.

Synthetic only: a made-up ShakeMap grid, made-up CPT sites, two made-up entries and a PLANTED label file. No
Canterbury data, no real label, no Kaikoura, no stamp. Every file lives in pytest's tmp_path.

```
 two recipes --> D4 submit_recipe (recipe check) --> D3 append_item --> receipt --> D3 verify_receipt: VERIFIED
 one ordinary entry --> D3 submit_entry (intake.read_entry) --> receipt        (D4 skips it: no recipe)
        v
 D4 freeze A, freeze B (register hash and length frozen) --> predict A, predict B from the D3 register
        v   audit hook: no open of the label file before both "predictions_written" lines
 PLANTED labels written --> score A, score B (two cards, never pooled) --> receipts still VERIFIED
```
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import forward as F  # noqa: E402
from scoreboard.evalkit import register as R  # noqa: E402
from scoreboard.evalkit.spec import Labels, SpecError, canonical_json, save_labels  # noqa: E402

EV = "synth-d34-1"
ORIGIN = "2030-01-01T00:00:00Z"
FLAG = {"id": EV, "time_utc": ORIGIN, "mag": 6.6, "mag_type": "mww", "lat": -40.0, "lon": 175.0, "depth_km": 9.0,
        "status": "CANDIDATE", "class_b": {"flag": "-"}}
TASK = {"target": "surface manifestation", "label_source": "PLANTED synthetic (d34 joint test)",
        "label_rule_sha256": "0" * 64, "tile_m": 1000.0, "clip": 1e-3, "seed": 5,
        "label_caveats": "PLANTED labels; a pipeline test, not a result", "n_boot": 60, "min_sites": 50,
        "min_per_class": 10, "min_tiles": 5}


def at(text):
    """A fixed intake clock for D3 (`now=`)."""
    return lambda: datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def write_grid(path, *, scale=1.0, version="1"):
    """A ShakeMap-shaped grid.xml for the planted earthquake: PGA (%g) falls with distance from (-40, 175)."""
    lons = np.round(np.arange(174.8, 175.2001, 0.01), 4)
    lats = np.round(np.arange(-40.2, -39.7999, 0.01), 4)
    rows = [f"{lo:.4f} {la:.4f} {scale * 55.0 / (1.0 + np.hypot((lo - 175.0) * 85.0, (la + 40.0) * 111.0) / 6.0):.4f}"
            for la in lats[::-1] for lo in lons]
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><shakemap_grid xmlns="http://earthquake.usgs.gov/eqcenter/'
           f'shakemap" event_id="{EV}" shakemap_version="{version}" process_timestamp="2030-01-01T00:20:00" '
           f'shakemap_originator="us" map_status="automatic"><event event_id="{EV}" magnitude="6.6"/>'
           f'<grid_specification lon_min="{lons[0]}" lat_min="{lats[0]}" lon_max="{lons[-1]}" lat_max="{lats[-1]}" '
           f'nlon="{lons.size}" nlat="{lats.size}"/><grid_field index="1" name="LON" units="dd"/>'
           f'<grid_field index="2" name="LAT" units="dd"/><grid_field index="3" name="PGA" units="%g"/>'
           f'<grid_data>\n' + "\n".join(rows) + "\n</grid_data></shakemap_grid>")
    Path(path).write_text(xml, encoding="utf-8")
    return path


def write_cpts(path, n=360, seed=21):
    rng = np.random.default_rng(seed)
    lat, lon = rng.uniform(-40.15, -39.85, n), rng.uniform(174.85, 175.15, n)
    F.write_cpt_list(path, [f"c{k:04d}" for k in range(n)], lat, lon, (lon - 175.0) * 85000.0,
                     (lat + 40.0) * 111000.0)
    return path


def near_is_high(inp):
    """Entry beta's model: closer to the epicentre ranks higher."""
    return 1.0 / (1.0 + np.hypot(inp.x_m, inp.y_m) / 5000.0)


def recipes():
    hook = "tests.test_evalkit_d34_joint:near_is_high"
    return {
        "alpha": {"name": "alpha pga logistic",
                  "runner": {"kind": "builtin", "name": "pga_logistic", "params": {"a": 0.5, "b": 1.8}},
                  "manifest": {"model": "alpha pga logistic", "output_kind": "target_probability",
                               "statement": "expit(0.5 + 1.8 ln(PGA / MSF(M))), fixed before any flag",
                               "exposure": {"*": "unseen"}, "house": False}},
        "beta": {"name": "beta distance score",
                 "runner": {"kind": "python", "callable": hook, "code_sha256": F.callable_sha256(hook)},
                 "manifest": {"model": "beta distance score", "output_kind": "score",
                              "statement": "1 / (1 + r / 5 km), ranking only", "exposure": {"*": "unseen"},
                              "house": False}},
    }


def planted_labels(run, path, seed=13):
    """PLANTED labels from a stated rule on the type A frozen PGA. Written only after both predictions hashes."""
    inp = F.frozen_inputs(run, "A")
    p = 1.0 / (1.0 + np.exp(-(0.3 + 2.0 * np.log(inp.pga_g / 1.1))))
    y = (np.random.default_rng(seed).random(p.size) < p).astype(float)
    save_labels(Labels(inp.site_ids, y, inp.x_m, inp.y_m), path)
    return path


@pytest.fixture
def joint(tmp_path):
    """Two recipes and one ordinary entry on a temp D3 register, with receipts, plus the planted inputs."""
    reg = tmp_path / "register" / "register.jsonl"
    rdir = tmp_path / "recipes"
    rdir.mkdir()
    receipts = {}
    for k, (who, e) in enumerate(recipes().items()):
        (rdir / f"{who}.json").write_text(json.dumps(e, indent=1), encoding="utf-8")
        receipts[who] = F.submit_recipe(reg, e, entrant=who, now=at(f"2029-12-2{k}T09:00:00Z"))
    plain = tmp_path / "plain"                       # an ordinary prediction entry, through D3's own intake
    plain.mkdir()
    ids = [f"p{k}" for k in range(5)]
    (plain / "gamma.json").write_text(json.dumps({"model": "gamma", "output_kind": "score", "statement": "s",
                                                  "exposure": {"other-quake": "unseen"}, "house": False}))
    (plain / "gamma.csv").write_text("site_id,p\n" + "".join(f"{s},0.{k}\n" for k, s in enumerate(ids)))
    receipts["gamma"] = R.submit_entry(reg, plain / "gamma.json", entrant="gamma", now=at("2029-12-25T09:00:00Z"))
    src = tmp_path / "src"
    src.mkdir()
    return {"reg": reg, "recipes": rdir, "receipts": receipts, "plain": plain, "run": tmp_path / "run",
            "gA": write_grid(src / "gA.xml"), "gB": write_grid(src / "gB.xml", scale=1.25, version="5"),
            "cpt": write_cpts(src / "cpts.csv"), "labels": src / "PLANTED-labels.csv"}


# ---------------------------------------------------------------------------------------------------------------
# Audit hook: who opened the label file, and what the run log held at that moment
# ---------------------------------------------------------------------------------------------------------------

_W = {"watch": None, "seen": []}


def _hook(event, args):
    w = _W["watch"]
    if event != "open" or w is None or not args or str(args[0]) != w["labels"] or w.get("busy"):
        return
    w["busy"] = True
    try:
        log = Path(w["log"])
        _W["seen"].append([json.loads(x)["step"] for x in log.read_text().splitlines()] if log.exists() else [])
    finally:
        w["busy"] = False


# Python cannot remove an audit hook, so this one stays for the rest of the pytest session. It is inert: it returns
# at once unless a test sets _W["watch"], and every test here resets it to None in a finally block.
sys.addaudithook(_hook)


# ---------------------------------------------------------------------------------------------------------------
# The joint run
# ---------------------------------------------------------------------------------------------------------------

def test_d3_receipts_verify_for_recipes_and_entries(joint):
    chk = R.check_register(joint["reg"])
    assert chk.status == R.VERIFIED and chk.n_lines == 3
    for who, e in recipes().items():
        rec = joint["receipts"][who]
        assert rec["item_type"] == "entry" and rec["entrant"] == who
        assert rec["entry_sha256"] == F.recipe_sha256(e)
        assert R.verify_receipt(joint["reg"], rec, entry_sha=F.recipe_sha256(e))["status"] == R.VERIFIED
    sha = R.entry_file_sha256(joint["plain"] / "gamma.json")
    assert R.verify_receipt(joint["reg"], joint["receipts"]["gamma"], entry_sha=sha)["status"] == R.VERIFIED


def test_joint_run_is_blind_and_scores_types_a_and_b_apart(joint):
    run, reg, rdir = joint["run"], joint["reg"], joint["recipes"]
    before = reg.read_bytes()
    F.freeze_inputs(run, FLAG, joint["gA"], joint["cpt"], input_type="A", register_path=reg)
    F.freeze_inputs(run, FLAG, joint["gB"], joint["cpt"], input_type="B", register_path=reg,
                    shakemap_note="final version (planted)")
    planted_labels(run, joint["labels"])        # on disk before predict runs, so an early open would succeed
    _W["seen"].clear()
    _W["watch"] = {"labels": str(joint["labels"]), "log": str(run / F.LOG)}
    try:
        pa = F.predict(run, reg, "A", recipes_dir=rdir)
        pb = F.predict(run, reg, "B", recipes_dir=rdir)
        assert _W["seen"] == []                                      # predict never opened the label file
        ca = F.score(run, joint["labels"], "A", TASK)
        cb = F.score(run, joint["labels"], "B", TASK)
    finally:
        _W["watch"] = None
    assert _W["seen"], "score never opened the label file"
    for steps in _W["seen"]:                                         # every open came after both hashes
        assert steps.count("predictions_written") == 2 and "labels_opened" not in steps[:steps.index(
            "predictions_written")]
    order = F.blind_order(run)
    assert order["A"]["blind"] and order["B"]["blind"]

    # Every counted entry traces back to a D3 receipt; the ordinary entry is skipped; nothing is LATE.
    want = {e["name"]: joint["receipts"][who] for who, e in recipes().items()}
    for p in (pa, pb):
        rec = p["record"]
        assert sorted(rec["entries"]) == sorted(want)
        assert not rec["late"] and not rec["run_failed"]
        assert rec["register_skipped"] == {"duplicate_hash": [], "name_taken": [],
                                           "unmatched_entry_lines": {"n": 1, "seqs": [3]}}   # gamma, by design
        for name, h in rec["entries"].items():
            assert h["register_entry_sha256"] == want[name]["entry_sha256"]
            assert F._utc(h["intake_utc"]) == F._utc(want[name]["intake_utc"])
        assert sorted(rec["counted_register_entries"]) == sorted(r["entry_sha256"] for r in want.values())

    # Two cards, one per type, never pooled; same planted label file; each names its own predictions hash.
    assert ca["forward"]["input_type"] == "A" and cb["forward"]["input_type"] == "B"
    assert ca["forward"]["predictions_sha256"] == pa["predictions_sha256"]
    assert cb["forward"]["predictions_sha256"] == pb["predictions_sha256"]
    assert ca["forward"]["input_manifest_sha256"] != cb["forward"]["input_manifest_sha256"]
    assert ca["forward"]["labels_file_sha256"] == cb["forward"]["labels_file_sha256"]
    a_sha = pa["record"]["entries"]["alpha pga logistic"]["entry_sha256"]
    b_sha = pb["record"]["entries"]["alpha pga logistic"]["entry_sha256"]
    assert a_sha != b_sha                                            # type B ran on its own (scaled) inputs
    assert [r["entry_sha256"] for r in cb["entries"] if r.get("name") == "alpha pga logistic"] == [b_sha]

    # The runner never wrote to the register; every receipt still verifies after the run.
    assert reg.read_bytes() == before
    for who in ("alpha", "beta"):
        e = recipes()[who]
        assert R.verify_receipt(reg, joint["receipts"][who], entry_sha=F.recipe_sha256(e))["status"] == R.VERIFIED


def test_score_before_predict_never_opens_the_planted_labels(joint):
    run = joint["run"]
    F.freeze_inputs(run, FLAG, joint["gA"], joint["cpt"], input_type="A", register_path=joint["reg"])
    joint["labels"].write_text("site_id,label,x_m,y_m\n")
    _W["seen"].clear()
    _W["watch"] = {"labels": str(joint["labels"]), "log": str(run / F.LOG)}
    try:
        with pytest.raises(F.RunRefused, match="no predictions hash"):
            F.score(run, joint["labels"], "A", TASK)
    finally:
        _W["watch"] = None
    assert _W["seen"] == []


# ---------------------------------------------------------------------------------------------------------------
# Seams the joint test found (each failed on main 85d068c; see docs/evals/D34-JOINT-TEST-2026-11-23.md)
# ---------------------------------------------------------------------------------------------------------------

def test_seam1_a_line_d3_calls_a_fault_is_refused_by_the_runner(joint, tmp_path):
    """A line with a good prev link but an intake time that goes back (a back-dated entry). D3 check says REGISTER
    FAULT. Before the fix, D4 checked only seq and prev links, so it counted the back-dated entry as on time."""
    reg, rdir = joint["reg"], joint["recipes"]
    late = {**recipes()["alpha"], "name": "back-dated", "runner": {"kind": "builtin", "name": "shaking_score"}}
    (rdir / "late.json").write_text(json.dumps(late))
    chk = R.check_register(reg)
    line = {"v": 1, "seq": chk.head_seq + 1, "intake_utc": "2029-11-01T00:00:00.000000Z", "item_type": "entry",
            "item_sha256": F.recipe_sha256(late), "prev_sha256": chk.head_sha256, "entrant": "mallory"}
    with open(reg, "a", encoding="utf-8") as f:
        f.write(canonical_json(line) + "\n")
    assert R.check_register(reg).status == R.REGISTER_FAULT
    with pytest.raises(SpecError, match="REGISTER FAULT"):
        F.register_entries(reg, rdir)


def test_seam2_recipe_intake_checks_the_recipe_before_d3_appends(joint):
    """Before the fix, a recipe went on by a bare append_item: a broken recipe got a VERIFIED receipt and later made
    predict refuse every entry of the run. submit_recipe checks the recipe first; nothing is appended on refusal."""
    reg = joint["reg"]
    n = R.check_register(reg).n_lines
    bad_kind = {**recipes()["alpha"], "name": "bad kind"}
    bad_kind["manifest"] = {**bad_kind["manifest"], "output_kind": "certainty"}
    bad_runner = {**recipes()["alpha"], "name": "bad runner", "runner": {"kind": "builtin", "name": "oracle"}}
    bad_code = {**recipes()["beta"], "name": "bad code"}
    bad_code["runner"] = {**bad_code["runner"], "code_sha256": "f" * 64}
    for e in (bad_kind, bad_runner, bad_code, {"name": "no runner", "manifest": {}}):
        with pytest.raises(SpecError):
            F.submit_recipe(reg, e, entrant="mallory", now=at("2029-12-28T00:00:00Z"))
    assert R.check_register(reg).n_lines == n


def test_seam3_a_line_appended_after_the_freeze_does_not_block_predict(joint):
    """D3 is one live, append-only register. Before the fix, D4 froze the whole file's hash, so any later intake
    (another entrant, another earthquake) between freeze and predict refused the run. Now D4 freezes the register
    prefix (its byte length and sha256) and reads only those lines; a change inside the prefix is still refused."""
    run, reg, rdir = joint["run"], joint["reg"], joint["recipes"]
    F.freeze_inputs(run, FLAG, joint["gA"], joint["cpt"], input_type="A", register_path=reg)
    after = {**recipes()["alpha"], "name": "after the freeze", "runner": {"kind": "builtin",
                                                                          "name": "shaking_score"}}
    (rdir / "after.json").write_text(json.dumps(after))
    F.submit_recipe(reg, after, entrant="delta", now=at("2030-01-01T06:00:00Z"))
    rec = F.predict(run, reg, "A", recipes_dir=rdir)["record"]
    assert sorted(rec["entries"]) == ["alpha pga logistic", "beta distance score"]     # the frozen set only
    assert rec["late"] == [{"name": "after the freeze", "register_entry_sha256": F.recipe_sha256(after),
                            "intake_utc": "2030-01-01T06:00:00.000000Z", "seq": 4,
                            "reason": "registered after the freeze"}]                  # listed, not lost
    m = json.loads((run / "manifests" / "A.json").read_text())
    assert m["register"]["n_bytes"] < reg.stat().st_size


def test_seam3_an_edit_inside_the_frozen_prefix_is_still_refused(joint):
    run, reg = joint["run"], joint["reg"]
    F.freeze_inputs(run, FLAG, joint["gA"], joint["cpt"], input_type="A", register_path=reg)
    lines = reg.read_text().splitlines()
    d = json.loads(lines[0])
    d["entrant"] = "mallory"
    reg.write_text("\n".join([canonical_json(d)] + lines[1:]) + "\n")
    with pytest.raises(F.RunRefused, match="not the one frozen"):
        F.predict(run, reg, "A", recipes_dir=joint["recipes"])


# ---------------------------------------------------------------------------------------------------------------
# Review 1 (2026-11-23): the rest of seam 2. A VERIFIED receipt must never make predict refuse the run.
# ---------------------------------------------------------------------------------------------------------------

def test_seam2_a_retry_of_the_same_recipe_gets_no_second_line(joint):
    """Review finding (a): an entrant retry of the same recipe got a second VERIFIED line, then predict refused
    every entry ("registered twice"). Now intake refuses the retry, and nothing is appended."""
    reg = joint["reg"]
    n = R.check_register(reg).n_lines
    with pytest.raises(SpecError, match="already registered"):
        F.submit_recipe(reg, recipes()["alpha"], entrant="alpha", now=at("2029-12-28T00:00:00Z"))
    assert R.check_register(reg).n_lines == n


def test_seam2_a_duplicate_hash_line_on_the_register_does_not_block_predict(joint):
    """The same retry made by a bare D3 append_item (or in a race with the intake read): predict counts the first
    line of the hash, lists the second, and runs."""
    run, reg, rdir = joint["run"], joint["reg"], joint["recipes"]
    rc = R.append_item(reg, "entry", F.recipe_sha256(recipes()["alpha"]), entrant="alpha",
                       now=at("2029-12-28T00:00:00Z"))
    assert R.verify_receipt(reg, rc, entry_sha=F.recipe_sha256(recipes()["alpha"]))["status"] == R.VERIFIED
    F.freeze_inputs(run, FLAG, joint["gA"], joint["cpt"], input_type="A", register_path=reg)
    rec = F.predict(run, reg, "A", recipes_dir=rdir)["record"]
    assert sorted(rec["entries"]) == ["alpha pga logistic", "beta distance score"]
    assert rec["entries"]["alpha pga logistic"]["intake_utc"] == joint["receipts"]["alpha"]["intake_utc"]
    assert rec["register_skipped"]["duplicate_hash"] == [
        {"seq": rc["seq"], "register_entry_sha256": rc["entry_sha256"], "intake_utc": rc["intake_utc"]}]


def test_seam2_a_taken_name_is_refused_at_intake_with_recipes_dir(joint):
    """Review finding (b), at intake: a second entrant's different recipe under a name already registered."""
    reg, rdir = joint["reg"], joint["recipes"]
    n = R.check_register(reg).n_lines
    twin = {**recipes()["alpha"], "runner": {"kind": "builtin", "name": "shaking_score"}}
    with pytest.raises(SpecError, match="already registered by another recipe"):
        F.submit_recipe(reg, twin, entrant="mallory", now=at("2029-12-28T00:00:00Z"), recipes_dir=rdir)
    assert R.check_register(reg).n_lines == n


def test_seam2_a_taken_name_without_recipes_dir_does_not_block_predict(joint):
    """Review finding (b), at predict: the intake had no recipes_dir, so the twin got a VERIFIED receipt. Predict
    runs the first recipe of the name (by seq), lists the twin under name_taken, and does not refuse the run."""
    run, reg, rdir = joint["run"], joint["reg"], joint["recipes"]
    twin = {**recipes()["alpha"], "runner": {"kind": "builtin", "name": "shaking_score"}}
    (rdir / "twin.json").write_text(json.dumps(twin))
    rc = F.submit_recipe(reg, twin, entrant="mallory", now=at("2029-12-28T00:00:00Z"))
    assert R.verify_receipt(reg, rc, entry_sha=F.recipe_sha256(twin))["status"] == R.VERIFIED
    F.freeze_inputs(run, FLAG, joint["gA"], joint["cpt"], input_type="A", register_path=reg)
    rec = F.predict(run, reg, "A", recipes_dir=rdir)["record"]
    assert rec["entries"]["alpha pga logistic"]["register_entry_sha256"] == F.recipe_sha256(recipes()["alpha"])
    assert rec["register_skipped"]["name_taken"] == [
        {"seq": rc["seq"], "name": "alpha pga logistic", "register_entry_sha256": rc["entry_sha256"],
         "intake_utc": rc["intake_utc"]}]


def test_a_registered_recipe_with_no_file_leaves_a_trace(joint):
    """Review nit: a registered recipe whose file is missing from recipes_dir is not run. It is counted with the
    entry lines that match no recipe (seqs listed), next to the ordinary entry that D4 skips by design."""
    run, reg, rdir = joint["run"], joint["reg"], joint["recipes"]
    (rdir / "beta.json").unlink()
    F.freeze_inputs(run, FLAG, joint["gA"], joint["cpt"], input_type="A", register_path=reg)
    rec = F.predict(run, reg, "A", recipes_dir=rdir)["record"]
    assert sorted(rec["entries"]) == ["alpha pga logistic"]
    assert rec["register_skipped"]["unmatched_entry_lines"] == {"n": 2, "seqs": [2, 3]}

"""evalkit forward runner (D4, eval M6, rule R21): freeze, hash, predict blind, score types A and B apart.

Synthetic only: a made-up ShakeMap grid, made-up CPT sites and a planted label file. No Canterbury data is read.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import forward as F  # noqa: E402
from scoreboard.evalkit.spec import Labels, SpecError, save_labels  # noqa: E402

EV = "synth-fwd-1"
FLAG = {"id": EV, "time_utc": "2030-01-01T00:00:00Z", "mag": 6.4, "mag_type": "mww", "lat": -40.0, "lon": 175.0,
        "depth_km": 8.0, "status": "CANDIDATE", "class_b": {"flag": "CLASS B"}, "cpts": {"n_cpt": 400}}
TASK = {"target": "surface manifestation", "label_source": "planted-synthetic", "label_rule_sha256": "0" * 64,
        "tile_m": 4000.0, "clip": 1e-6, "seed": 20261106, "label_caveats": "planted synthetic labels; a test, not a result",
        "n_boot": 60, "min_sites": 50, "min_per_class": 10, "min_tiles": 5}


def write_grid(path, *, scale=1.0, version="1", event_id=EV, gz=False):
    """A ShakeMap-shaped grid.xml: PGA (%g) falls with distance from (-40, 175)."""
    lons = np.round(np.arange(174.8, 175.2001, 0.01), 4)
    lats = np.round(np.arange(-40.2, -39.7999, 0.01), 4)
    rows = []
    for la in lats[::-1]:
        for lo in lons:
            r = np.hypot((lo - 175.0) * 85.0, (la + 40.0) * 111.0)
            rows.append(f"{lo:.4f} {la:.4f} {scale * 60.0 / (1.0 + r / 5.0):.4f}")
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><shakemap_grid xmlns="http://earthquake.usgs.gov/eqcenter/'
           f'shakemap" event_id="{event_id}" shakemap_version="{version}" process_timestamp="2030-01-01T00:10:00" '
           f'shakemap_originator="us" map_status="automatic"><event event_id="{event_id}" magnitude="6.4"/>'
           f'<grid_specification lon_min="{lons[0]}" lat_min="{lats[0]}" lon_max="{lons[-1]}" lat_max="{lats[-1]}" '
           f'nlon="{lons.size}" nlat="{lats.size}"/><grid_field index="1" name="LON" units="dd"/>'
           f'<grid_field index="2" name="LAT" units="dd"/><grid_field index="3" name="PGA" units="%g"/>'
           f'<grid_data>\n' + "\n".join(rows) + "\n</grid_data></shakemap_grid>")
    data = xml.encode()
    if gz:
        import gzip
        data = gzip.compress(data)
    Path(path).write_bytes(data)
    return path


def write_cpts(path, n=400, seed=3, outside=5):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-40.15, -39.85, n)
    lon = rng.uniform(174.85, 175.15, n)
    lat[:outside] = -41.0                                              # outside the grid
    x = (lon - 175.0) * 85000.0
    y = (lat + 40.0) * 111000.0
    F.write_cpt_list(path, [f"s{k:04d}" for k in range(n)], lat, lon, x, y)
    return path


def entries():
    return [
        {"name": "shaking score", "runner": {"kind": "builtin", "name": "shaking_score"},
         "manifest": {"model": "shaking score", "output_kind": "score", "statement": "PGA / MSF(M), ranking only",
                      "exposure": {"*": "unseen"}, "house": False}},
        {"name": "pga logistic", "runner": {"kind": "builtin", "name": "pga_logistic",
                                             "params": {"a": 1.0, "b": 2.0}},
         "manifest": {"model": "pga logistic", "output_kind": "target_probability",
                      "statement": "expit(1 + 2 ln(PGA / MSF(M))), fixed before the flag",
                      "exposure": {"*": "unseen"}, "house": True}},
        python_entry("python hook", "far_is_low"),
    ]


def python_entry(name, fn):
    spec = f"tests.test_evalkit_forward:{fn}"
    return {"name": name, "runner": {"kind": "python", "callable": spec, "code_sha256": F.callable_sha256(spec)},
            "manifest": {"model": name, "output_kind": "score", "statement": "a python hook entry",
                         "exposure": {"*": "unseen"}, "house": False}}


def far_is_low(inp):
    return -np.hypot(inp.x_m, inp.y_m)


def noisy(inp):
    """Not frozen: a second run differs (R21 RUN FAILED)."""
    return np.random.default_rng().random(len(inp.site_ids))


def planted_labels(run_dir, path, seed=11):
    """Planted labels drawn from the true model on the type A frozen PGA. Written only after predictions."""
    inp = F.frozen_inputs(run_dir, "A")
    rng = np.random.default_rng(seed)
    p = 1.0 / (1.0 + np.exp(-(1.0 + 2.0 * np.log(inp.pga_g / 1.1))))
    y = (rng.random(p.size) < p).astype(float)
    y[::17] = np.nan
    save_labels(Labels(inp.site_ids, y, inp.x_m, inp.y_m), path)
    return path


@pytest.fixture
def setup(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    run = tmp_path / "run"
    reg = src / "register.jsonl"
    F.write_register(reg, entries())
    return {"run": run, "reg": reg, "gA": write_grid(src / "gA.xml"), "gB": write_grid(src / "gB.xml", scale=1.3,
            version="4"), "cpt": write_cpts(src / "cpts.csv"), "labels": src / "PLANTED-labels.csv"}


# ---------------------------------------------------------------------------------------------------------------
# Watcher flags (read only)
# ---------------------------------------------------------------------------------------------------------------

def test_flagged_events_reads_the_watcher_shape(tmp_path):
    S = {"events": {
        "a": {"id": "a", "time_utc": "2030-01-02T00:00:00Z", "status": "CANDIDATE", "class_b": {"flag": "-"}},
        "b": {"id": "b", "time_utc": "2030-01-01T00:00:00Z", "status": "BELOW M6", "class_b": {"flag": "CLASS B"}},
        "c": {"id": "c", "time_utc": "2030-01-03T00:00:00Z", "status": "FEW CPTS", "class_b": {"flag": "-"}},
        "d": {"id": "d", "time_utc": "2030-01-04T00:00:00Z", "status": "ALERT UNKNOWN"}}}
    p = tmp_path / "state.json"
    p.write_text(json.dumps(S))
    before = p.read_bytes()
    assert [r["id"] for r in F.flagged_events(p)] == ["b", "a", "d"]
    assert p.read_bytes() == before


def test_watcher_constants_match_the_watcher():
    root = Path(__file__).resolve().parents[1]
    w = root.parents[1] / "repos" / "event-scoreboard" / "scripts" / "g5_watch.py"
    b = root.parents[1] / "repos" / "event-scoreboard" / "scoreboard" / "g5_class_b.py"
    if not w.exists():                      # the worktree sits under projects/, the repos under the pillar root
        w = root.parents[3] / "repos" / "event-scoreboard" / "scripts" / "g5_watch.py"
        b = root.parents[3] / "repos" / "event-scoreboard" / "scoreboard" / "g5_class_b.py"
    if not w.exists():
        pytest.skip("event-scoreboard not on disk")
    assert 'FLAG_STATUSES = ("CANDIDATE", "ALERT", "ALERT UNKNOWN")' in w.read_text()
    assert 'FLAG = "CLASS B"' in b.read_text()


# ---------------------------------------------------------------------------------------------------------------
# Freeze and hash
# ---------------------------------------------------------------------------------------------------------------

def test_freeze_writes_a_hashed_manifest_and_logs_it(setup):
    out = F.freeze_inputs(setup["run"], FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    m = out["manifest"]
    assert m["input_type"] == "A" and m["shakemap"]["shakemap_version"] == "1"
    assert m["sites"]["n_sites"] == 395 and m["sites"]["n_outside_grid"] == 5
    log = F.read_log(setup["run"])
    assert log[0]["step"] == "inputs_frozen" and log[0]["sha256"] == out["manifest_sha256"]
    assert F.load_frozen(setup["run"], "A")[1] == out["manifest_sha256"]


def test_gzip_grid_and_bilinear_sample(tmp_path):
    g = F.read_grid(write_grid(tmp_path / "g.xml.gz", gz=True))
    assert g.sample([-40.0], [175.0])[0] == pytest.approx(0.60, abs=1e-9)
    assert np.isnan(g.sample([-41.0], [175.0])[0])


def test_a_frozen_type_is_never_replaced_and_edits_are_caught(setup):
    F.freeze_inputs(setup["run"], FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    with pytest.raises(F.RunRefused, match="already frozen"):
        F.freeze_inputs(setup["run"], FLAG, setup["gB"], setup["cpt"], input_type="A", register_path=setup["reg"])
    p = setup["run"] / "inputs" / "A" / "package" / "cpt_list.csv"
    p.write_text(p.read_text().replace("s0100", "s9999"))
    with pytest.raises(F.RunRefused, match="changed after the freeze"):
        F.load_frozen(setup["run"], "A")


def test_wrong_event_grid_is_refused(setup, tmp_path):
    g = write_grid(tmp_path / "other.xml", event_id="someone-else")
    with pytest.raises(F.RunRefused, match="not the flagged"):
        F.freeze_inputs(setup["run"], FLAG, g, setup["cpt"], input_type="A", register_path=setup["reg"])


def test_log_chain_detects_an_edited_line(setup):
    F.freeze_inputs(setup["run"], FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    F.freeze_inputs(setup["run"], FLAG, setup["gB"], setup["cpt"], input_type="B", register_path=setup["reg"])
    p = setup["run"] / F.LOG
    lines = p.read_text().splitlines()
    d = json.loads(lines[0])
    d["sha256"] = "f" * 64
    p.write_text("\n".join([json.dumps(d, sort_keys=True)] + lines[1:]) + "\n")
    with pytest.raises(F.RunRefused, match="broken chain"):
        F.read_log(setup["run"])


def test_publish_is_wired_but_not_run(setup, monkeypatch):
    F.freeze_inputs(setup["run"], FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    called = []
    monkeypatch.setattr(F.subprocess, "run", lambda *a, **k: called.append(a))
    out = F.publish_input_hash(setup["run"], "A")
    assert out["ran"] is False and out["command"][:2] == ["ots", "stamp"] and not called
    assert out["manifest_sha256"] in Path(out["anchor"]).read_text()
    with pytest.raises(F.RunRefused, match="founder go"):
        F.publish_input_hash(setup["run"], "A", run=True)
    assert not called
    assert [d["step"] for d in F.read_log(setup["run"])] == ["inputs_frozen", "publish_planned"]


# ---------------------------------------------------------------------------------------------------------------
# Register (D3 shape)
# ---------------------------------------------------------------------------------------------------------------

def test_register_checks_hashes_chain_and_names(tmp_path):
    p = tmp_path / "r.jsonl"
    F.write_register(p, entries())
    assert [e["name"] for e in F.register_entries(p)] == ["shaking score", "pga logistic", "python hook"]
    lines = p.read_text().splitlines()
    d = json.loads(lines[1])
    d["entry"]["runner"]["params"]["a"] = 5.0
    p.write_text("\n".join([lines[0], json.dumps(d), lines[2]]) + "\n")
    with pytest.raises(SpecError, match="entry_sha256"):
        F.register_entries(p)
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(SpecError, match="prev_sha256"):
        F.register_entries(p)
    F.write_register(p, entries()[:1] * 2)
    with pytest.raises(SpecError, match="registered twice"):
        F.register_entries(p)


# ---------------------------------------------------------------------------------------------------------------
# Blindness: the label file is not opened before the predictions hash is written
# ---------------------------------------------------------------------------------------------------------------

_OPENS = {"watch": None, "seen": []}


def _hook(event, args):
    if event == "open" and _OPENS["watch"] is not None and args and str(args[0]) == _OPENS["watch"]["labels"]:
        if _OPENS["watch"].get("busy"):
            return
        _OPENS["watch"]["busy"] = True
        try:
            log = Path(_OPENS["watch"]["log"]).read_text() if Path(_OPENS["watch"]["log"]).exists() else ""
            _OPENS["seen"].append([json.loads(ln)["step"] for ln in log.splitlines()])
        finally:
            _OPENS["watch"]["busy"] = False


sys.addaudithook(_hook)


def _watch(setup):
    _OPENS["seen"].clear()
    _OPENS["watch"] = {"labels": str(setup["labels"]), "log": str(setup["run"] / F.LOG)}


def test_replay_end_to_end_is_blind_and_scores_types_apart(setup):
    run = setup["run"]
    _watch(setup)
    try:
        F.freeze_inputs(run, FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
        F.publish_input_hash(run, "A")
        F.freeze_inputs(run, FLAG, setup["gB"], setup["cpt"], input_type="B", register_path=setup["reg"],
                        shakemap_note="final version")
        pa = F.predict(run, setup["reg"], "A")
        pb = F.predict(run, setup["reg"], "B")
        assert _OPENS["seen"] == []                                    # labels never touched while predicting
        planted_labels(run, setup["labels"])                           # labels "arrive" only now
        _OPENS["seen"].clear()
        ca = F.score(run, setup["labels"], "A", TASK)
        cb = F.score(run, setup["labels"], "B", TASK)
    finally:
        _OPENS["watch"] = None
    assert _OPENS["seen"], "the label file was never opened by score"
    for steps in _OPENS["seen"]:                                       # every open came after both hashes
        assert steps.count("predictions_written") == 2
    assert ca["forward"]["input_type"] == "A" and cb["forward"]["input_type"] == "B"
    assert ca["forward"]["predictions_sha256"] == pa["predictions_sha256"]
    assert cb["forward"]["predictions_sha256"] == pb["predictions_sha256"]
    assert ca["forward"]["input_manifest_sha256"] != cb["forward"]["input_manifest_sha256"]
    assert ca["forward"]["labels_file_sha256"] == cb["forward"]["labels_file_sha256"]
    assert (run / "scores" / "A" / "card.json").exists() and (run / "scores" / "B" / "card.json").exists()
    assert pa["record"]["entries"]["pga logistic"]["entry_sha256"] != pb["record"]["entries"]["pga logistic"][
        "entry_sha256"]                                                # type B ran on its own inputs
    order = F.blind_order(run)
    assert order["A"]["blind"] and order["B"]["blind"]
    assert [d["step"] for d in F.read_log(run)][-4:] == ["labels_opened", "scored", "labels_opened", "scored"]


def test_score_before_predictions_is_refused_without_opening_labels(setup):
    run = setup["run"]
    F.freeze_inputs(run, FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    setup["labels"].write_text("site_id,label,x_m,y_m\n")
    _watch(setup)
    try:
        with pytest.raises(F.RunRefused, match="no predictions hash"):
            F.score(run, setup["labels"], "A", TASK)
    finally:
        _OPENS["watch"] = None
    assert _OPENS["seen"] == []
    assert "labels_opened" not in [d["step"] for d in F.read_log(run)]


def test_edited_predictions_refuse_the_score_without_opening_labels(setup):
    run = setup["run"]
    F.freeze_inputs(run, FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    F.predict(run, setup["reg"], "A")
    planted_labels(run, setup["labels"])
    p = run / "predictions" / "A" / "pga_logistic.csv"
    lines = p.read_text().splitlines()
    lines[1] = lines[1].split(",")[0] + ",0.999"
    p.write_text("\n".join(lines) + "\n")
    _watch(setup)
    try:
        with pytest.raises(F.RunRefused, match="changed after the predictions hash"):
            F.score(run, setup["labels"], "A", TASK)
    finally:
        _OPENS["watch"] = None
    assert _OPENS["seen"] == []


def test_predict_after_labels_opened_is_refused_and_runs_once(setup):
    run = setup["run"]
    F.freeze_inputs(run, FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    F.freeze_inputs(run, FLAG, setup["gB"], setup["cpt"], input_type="B", register_path=setup["reg"])
    F.predict(run, setup["reg"], "A")
    with pytest.raises(F.RunRefused, match="run once"):
        F.predict(run, setup["reg"], "A")
    planted_labels(run, setup["labels"])
    F.score(run, setup["labels"], "A", TASK)
    with pytest.raises(F.RunRefused, match="not blind"):
        F.predict(run, setup["reg"], "B")


def test_predict_takes_no_labels_and_entries_see_no_labels():
    import inspect
    assert list(inspect.signature(F.predict).parameters) == ["run_dir", "register_path", "input_type", "recipes_dir",
                                                             "sandbox", "sandbox_opts"]   # B4 sandbox (ruling 59a)
    assert not any("label" in f for f in F.FrozenInputs.__dataclass_fields__)


def test_task_identity_comes_from_the_frozen_inputs(setup):
    run = setup["run"]
    F.freeze_inputs(run, FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    F.predict(run, setup["reg"], "A")
    planted_labels(run, setup["labels"])
    with pytest.raises(F.RunRefused, match="comes from the frozen inputs"):
        F.score(run, setup["labels"], "A", {**TASK, "event_id": "other"})
    with pytest.raises(F.RunRefused, match="missing"):
        F.score(run, setup["labels"], "A", {k: v for k, v in TASK.items() if k != "target"})


def test_types_must_score_the_same_labels(setup, tmp_path):
    run = setup["run"]
    F.freeze_inputs(run, FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    F.freeze_inputs(run, FLAG, setup["gB"], setup["cpt"], input_type="B", register_path=setup["reg"])
    F.predict(run, setup["reg"], "A")
    F.predict(run, setup["reg"], "B")
    planted_labels(run, setup["labels"])
    F.score(run, setup["labels"], "A", TASK)
    other = planted_labels(run, tmp_path / "other.csv", seed=99)
    with pytest.raises(F.RunRefused, match="same label file"):
        F.score(run, other, "B", TASK)


def test_reads_a_d3_register_of_hashes_with_recipes(tmp_path, setup):
    """A register in the D3 layout: canonical JSON lines, prev_sha256 is the sha256 of the line text before.

    Intake times never go back (D3 R20); the D3 and D4 joint test made the runner run D3's own check first."""
    import hashlib
    from scoreboard.evalkit.spec import canonical_json
    rec = tmp_path / "recipes"
    rec.mkdir()
    late = {**entries()[0], "name": "late entry"}
    items = [("entry", e, "2029-12-30T00:00:00.000000Z") for e in entries()]
    items.insert(1, ("entry", None, "2029-12-30T00:00:00.000000Z"))   # an ordinary entry, as submit_entry writes
    items.insert(2, ("task", None, "2029-12-30T00:00:00.000000Z"))
    items.append(("entry", late, "2030-01-01T00:00:00.000000Z"))      # the origin itself: LATE
    prev, lines = "0" * 64, []
    for k, (typ, e, t) in enumerate(items):
        if e is not None:
            (rec / f"e{k}.json").write_text(json.dumps(e))
        h = F.recipe_sha256(e) if e is not None else hashlib.sha256(f"other {k}".encode()).hexdigest()
        ln = canonical_json({"v": 1, "seq": k + 1, "intake_utc": t, "item_type": typ, "item_sha256": h,
                             "prev_sha256": prev, "entrant": None})
        lines.append(ln)
        prev = hashlib.sha256(ln.encode()).hexdigest()
    reg = tmp_path / "d3.jsonl"
    reg.write_text("\n".join(lines) + "\n")
    got = F.register_entries(reg, rec)
    assert [e["name"] for e in got] == ["shaking score", "pga logistic", "python hook", "late entry"]
    with pytest.raises(SpecError, match="recipes_dir"):
        F.register_entries(reg)
    F.freeze_inputs(setup["run"], FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=reg)
    out = F.predict(setup["run"], reg, "A", recipes_dir=rec)["record"]
    assert sorted(out["entries"]) == ["pga logistic", "python hook", "shaking score"]
    assert [d["name"] for d in out["late"]] == ["late entry"]          # not run, listed with its intake time
    assert out["late"][0]["intake_utc"] == "2030-01-01T00:00:00.000000Z"
    (rec / "e3.json").write_text(json.dumps({**entries()[1], "name": "changed"}))
    with pytest.raises(SpecError, match="on no register line"):
        F.register_entries(reg, rec)
    (rec / "e3.json").write_text(json.dumps(entries()[1]))
    reg.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(SpecError, match="REGISTER FAULT"):            # D3's own check refuses it first
        F.register_entries(reg, rec)


# ---------------------------------------------------------------------------------------------------------------
# Review 1 fixes
# ---------------------------------------------------------------------------------------------------------------

def _two_types(setup):
    run = setup["run"]
    F.freeze_inputs(run, FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    F.freeze_inputs(run, FLAG, setup["gB"], setup["cpt"], input_type="B", register_path=setup["reg"])
    pa, pb = F.predict(run, setup["reg"], "A"), F.predict(run, setup["reg"], "B")
    planted_labels(run, setup["labels"])
    return run, pa, pb


def test_a_type_is_scored_once_and_never_on_other_labels(setup, tmp_path):
    run, _, _ = _two_types(setup)
    F.score(run, setup["labels"], "A", TASK)
    card = (run / "scores" / "A" / "card.json").read_bytes()
    with pytest.raises(F.RunRefused, match="already scored"):
        F.score(run, setup["labels"], "A", TASK)
    other = planted_labels(run, tmp_path / "other.csv", seed=99)
    with pytest.raises(F.RunRefused, match="already scored"):
        F.score(run, other, "A", TASK)
    assert (run / "scores" / "A" / "card.json").read_bytes() == card
    assert [d["step"] for d in F.read_log(run)].count("labels_opened") == 1


def test_label_shopping_on_one_type_is_refused(setup, tmp_path, monkeypatch):
    run, _, _ = _two_types(setup)
    import scoreboard.evalkit.card as C
    real = C.build_card
    monkeypatch.setattr(C, "build_card", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        F.score(run, setup["labels"], "A", TASK)                     # labels opened, no card written
    monkeypatch.setattr(C, "build_card", real)
    other = planted_labels(run, tmp_path / "other.csv", seed=99)
    with pytest.raises(F.RunRefused, match="another label file"):
        F.score(run, other, "A", TASK)
    assert not (run / "scores" / "A" / "card.json").exists()
    assert F.score(run, setup["labels"], "A", TASK)["forward"]["input_type"] == "A"


def test_card_b_uses_type_b_predictions(setup):
    run, pa, pb = _two_types(setup)
    ca, cb = F.score(run, setup["labels"], "A", TASK), F.score(run, setup["labels"], "B", TASK)

    def sha(card, model):
        return [r["entry_sha256"] for r in card["entries"] if r.get("name") == model]
    assert sha(cb, "pga logistic") == [pb["record"]["entries"]["pga logistic"]["entry_sha256"]]
    assert sha(ca, "pga logistic") == [pa["record"]["entries"]["pga logistic"]["entry_sha256"]]
    assert sha(ca, "pga logistic") != sha(cb, "pga logistic")
    rec = json.loads((run / "predictions" / "B" / "predictions.json").read_text())
    assert rec["input_type"] == "B"


def test_edited_manifest_is_caught(setup):
    F.freeze_inputs(setup["run"], FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    p = setup["run"] / "manifests" / "A.json"
    m = json.loads(p.read_text())
    m["event"]["mag"] = 7.9
    p.write_text(json.dumps(m))
    with pytest.raises(F.RunRefused, match="input manifest changed after the freeze"):
        F.load_frozen(setup["run"], "A")


def test_unflagged_record_needs_a_logged_override(setup):
    rec = {**FLAG, "status": "BELOW M6", "class_b": {"flag": "no"}}
    with pytest.raises(F.RunRefused, match="not flagged"):
        F.freeze_inputs(setup["run"], rec, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    out = F.freeze_inputs(setup["run"], rec, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"],
                          override_reason="test of the override path")
    assert out["manifest"]["flag"] == {"flagged": False, "override_reason": "test of the override path"}
    assert F.read_log(setup["run"])[0]["override_reason"] == "test of the override path"


def test_register_is_frozen_at_the_freeze(setup):
    F.freeze_inputs(setup["run"], FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    F.write_register(setup["reg"], entries()[:2])
    with pytest.raises(F.RunRefused, match="not the one frozen"):
        F.predict(setup["run"], setup["reg"], "A")


def test_types_must_run_the_same_entries(setup, tmp_path):
    run = setup["run"]
    reg_b = tmp_path / "reg_b.jsonl"
    F.write_register(reg_b, entries()[:2])
    F.freeze_inputs(run, FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=setup["reg"])
    F.freeze_inputs(run, FLAG, setup["gB"], setup["cpt"], input_type="B", register_path=reg_b)
    F.predict(run, setup["reg"], "A")
    with pytest.raises(F.RunRefused, match="same counted entries"):
        F.predict(run, reg_b, "B")


def test_late_entries_are_not_run(setup, tmp_path):
    reg = tmp_path / "late.jsonl"
    F.write_register(reg, entries(), submitted_utc="2029-12-31T12:00:00Z")   # 12 h before the origin
    F.freeze_inputs(setup["run"], FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=reg)
    with pytest.raises(F.RunRefused, match="no entry counts"):
        F.predict(setup["run"], reg, "A")
    counted, late = F.split_late(F.register_entries(reg), "2030-01-01T12:00:00Z")
    assert len(counted) == 3 and not late


def test_python_entry_code_is_frozen(setup, tmp_path):
    e = python_entry("python hook", "far_is_low")
    e["runner"]["code_sha256"] = "f" * 64
    reg = tmp_path / "code.jsonl"
    F.write_register(reg, [e])
    F.freeze_inputs(setup["run"], FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=reg)
    with pytest.raises(SpecError, match="changed after registration"):
        F.predict(setup["run"], reg, "A")
    del e["runner"]["code_sha256"]
    F.write_register(reg, [e])
    with pytest.raises(F.RunRefused, match="not the one frozen"):
        F.predict(setup["run"], reg, "A")


def test_python_entry_without_code_hash_is_refused(setup, tmp_path):
    e = python_entry("python hook", "far_is_low")
    del e["runner"]["code_sha256"]
    reg = tmp_path / "nocode.jsonl"
    F.write_register(reg, [e])
    F.freeze_inputs(setup["run"], FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=reg)
    with pytest.raises(SpecError, match="needs code_sha256"):
        F.predict(setup["run"], reg, "A")


def test_entry_whose_second_run_differs_is_run_failed(setup, tmp_path):
    reg = tmp_path / "noisy.jsonl"
    F.write_register(reg, entries() + [python_entry("noisy hook", "noisy")])
    F.freeze_inputs(setup["run"], FLAG, setup["gA"], setup["cpt"], input_type="A", register_path=reg)
    rec = F.predict(setup["run"], reg, "A")["record"]
    assert "noisy hook" not in rec["entries"] and len(rec["entries"]) == 3
    assert [d["name"] for d in rec["run_failed"]] == ["noisy hook"]
    assert not (setup["run"] / "predictions" / "A" / "noisy_hook.csv").exists()


def test_score_writes_package_sha256_on_a_closed_task_from_the_frozen_inputs(setup, capsys):
    """D5 follow-ups (wave 10): forward.score builds a closed task with package_sha256 (R2, R15)."""
    from scoreboard.evalkit import package as P
    from scoreboard.evalkit.spec import Task
    run, _, _ = _two_types(setup)
    _watch(setup)
    try:
        with pytest.raises(F.RunRefused, match="package_sha256 comes from the frozen inputs"):
            F.score(run, setup["labels"], "A", {**TASK, "package_sha256": "3" * 64})
    finally:
        _OPENS["watch"] = None
    assert _OPENS["seen"] == []                                  # the label path is never opened (review 1, Q11)
    assert not [d for d in F.read_log(run) if d["step"] == "labels_opened"]      # refused before the label open
    cards = {t: F.score(run, setup["labels"], t, TASK) for t in ("A", "B")}
    for t, card in cards.items():
        m = json.loads((run / "manifests" / f"{t}.json").read_text())
        table = run / "scores" / t / "package_table.csv"
        rows, pkg = P.read_table(table)
        assert rows == m["inputs"]["files"] and pkg == m["inputs"]["package_sha256"]     # B4 layout (ruling 58c)
        assert list(rows) == ["event.json", "package/cpt_list.csv", "shakemap/grid.xml", "sites.csv"]
        assert card["task"]["division"] == "closed" and card["task"]["package_sha256"] == pkg
        assert card["forward"]["package_sha256"] == pkg and card["forward"]["package_table"] == "package_table.csv"
        assert card["task_sha256"] == Task.from_dict(card["task"]).sha256()
        assert P.check_files(table, run / "inputs" / t)["status"] == "PASS"      # the frozen inputs check out
        assert P.check_files(table, run / "inputs" / t)["not_in_table"] == []     # /inputs/ holds inputs only
    assert cards["A"]["task"]["package_sha256"] != cards["B"]["task"]["package_sha256"]   # own inputs per type


def test_score_on_an_open_task_names_no_package(setup):
    run, _, _ = _two_types(setup)
    card = F.score(run, setup["labels"], "A", {**TASK, "division": "open"})
    assert "package_sha256" not in card["task"] and card["forward"]["package_sha256"] is None
    assert not (run / "scores" / "A" / "package_table.csv").exists()

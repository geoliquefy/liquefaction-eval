"""D8 ruling 50 Elbistan PGA check (eval M5, 2026-11-24): inputs v2 withholds an NGL station PGA of the other doublet
event. Synthetic data only. No em dashes."""
from __future__ import annotations

import inspect
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scoreboard.evalkit import world_inputs as W

ROOT = Path(__file__).resolve().parents[1]
_OPENS: list[str] | None = None


def _hook(event, args):
    if _OPENS is not None and event == "open" and args and isinstance(args[0], (str, bytes, Path)):
        _OPENS.append(str(args[0] if not isinstance(args[0], bytes) else args[0].decode(errors="ignore")))


sys.addaudithook(_hook)


def _fixture(tmp_path: Path):
    d = tmp_path / "inputs"
    d.mkdir()
    n = 5
    cases = pd.DataFrame({
        "case_id": [f"c{i}" for i in range(n)], "source": ["NGL-API-2026-11-20"] * 4 + ["PRJ-6544"],
        "earthquake": ["Elbistan", "Elbistan", "Pazarcik", "Pazarcik", "Loma Prieta"], "mw": [7.5, 7.5, 7.8, 7.8, 6.9],
        "pga_primary": [0.70, 0.30, 0.60, np.nan, 0.2],
        "pga_primary_source": ["NGL GMIM PGA tag None"] * 3 + ["none", "PRJ-6544 USGS PGA"],
        "pga_ngl_gmim": [0.70, 0.30, 0.60, np.nan, np.nan], "pga_prj4022": [np.nan] * 5,
        "trace_row": [0, 0, 1, 1, -1],
        "lat": [37.0, 38.0, 37.2, 37.3, 37.0], "lon": [37.0, 37.1, 37.2, 37.3, -121.9],
    })
    cases.to_parquet(d / "cases.parquet", index=False)
    np.savez(d / "traces.npz", offsets=np.array([0, 3, 6]), depth_m=np.tile([1.0, 2.0, 3.0], 2),
             qc_kpa=np.full(6, 5000.0), fs_kpa=np.full(6, 50.0))
    man = {"schema": W.SCHEMA_INPUTS, "files": {"cases.parquet": W.sha_file(d / "cases.parquet"),
                                                "traces.npz": W.sha_file(d / "traces.npz")},
           "rules": {"pga_primary": "first rule"}}
    (d / W.INPUTS_MANIFEST_V1).write_text(json.dumps(man))
    lab = tmp_path / "labels-sealed"
    lab.mkdir()
    pd.DataFrame({"case_id": cases["case_id"], "observation_raw": ["1", "0", "1", "0", "1"]}).to_parquet(
        lab / "labels.parquet")
    return d, lab


def test_ngl_station_reads_the_method_string():
    assert W.ngl_station("Ground Motion Station TK-2718") == ("TK", "2718")
    assert W.ngl_station("Ground Motion Station TK-0.213") == ("TK", "0.213")
    assert W.ngl_station("Estimated based on nearby SGMS recordings") is None
    assert W.ngl_station(None) is None


def test_station_event_check_words():
    own, other = {("TK", "1"): 0.07}, {("TK", "2"): 0.715, ("TK", "3"): 0.59}
    assert W.station_event_check(0.07, ("TK", "1"), own, other) == "OWN EVENT LIST"
    assert W.station_event_check(0.70, ("TK", "2"), own, other) == "WRONG EVENT"
    assert W.station_event_check(0.07, ("TK", "3"), own, other) == "OTHER EVENT, VALUE DIFFERS"
    assert W.station_event_check(0.13, ("TK", "9"), own, other) == "IN NO LIST"
    assert W.station_event_check(0.3, None, own, other) == "NOT A STATION"
    assert W.station_event_check(0.60, ("TK", "2"), own, other) == "OTHER EVENT, VALUE DIFFERS"   # 16% off
    assert set(W.STATION_CHECK_WORDS) >= {"WRONG EVENT", "OWN EVENT LIST"}


def test_supersede_never_opens_the_labels_and_keeps_the_first_files(tmp_path):
    global _OPENS
    inp, lab = _fixture(tmp_path)
    before = {p.name: W.sha_file(p) for p in inp.iterdir()}
    lab_file = lab / "labels.parquet"
    modes = (lab_file.stat().st_mode, lab.stat().st_mode)
    os.chmod(lab_file, 0)
    os.chmod(lab, 0)
    _OPENS = []
    try:
        r = W.supersede_inputs(inp, {"c0": "WRONG EVENT"}, check={"n": 1}, kit_commit="test")
    finally:
        opened, _OPENS = _OPENS, None
        os.chmod(lab, stat.S_IMODE(modes[1]))
        os.chmod(lab_file, stat.S_IMODE(modes[0]))
    assert not any(str(lab) in p for p in opened), "the supersede step opened the sealed label folder"
    ours = [p for p in opened if str(tmp_path) in p]
    assert ours and all(p.startswith(str(inp)) for p in ours)
    assert {n: W.sha_file(inp / n) for n in before} == before            # first files byte for byte
    assert r["withheld"] == 1 and r["per_earthquake"] == {"Elbistan": 1}
    m = json.loads((inp / W.INPUTS_MANIFEST_V2).read_text())
    assert m["manifest_version"] == 2 and m["cases_file"] == W.CASES_V2_FILE
    assert m["supersedes"]["manifest"]["sha256"] == before[W.INPUTS_MANIFEST_V1]
    assert m["files"]["traces.npz"] == before["traces.npz"] and m["labels_opened"] is False
    assert m["rules"]["pga_primary"].startswith("first rule; v2: withheld")
    assert (inp / W.INPUTS_SUPERSEDED_NOTE).read_text().startswith(W.INPUTS_MANIFEST_V1)
    # the v2 inputs load through their own manifest; only the withheld value moved
    _, old, _ = W.load_inputs(inp)
    _, new, _ = W.load_inputs(inp, W.INPUTS_MANIFEST_V2)
    assert np.isnan(new.loc[0, "pga_primary"]) and new.loc[0, "pga_primary_source"] == W.WITHHELD_SOURCE
    assert new.loc[0, "pga_ngl_gmim"] == 0.70                            # the raw source value stays visible
    rest = [c for c in old.columns if c not in W.INPUTS_V2_CHANGEABLE]
    assert new[rest].equals(old[rest]) and new.iloc[1:].equals(old.iloc[1:])


def test_supersede_never_rewrites(tmp_path):
    inp, _ = _fixture(tmp_path)
    W.supersede_inputs(inp, {"c0": "WRONG EVENT"}, check={})
    sha = W.sha_file(inp / W.INPUTS_MANIFEST_V2)
    with pytest.raises(W.D8Refused, match="never rewritten"):
        W.supersede_inputs(inp, {"c0": "WRONG EVENT"}, check={})
    assert W.sha_file(inp / W.INPUTS_MANIFEST_V2) == sha


@pytest.mark.parametrize("withheld,msg", [
    ({}, "nothing to withhold"),
    ({"c0": "IN NO LIST"}, "only a WRONG EVENT"),
    ({"zz": "WRONG EVENT"}, "not in the inputs"),
    ({"c4": "WRONG EVENT"}, "only an NGL GMIM PGA"),
    ({"c3": "WRONG EVENT"}, "only an NGL GMIM PGA"),
])
def test_supersede_refusals_leave_no_file(tmp_path, withheld, msg):
    inp, _ = _fixture(tmp_path)
    before = sorted(p.name for p in inp.iterdir())
    with pytest.raises(W.D8Refused, match=msg):
        W.supersede_inputs(inp, withheld, check={})
    assert sorted(p.name for p in inp.iterdir()) == before


def test_supersede_refuses_a_changed_first_file(tmp_path):
    inp, _ = _fixture(tmp_path)
    with open(inp / "traces.npz", "ab") as f:
        f.write(b"x")
    with pytest.raises(W.D8Refused, match="differs from its frozen sha256"):
        W.supersede_inputs(inp, {"c0": "WRONG EVENT"}, check={})
    assert not (inp / W.INPUTS_MANIFEST_V2).exists()


def test_load_inputs_refuses_a_v2_manifest_whose_cases_file_has_no_hash(tmp_path):
    inp, _ = _fixture(tmp_path)
    m = json.loads((inp / W.INPUTS_MANIFEST_V1).read_text())
    m["cases_file"] = "other.parquet"
    (inp / "bad.json").write_text(json.dumps(m))
    with pytest.raises(W.D8Refused, match="no frozen sha256"):
        W.load_inputs(inp, "bad.json")


def test_driver_elbistan_step_names_no_label_path():
    src = (ROOT / "scripts" / "d8_world_inputs_2026_11_23.py").read_text()
    body = src.split("def cmd_elbistan_pga", 1)[1].split("\ndef ", 1)[0]
    assert "label" not in body.lower().replace("labels_opened", "")
    assert "FLDM" not in body.replace("no FLDM", "")
    code = inspect.getsource(W.supersede_inputs)
    assert "label" not in code.lower().replace("labels_opened", "")


# ----------------------------------------------------------------------------------------------------------------
# Review 1 fixes (wave 5, 2026-11-24)
# ----------------------------------------------------------------------------------------------------------------

def test_station_event_check_needs_every_list_to_agree():
    """N3: a station in two lists of the other event is WRONG EVENT only when the value matches both."""
    other = {("TK", "2"): [0.715, 0.715], ("TK", "5"): [0.715, 0.40]}
    assert W.station_event_check(0.70, ("TK", "2"), {}, other) == "WRONG EVENT"
    assert W.station_event_check(0.70, ("TK", "5"), {}, other) == "OTHER EVENT, VALUE DIFFERS"
    assert W.station_event_check(0.70, ("TK", "2"), {("TK", "2"): [0.70]}, other) == "OWN EVENT LIST"


def _pred_manifest(p: Path, inputs_sha: str) -> Path:
    p.write_text(json.dumps({"inputs_manifest_sha256": inputs_sha, "files": {}}))
    return p


def test_stamp_refuses_predictions_on_superseded_inputs(tmp_path):
    """S1: once inputs v2 supersede the first inputs, predictions built on the first ones cannot be stamped."""
    inp, _ = _fixture(tmp_path)
    old = _pred_manifest(tmp_path / "PREDICTIONS-MANIFEST-v2.json", W.sha_file(inp / W.INPUTS_MANIFEST_V1))
    calls = []
    assert W.stamp_plan(old, inp, runner=lambda *a, **k: calls.append(a))["ran"] is False    # not superseded yet
    W.supersede_inputs(inp, {"c0": "WRONG EVENT"}, check={})
    assert W.sha_file(inp / W.INPUTS_MANIFEST_V1) in W.superseded_inputs_sha256(inp)
    for run, go in ((False, ""), (True, "a founder go")):
        with pytest.raises(W.D8Refused, match="superseded"):
            W.stamp_plan(old, inp, run=run, go=go, runner=lambda *a, **k: calls.append(a))
    assert not calls
    new = _pred_manifest(tmp_path / "p-new.json", W.sha_file(inp / W.INPUTS_MANIFEST_V2))
    assert W.stamp_plan(new, inp)["ran"] is False
    with pytest.raises(W.D8Refused, match="names no inputs manifest"):
        W.stamp_plan(_pred_manifest(tmp_path / "p-none.json", ""), inp)
    with pytest.raises(W.D8Refused, match="is not in"):
        W.stamp_plan(_pred_manifest(tmp_path / "p-other.json", "0" * 64), inp)


def test_stamp_guard_reads_the_superseded_note_alone(tmp_path):
    """S1: the note SUPERSEDED-INPUTS.txt alone is enough to refuse (no v2 manifest needed)."""
    inp, _ = _fixture(tmp_path)
    h = W.sha_file(inp / W.INPUTS_MANIFEST_V1)
    (inp / W.INPUTS_SUPERSEDED_NOTE).write_text(f"{W.INPUTS_MANIFEST_V1} (sha256 {h}) is SUPERSEDED by x (sha256 "
                                                f"{'1' * 64}).\n")
    assert W.superseded_inputs_sha256(inp) == {h}
    with pytest.raises(W.D8Refused, match="superseded"):
        W.stamp_plan(_pred_manifest(tmp_path / "p.json", h), inp)


def test_predict_takes_a_manifest_and_refuses_a_superseded_one(tmp_path):
    """S1: predict runs on inputs v2 through its manifest, names it, and refuses the superseded first manifest."""
    global _OPENS
    inp, lab = _fixture(tmp_path)
    W.supersede_inputs(inp, {"c0": "WRONG EVENT"}, check={})
    with pytest.raises(W.D8Refused, match="superseded"):
        W.predict(inp, tmp_path / "pred-v1")
    assert not (tmp_path / "pred-v1").exists()
    _OPENS = []
    try:
        r = W.predict(inp, tmp_path / "pred", manifest=W.INPUTS_MANIFEST_V2)
    finally:
        opened, _OPENS = _OPENS, None
    assert not any(str(lab) in p for p in opened)
    pm = json.loads(Path(r["manifest"]).read_text())
    assert pm["inputs_manifest"] == W.INPUTS_MANIFEST_V2
    assert pm["inputs_manifest_sha256"] == W.sha_file(inp / W.INPUTS_MANIFEST_V2)
    txt = (tmp_path / "pred" / "baseline--shaking-score__pga_primary.csv").read_text().splitlines()
    assert txt[1] == "c0," and txt[2].split(",")[1] != ""        # the withheld case has no value; the next has one
    assert W.stamp_plan(r["manifest"], inp)["ran"] is False


# --- N1: the driver step end to end on a synthetic NGL API -------------------------------------------------------

def _load_driver():
    import importlib.util
    spec = importlib.util.spec_from_file_location("d8_driver_test", ROOT / "scripts" / "d8_world_inputs_2026_11_23.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _station_list(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"features": [
        {"properties": {"network": n, "code": c, "station_type": t, "pga": pga}} for n, c, t, pga in rows]}))


def test_driver_elbistan_step_end_to_end_on_a_synthetic_api(tmp_path, monkeypatch):
    D = _load_driver()
    api = tmp_path / "ngl" / "api"
    api.mkdir(parents=True)
    st = "Ground Motion Station "
    tables = {
        "EVNT": [{"EVNT_ID": "1", "EVNT_NM": "Pazarcik "}, {"EVNT_ID": "2", "EVNT_NM": "Elbistan"},
                 {"EVNT_ID": "3", "EVNT_NM": "Loma Prieta"}],
        "FLDO": [{"FLDO_ID": f"o{i}", "EVNT_ID": e, "SITE_ID": f"s{i}"}
                 for i, e in ((1, "2"), (2, "2"), (3, "1"), (4, "2"), (5, "2"), (6, "3"))],
        "TEST": [{"SITE_ID": f"s{i}", "TEST_ID": f"t{i}", "TEST_TYPE": "SCPG"} for i in (1, 2, 4, 5, 6)],
        "SCPG": [{"TEST_ID": f"t{i}", "SCPG_ID": f"g{i}"} for i in (1, 2, 4, 5, 6)],
        # file order differs from GMIM_ID order on purpose: the build rule takes the lowest GMIM_ID (PGA, g)
        "GMIM": [
            {"FLDO_ID": "o1", "GMIM_ID": "9", "GMIM_TYPE": "PGA", "GMIM_UNIT": "g", "GMIM_VALUE": "0.30",
             "GMIM_METHOD": "Estimated based on nearby SGMS recordings"},
            {"FLDO_ID": "o1", "GMIM_ID": "3", "GMIM_TYPE": "PGA", "GMIM_UNIT": "g", "GMIM_VALUE": "0.70",
             "GMIM_METHOD": st + "TK-2"},
            {"FLDO_ID": "o1", "GMIM_ID": "1", "GMIM_TYPE": "PGV", "GMIM_UNIT": "cm/s", "GMIM_VALUE": "50",
             "GMIM_METHOD": st + "TK-9"},
            {"FLDO_ID": "o2", "GMIM_ID": "10", "GMIM_TYPE": "PGA", "GMIM_UNIT": "g", "GMIM_VALUE": "0.70",
             "GMIM_METHOD": st + "TK-2"},
            {"FLDO_ID": "o2", "GMIM_ID": "4", "GMIM_TYPE": "PGA", "GMIM_UNIT": "g", "GMIM_VALUE": "0.30",
             "GMIM_METHOD": "Estimated based on nearby SGMS recordings"},
            {"FLDO_ID": "o3", "GMIM_ID": "5", "GMIM_TYPE": "PGA", "GMIM_UNIT": "g", "GMIM_VALUE": "0.70",
             "GMIM_METHOD": st + "TK-2"},
            {"FLDO_ID": "o4", "GMIM_ID": "6", "GMIM_TYPE": "PGA", "GMIM_UNIT": "g", "GMIM_VALUE": "0.07",
             "GMIM_METHOD": st + "TK-3"},
            {"FLDO_ID": "o5", "GMIM_ID": "7", "GMIM_TYPE": "PGA", "GMIM_UNIT": "g", "GMIM_VALUE": "0.70",
             "GMIM_METHOD": st + "TK-5"},
            {"FLDO_ID": "o6", "GMIM_ID": "8", "GMIM_TYPE": "PGA", "GMIM_UNIT": "g", "GMIM_VALUE": "0.70",
             "GMIM_METHOD": st + "TK-2"},
        ],
    }
    for t, rows in tables.items():
        (api / f"{t}_all.json").write_text(json.dumps(rows))
    data = tmp_path / "data"
    lists = {"Pazarcik": ["paz/v19.json", "paz/v17.json"], "Elbistan": ["elb/v12.json", "elb/v11.json"]}
    _station_list(data / "paz/v19.json", [("TK", "2", "seismic", 71.5), ("TK", "3", "seismic", 59.0),
                                          ("TK", "5", "seismic", 71.5)])
    _station_list(data / "paz/v17.json", [("TK", "2", "seismic", 71.5), ("TK", "5", "seismic", 40.0)])
    # a non-seismic TK-2 entry in the Elbistan list must not count as an own-list station
    _station_list(data / "elb/v12.json", [("TK", "9", "seismic", 20.0), ("TK", "2", "macroseismic", 70.0)])
    _station_list(data / "elb/v11.json", [("TK", "9", "seismic", 24.0)])
    ids = {k: W.id_hash("NGL", k) for k in ("o1|g1", "o2|g2", "o3|None", "o4|g4", "o5|g5")}
    out = tmp_path / "world"
    inp = out / "inputs"
    inp.mkdir(parents=True)
    n = len(ids)
    cases = pd.DataFrame({"case_id": list(ids.values()), "source": ["NGL-API-2026-11-20"] * n,
                          "earthquake": ["Elbistan", "Elbistan", "Pazarcik", "Elbistan", "Elbistan"],
                          "mw": [7.5, 7.5, 7.8, 7.5, 7.5], "pga_primary": [0.70, 0.30, 0.70, 0.07, 0.70],
                          "pga_primary_source": ["NGL GMIM PGA tag None"] * n,
                          "pga_ngl_gmim": [0.70, 0.30, 0.70, 0.07, 0.70], "trace_row": [-1] * n})
    cases.to_parquet(inp / "cases.parquet", index=False)
    np.savez(inp / "traces.npz", offsets=np.array([0]), depth_m=np.zeros(0), qc_kpa=np.zeros(0), fs_kpa=np.zeros(0))
    (inp / W.INPUTS_MANIFEST_V1).write_text(json.dumps({"schema": W.SCHEMA_INPUTS, "files": {
        "cases.parquet": W.sha_file(inp / "cases.parquet"), "traces.npz": W.sha_file(inp / "traces.npz")}}))
    agg = tmp_path / "agg.json"
    for name, v in (("NGL_API", tmp_path / "ngl"), ("DATA", data), ("OUT", out), ("AGG_ELBISTAN", agg),
                    ("DOUBLET_STATIONLISTS", lists), ("_kit_commit", lambda: "test")):
        monkeypatch.setattr(D, name, v)
    D.main(["elbistan-pga"])
    a = json.loads(agg.read_text())
    assert a["withheld"] == 1 and a["withheld_per_earthquake"] == {"Elbistan": 1}
    assert a["station_check"]["cases_by_word"] == {
        "Elbistan: NOT A STATION": 1, "Elbistan: OTHER EVENT, VALUE DIFFERS": 2, "Elbistan: WRONG EVENT": 1,
        "Pazarcik: OWN EVENT LIST": 1}
    v2 = pd.read_parquet(inp / W.CASES_V2_FILE)
    gone = set(v2.loc[v2["pga_primary"].isna(), "case_id"])
    assert gone == {ids["o1|g1"]}                                 # lowest GMIM_ID row, not file order, not the last
    text = agg.read_text()
    assert "TK" not in text and not any(c in text for c in ids.values())   # counts only: no station code, no case id

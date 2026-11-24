"""D8 predictions v3 on inputs v2 (eval M5, founder ruling 52a, 2026-11-24): every v2 model rerun on inputs v2, the
changed files under new names, the v1 and v2 manifests kept and named superseded. Synthetic data only. No em dashes."""
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
from scoreboard.evalkit.pgv import Grid

ROOT = Path(__file__).resolve().parents[1]
_OPENS: list[str] | None = None


def _hook(event, args):
    if _OPENS is not None and event == "open" and args and isinstance(args[0], (str, bytes, Path)):
        _OPENS.append(str(args[0] if not isinstance(args[0], bytes) else args[0].decode(errors="ignore")))


sys.addaudithook(_hook)

SHAKING = "baseline--shaking-score__pga_primary.csv"


def _grid():
    x, y = np.linspace(-122.5, -121.5, 11), np.linspace(36.5, 37.5, 11)
    model = np.full((11, 11), 0.01)
    model[8:, :] = 0.5
    model[2, 6] = np.nan
    return {"x": x, "y": y, "sha256": "0" * 64, "path": "synthetic", "model": model,
            "row_order": W.ZHU_REFREEZE_ROW_ORDER, "orientation": {"row0": "north", "source": "test"}}


def _shakemap():
    pga = np.full((11, 11), 30.0)
    pga[2, 6] = 5.0
    return Grid("synthetic", 6.9, "1", -122.5, 36.5, 0.1, 0.1, 11, 11,
                {"PGA": pga, "PGV": np.full((11, 11), 20.0), "SVEL": np.full((11, 11), 300.0)}, {"PGV": "cms"},
                "1" * 64)


def _world(tmp_path: Path):
    """Inputs v1, predictions v1 (first freeze), predictions v2 (Zhu refreeze), then inputs v2 (c0 withheld)."""
    inp = tmp_path / "inputs"
    inp.mkdir()
    n = 6
    cases = pd.DataFrame({
        "case_id": [f"c{i}" for i in range(n)],
        "source": ["NGL-API-2026-11-20"] * 3 + ["PRJ-6544"] * 3,
        "earthquake": ["Elbistan", "Elbistan", "Pazarcik", "Loma Prieta", "Loma Prieta", "Loma Prieta"],
        "mw": [7.5, 7.5, 7.8, 6.9, 6.9, 6.9],
        "pga_primary": [0.70, 0.30, 0.60, 0.2, 0.3, 0.4],
        "pga_primary_source": ["NGL GMIM PGA tag None"] * 3 + ["PRJ-6544 USGS PGA"] * 3,
        "pga_ngl_gmim": [0.70, 0.30, 0.60, np.nan, np.nan, np.nan],
        "pga_prj4022": [np.nan, np.nan, np.nan, 0.25, np.nan, 0.35],
        "gwt_m": [np.nan] * n, "trace_row": [-1] * n,
        "lat": [38.2, 38.3, 37.2, 36.7, 37.4, 36.6], "lon": [37.2, 37.3, 37.0, -121.9, -121.9, -121.9],
    })
    cases.to_parquet(inp / "cases.parquet", index=False)
    np.savez(inp / "traces.npz", offsets=np.array([0]), depth_m=np.zeros(0), qc_kpa=np.zeros(0), fs_kpa=np.zeros(0))
    (inp / W.INPUTS_MANIFEST_V1).write_text(json.dumps({"schema": W.SCHEMA_INPUTS, "files": {
        "cases.parquet": W.sha_file(inp / "cases.parquet"), "traces.npz": W.sha_file(inp / "traces.npz")}}))
    lab = tmp_path / "labels-sealed"
    lab.mkdir()
    pd.DataFrame({"case_id": cases["case_id"], "observation_raw": ["1", "0"] * 3}).to_parquet(lab / "labels.parquet")
    pred = tmp_path / "pred"
    as_stored = {**_grid(), "model": _grid()["model"][::-1]}
    as_stored.pop("row_order")
    W.predict(inp, pred, zhu_grids={"Loma Prieta": as_stored})
    W.refreeze_zhu(inp, pred, {"Loma Prieta": _grid()}, {"Loma Prieta": _shakemap()})
    W.supersede_inputs(inp, {"c0": "WRONG EVENT"}, check={})
    return inp, pred, lab


def _run(inp, pred, **kw):
    return W.predict_v3(inp, pred, zhu_grids={"Loma Prieta": _grid()}, shakemaps={"Loma Prieta": _shakemap()},
                        kit_commit="test", **kw)


def test_predict_v3_never_opens_the_labels_and_keeps_every_earlier_file(tmp_path):
    global _OPENS
    inp, pred, lab = _world(tmp_path)
    before = {p.name: W.sha_file(p) for p in pred.iterdir()}
    lab_file = lab / "labels.parquet"
    modes = (lab_file.stat().st_mode, lab.stat().st_mode)
    os.chmod(lab_file, 0)
    os.chmod(lab, 0)
    _OPENS = []
    try:
        r = _run(inp, pred)
    finally:
        opened, _OPENS = _OPENS, None
        os.chmod(lab, stat.S_IMODE(modes[1]))
        os.chmod(lab_file, stat.S_IMODE(modes[0]))
    assert not any(str(lab) in p for p in opened), "predictions v3 opened the sealed label folder"
    ours = [p for p in opened if str(tmp_path) in p]
    assert ours and all(p.startswith(str(inp)) or p.startswith(str(pred)) for p in ours)
    assert {n: W.sha_file(pred / n) for n in before} == before                   # every earlier file byte for byte
    # only the withheld case moved, and only in the pga_primary file that has a value for it (P=None: shaking only)
    assert r["changed_files"] == [SHAKING]
    assert r["per_earthquake"] == {"Elbistan": {"values_changed": 1, "value_to_empty": 1, "empty_to_value": 0,
                                                "value_to_other_value": 0, "reason_changed": 0, "files_changed": 1}}
    assert r["cases_changed"] == 1 and r["inputs_changed"]["columns"] == ["pga_primary", "pga_primary_source"]
    m = json.loads(Path(r["manifest"]).read_text())
    assert m["manifest_version"] == 3 and m["scored"] is False and m["labels_opened"] is False
    assert m["inputs_manifest"] == W.INPUTS_MANIFEST_V2
    assert m["inputs_manifest_sha256"] == W.sha_file(inp / W.INPUTS_MANIFEST_V2)
    assert m["supersedes"]["manifests"] == {W.PREDICTIONS_MANIFEST_V1: before[W.PREDICTIONS_MANIFEST_V1],
                                            W.PREDICTIONS_MANIFEST_V2: before[W.PREDICTIONS_MANIFEST_V2]}
    new = W.v3_file_name(SHAKING)
    assert SHAKING not in m["files"] and new in m["files"] and m["files"][new]["replaces"] == SHAKING
    assert m["supersedes"]["files"][SHAKING]["replaced_by"] == new
    m2 = json.loads((pred / W.PREDICTIONS_MANIFEST_V2).read_text())
    assert len(m["files"]) == len(m2["files"])
    assert sorted(m["supersedes"]["carried_over_unchanged"]) == sorted(set(m2["files"]) - {SHAKING})
    for name, row in m["files"].items():
        assert W.sha_file(pred / name) == row["sha256"]
    lines = (pred / new).read_text().splitlines()
    old = (pred / SHAKING).read_text().splitlines()
    assert lines[1] == "c0," and old[1] != "c0," and lines[2:] == old[2:]
    assert m["files"][new]["n_values"] == m2["files"][SHAKING]["n_values"] - 1
    note = (pred / W.SUPERSEDED_NOTE_V3).read_text()
    assert note.startswith(W.PREDICTIONS_MANIFEST_V1) and SHAKING in note
    assert W.SUPERSEDED_NOTE in {p.name for p in pred.iterdir()}                  # the v2 note stays as it was


def test_stamp_plan_takes_v3_and_refuses_v1_and_v2(tmp_path):
    inp, pred, _ = _world(tmp_path)
    calls = []
    run = lambda *a, **k: calls.append(a)  # noqa: E731
    _run(inp, pred)
    sup = W.superseded_predictions_sha256(pred)
    assert W.sha_file(pred / W.PREDICTIONS_MANIFEST_V1) in sup and W.sha_file(pred / W.PREDICTIONS_MANIFEST_V2) in sup
    assert W.sha_file(pred / W.PREDICTIONS_MANIFEST_V3) not in sup
    plan = W.stamp_plan(pred / W.PREDICTIONS_MANIFEST_V3, inp, runner=run)
    assert plan["ran"] is False and plan["command"][-1].endswith(W.PREDICTIONS_MANIFEST_V3)
    for old in (W.PREDICTIONS_MANIFEST_V1, W.PREDICTIONS_MANIFEST_V2):
        for r, go in ((False, ""), (True, "a founder go")):
            with pytest.raises(W.D8Refused, match="superseded"):
                W.stamp_plan(pred / old, inp, run=r, go=go, runner=run)
    assert not calls


def test_stamp_refuses_a_superseded_predictions_manifest_even_on_current_inputs(tmp_path):
    """The predictions guard stands on its own: a manifest named superseded in its folder is refused although its
    inputs manifest is current."""
    inp, pred, _ = _world(tmp_path)
    p = pred / "PREDICTIONS-MANIFEST-x.json"
    p.write_text(json.dumps({"inputs_manifest_sha256": W.sha_file(inp / W.INPUTS_MANIFEST_V2), "files": {}}))
    assert W.stamp_plan(p, inp)["ran"] is False
    (pred / "SUPERSEDED-x.txt").write_text(f"x (sha256 {W.sha_file(p)}) is SUPERSEDED by y (sha256 {'2' * 64}).\n")
    with pytest.raises(W.D8Refused, match="superseded by a later predictions manifest"):
        W.stamp_plan(p, inp)


def _assert_no_v3_file(pred):
    assert not [p.name for p in pred.iterdir() if W.PRED_V3_TAG in p.name or p.name in (
        W.PREDICTIONS_MANIFEST_V3, W.SUPERSEDED_NOTE_V3)]


def test_predict_v3_never_rewrites(tmp_path):
    inp, pred, _ = _world(tmp_path)
    r = _run(inp, pred)
    with pytest.raises(W.D8Refused, match="never rewritten"):
        _run(inp, pred)
    assert W.sha_file(pred / W.PREDICTIONS_MANIFEST_V3) == r["sha256"]


def test_predict_v3_refuses_a_change_outside_the_moved_inputs_and_writes_nothing(tmp_path, monkeypatch):
    """Any value that moves on a case whose model inputs did not move is a finding: refused before any write."""
    inp, pred, _ = _world(tmp_path)
    real = W.run_models

    def drift(cases, traces, pga_col, **k):
        out = real(cases, traces, pga_col, **k)
        out["baseline: shaking score"] = out["baseline: shaking score"] + 1e-9
        return out
    monkeypatch.setattr(W, "run_models", drift)
    with pytest.raises(W.D8Refused, match="a finding"):
        _run(inp, pred)
    _assert_no_v3_file(pred)
    monkeypatch.setattr(W, "run_models", real)
    assert Path(_run(inp, pred)["manifest"]).exists()


def test_predict_v3_input_columns_are_keyed_per_model():
    """Review 1, N1: the guard map is pinned, and it is keyed per model, not per PGA variant."""
    assert W.PRED_V3_INPUT_COLUMNS == {"shaking": ("{pga}", "mw"), "cpt": ("{pga}", "mw", "gwt_m", "trace_row"),
                                       "zhu": ("earthquake", "lat", "lon")}
    for var in W.PGA_VARIANTS:
        assert W.pred_v3_input_columns("baseline: shaking score", var) == (var, "mw")
        for m in W.PGA_MODELS:
            if m != "baseline: shaking score":
                assert W.pred_v3_input_columns(m, var) == (var, "mw", "gwt_m", "trace_row")
    assert W.pred_v3_input_columns(W.ZHU_MODEL, "usgs-grid-georef-2026-11-23") == ("earthquake", "lat", "lon")
    with pytest.raises(W.D8Refused, match="no input column list"):
        W.pred_v3_input_columns("a model the kit does not know", "pga_primary")
    with pytest.raises(W.D8Refused, match="no input column list"):
        W.pred_v3_input_columns("Level 0", "pga_other")


def test_predict_v3_refuses_a_shaking_change_where_only_gwt_moved(tmp_path, monkeypatch):
    """Review 1, N1: a move in gwt_m alone must not excuse a shaking score change (the shaking score reads PGA, Mw)."""
    inp, pred, _ = _world(tmp_path)
    real_load, real_run = W.load_inputs, W.run_models

    def load(d, manifest=W.INPUTS_MANIFEST_V1):
        man, cases, traces = real_load(d, manifest)
        if manifest == W.INPUTS_MANIFEST_V1:              # inputs v1 seen with another water table on c1 only
            cases = cases.copy()
            cases.loc[cases["case_id"] == "c1", "gwt_m"] = 1.5
        return man, cases, traces

    def drift(cases, traces, pga_col, **k):
        out = real_run(cases, traces, pga_col, **k)
        out["baseline: shaking score"][1] += 1e-3        # c1: its PGA and Mw did not move
        return out
    monkeypatch.setattr(W, "load_inputs", load)
    monkeypatch.setattr(W, "run_models", drift)
    with pytest.raises(W.D8Refused, match="a finding"):
        _run(inp, pred)
    _assert_no_v3_file(pred)


def test_predict_v3_refuses_a_zhu_change_where_only_the_pga_moved(tmp_path, monkeypatch):
    """Review 1, N1 (plant P5): Zhu reads earthquake, lat and lon, so a PGA move must not excuse a Zhu change."""
    inp, pred, _ = _world(tmp_path)
    real_zhu, real_ruled = W.run_zhu, W.zhu_ruling46_text
    seen = {}

    def zhu(cases, grids):
        zv, rec = real_zhu(cases, grids)
        seen["zv"] = zv.copy()
        zv = zv.copy()
        zv[0] = 0.25                                   # c0: only its PGA moved between inputs v1 and v2
        return zv, rec

    def ruled(cases, grids, shakemaps, zv):
        return real_ruled(cases, grids, shakemaps, seen["zv"])
    monkeypatch.setattr(W, "run_zhu", zhu)
    monkeypatch.setattr(W, "zhu_ruling46_text", ruled)
    with pytest.raises(W.D8Refused, match="a finding"):
        _run(inp, pred)
    _assert_no_v3_file(pred)


def test_predict_v3_refuses_changed_v2_files_options_and_grids(tmp_path, monkeypatch):
    inp, pred, _ = _world(tmp_path)
    with pytest.raises(W.D8Refused, match="Zhu grids and their ShakeMaps"):
        W.predict_v3(inp, pred, zhu_grids={"Loma Prieta": _grid()})
    with pytest.raises(W.D8Refused, match="georef"):
        W.predict_v3(inp, pred, zhu_grids={"Loma Prieta": {**_grid(), "row_order": None}},
                     shakemaps={"Loma Prieta": _shakemap()})
    with pytest.raises(W.D8Refused, match="no ShakeMap"):
        W.predict_v3(inp, pred, zhu_grids={"Loma Prieta": _grid()}, shakemaps={})
    monkeypatch.setitem(W.MODELS["baseline: shaking score"], "rule", "another rule")
    with pytest.raises(W.D8Refused, match="options differ"):
        _run(inp, pred)
    monkeypatch.undo()
    p = pred / "level-0__pga_prj4022.csv"
    p.write_text(p.read_text() + "x,1\n")
    with pytest.raises(W.D8Refused, match="differs from"):
        _run(inp, pred)
    _assert_no_v3_file(pred)


def test_predict_v3_needs_inputs_v2_over_the_inputs_of_predictions_v2(tmp_path):
    inp, pred, _ = _world(tmp_path)
    (inp / W.INPUTS_MANIFEST_V2).rename(inp / "moved.json")
    with pytest.raises(W.D8Refused, match="no INPUTS-MANIFEST-v2.json"):
        _run(inp, pred)
    _assert_no_v3_file(pred)


def test_superseded_predictions_ignores_the_inputs_note(tmp_path):
    (tmp_path / "SUPERSEDED-INPUTS.txt").write_text(f"x (sha256 {'3' * 64}) is SUPERSEDED by y.\n")
    assert W.superseded_predictions_sha256(tmp_path) == set()


def test_driver_predict_v3_step_names_no_label_path():
    src = (ROOT / "scripts" / "d8_world_inputs_2026_11_23.py").read_text()
    body = src.split("def cmd_predict_v3", 1)[1].split("\ndef ", 1)[0]
    assert "label" not in body.lower().replace("labels_opened", "")
    code = inspect.getsource(W.predict_v3)
    assert "label" not in code.lower().replace("labels_opened", "")


def test_driver_stamp_plan_names_v3_first():
    src = (ROOT / "scripts" / "d8_world_inputs_2026_11_23.py").read_text()
    body = src.split("def cmd_stamp", 1)[1].split("\ndef ", 1)[0]
    assert (body.index("PREDICTIONS_MANIFEST_V3") < body.index("PREDICTIONS_MANIFEST_V2")
            < body.index("PREDICTIONS_MANIFEST_V1"))

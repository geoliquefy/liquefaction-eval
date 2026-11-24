"""D8 Zhu refreeze (eval M5, 2026-11-23): only USGS Zhu 2017 is rerun, under new names, beside the first files.
Synthetic data only. No em dashes."""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scoreboard.evalkit import world_inputs as W
from scoreboard.evalkit import zhu_cells as Z
from scoreboard.evalkit.pgv import Grid

ROOT = Path(__file__).resolve().parents[1]
_OPENS: list[str] | None = None


def _hook(event, args):
    if _OPENS is not None and event == "open" and args and isinstance(args[0], (str, bytes, Path)):
        _OPENS.append(str(args[0] if not isinstance(args[0], bytes) else args[0].decode(errors="ignore")))


sys.addaudithook(_hook)


def _fixture(tmp_path: Path):
    d = tmp_path / "inputs"
    d.mkdir()
    n = 6
    cases = pd.DataFrame({
        "case_id": [f"c{i}" for i in range(n)], "source": ["PRJ-6544"] * n, "earthquake": ["Loma Prieta"] * n,
        "mw": [6.9] * n, "pga_primary": np.linspace(0.1, 0.6, n), "pga_prj4022": [np.nan] * n,
        "gwt_m": [np.nan] * n, "trace_row": [-1] * n,
        "lat": [36.6, 36.7, 37.4, 37.3, np.nan, 37.2], "lon": [-121.9] * n,
    })
    cases.to_parquet(d / "cases.parquet", index=False)
    np.savez(d / "traces.npz", offsets=np.array([0]), depth_m=np.zeros(0), qc_kpa=np.zeros(0), fs_kpa=np.zeros(0))
    man = {"schema": W.SCHEMA_INPUTS, "files": {"cases.parquet": W.sha_file(d / "cases.parquet"),
                                                "traces.npz": W.sha_file(d / "traces.npz")}}
    (d / "INPUTS-MANIFEST.json").write_text(json.dumps(man))
    lab = tmp_path / "labels-sealed"
    lab.mkdir()
    pd.DataFrame({"case_id": cases["case_id"], "observation_raw": ["1", "0"] * 3}).to_parquet(lab / "labels.parquet")
    return d, lab


def _grid(row_order):
    """A north-band hazard grid: rows aligned to ascending y. Stored as the USGS writer does, row 0 is north."""
    x, y = np.linspace(-122.5, -121.5, 11), np.linspace(36.5, 37.5, 11)
    model = np.full((11, 11), 0.01)
    model[8:, :] = 0.5                         # north band (lat >= 37.3)
    model[9, 6] = np.nan                       # empty cell at lat 37.4, lon -121.9
    model[2, 6] = np.nan                       # empty cell at lat 36.7, lon -121.9
    g = {"x": x, "y": y, "sha256": "0" * 64, "path": "synthetic"}
    if row_order == W.ZHU_REFREEZE_ROW_ORDER:
        return {**g, "model": model, "row_order": row_order, "orientation": {"row0": "north", "source": "test"}}
    return {**g, "model": model[::-1]}         # the first-slice read: rows as stored, mirrored


def _shakemap():
    nlon, nlat = 11, 11
    pga = np.full((nlat, nlon), 30.0)          # %g
    pgv = np.full((nlat, nlon), 20.0)
    vs = np.full((nlat, nlon), 300.0)
    pga[2, 6] = 5.0                            # the empty cell at 36.7 fails PGA < 0.10 g: MODEL ZERO
    return Grid("synthetic", 6.9, "1", -122.5, 36.5, 0.1, 0.1, nlon, nlat, {"PGA": pga, "PGV": pgv, "SVEL": vs},
                {"PGV": "cms"}, "1" * 64)


def _first_freeze(tmp_path, inp):
    pred = tmp_path / "pred"
    W.predict(inp, pred, zhu_grids={"Loma Prieta": _grid(None)})
    return pred


def test_refreeze_never_opens_the_labels_and_leaves_the_first_files(tmp_path, monkeypatch):
    global _OPENS
    inp, lab = _fixture(tmp_path)
    pred = _first_freeze(tmp_path, inp)
    before = {p.name: W.sha_file(p) for p in pred.iterdir()}
    lab_file = lab / "labels.parquet"
    modes = (lab_file.stat().st_mode, lab.stat().st_mode)
    os.chmod(lab_file, 0)
    os.chmod(lab, 0)
    _OPENS = []
    try:
        r = W.refreeze_zhu(inp, pred, {"Loma Prieta": _grid(W.ZHU_REFREEZE_ROW_ORDER)},
                           {"Loma Prieta": _shakemap()}, kit_commit="test")
    finally:
        opened, _OPENS = _OPENS, None
        os.chmod(lab, stat.S_IMODE(modes[1]))
        os.chmod(lab_file, stat.S_IMODE(modes[0]))
    assert not any(str(lab) in p for p in opened), "the refreeze opened the sealed label folder"
    ours = [p for p in opened if str(tmp_path) in p]
    assert ours and all(p.startswith(str(inp)) or p.startswith(str(pred)) for p in ours)
    # the first manifest and every first file are byte for byte as they were
    assert {n: W.sha_file(pred / n) for n in before} == before
    m = json.loads(Path(r["manifest"]).read_text())
    assert m["manifest_version"] == 2 and m["scored"] is False and m["labels_opened"] is False
    assert m["supersedes"]["manifest"]["sha256"] == before[W.PREDICTIONS_MANIFEST_V1]
    assert W.ZHU_V1_FILE in m["supersedes"]["files"] and W.ZHU_V1_FILE not in m["files"]
    for name, row in m["files"].items():
        assert W.sha_file(pred / name) == row["sha256"]
    assert len(m["supersedes"]["carried_over_unchanged"]) == len(W.PGA_MODELS) * len(W.PGA_VARIANTS)
    assert (pred / W.SUPERSEDED_NOTE).read_text().startswith(W.PREDICTIONS_MANIFEST_V1)
    # values: the mirrored read put the north band in the south
    new = (pred / W.ZHU_REFREEZE_FILE).read_text().splitlines()
    assert new[1] == "c0,0.01" and new[3] == "c2," and new[4] == "c3,0.5" and new[5] == "c4," and new[6] == "c5,0.01"
    c = r["compare"]["Loma Prieta"]
    assert c["cases"] == 6 and c["values_refrozen"] == 3
    # ruling 46 split, label free
    ruled = (pred / W.ZHU_RULED_FILE).read_text().splitlines()
    assert ruled[0] == "case_id,value,reason"
    assert ruled[2] == f"c1,0.0,{Z.MODEL_ZERO}" and ruled[3] == f"c2,,{Z.OFF_GRID}"
    assert ruled[5] == "c4,,NO COORDINATES"
    s = r["ruling46"]["Loma Prieta"]
    assert (s[Z.ON_GRID_VALUE], s[Z.MODEL_ZERO], s[Z.OFF_GRID], s[Z.NOT_CHECKABLE]) == (3, 1, 1, 0)
    assert s["model_zero_by_limit"] == {"pga": 1, "pgv": 0, "vs30": 0}


def test_refreeze_never_rewrites_a_file(tmp_path):
    inp, _ = _fixture(tmp_path)
    pred = _first_freeze(tmp_path, inp)
    g = {"Loma Prieta": _grid(W.ZHU_REFREEZE_ROW_ORDER)}
    W.refreeze_zhu(inp, pred, g, None)
    with pytest.raises(W.D8Refused, match="never rewrites"):
        W.refreeze_zhu(inp, pred, g, None)


def test_refreeze_refuses_an_as_stored_grid(tmp_path):
    inp, _ = _fixture(tmp_path)
    pred = _first_freeze(tmp_path, inp)
    with pytest.raises(W.D8Refused, match="georef"):
        W.refreeze_zhu(inp, pred, {"Loma Prieta": _grid(None)}, None)
    _assert_no_new_file(pred)


def test_refreeze_stops_when_a_first_file_changed(tmp_path):
    inp, _ = _fixture(tmp_path)
    pred = _first_freeze(tmp_path, inp)
    p = pred / "level-0__pga_primary.csv"
    p.write_text(p.read_text() + "x,1\n")
    with pytest.raises(W.D8Refused, match="differs from the first manifest"):
        W.refreeze_zhu(inp, pred, {"Loma Prieta": _grid(W.ZHU_REFREEZE_ROW_ORDER)}, None)
    assert not (pred / W.ZHU_REFREEZE_FILE).exists()


def test_refreeze_needs_a_shakemap_per_grid_for_the_split(tmp_path):
    inp, _ = _fixture(tmp_path)
    pred = _first_freeze(tmp_path, inp)
    with pytest.raises(W.D8Refused, match="no ShakeMap"):
        W.refreeze_zhu(inp, pred, {"Loma Prieta": _grid(W.ZHU_REFREEZE_ROW_ORDER)}, {})
    _assert_no_new_file(pred)


def _assert_no_new_file(pred):
    for name in (W.ZHU_REFREEZE_FILE, W.ZHU_RULED_FILE, W.PREDICTIONS_MANIFEST_V2, W.SUPERSEDED_NOTE):
        assert not (pred / name).exists(), name


def test_refreeze_split_refusal_leaves_no_file_and_a_rerun_works(tmp_path, monkeypatch):
    """Review 1 nit (wave 4): every check runs before the first write, so a refusal never blocks the next run."""
    inp, _ = _fixture(tmp_path)
    pred = _first_freeze(tmp_path, inp)
    g = {"Loma Prieta": _grid(W.ZHU_REFREEZE_ROW_ORDER)}
    real = Z.split_by_reason

    def wrong(*a, **k):
        v, why = real(*a, **k)
        return v + 1.0, why
    monkeypatch.setattr(Z, "split_by_reason", wrong)
    with pytest.raises(W.D8Refused, match="split disagrees"):
        W.refreeze_zhu(inp, pred, g, {"Loma Prieta": _shakemap()})
    _assert_no_new_file(pred)
    monkeypatch.setattr(Z, "split_by_reason", real)
    r = W.refreeze_zhu(inp, pred, g, {"Loma Prieta": _shakemap()})
    assert Path(r["manifest"]).exists()


def test_split_by_reason_has_the_masks_of_cell_input_ruled():
    gx, gy = np.array([172.0, 172.1, 172.2]), np.array([-43.6, -43.5])
    m = np.array([[0.2, np.nan, 0.4], [np.nan, 0.6, np.nan]])
    lon = np.array([172.0, 172.1, 172.0, 172.2, 172.2, 175.0])
    lat = np.array([-43.6, -43.6, -43.5, -43.5, -43.5, -43.5])
    pga = np.array([0.3, 0.05, 0.3, 0.3, 0.3, 0.3])
    pgv = np.array([20.0, 20.0, 20.0, 20.0, np.nan, 20.0])
    vs = np.array([200.0, 200.0, 700.0, 200.0, 200.0, 200.0])
    y = np.array([1, 0, 0, 1, 0, 1])
    c = Z.cell_input_ruled(gx, gy, m, lon, lat, y, pga, pgv, vs)
    v, why = Z.split_by_reason(gx, gy, m, lon, lat, pga, pgv, vs)
    keep = (why == Z.ON_GRID_VALUE) | (why == Z.MODEL_ZERO)
    assert v[keep].tolist() == c.values.tolist() and y[keep].tolist() == c.label.tolist()
    assert int((why == Z.MODEL_ZERO).sum()) == c.rec["n_model_zero"]
    assert int((why == Z.OFF_GRID).sum()) == c.rec["n_off_grid"]
    assert int((why == Z.NOT_CHECKABLE).sum()) == c.rec["n_not_checkable"]
    assert np.isnan(v[~keep]).all()


def test_driver_refreeze_step_names_no_label_path():
    src = (ROOT / "scripts" / "d8_world_inputs_2026_11_23.py").read_text()
    body = src.split("def cmd_refreeze_zhu", 1)[1].split("\ndef ", 1)[0]
    assert "label" not in body.lower().replace("labels_opened", "")
    import inspect
    assert "label" not in inspect.getsource(W.refreeze_zhu).lower().replace("labels_opened", "")
    code = inspect.getsource(Z.split_by_reason).split('"""')[2]     # the body after the docstring
    assert "label" not in code.lower() and "label" not in inspect.signature(Z.split_by_reason).parameters


def test_driver_stamp_plan_names_the_v2_manifest_first():
    src = (ROOT / "scripts" / "d8_world_inputs_2026_11_23.py").read_text()
    body = src.split("def cmd_stamp", 1)[1].split("\ndef ", 1)[0]
    assert body.index("PREDICTIONS_MANIFEST_V2") < body.index("PREDICTIONS_MANIFEST_V1")

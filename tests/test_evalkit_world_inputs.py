"""D8 first slice (eval M5): world set v2 inputs and frozen predictions. Synthetic data only. No em dashes."""
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


def _fixture(tmp_path: Path, extra_cols: dict | None = None) -> tuple[Path, Path]:
    d = tmp_path / "inputs"
    d.mkdir()
    n = 6
    cases = pd.DataFrame({
        "case_id": [f"c{i}" for i in range(n)], "source": ["PRJ-6544"] * n, "earthquake": ["Loma Prieta"] * n,
        "mw": [6.9] * n, "pga_primary": np.linspace(0.1, 0.6, n), "pga_prj4022": [np.nan] * (n - 2) + [0.2, 0.3],
        "gwt_m": [1.0, 2.0] + [np.nan] * (n - 2), "trace_row": [0, 1] + [-1] * (n - 2),
        "lat": [37.0, 37.1, 37.2, 37.3, np.nan, 37.0], "lon": [-121.9] * n,
    })
    for k, v in (extra_cols or {}).items():
        cases[k] = v
    cases.to_parquet(d / "cases.parquet", index=False)
    np.savez(d / "traces.npz", offsets=np.array([0, 3, 6]), depth_m=np.tile([1.0, 2.0, 3.0], 2),
             qc_kpa=np.full(6, 5000.0), fs_kpa=np.full(6, 50.0))
    man = {"schema": W.SCHEMA_INPUTS, "files": {"cases.parquet": W.sha_file(d / "cases.parquet"),
                                                "traces.npz": W.sha_file(d / "traces.npz")}}
    (d / "INPUTS-MANIFEST.json").write_text(json.dumps(man))
    lab = tmp_path / "labels-sealed"
    lab.mkdir()
    pd.DataFrame({"case_id": cases["case_id"], "observation_raw": ["1", "0"] * 3}).to_parquet(lab / "labels.parquet")
    return d, lab


def _fake_helpers(monkeypatch):
    """Stand-ins for the pinned procedures, so the CPT branch of run_models runs under the watch (review 1 fix)."""
    from scoreboard.evalkit import chain_settings as CS
    monkeypatch.setattr(CS, "run", lambda P, Cq, Z, qc, fs, g, a, m, sid: {"lpi": a * 10.0, "lsn": a * 20.0})
    return {"P": object(), "Cq": object(), "Z": np.array([1.0, 2.0, 3.0]),
            "to_grid": lambda d, qc, fs: (np.asarray(qc, float), np.asarray(fs, float)),
            "lnk_star": lambda P, Z, qc, fs, i, g, a, m: np.log(a)}


def _fake_zhu():
    x, y = np.linspace(-122.5, -121.5, 11), np.linspace(36.5, 37.5, 11)
    model = np.full((11, 11), 0.2)
    model[7, 6] = np.nan                       # the cell nearest lat 37.2, lon -121.9 is empty
    return {"Loma Prieta": {"x": x, "y": y, "model": model, "sha256": "0" * 64, "path": "synthetic"}}


def _watch_readers(monkeypatch, seen: list):
    """Record every path handed to the readers that can bypass the Python 'open' audit event (C++ file opens)."""
    import pyarrow.parquet as pq

    def wrap(mod, name):
        f = getattr(mod, name)

        def w(src, *a, **k):
            seen.append(str(src))
            return f(src, *a, **k)
        monkeypatch.setattr(mod, name, w)
    for mod, name in ((pd, "read_parquet"), (pd, "read_csv"), (pq, "read_table"), (pq, "ParquetFile"),
                      (np, "load"), (np, "loadtxt"), (np, "genfromtxt")):
        wrap(mod, name)


def test_predict_never_opens_the_label_file(tmp_path, monkeypatch):
    global _OPENS
    inp, lab = _fixture(tmp_path)
    helpers = _fake_helpers(monkeypatch)
    seen: list[str] = []
    _watch_readers(monkeypatch, seen)
    cols: list[set] = []
    real_run = W.run_models
    monkeypatch.setattr(W, "run_models", lambda cases, *a, **k: (cols.append(set(cases.columns)),
                                                                 real_run(cases, *a, **k))[1])
    lab_file = lab / "labels.parquet"
    modes = (lab_file.stat().st_mode, lab.stat().st_mode)
    os.chmod(lab_file, 0)                      # any open of the label file now raises, C++ readers included
    os.chmod(lab, 0)
    _OPENS = []
    try:
        r = W.predict(inp, tmp_path / "pred", helpers=helpers, zhu_grids=_fake_zhu())
    finally:
        opened, _OPENS = _OPENS, None
        os.chmod(lab, stat.S_IMODE(modes[1]))
        os.chmod(lab_file, stat.S_IMODE(modes[0]))
    assert not any(str(lab) in p for p in opened + seen), "the prediction step opened the sealed label folder"
    ours = [p for p in opened + seen if str(tmp_path) in p]
    assert ours and all(p.startswith(str(inp)) or p.startswith(str(tmp_path / "pred")) for p in ours)
    assert cols and all(c <= W.INPUT_ALLOWED_COLUMNS for c in cols), "run_models saw a column outside the allowlist"
    m = json.loads(Path(r["manifest"]).read_text())
    assert m["scored"] is False and m["labels_opened"] is False
    for name, row in m["files"].items():
        assert W.sha_file(tmp_path / "pred" / name) == row["sha256"]
    s = m["files"]["baseline--shaking-score__pga_primary.csv"]
    assert s["n_values"] == 6
    assert m["files"]["baseline--shaking-score__pga_prj4022.csv"]["n_values"] == 2
    assert m["files"]["level-0__pga_primary.csv"]["n_values"] == 2   # the CPT branch ran under the watch
    z = m["files"]["usgs-zhu-2017__usgs-grid.csv"]
    assert z["n_rows"] == 6 and z["n_values"] == 4                     # one empty cell, one case with no lat
    assert m["zhu_grid_read"]["Loma Prieta"] == {"cases": 6, "no_coordinates": 1, "on_grid": 4, "empty_cell": 1,
                                                 "outside_extent": 0, "grid_sha256": "0" * 64,
                                                 "grid_path": "synthetic"}
    zv = (tmp_path / "pred" / "usgs-zhu-2017__usgs-grid.csv").read_text().splitlines()
    assert zv[3] == "c2," and zv[5] == "c4,"                           # empty stays empty, never 0


def test_predict_without_helpers_still_runs_the_shaking_score(tmp_path):
    inp, _ = _fixture(tmp_path)
    m = json.loads(Path(W.predict(inp, tmp_path / "pred")["manifest"]).read_text())
    assert m["files"]["level-0__pga_primary.csv"]["n_values"] == 0
    assert m["files"]["baseline--shaking-score__pga_primary.csv"]["n_values"] == 6
    assert m["zhu_grid_read"] is None and not any("zhu" in f for f in m["files"])


def test_an_input_column_outside_the_allowlist_is_refused(tmp_path):
    inp, _ = _fixture(tmp_path, {"site_flag_v2": [1, 0, 1, 0, 1, 0]})
    with pytest.raises(W.D8Refused, match="allowlist"):
        W.load_inputs(inp)


def test_authors_rule_says_u2_was_not_passed():
    for name, spec in W.MODELS.items():
        if spec.get("chain") == "authors-maurer-2026-11-19":
            assert "u2 not passed: tip = qc" in spec["rule"], name


def test_zhu_grid_map_holds_no_kaikoura_or_canterbury():
    assert not any(W.is_kaikoura(k) or W.is_canterbury(k) for k in W.ZHU_GRIDS)
    assert not any(W.is_kaikoura(v) for v in W.ZHU_GRIDS.values())


def test_driver_predict_step_names_no_label_path():
    src = (ROOT / "scripts" / "d8_world_inputs_2026_11_23.py").read_text()
    body = src.split("def cmd_predict", 1)[1].split("\ndef ", 1)[0]
    assert "label" not in body.lower()
    assert "label" not in inspect.getsource(W.predict).lower().replace("labels_opened", "")
    assert "label" not in inspect.getsource(W.run_models).lower()


@pytest.mark.parametrize("col", ["label", "Observation", "liq_observed", "fldm_json", "FLDO_DESC"])
def test_an_input_table_with_a_label_column_is_refused(tmp_path, col):
    inp, _ = _fixture(tmp_path, {col: [1, 0, 1, 0, 1, 0]})
    with pytest.raises(W.D8Refused, match="correction 6"):
        W.load_inputs(inp)


def test_a_changed_input_file_is_refused(tmp_path):
    inp, _ = _fixture(tmp_path)
    (inp / "traces.npz").write_bytes(b"changed")
    with pytest.raises(W.D8Refused, match="frozen sha256"):
        W.load_inputs(inp)


@pytest.mark.parametrize("name", ["Kaikoura, New Zealand", "Kaikōura 2016", "KAIKOURA"])
def test_kaikoura_stays_out(name):
    assert W.is_kaikoura(name)
    with pytest.raises(W.D8Refused, match="Kaikoura"):
        W.check_no_kaikoura(["Loma Prieta", name])


def test_canterbury_events_are_recognised():
    for n in ("Darfield, New Zealand", "Christchurch SE Earthquake 0613, New Zealand", "Lyttleton Earthquake"):
        assert W.is_canterbury(n)
    assert not W.is_canterbury("New Zealand-02 (Edgecumbe)")


def test_stamp_is_wired_never_run(tmp_path):
    (tmp_path / "INPUTS-MANIFEST.json").write_text("{}")
    p = tmp_path / "PREDICTIONS-MANIFEST.json"
    p.write_text(json.dumps({"inputs_manifest_sha256": W.sha_file(tmp_path / "INPUTS-MANIFEST.json")}))
    calls = []
    r = W.stamp_plan(p, tmp_path, runner=lambda *a, **k: calls.append(a))
    assert r["ran"] is False and r["command"][:2] == ["ots", "stamp"] and not calls
    with pytest.raises(W.D8Refused, match="founder go"):
        W.stamp_plan(p, tmp_path, run=True, go="  ", runner=lambda *a, **k: calls.append(a))
    assert not calls


def test_exposure_tags_follow_the_v1_rules():
    t = W.exposure_tags("Loma Prieta", "NGL-API-2026-11-20")
    assert t["exp_bi14"] == "POSSIBLE" and t["exp_zhu2017"] == "EVENT IN TRAINING"
    u = W.exposure_tags("Ridgecrest", "PRJ-6544")
    assert u["exp_bi14"] == "UNSEEN" and u["exp_zhu2017"].startswith("UNSEEN")
    assert "Zhu-family" in u["exp_zhu2017"]
    from scoreboard.evalkit import card
    assert t["kit_exp_bi14"] == "fitted_on" and t["kit_exp_zhu2017"] == "fitted_on"
    assert u["kit_exp_bi14"] == "unseen" and u["kit_exp_zhu2017"] == "unknown" and u["kit_exp_shaking"] == "unseen"
    assert all(v in card.EXPOSURE_MARKS for k, v in {**t, **u}.items() if k.startswith("kit_"))


def test_every_source_has_a_licence_line():
    for k, v in W.LICENCES.items():
        assert v["licence"] and v["url"].startswith("https://") and v["attribution"], k


def test_existing_chain_ids_are_untouched():
    from scoreboard.evalkit import chain_settings as CS
    assert CS.SETTINGS["default-2026-11"] == {"ic_cut": 2.6, "fc": "bi14-cfc0", "tip": "qc", "area_ratio": None,
                                              "lpi_window_m": 20.0, "lsn_window_m": 15.0}
    assert CS.SETTINGS["authors-maurer-2026-11-19"]["fc"] == "maurer2019-christchurch"
    assert {m.get("chain") for m in W.MODELS.values()} - {None} <= set(CS.SETTINGS)


def test_prj6544_names_map_onto_the_canonical_names():
    assert W.PRJ6544_TO_CANON["LomaPrieta"] == "Loma Prieta"
    assert W.exposure_tags(W.PRJ6544_TO_CANON["LomaPrieta"], "PRJ-6544")["exp_bi14"] == "POSSIBLE"
    assert W.PRJ6544_TO_CANON["1906"] == "1906 San Francisco"

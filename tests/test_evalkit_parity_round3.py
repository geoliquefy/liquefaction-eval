"""M3 attempt 3 (item A4, G1+G5+G6, eval M3): planted cases for the parity-on-the-global-set code. No em dashes.

Synthetic and planted data only. No test opens the PRJ-3012 file, computes our value on it, or uses the printed
target; the printed values below are planted ("0.62", "0.835", "0.5", or read off a planted AUC). The target never
appears in this file. One test only checks that the runner constant equals the value its header and the
registration print; nothing is computed on it.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from scoreboard.evalkit import chain_settings as CS  # noqa: E402
from scoreboard.evalkit import parity_global as PG  # noqa: E402
from scoreboard.scorer import auc as kit_auc  # noqa: E402


def _load_script():
    spec = importlib.util.spec_from_file_location("r3run", KIT / "scripts" / "run_m3_parity_round3_2026_11_23.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _planted_rows(yes_total=160):
    """275 planted records with the Table 3.2 counts and `yes_total` yes labels (Manifestation 2)."""
    events = [e for e, n in PG.PEER_TABLE_3_2.items() for _ in range(n)]
    man = np.zeros(len(events))
    man[:yes_total] = 2.0
    return events, man


# --- rounding band and decision ---------------------------------------------------------------------------------

def test_rounding_band_is_half_a_unit_of_the_last_printed_digit():
    assert PG.rounding_band("0.62") == pytest.approx((0.615, 0.625), abs=1e-15)
    assert PG.rounding_band("0.835") == pytest.approx((0.8345, 0.8355), abs=1e-15)
    assert PG.rounding_band("0.5") == pytest.approx((0.45, 0.55), abs=1e-15)
    with pytest.raises(ValueError):
        PG.rounding_band("1")


def test_decide_includes_both_band_ends_and_refuses_just_outside():
    assert PG.decide(0.62, "0.62") == "AGREES"
    assert PG.decide(0.615, "0.62") == "AGREES"
    assert PG.decide(0.625, "0.62") == "AGREES"
    assert PG.decide(0.6149, "0.62") == "DISAGREES"
    assert PG.decide(0.6251, "0.62") == "DISAGREES"
    with pytest.raises(PG.GateError):
        PG.decide(float("nan"), "0.62")


def test_more_printed_digits_give_a_tighter_band():
    assert PG.decide(0.8352, "0.835") == "AGREES"
    assert PG.decide(0.8357, "0.835") == "DISAGREES"
    assert PG.decide(0.8357, "0.84") == "AGREES"


# --- AUC and its tie variants -----------------------------------------------------------------------------------

def test_auc_tie_variants_on_a_planted_tie():
    a = PG.auc_variants([1, 1, 0, 0], [2.0, 1.0, 1.0, 0.0])
    assert a["half"] == pytest.approx(3.5 / 4)
    assert a["strict"] == pytest.approx(3 / 4)
    assert a["ties_one"] == pytest.approx(4 / 4)
    assert (a["tied_pairs"], a["pairs"], a["n_yes"], a["n_no"]) == (1, 4, 2, 2)


def test_auc_half_equals_the_kit_scorer_with_many_zero_ties():
    rng = np.random.default_rng(20261123)
    y = rng.random(300) < 0.58
    s = np.where(rng.random(300) < 0.3, 0.0, rng.gamma(2.0, 3.0, 300) + y * 2.0)
    a = PG.auc_variants(y, s)
    assert a["half"] == pytest.approx(kit_auc(y, s), abs=1e-12)
    assert a["strict"] <= a["half"] <= a["ties_one"]


def test_auc_refuses_non_finite_scores_and_one_class():
    with pytest.raises(PG.GateError):
        PG.auc_variants([1, 0], [1.0, np.nan])
    with pytest.raises(PG.GateError):
        PG.auc_variants([1, 1], [1.0, 2.0])


def test_planted_end_to_end_word_follows_the_band():
    """A planted 275-case world: scores built so the AUC is known; the word follows only the printed rounding."""
    events, man = _planted_rows()
    rows = PG.gate_rows(events, man)
    y = man == 2.0
    rng = np.random.default_rng(7)
    s = rng.normal(0, 1, y.size) + 1.2 * y
    a = PG.auc_variants(y, s)["half"]
    printed = f"{a:.2f}"
    assert PG.decide(a, printed) == "AGREES"
    off = f"{round(a, 2) + 0.01:.2f}"
    assert PG.decide(a, off) == "DISAGREES"
    assert PG.pair_units(a, printed, rows["yes"], rows["no"]) <= 0.005 * rows["yes"] * rows["no"] + 1e-6


# --- gates ------------------------------------------------------------------------------------------------------

def test_gate_rows_passes_planted_table_counts():
    events, man = _planted_rows()
    r = PG.gate_rows(events, man)
    assert (r["n"], r["yes"], r["no"]) == (275, 160, 115)
    assert r["counts"] == PG.PEER_TABLE_3_2


def test_gate_rows_refuses_a_moved_case_a_bad_label_and_a_wrong_share():
    events, man = _planted_rows()
    moved = list(events)
    moved[0] = "Emilia"
    with pytest.raises(PG.GateError, match="counts"):
        PG.gate_rows(moved, man)
    bad = man.copy()
    bad[5] = 1.0
    with pytest.raises(PG.GateError, match="0 or 2"):
        PG.gate_rows(events, bad)
    with pytest.raises(PG.GateError, match="share"):
        PG.gate_rows(*_planted_rows(yes_total=140))
    with pytest.raises(PG.GateError, match="records"):
        PG.gate_rows(events[:-1], man[:-1])


def test_table_counts_equal_the_g1_curated_readme_counts():
    assert sum(PG.PEER_TABLE_3_2.values()) == PG.N_CASES == 275
    assert len(PG.PEER_TABLE_3_2) == 21
    spec = importlib.util.spec_from_file_location("g1c", KIT / "scripts" / "run_g1_curated.py")
    g1c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g1c)
    assert PG.PEER_TABLE_3_2 == g1c.README_COUNTS


def test_gate_inputs_lpi_and_chain():
    PG.gate_inputs([7.0, 6.5], [0.2, 0.3], [1.0, 2.0], [20, 400])
    with pytest.raises(PG.GateError, match="GWT"):
        PG.gate_inputs([7.0], [0.2], [np.nan], [100])
    with pytest.raises(PG.GateError, match="readings"):
        PG.gate_inputs([7.0], [0.2], [1.0], [19])
    with pytest.raises(PG.GateError):
        PG.gate_lpi(np.r_[np.zeros(274), np.nan])
    PG.gate_lpi(np.zeros(275))
    assert PG.gate_chain([1.0, 2.0], [1.0, 2.0 + 5e-7]) == pytest.approx(5e-7)
    with pytest.raises(PG.GateError):
        PG.gate_chain([1.0, 2.0], [1.0, 2.1])
    with pytest.raises(PG.GateError):
        PG.gate_chain([1.0, np.nan], [1.0, 2.0])


# --- chain ids (correction 2) -----------------------------------------------------------------------------------

def test_existing_chain_ids_unchanged_and_new_id_as_registered():
    assert CS.SETTINGS[CS.DEFAULT_ID] == {"ic_cut": 2.6, "fc": "bi14-cfc0", "tip": "qc", "area_ratio": None,
                                          "lpi_window_m": 20.0, "lsn_window_m": 15.0}
    assert CS.SETTINGS[CS.AUTHORS_ID] == {"ic_cut": 2.50, "fc": "maurer2019-christchurch", "tip": "qt",
                                          "area_ratio": 0.80, "lpi_window_m": 20.0, "lsn_window_m": 20.0}
    assert CS.PEER_GLOBAL_ID == "peer-global-2026-11-23"
    assert CS.setting(CS.PEER_GLOBAL_ID) == {"ic_cut": 2.50, "fc": "bi14-cfc0", "tip": "qc", "area_ratio": None,
                                             "lpi_window_m": 20.0, "lsn_window_m": 20.0}


# --- runner -----------------------------------------------------------------------------------------------------

def test_runner_refuses_without_go_before_opening_any_file(monkeypatch, tmp_path):
    mod = _load_script()
    monkeypatch.setattr(mod, "PRJ", tmp_path / "absent.mat")
    opened = []
    monkeypatch.setattr(mod, "sha", lambda p: opened.append(p) or "x")
    with pytest.raises(SystemExit, match="refused"):
        mod.main([])
    assert opened == []


def test_runner_chain_path_on_planted_profiles():
    """The runner's chain path (grid, chain id, G-chain against the G1 curated path) on planted profiles only."""
    mod = _load_script()
    if not mod.BENCH_PY.exists():
        pytest.skip("bench chain not on disk (container)")
    g1c = mod.load_g1c()
    bench, P, RC, C = mod.bench_chain(g1c)
    rng = np.random.default_rng(11)
    d = {k: [] for k in ("depth", "qc", "fs", "gwt", "pga", "mw")}
    for k in range(6):
        dep = np.arange(0.02, 15.0, 0.02)
        qc = 1500.0 + 4000.0 * rng.random() + 300.0 * dep  # kPa, sandy to medium
        d["depth"].append(dep)
        d["qc"].append(qc)
        d["fs"].append(qc * (0.005 + 0.02 * (k % 3)))
        d["gwt"].append(1.0 + 0.5 * k)
        d["pga"].append(0.15 + 0.08 * k)
        d["mw"].append(6.0 + 0.3 * k)
    lpi_new, qcg, fsg = mod.chain_lpi(d, g1c, P, C, CS.PEER_GLOBAL_ID)
    lpi_def, _, _ = mod.chain_lpi(d, g1c, P, C, CS.DEFAULT_ID)
    assert np.all(np.isfinite(lpi_new)) and np.all(lpi_new >= 0)
    ref = [g1c.grid_values({"qc": qcg[k], "fs": fsg[k], "gwt": d["gwt"][k], "pga": d["pga"][k], "mw": d["mw"][k]},
                           bench, P, RC, C)["BI14-LPI"] for k in range(6)]
    assert PG.gate_chain(lpi_def, ref) <= 1e-5
    assert np.all(lpi_new <= lpi_def + 1e-9)  # a stricter Ic cut (2.50 < 2.6) can only remove susceptible layers


# --- runner decision path, end to end on a planted world (review 1, should-fix 1) ----------------------------------

N_YES, N_NO = 160, 115


def _planted_world():
    """A planted 275-case world with many zero-LPI ties, so the three tie rules give clearly different AUCs.

    Yes: 60 zeros, then 100 values. No: 80 zeros, then 35 values. Tied pairs (zero with zero): 60 x 80 = 4,800 of
    18,400, so ties 0 and ties 1 sit 0.13 below and above ties 1/2.
    """
    events, man = _planted_rows(yes_total=N_YES)
    y = man == 2.0
    lpi = np.zeros(man.size)
    yes_idx, no_idx = np.flatnonzero(y), np.flatnonzero(~y)
    lpi[yes_idx[60:]] = np.linspace(1.0, 30.0, N_YES - 60)
    lpi[no_idx[80:]] = np.linspace(0.5, 12.0, N_NO - 80)
    d = {"event": events, "man": list(man), "mw": [7.0] * man.size, "pga": [0.3] * man.size,
         "gwt": [1.5] * man.size, "n_ok": [400] * man.size, "depth": [None] * man.size,
         "qc": [None] * man.size, "fs": [None] * man.size}
    lpi_def = lpi * 1.1 + 0.2 * (lpi > 0)  # the default id: same order, not the same values
    return d, y, {CS.PEER_GLOBAL_ID: lpi, CS.DEFAULT_ID: lpi_def}


def _patch_world(monkeypatch, mod, d, lpis, printed, inverse_fault=False):
    """Swap every file read and the bench chain for the planted world. The runner's own logic runs unchanged."""
    import types
    monkeypatch.setattr(mod, "sha", lambda p: mod.PRJ_SHA)
    monkeypatch.setattr(mod, "PUBLISHED", printed)

    def load_prj(inverse):
        out = {k: list(v) for k, v in d.items()}
        if inverse and inverse_fault:
            out["n_ok"][3] = 5  # planted: one case short of readings, inverse-filtered only
        return out

    def chain_lpi(dd, g1c, P, C, setting_id):
        idx = np.arange(len(dd["mw"]), dtype=float)[:, None]  # the case index rides in the grid, for grid_values
        return lpis[setting_id].copy(), idx, idx

    g1c = types.SimpleNamespace(grid_values=lambda prof, *a: {"BI14-LPI": lpis[CS.DEFAULT_ID][int(prof["qc"][0])]})
    monkeypatch.setattr(mod, "load_prj", load_prj)
    monkeypatch.setattr(mod, "chain_lpi", chain_lpi)
    monkeypatch.setattr(mod, "load_g1c", lambda: g1c)
    monkeypatch.setattr(mod, "bench_chain", lambda g: (None, None, None, None))


def _go(mod, tmp_path):
    out = tmp_path / "res.json"
    assert mod.main(["--go", "--out", str(out)]) == 0
    import json
    return json.loads(out.read_text())


def test_runner_word_uses_ties_half_and_the_published_string(monkeypatch, tmp_path):
    mod = _load_script()
    d, y, lpis = _planted_world()
    a = PG.auc_variants(y, lpis[CS.PEER_GLOBAL_ID])
    printed = f"{a['half']:.2f}"
    lo, hi = PG.rounding_band(printed)
    # The planted world must separate the tie rules: only ties 1/2 lies in the planted band.
    assert lo + 1e-3 < a["half"] < hi - 1e-3
    assert not (lo <= a["strict"] <= hi) and not (lo <= a["ties_one"] <= hi)
    assert printed != mod.PUBLISHED  # never the registered target

    calls = []
    real_decide = PG.decide
    monkeypatch.setattr(mod.PG, "decide", lambda v, p: calls.append((v, p)) or real_decide(v, p))
    _patch_world(monkeypatch, mod, d, lpis, printed)
    res = _go(mod, tmp_path)
    assert res["word"] == "AGREES"
    assert calls == [(pytest.approx(a["half"], abs=1e-15), printed)]  # decide gets ties 1/2 and PUBLISHED itself
    assert res["auc_ties_half"] == pytest.approx(a["half"], abs=1e-15)
    assert res["published"] == printed and res["band"] == pytest.approx([lo, hi], abs=1e-15)
    assert res["beside"]["auc_ties_zero"] == pytest.approx(a["strict"], abs=1e-15)
    assert res["beside"]["auc_ties_one"] == pytest.approx(a["ties_one"], abs=1e-15)
    assert (res["beside"]["n"], res["beside"]["yes"], res["beside"]["no"]) == (275, N_YES, N_NO)
    assert res["beside"]["not_computed_because"] == {}

    # A printed value one unit off flips the word; nothing else changes.
    off = f"{round(a['half'], 2) + 0.01:.2f}"
    _patch_world(monkeypatch, mod, d, lpis, off)
    assert _go(mod, tmp_path)["word"] == "DISAGREES"


def test_runner_stops_with_no_word_on_a_planted_row_fault(monkeypatch, tmp_path):
    mod = _load_script()
    d, y, lpis = _planted_world()
    printed = f"{PG.auc_variants(y, lpis[CS.PEER_GLOBAL_ID])['half']:.2f}"
    moved = dict(d)
    moved["event"] = ["Emilia"] + list(d["event"][1:])  # one case moved to another earthquake
    _patch_world(monkeypatch, mod, moved, lpis, printed)
    res = _go(mod, tmp_path)
    assert res["word"] is None and res["stopped_at_gate"].startswith("G-rows")
    assert "auc_ties_half" not in res
    _patch_world(monkeypatch, mod, d, lpis, printed)
    monkeypatch.setattr(mod, "sha", lambda p: "0" * 64)
    res = _go(mod, tmp_path)
    assert res["word"] is None and res["stopped_at_gate"].startswith("G-hash")


def test_runner_beside_fault_keeps_the_word(monkeypatch, tmp_path):
    """Review 1 nit: a gate error on a beside value is recorded beside; the registered word stands."""
    mod = _load_script()
    d, y, lpis = _planted_world()
    printed = f"{PG.auc_variants(y, lpis[CS.PEER_GLOBAL_ID])['half']:.2f}"
    _patch_world(monkeypatch, mod, d, lpis, printed, inverse_fault=True)
    res = _go(mod, tmp_path)
    assert res["word"] == "AGREES"
    assert res["beside"]["auc_inverse_filtered"] is None
    assert res["beside"]["not_computed_because"]["auc_inverse_filtered"].startswith("G-inputs")
    assert res["beside"]["auc_default_id"] is not None


def test_runner_published_constant_matches_its_header_and_the_registration():
    """The runner constant is the value its header and the registration print. Read only; nothing is computed on it."""
    mod = _load_script()
    assert f"printed {mod.PUBLISHED}." in mod.__doc__
    reg = mod.PILLAR / mod.REGISTRATION
    if not reg.exists():
        pytest.skip("pillar registration not on disk (container)")
    assert f"P = {mod.PUBLISHED}, printed to 2 decimals." in reg.read_text()

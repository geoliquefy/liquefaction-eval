"""Tests for the simulation lab (scoreboard/evalkit/lab; protocol docs/evals/LAB-PROTOCOL-2026-11-19.md). No em dashes.

Synthetic data only, except the simulator acceptance test, which reads the real positions and labels in memory
(skipped when the data or the committed fit file are not on this machine).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scoreboard.evalkit import metrics, resample
from scoreboard.evalkit.lab import engine, rules
from scoreboard.evalkit.lab.fields import FieldSampler
from scoreboard.scorer import auc

PILLAR = Path(__file__).resolve().parents[3]
FIT_JSON = PILLAR / "docs" / "evals" / "LAB-FIT-2026-11-19.json"


def _toy(n=600, seed=3, ties=False):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 9000, n), rng.uniform(0, 7000, n)
    lab = rng.random(n) < 0.4
    s1 = lab * 0.8 + rng.normal(0, 1, n)
    s2 = lab * 0.5 + rng.normal(0, 1, n)
    if ties:
        s1 = np.round(s1, 1)
    return x, y, lab, s1, s2


@pytest.mark.parametrize("ties", [False, True])
def test_engine_equals_kit_bootstrap_draw_for_draw(ties):
    x, y, lab, s1, s2 = _toy(ties=ties)
    tiles = resample.tiles_for(x, y, 2000.0)
    p1 = 1 / (1 + np.exp(-s1))
    kit = resample.bootstrap(lambda i: {"auc": auc(lab[i], s1[i]),
                                        "ll": metrics.log_loss(lab[i].astype(float), p1[i], 1e-6)}, tiles, 50, 11)
    w = engine.weights(tiles, 50, 11)
    fast_auc = engine.Ranked(lab, s1).auc(w)
    fast_ll = engine.mean_loss(w, engine.site_log_loss(lab.astype(float), p1))
    np.testing.assert_allclose(fast_auc, kit["auc"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fast_ll, kit["ll"], rtol=0, atol=1e-12)


def test_delete_one_tile_weights_match_direct_leave_out():
    """Amendment 1 jackknife: a zero-weight row per tile equals the AUC with that tile removed."""
    x, y, lab, s1, _ = _toy()
    tiles = resample.tiles_for(x, y, 4000.0)
    inv, g = engine.tile_index(tiles)
    w = np.ones((g, lab.size))
    w[inv, np.arange(lab.size)] = 0.0
    fast = engine.Ranked(lab, s1).auc(w)
    for k in range(g):
        keep = inv != k
        assert abs(fast[k] - auc(lab[keep], s1[keep])) < 1e-12


def test_placement_values_average_to_auc():
    _, _, lab, s1, _ = _toy(ties=True)
    pv = engine.placement_values(lab, s1)
    assert abs(pv[lab].mean() - auc(lab, s1)) < 1e-12
    assert abs(pv[~lab].mean() - auc(lab, s1)) < 1e-12


def test_field_unit_variance_and_range():
    rng = np.random.default_rng(5)
    x, y = rng.uniform(0, 20000, 400), rng.uniform(0, 20000, 400)
    fs = FieldSampler(np.r_[x, x + 1000.0], np.r_[y, y], 1000.0)
    draws = np.array([fs.draw(rng) for _ in range(300)])
    assert abs(draws.var() - 1.0) < 0.1
    # correlation at the practical range (1 km apart) is near exp(-3) = 0.05
    c = np.mean([np.corrcoef(draws[:, k], draws[:, k + 400])[0, 1] for k in range(400)])
    assert abs(c - np.exp(-3.0)) < 0.08


def test_window_fit_flags_edge_and_takes_fallback():
    lag = np.linspace(250, 11750, 24)
    vg = {"lag_m": lag.tolist(), "gamma": (0.05 + 0.01 * lag / 1000.0).tolist(), "n_pairs": [100] * 24}
    fit = rules.fit_range_in_window(vg)
    assert not fit["identified"]
    counts = {500.0: 300, 1000.0: 120, 2000.0: 50, 4000.0: 25, 8000.0: 10}
    ch = rules.choose_v2(fit, counts)
    assert ch["tile_m"] == 4000.0 and ch["word"] == "RANGE NOT IDENTIFIED"
    counts[4000.0] = 12
    assert rules.choose_v2(fit, counts)["tile_m"] == 2000.0


def test_window_fit_finds_a_short_range():
    lag = np.linspace(250, 11750, 24)
    vg = {"lag_m": lag.tolist(), "gamma": (0.02 + 0.15 * (1 - np.exp(-3 * lag / 1500.0))).tolist(),
          "n_pairs": [100] * 24}
    fit = rules.fit_range_in_window(vg)
    assert fit["identified"] and 1000 < fit["range_m"] < 2200
    ch = rules.choose_v2(fit, {500.0: 300, 1000.0: 120, 2000.0: 50, 4000.0: 25, 8000.0: 10})
    assert ch["tile_m"] == 2000.0 and ch["word"] is None


def test_choose_v2_never_picks_a_side_under_20_tiles_when_one_exists():
    fit = {"identified": True, "range_m": 6000.0}
    ch = rules.choose_v2(fit, {500.0: 300, 1000.0: 120, 2000.0: 50, 4000.0: 25, 8000.0: 10})
    assert ch["tile_m"] == 4000.0 and ch["word"] is not None


def test_pairs_variogram_matches_kit_semivariogram():
    x, y, lab, s1, _ = _toy(n=300)
    r = s1 - s1.mean()
    kit = resample.semivariogram(x, y, r, max_points=10_000)
    mine = rules.Pairs(x, y).variogram(r)
    np.testing.assert_allclose(np.nan_to_num(mine["gamma"]), np.nan_to_num(kit["gamma"]), atol=1e-12)
    assert mine["n_pairs"] == kit["n_pairs"]


def test_trend_residual_removes_a_planted_quadratic_trend():
    rng = np.random.default_rng(9)
    n = 3000
    x, y = rng.uniform(0, 20000, n), rng.uniform(0, 20000, n)
    lp = np.log(rng.uniform(0.1, 0.5, n))
    eta = -0.5 + 1.5 * lp + 0.02 * ((x - 1e4) / 1000) ** 2
    lab = rng.random(n) < 1 / (1 + np.exp(-eta))
    r = rules.trend_residual(lab.astype(float), x, y, lp, (1e4, 1e4))
    fit = rules.fit_range_in_window(rules.Pairs(x, y).variogram(r))
    assert fit["range_m"] == 0.0 or fit["structured_share"] < 0.2


@pytest.mark.skipif(not FIT_JSON.exists(), reason="committed lab fit not present")
def test_simulator_reproduces_real_range_and_prevalence():
    """Protocol D1 acceptance on the committed fit: rerun the check on the real positions (memory only)."""
    from scoreboard.evalkit.lab import fit as F
    try:
        real = F.load_real()
    except Exception as exc:       # data not on this machine (container, CI)
        pytest.skip(f"real data not available: {exc}")
    fitd = json.loads(FIT_JSON.read_text())
    from scoreboard.evalkit.lab.world import EVENTS
    geom = F.geometry(real, fitd["glm"]["quad"])
    pairs = {e: rules.Pairs(real["x"][real["masks"][e]], real["y"][real["masks"][e]]) for e in EVENTS}
    acc = F.acceptance(geom, pairs, real, fitd["glm"], fitd["prevalence"], fitd["real_residual_structure"],
                       fitd["chosen"]["sigma_s"], fitd["chosen"]["rho_s"])
    assert acc["pass"], acc


# ---------------------------------------------------------------------------------------------------------------
# Amendment 1 code: certify.py and measure.py (step 2 review 1, MINOR 2). Hand-built inputs only.
# ---------------------------------------------------------------------------------------------------------------
def _run(W=4, E=2, se=1.0, d_auc=2.0, d_ll=3.0, n_tiles=(30, 30, 30, 25, 12)):
    """A hand-built lab run: res (W, E, S, 3, 2, 6) with alternating-sign differences and a fixed SE."""
    from scoreboard.evalkit.lab.aggregate import D, SE
    res = np.zeros((W, E, 5, 3, 2, 6))
    sign = np.where(np.arange(W) % 2 == 0, 1.0, -1.0)[:, None, None, None]
    res[..., 0, D] = sign * d_auc
    res[..., 1, D] = sign * d_ll
    res[..., SE] = se
    jack = np.zeros((W, E, 3, 2, 2))
    jack[..., 0] = res[:, :, 4, :, :, D]
    jack[..., 1] = 2.0 * se
    return {"res": res, "jack": jack, "n": np.full((W, E), 500), "n_pos": np.full((W, E), 100),
            "n_tiles": np.broadcast_to(np.array(n_tiles), (W, E, 5)).copy(), "slot": np.tile(np.arange(E), (W, 1))}


def test_certify_calibrate_is_rms_ratio_over_exact_cells(tmp_path):
    from scoreboard.evalkit.lab import certify
    from scoreboard.evalkit.lab.aggregate import D
    null = _run()
    med = _run()
    med["res"][:, :, :, :2, :, D] = 50.0          # non-exact comparisons in C-MED must not enter k
    np.savez(tmp_path / "C-NULL.npz", **null)
    np.savez(tmp_path / "C-MED.npz", **med)
    k = certify.calibrate(tmp_path)
    assert k == {"auc": 2.0, "log_loss": 3.0}


def test_certify_intervals_match_their_formulas():
    from scipy import stats
    from scoreboard.evalkit.lab import certify
    r = _run()
    d, half, p, gate = certify.intervals(r, "tile4-inflated", {"auc": 1.5, "log_loss": 2.0})
    np.testing.assert_allclose(half[..., 0], 1.959963984540054 * 1.5)
    np.testing.assert_allclose(half[..., 1], 1.959963984540054 * 2.0)
    np.testing.assert_allclose(p[..., 0], 2 * stats.norm.sf(2.0 / 1.5))
    assert gate.all()                               # 25 tiles at 4 km >= 20
    G = 12.0
    d, half, p, gate = certify.intervals(r, "tile8-t", {})
    np.testing.assert_allclose(half, stats.t.ppf(0.975, G - 1) * np.sqrt(G / (G - 1)))
    assert gate.all()                               # 12 tiles at 8 km >= GATE8
    d, half, p, gate = certify.intervals(r, "jackknife8-t", {})
    np.testing.assert_allclose(half, stats.t.ppf(0.975, G - 1) * 2.0)
    np.testing.assert_allclose(d, r["res"][:, :, 4, :, :, 0])
    r2 = _run(n_tiles=(30, 30, 30, 19, 7))
    assert not certify.intervals(r2, "tile4-inflated", {"auc": 1.0, "log_loss": 1.0})[3].any()
    assert not certify.intervals(r2, "tile8-t", {})[3].any()


def test_certify_holm_fwer_counts_families_and_cards():
    from scoreboard.evalkit.lab import certify
    p = np.full((1, 1, 3, 2), 0.5)
    p[0, 0, 0, 0] = 0.001                           # one AUC comparison separates
    gate = np.ones_like(p, bool)
    exact = np.ones((1, 3, 2), bool)
    assert certify.holm_fwer(p, gate, exact, across=False) == (0.5, 2)
    assert certify.holm_fwer(p, gate, exact, across=True) == (1.0, 1)
    exact[0, 0, 0] = False                          # a planted gap is not a false SEPARATES
    assert certify.holm_fwer(p, gate, exact, across=False) == (0.0, 2)
    gate[:] = False
    assert certify.holm_fwer(p, gate, exact, across=False) == (0.0, 0)


def test_measure_detrend_fragility_and_summaries():
    from scoreboard.evalkit.lab import measure
    rng = np.random.default_rng(1)
    X = np.c_[np.ones(200), rng.normal(size=(200, 2))]
    v = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 200)
    r = measure.detrend(v, X)
    np.testing.assert_allclose(X.T @ r, 0.0, atol=1e-9)
    f = measure.fragility([0.0, 8.314, 1e6], 8.314, 1.113)
    np.testing.assert_allclose(f, [0.0, 0.5, 1.0], atol=1e-12)
    s = measure.summarise([(1000.0, True), (3000.0, True), (12000.0, False)])
    assert s == {"n": 3, "share_not_identified": 0.333, "identified_median_m": 2000.0}
    assert measure.summarise([(12000.0, False)])["identified_median_m"] is None
    meas = {"sets": {"B": {"sep2010": {"n": 10, "pair LL | a - b": {"range_m": 2000.0, "identified": True},
                                       "pair AUC | a - b": {"range_m": 12000.0, "identified": False},
                                       "residual | a": {"range_m": 900.0, "identified": True}}}}}
    rs = measure.real_summary(meas)
    assert rs["pair LL"] == {"n": 1, "share_not_identified": 0.0, "identified_median_m": 2000.0}
    assert rs["pair AUC"]["share_not_identified"] == 1.0

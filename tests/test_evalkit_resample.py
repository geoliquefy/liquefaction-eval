"""evalkit resampling (rule book R11): tile bootstrap, pairing, coverage, MDD, Holm, tile-size chooser. Synthetic."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import resample as R  # noqa: E402
from scoreboard.evalkit import synth  # noqa: E402
from scoreboard.scorer import Z_ALPHA, Z_POWER, EventResult, apply_holm, auc  # noqa: E402


def model(ev, sd, rng):
    return expit(ev["intercept"] + ev["slope"] * (ev["s"] + rng.normal(0.0, sd, ev["s"].size)))


def test_bootstrap_is_deterministic_and_paired():
    ev = synth.synthetic_event(500, 1)
    y, tiles = ev["label"].astype(bool), R.tiles_for(ev["x_m"], ev["y_m"], 2000.0)
    p = ev["p_target"]
    fn = lambda idx: {"auc": auc(y[idx], p[idx])}  # noqa: E731
    a = R.bootstrap(fn, tiles, 100, 7)["auc"]
    b = R.bootstrap(fn, tiles, 100, 7)["auc"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, R.bootstrap(fn, tiles, 100, 8)["auc"])
    d = R.paired(auc(y, p), auc(y, p), a, b)
    assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0 and d["p"] == 1.0


def test_paired_mdd_formula():
    rng = np.random.default_rng(0)
    da, db = rng.normal(0.7, 0.01, 500), rng.normal(0.65, 0.01, 500)
    r = R.paired(0.7, 0.65, da, db)
    assert r["mdd"] == pytest.approx((Z_ALPHA + Z_POWER) * np.std(da - db, ddof=1))
    assert r["diff"] == pytest.approx(0.05, abs=1e-12) and r["lo"] > 0 and r["p"] < 0.05


def test_paired_interval_coverage_on_synthetic_null():
    """Quick coverage study. Two exchangeable entries; the 95% paired AUC interval should cover 0.

    120 replicates, 200 draws each, 400 sites on 2 km tiles. Tolerance: coverage in [0.85, 1.0] (binomial SD at 0.95
    is 0.02; the percentile tile bootstrap is known to run a little narrow with few sites per tile).
    """
    covered = 0
    reps = 120
    for k in range(reps):
        ev = synth.synthetic_event(400, 100 + k)
        rng = np.random.default_rng(500 + k)
        y, tiles = ev["label"].astype(bool), R.tiles_for(ev["x_m"], ev["y_m"], 2000.0)
        a, b = model(ev, 0.6, rng), model(ev, 0.6, rng)
        da = R.bootstrap(lambda idx: {"v": auc(y[idx], a[idx])}, tiles, 200, k)["v"]
        db = R.bootstrap(lambda idx: {"v": auc(y[idx], b[idx])}, tiles, 200, k)["v"]
        r = R.paired(auc(y, a), auc(y, b), da, db)
        covered += int(r["lo"] <= 0.0 <= r["hi"])
    assert 0.85 <= covered / reps <= 1.0, covered / reps


def test_holm_matches_scorer_apply_holm():
    ps = [0.001, 0.04, 0.012, 0.3, 0.0125, 0.02]
    board = [EventResult(event=k, n_pos=50, n_neg=50, n_clusters=100, auc={}, auc_ci={},
                         p={"m": pv}, diff_ci90={"m": (0.1, 0.2)}) for k, pv in enumerate(ps)]
    apply_holm(board, min_clusters=0)
    assert R.holm(ps) == [r.verdict["m"] == "SEPARATES" for r in board]
    assert R.holm(ps) == [True, False, False, False, False, False]


def test_interval_ignores_undefined_draws():
    iv = R.interval(0.5, np.array([0.4, np.nan, 0.6, 0.5]))
    assert iv["n_draws"] == 3 and iv["lo"] >= 0.4 and iv["hi"] <= 0.6


def test_chooser_picks_larger_tile_for_correlated_residuals():
    rng = np.random.default_rng(11)
    n, extent = 1500, 20000.0
    x, y = rng.uniform(0, extent, n), rng.uniform(0, extent, n)
    iid = rng.normal(0.0, 1.0, n)
    field = synth.spatial_field(x, y, rng, cell_m=4000.0, sd=1.0, extent_m=extent)
    corr = field + rng.normal(0.0, 0.5, n)
    c_iid = R.choose_tile(x, y, iid, seed=1)
    c_corr = R.choose_tile(x, y, corr, seed=1)
    assert c_iid["range_m"] == 0.0 and c_iid["tile_m"] == R.TILE_CANDIDATES_M[0]
    assert c_corr["range_m"] > 1500.0 and c_corr["tile_m"] > c_iid["tile_m"]
    assert len(c_corr["variogram"]["gamma"]) == 24 and "smallest" in c_corr["rule"]


def test_chooser_on_binary_residuals_of_a_model_that_misses_the_field():
    ev = synth.synthetic_event(1500, 12, field_sd=1.2, field_cell_m=4000.0)
    resid = ev["label"] - ev["fragility"] * expit(ev["intercept"] + ev["slope"] * ev["s"])
    resid_oracle = ev["label"] - ev["p_target"]
    assert R.choose_tile(ev["x_m"], ev["y_m"], resid)["tile_m"] > \
        R.choose_tile(ev["x_m"], ev["y_m"], resid_oracle)["tile_m"]


def test_tile_sensitivity_prints_1_2_4_km():
    ev = synth.synthetic_event(600, 13)
    y, p = ev["label"].astype(bool), ev["p_target"]
    out = R.tile_sensitivity(lambda idx: {"auc": auc(y[idx], p[idx])}, ev["x_m"], ev["y_m"], "auc", n_boot=100, seed=1)
    assert list(out) == ["1000", "2000", "4000"]
    assert all(v["value"] == auc(y, p) and v["lo"] < v["value"] < v["hi"] for v in out.values())

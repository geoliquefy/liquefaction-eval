"""evalkit candidate intervals (own-work item C1, eval M2): defaults unchanged, determinism, coverage on known-truth
toys. Synthetic data only. No em dashes."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import intervals as I  # noqa: E402
from scoreboard.evalkit import pool, resample as R, synth  # noqa: E402
from scoreboard.evalkit.lab import engine  # noqa: E402
from scoreboard.scorer import auc, grid_clusters  # noqa: E402


def mean_fn(v):
    return lambda idx: {"m": float(v[idx].mean())}


def covers(lo, hi, truth=0.0):
    return int(lo <= truth <= hi)


# ---------------------------------------------------------------------------------------------------------------
# Flags: the defaults are the current kit methods
# ---------------------------------------------------------------------------------------------------------------

def test_default_within_method_is_the_tile_bootstrap_unchanged():
    ev = synth.synthetic_event(400, 3)
    y, p = ev["label"].astype(bool), ev["p_target"]
    tiles = R.tiles_for(ev["x_m"], ev["y_m"], 4000.0)
    fn = lambda idx: {"auc": auc(y[idx], p[idx])}  # noqa: E731
    got = I.within_event_draws(fn, tiles=tiles, n_boot=50, seed=4)
    assert got["method"] == I.DEFAULT_WITHIN == "tile"
    np.testing.assert_array_equal(got["draws"]["auc"], R.bootstrap(fn, tiles, 50, 4)["auc"])


def test_default_pooled_method_is_hartung_knapp_unchanged():
    d, s = [0.03, 0.01, 0.02, -0.005], [0.01, 0.012, 0.015, 0.02]
    assert I.DEFAULT_POOLED == "hartung-knapp"
    assert I.pooled_interval(d, s) == pool.random_effects(d, s)


def test_percentile_form_is_resample_paired_unchanged():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.70, 0.01, 300), rng.normal(0.68, 0.01, 300)
    got = I.paired(0.70, 0.68, a, b)
    ref = R.paired(0.70, 0.68, a, b)
    assert got.pop("form") == "percentile" and got == ref


def test_unknown_flags_are_refused():
    with pytest.raises(ValueError, match="method must be"):
        I.within_event_draws(mean_fn(np.zeros(3)), method="jackknife", n_boot=5, seed=0)
    with pytest.raises(ValueError, match="method must be"):
        I.pooled_interval([0.1, 0.2, 0.3], [0.1] * 3, method="dl")
    with pytest.raises(ValueError, match="form must be"):
        I.paired(0.0, 0.0, np.zeros(3), np.zeros(3), form="bca")


def test_t_form_matches_the_hand_formula():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0.0, 0.02, 400), rng.normal(0.0, 0.02, 400)
    g = 12
    r = I.paired(0.01, 0.0, a, b, form="t", n_units=g)
    se = np.std(a - b, ddof=1) * np.sqrt(g / (g - 1))
    tq = stats.t.ppf(0.975, g - 1)
    assert r["se"] == pytest.approx(se) and r["df"] == g - 1
    assert r["lo"] == pytest.approx(0.01 - tq * se) and r["hi"] == pytest.approx(0.01 + tq * se)
    assert r["p"] == pytest.approx(2 * stats.t.sf(0.01 / se, g - 1))
    with pytest.raises(ValueError, match="n_units"):
        I.paired(0.01, 0.0, a, b, form="t")


# ---------------------------------------------------------------------------------------------------------------
# (a) spatial block bootstrap
# ---------------------------------------------------------------------------------------------------------------

def test_block_side_comes_from_the_measured_ranges():
    assert I.MEASURED_PAIRDIFF_RANGE_M == {"auc": 5500.0, "log_loss": 2700.0}
    assert I.block_side(5500.0) == 5500.0 and I.block_side(2700.0) == 3000.0
    assert I.block_side(0.0) == 500.0 and I.block_side(501.0) == 1000.0
    with pytest.raises(ValueError):
        I.block_side(float("nan"))


def test_spatial_block_uses_the_metric_range_and_is_seeded_and_paired():
    ev = synth.synthetic_event(600, 5)
    y, p = ev["label"].astype(bool), ev["p_target"]
    fn = lambda idx: {"auc": auc(y[idx], p[idx])}  # noqa: E731
    a = I.spatial_block_bootstrap(fn, ev["x_m"], ev["y_m"], metric="auc", n_boot=60, seed=9)
    b = I.spatial_block_bootstrap(fn, ev["x_m"], ev["y_m"], metric="auc", n_boot=60, seed=9)
    c = I.spatial_block_bootstrap(fn, ev["x_m"], ev["y_m"], metric="auc", n_boot=60, seed=10)
    np.testing.assert_array_equal(a["draws"]["auc"], b["draws"]["auc"])
    assert not np.array_equal(a["draws"]["auc"], c["draws"]["auc"])
    blocks = grid_clusters(ev["x_m"], ev["y_m"], 5500.0)
    np.testing.assert_array_equal(a["draws"]["auc"], R.bootstrap(fn, blocks, 60, 9)["auc"])
    assert a["block_m"] == 5500.0 and a["n_blocks"] == np.unique(blocks).size and "11.1" in a["range_source"]
    assert a["under_min_blocks"] == (a["n_blocks"] < 20)
    ll = I.spatial_block_bootstrap(fn, ev["x_m"], ev["y_m"], metric="log_loss", n_boot=5, seed=9)
    assert ll["block_m"] == 3000.0 and ll["n_blocks"] > a["n_blocks"]
    with pytest.raises(ValueError, match="exactly one"):
        I.spatial_block_bootstrap(fn, ev["x_m"], ev["y_m"], n_boot=5, seed=0)
    with pytest.raises(ValueError, match="no measured range"):
        I.spatial_block_bootstrap(fn, ev["x_m"], ev["y_m"], metric="brier", n_boot=5, seed=0)


def _spatial_toy(seed, n=800, extent=40000.0, cell=2500.0):
    """Known truth 0: a smooth Gaussian field (bilinear on a 2.5 km grid, so correlated to about 5 km) plus noise."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, extent, n), rng.uniform(0, extent, n)
    return x, y, synth.spatial_field(x, y, rng, cell_m=cell, sd=1.0, extent_m=extent) + rng.normal(0, 0.5, n)


def test_spatial_block_covers_a_known_truth_that_site_resampling_misses():
    """150 toys (seeds 0 to 149), 200 draws, deterministic: 5.5 km blocks 0.91, site-level 0.63. Bounds: blocks
    >= 0.85; site-level at least 0.15 below the blocks (binomial SD about 0.02 at 0.91)."""
    reps, cov_block, cov_site = 150, 0, 0
    for k in range(reps):
        x, y, v = _spatial_toy(k)
        fn = mean_fn(v)
        db = I.spatial_block_bootstrap(fn, x, y, range_m=5500.0, n_boot=200, seed=k)["draws"]["m"]
        ds = R.bootstrap(fn, np.arange(v.size), 200, k)["m"]
        cov_block += covers(*_lohi(R.interval(v.mean(), db)))
        cov_site += covers(*_lohi(R.interval(v.mean(), ds)))
    assert cov_block / reps >= 0.85, cov_block / reps
    assert cov_site / reps <= cov_block / reps - 0.15, (cov_site / reps, cov_block / reps)


def _lohi(iv):
    return iv["lo"], iv["hi"]


# ---------------------------------------------------------------------------------------------------------------
# (b) cluster bootstrap on survey units
# ---------------------------------------------------------------------------------------------------------------

def test_unit_ids_are_required_for_every_site():
    for missing in (None, float("nan"), " ", np.float32("nan"), np.float64("nan"), np.datetime64("NaT", "D")):
        with pytest.raises(ValueError, match="survey-unit id"):
            I.unit_codes(["a", missing, "b"], 3)
    pd = pytest.importorskip("pandas")
    with pytest.raises(ValueError, match="survey-unit id"):
        I.unit_codes(["a", pd.NA, "b"], 3)
    with pytest.raises(ValueError, match="one unit id per site"):
        I.unit_codes(["a", "b"], n_sites=3)
    np.testing.assert_array_equal(I.unit_codes(["b", "a", "b", "c"], 4), [1, 0, 1, 2])
    np.testing.assert_array_equal(I.unit_codes([10, 9, np.int64(10), 2], 4), [2, 1, 2, 0])


def test_mixed_unit_id_kinds_are_refused_not_merged():
    for ids in ([7, "7"], [1.0, "1.0"], [1, 1.0], ["a", 2]):
        with pytest.raises(ValueError, match="one kind"):
            I.unit_codes(ids, 2)


def test_a_short_id_list_is_refused_on_every_path():
    v = np.zeros(100)
    ids = [f"u{j % 5}" for j in range(40)]
    with pytest.raises(ValueError, match="one unit id per site: 40 ids for 100 sites"):
        I.within_event_draws(mean_fn(v), method="survey-cluster", unit_ids=ids, n_sites=100, n_boot=5, seed=0)
    with pytest.raises(ValueError, match="one unit id per site"):
        I.cluster_bootstrap(mean_fn(v), ids, n_sites=100, n_boot=5, seed=0)
    with pytest.raises(ValueError, match="needs unit_ids and n_sites"):
        I.within_event_draws(mean_fn(v), method="survey-cluster", unit_ids=ids, n_boot=5, seed=0)
    with pytest.raises(TypeError):
        I.cluster_bootstrap(mean_fn(v), ids, n_boot=5, seed=0)


def test_cluster_bootstrap_is_seeded_and_matches_the_lab_engine():
    rng = np.random.default_rng(6)
    units = [f"team-{j % 25}" for j in range(500)]
    v = rng.normal(0, 1, 500)
    a = I.cluster_bootstrap(mean_fn(v), units, n_sites=500, n_boot=80, seed=3)
    b = I.cluster_bootstrap(mean_fn(v), units, n_sites=500, n_boot=80, seed=3)
    np.testing.assert_array_equal(a["draws"]["m"], b["draws"]["m"])
    c = I.cluster_bootstrap(mean_fn(v), units, n_sites=500, n_boot=80, seed=4)
    assert not np.array_equal(a["draws"]["m"], c["draws"]["m"])
    assert a["n_units"] == 25 and not a["under_min_units"] and a["unit_sizes"] == {"min": 20, "median": 20.0, "max": 20}
    w = engine.weights(I.unit_codes(units, 500), 80, 3)
    np.testing.assert_allclose(engine.mean_loss(w, v), a["draws"]["m"], rtol=1e-12)
    flag = I.within_event_draws(mean_fn(v), method="survey-cluster", unit_ids=units, n_sites=500, n_boot=80, seed=3)
    np.testing.assert_array_equal(flag["draws"]["m"], a["draws"]["m"])
    small = I.cluster_bootstrap(mean_fn(v[:15]), units[:15], n_sites=15, n_boot=5, seed=0)
    assert small["under_min_units"]


def _unit_toy(seed, n_units=60, per=10, extent=20000.0):
    """Known truth 0: each survey unit shares one label-error offset; units are scattered, not spatial blocks."""
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n_units), per)
    x, y = rng.uniform(0, extent, units.size), rng.uniform(0, extent, units.size)
    return x, y, units, rng.normal(0, 1.0, n_units)[units] + rng.normal(0, 0.5, units.size)


def test_cluster_bootstrap_covers_a_unit_effect_that_tiles_miss():
    """150 toys (seeds 0 to 149), 200 draws, deterministic: units 0.93, 2 km tiles 0.46. Bounds: units >= 0.85;
    tiles at least 0.25 below the units."""
    reps, cov_unit, cov_tile = 150, 0, 0
    for k in range(reps):
        x, y, u, v = _unit_toy(k)
        fn = mean_fn(v)
        du = I.cluster_bootstrap(fn, u.tolist(), n_sites=u.size, n_boot=200, seed=k)["draws"]["m"]
        dt = R.bootstrap(fn, grid_clusters(x, y, 2000.0), 200, k)["m"]
        cov_unit += covers(*_lohi(R.interval(v.mean(), du)))
        cov_tile += covers(*_lohi(R.interval(v.mean(), dt)))
    assert cov_unit / reps >= 0.85, cov_unit / reps
    assert cov_tile / reps <= cov_unit / reps - 0.25, (cov_tile / reps, cov_unit / reps)


# ---------------------------------------------------------------------------------------------------------------
# (c) event-level resampling for a pooled line
# ---------------------------------------------------------------------------------------------------------------

def test_event_resampling_needs_three_earthquakes_and_prints_the_small_k_caveat():
    with pytest.raises(ValueError, match="NOT ALLOWED"):
        I.event_bootstrap([0.01, 0.02])
    r = I.event_bootstrap([0.01, 0.02, 0.04])
    assert r["k"] == 3 and r["n_distinct_resamples"] == 10 and r["scheme"] == "exact enumeration"
    assert r["n_draws"] == 27 and "10 distinct resamples" in r["caveat"] and "2/3" in r["caveat"]
    assert I.n_distinct_resamples(4) == 35
    assert "not certified" in r["caveat"]


def test_one_stage_exact_is_seed_free_and_its_t_form_is_the_equal_weight_line():
    d = [0.031, 0.004, 0.018, -0.006]
    a = I.event_bootstrap(d, seed=1, form="t")
    b = I.event_bootstrap(d, seed=2, form="t")
    assert a == b
    ew = pool.equal_weight(d)
    assert a["mean"] == pytest.approx(ew["mean"]) and a["se"] == pytest.approx(ew["se"])
    assert a["lo"] == pytest.approx(ew["lo"]) and a["hi"] == pytest.approx(ew["hi"]) and a["p"] == pytest.approx(ew["p"])
    pct = I.event_bootstrap(d, seed=1)
    assert pct == I.event_bootstrap(d, seed=2)
    all_means = np.asarray(d)[np.array(list(itertools.product(range(4), repeat=4)))].mean(axis=1)
    assert all_means.size == 4 ** 4 and pct["se"] == pytest.approx(np.std(all_means, ddof=0), rel=1e-12)
    assert not pct["zero_width"] and not a["zero_width"]


def test_equal_differences_are_marked_zero_width_not_decisive():
    for form in ("percentile", "t"):
        r = I.event_bootstrap([0.02, 0.02, 0.02], form=form)
        assert r["zero_width"] and r["lo"] == r["hi"] == pytest.approx(0.02) and r["p"] == 0.0
        assert "ZERO WIDTH" in r["caveat"] and "not a decisive result" in r["caveat"]


def test_two_stage_and_large_k_are_seeded():
    rng = np.random.default_rng(7)
    d = rng.normal(0.02, 0.01, 4)
    within = [di + rng.normal(0, 0.01, 200) for di in d]
    a = I.event_bootstrap(d, within=within, n_boot=300, seed=5)
    assert a == I.event_bootstrap(d, within=within, n_boot=300, seed=5)
    assert a["lo"] != I.event_bootstrap(d, within=within, n_boot=300, seed=6)["lo"]
    assert a["scheme"].startswith("two stage")
    d8 = rng.normal(0.02, 0.01, 8)
    b = I.event_bootstrap(d8, n_boot=300, seed=5)
    assert b["scheme"] == "300 seeded draws" and b == I.event_bootstrap(d8, n_boot=300, seed=5)
    with pytest.raises(ValueError, match="one draw array per earthquake"):
        I.event_bootstrap(d, within=within[:3])
    assert I.pooled_interval(d, method="event-resample", within=within, n_boot=300, seed=5) == a


def test_event_resampling_coverage_on_a_known_truth_shows_the_small_k_caveat():
    """d_i = mu + tau z_i + s e_i, mu 0.02, tau 0.02, s 0.01; 300 meta-analyses per k, deterministic: one-stage
    percentile 0.68 at k = 3 and 0.78 at k = 4 (the caveat); one-stage t 0.96 and 0.97; two-stage percentile 0.86 at
    k = 3, 0.89 at k = 4. Bounds below leave about 3 binomial SDs."""
    mu, tau, s, reps = 0.02, 0.02, 0.01, 300
    cov = {k: {"p1": 0, "t1": 0, "p2": 0} for k in (3, 4)}
    for k in (3, 4):
        for r in range(reps):
            rng = np.random.default_rng(1000 * k + r)
            d = mu + tau * rng.normal(size=k) + s * rng.normal(size=k)
            within = [di + s * rng.normal(size=200) for di in d]
            for key, kw in (("p1", {}), ("t1", {"form": "t"}), ("p2", {"within": within})):
                o = I.event_bootstrap(d, seed=r, n_boot=500, **kw)
                cov[k][key] += covers(o["lo"], o["hi"], mu)
    rate = {k: {m: v / reps for m, v in c.items()} for k, c in cov.items()}
    assert rate[3]["p1"] <= 0.78, rate        # the small-k caveat, measured: the one-stage percentile undercovers
    assert rate[3]["p1"] < rate[4]["p1"], rate
    assert min(rate[3]["t1"], rate[4]["t1"]) >= 0.91, rate
    assert min(rate[3]["p2"], rate[4]["p2"]) >= 0.82, rate

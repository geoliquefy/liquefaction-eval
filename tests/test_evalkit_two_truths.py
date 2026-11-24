"""Tests for C2 step 1 (scoreboard/evalkit/lab/two_truths.py): the two-truths coverage table. No em dashes.

Synthetic geometry and hand-built arrays only. No real data is read.
"""
from __future__ import annotations

import numpy as np

from scoreboard.evalkit import metrics, resample
from scoreboard.evalkit.lab import two_truths as T
from scoreboard.evalkit.lab.run import CLIP, COMPS
from scoreboard.evalkit.lab.world import EVENTS, Geometry, World
from scoreboard.scorer import auc, grid_clusters


def _world(n=500, seed=11, **params):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 12000, n), rng.uniform(0, 9000, n)
    masks = {e: rng.random(n) < 0.8 for e in EVENTS}
    ln_pga = {e: np.log(rng.uniform(0.15, 0.45, n)) for e in EVENTS}
    geom = Geometry(x, y, masks, ln_pga, {e: 0.3 for e in EVENTS}, 0.1 * rng.normal(size=n),
                    (float(x.mean()), float(y.mean())))
    p = {"sigma_s": 1.0, "rho_s": 1200.0, "rho_m": 3000.0, "f1": 0.35, **params}
    return World(geom, p, {e: 2.0 for e in EVENTS}, {e: 0.4 for e in EVENTS}, calib_seed=3)


def test_split_draw_equals_world_draw_bit_for_bit():
    w = _world()
    ref = w.draw(123)
    rng = np.random.default_rng(123)
    w._reset()
    got = T.draw_events(w, T.draw_fixed(w, rng), rng)
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert a["event"] == b["event"]
        assert np.array_equal(a["idx"], b["idx"]) and np.array_equal(a["y"], b["y"]) and np.array_equal(a["z"], b["z"])
        assert all(np.array_equal(p, q) for p, q in zip(a["p"], b["p"]))


def test_replicates_keep_models_and_ground_and_redraw_labels():
    w = _world()
    rng = np.random.default_rng(5)
    w._reset()
    fixed = T.draw_fixed(w, rng)
    first = T.draw_events(w, fixed, rng)
    T.reset_redrawn(w)
    second = T.draw_events(w, fixed, rng)
    for a, b in zip(first, second):
        assert not np.array_equal(a["y"], b["y"])                    # shaking, outcomes and labels redrawn
        # entries see only ground and their fixed model error: identical probabilities
        assert all(np.array_equal(p, q) for p, q in zip(a["p"], b["p"]))


def test_reset_redrawn_forgets_spare_shaking_and_label_error_draws():
    w = _world()
    spare = np.zeros(w.g.x.size)
    w.s_pga._spare, w.s_err._spare = spare, spare
    T.reset_redrawn(w)
    assert w.s_pga._spare is None and w.s_err._spare is None


def test_tile_is_4km_as_on_the_card():
    assert T.TILE_M == 4000.0


def test_interval_event_is_the_kit_paired_percentile_interval():
    rng = np.random.default_rng(2)
    n = 400
    x, y = rng.uniform(0, 16000, n), rng.uniform(0, 12000, n)
    lab = rng.random(n) < 0.4
    ps = [1 / (1 + np.exp(-(lab * s + rng.normal(0, 1, n)))) for s in (1.0, 0.6, 0.6)]
    tiles = grid_clusters(x, y, T.TILE_M)
    out = T.interval_event(lab, ps, tiles, seed=9, n_boot=200)
    assert out.shape == (3, 2, 5)
    assert np.all(out[..., T.LOW] < out[..., T.HIGH])
    assert np.allclose(T.point_diffs(lab, ps), out[..., T.DIFF])
    # independent path (review 1): the kit's row-index tile bootstrap (resample.bootstrap, same seed), scored by
    # scorer.auc and metrics.log_loss on the drawn rows, then resample.paired. Percentile lo, hi, p and SE must agree.
    yf = lab.astype(float)

    def stat(idx):
        out_ = {}
        for k, p in enumerate(ps):
            out_[f"auc{k}"] = auc(lab[idx], p[idx])
            out_[f"log_loss{k}"] = metrics.log_loss(yf[idx], p[idx], CLIP)
        return out_

    draws = resample.bootstrap(stat, tiles, 200, 9)
    point = {"auc": [auc(lab, p) for p in ps], "log_loss": [metrics.log_loss(yf, p, CLIP) for p in ps]}
    for ci, (a, b) in enumerate(COMPS):
        for mi, m in enumerate(("auc", "log_loss")):
            ref = resample.paired(point[m][a], point[m][b], draws[f"{m}{a}"], draws[f"{m}{b}"])
            got = out[ci, mi]
            want = [ref["diff"], ref["lo"], ref["hi"], ref["p"], ref["se"]]
            assert np.allclose([got[T.DIFF], got[T.LOW], got[T.HIGH], got[T.PVAL], got[T.SEB]], want,
                               rtol=0.0, atol=1e-9), (ci, m)
    assert np.isnan(T.interval_event(np.zeros(n, bool), ps, tiles, seed=9, n_boot=20)).all()


def _hand_run(W=6, E=2):
    """Asymmetric hand fixture (review 1). Intervals [-1, 1] on comparisons 0 and 1, [0.5, 1] on comparison 2.

    Planted SEPARATES: A on slot 0, comparison 0, AUC (estimate +0.4) in every world; B on slot 1, comparison 2, AUC
    (estimate +0.75) in worlds 0 to 2. theta_w: worlds 0 and 1 are 2.0 everywhere. In worlds 2 to 5, comparisons 0
    and 1 are 0, except cell A at -0.3 (worlds 2, 3: wrong sign) and +0.3 (worlds 4, 5: right sign); comparison 2 is
    +0.2. Means over worlds: 0.667 on comparisons 0 and 1, 0.8 on comparison 2, so exact 0 and the mean differ.
    """
    res = np.zeros((W, E, 3, 2, 5))
    res[..., T.LOW], res[..., T.HIGH], res[..., T.PVAL], res[..., T.SEB] = -1.0, 1.0, 0.5, 0.5
    res[:, :, 2, :, T.LOW], res[:, :, 2, :, T.HIGH], res[:, :, 2, :, T.DIFF] = 0.5, 1.0, 0.75
    res[:, 0, 0, 0, T.DIFF] = 0.4
    res[:, 0, 0, 0, T.PVAL] = 0.001
    res[:3, 1, 2, 0, T.PVAL] = 0.001
    theta_w = np.zeros((W, E, 3, 2))
    theta_w[:2] = 2.0
    theta_w[2:4, 0, 0, :] = -0.3
    theta_w[4:, 0, 0, :] = 0.3
    theta_w[2:, :, 2, :] = 0.2
    return {"res": res, "theta_w": theta_w, "theta_w_sd": np.full(theta_w.shape, 0.1),
            "n": np.full((W, E), 500), "n_pos": np.full((W, E), 100), "n_tiles": np.full((W, E), 25),
            "slot": np.tile(np.arange(E), (W, 1)), "r_theta": np.array(10), "seconds": np.array(1.0)}


def _close(c: dict, want: float) -> bool:
    return abs(c["value"] - want) < 1e-4


def test_summary_models_fixed_coverage_and_wrong_sign_rule():
    a = T.summarise_condition("R-MED", _hand_run())["metrics"]["auc"]
    mf = a["models_fixed"]
    # worlds 0, 1: truth 2.0 never covered; worlds 2-5: comparisons 0, 1 covered, comparison 2 (0.2) not
    assert _close(mf["coverage"], 4 * (4 / 6) / 6)
    # wrong sign only on cell A in worlds 2 and 3 (truth -0.3, estimate +0.4); B has truth 2.0 or 0.2: right sign
    assert _close(mf["false_separates_per_comparison"], 2 / 36)
    assert _close(mf["false_separates_family"], 1 / 6)
    assert _close(a["separates_rate_all"], 9 / 36)


def test_summary_models_averaged_uses_exact_zero_where_exchangeable():
    med = T.summarise_condition("R-MED", _hand_run())["metrics"]["auc"]["models_averaged"]
    # comparisons 0, 1: truth 0.667 inside [-1, 1]; comparison 2: exact 0 (not the mean 0.8) outside [0.5, 1]
    assert _close(med["coverage"], 4 / 6)
    # false SEPARATES only on exact-0 cells (comparison 2): B in worlds 0-2, 1 of 2 cells, 1 of 2 families
    assert _close(med["false_separates_per_comparison"], 0.25)
    assert _close(med["false_separates_family"], 0.25)
    null = T.summarise_condition("R-NULL", _hand_run())["metrics"]["auc"]["models_averaged"]
    # NULL: every cell is exact 0; the comparison 2 interval [0.5, 1] misses it
    assert _close(null["coverage"], 4 / 6)
    # worlds 0-2: 2 of 6 cells and both families; worlds 3-5: 1 of 6 cells and 1 of 2 families
    assert _close(null["false_separates_per_comparison"], 0.25)
    assert _close(null["false_separates_family"], 0.75)


def test_summary_gate_passing_coverage_drops_failing_slots():
    r = _hand_run()
    r["n_tiles"][:, 1] = 17                    # slot 1 fails the count gate (20 tiles), as Dec does
    mf = T.summarise_condition("R-MED", r)["metrics"]["auc"]["models_fixed"]
    # slot 0 only: worlds 0, 1 none covered; worlds 2-5 comparisons 0 and 1 of 3
    assert _close(mf["coverage_gate_passing"], 4 * (2 / 3) / 6)


def test_exact_mask_and_holm_match_the_lab():
    assert T.exact_mask("R-NULL", 4).all()
    m = T.exact_mask("R-MED", 4)
    assert m[:, 2].all() and not m[:, :2].any()
    p = np.random.default_rng(1).random((3, 2, 3, 2)) * 0.06
    sep = T.holm_separates(p)
    for w in range(3):
        for e in range(2):
            for mi in range(2):
                assert list(sep[w, e, :, mi]) == resample.holm(list(p[w, e, :, mi]))


def test_band_reads_the_target_with_monte_carlo_error():
    assert T.band({"value": 0.95, "se": 0.01}) == "IN BAND"
    assert T.band({"value": 0.90, "se": 0.01}) == "UNDER"
    assert T.band({"value": 0.92, "se": 0.01}) == "IN BAND"      # upper end 0.9396 reaches 0.93
    assert T.band({"value": 0.995, "se": 0.002}) == "OVER"

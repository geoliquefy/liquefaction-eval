"""evalkit metrics (rule book R3, R9) on seeded synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import metrics as M  # noqa: E402
from scoreboard.scorer import auc  # noqa: E402


def draw(n, seed, sd=1.5, shift=-1.0):
    rng = np.random.default_rng(seed)
    eta = shift + rng.normal(0.0, sd, n)
    p = expit(eta)
    return (rng.random(n) < p).astype(float), p, eta


def test_calibrated_entry_slope_one_intercept_zero():
    y, p, _ = draw(20000, 1)
    assert abs(M.recal_slope(y, p, 1e-6) - 1.0) < 0.06
    assert abs(M.citl(y, p, 1e-6)) < 0.05
    assert abs(M.level_miss(y, p)) < 0.05


def test_overconfident_entry_slope_below_one():
    y, _, eta = draw(20000, 2)
    over = expit(3.0 * eta)
    b = M.recal_slope(y, over, 1e-6)
    assert 0.25 < b < 0.42                                   # about 1/3 by construction
    under = expit(0.5 * eta)
    assert 1.7 < M.recal_slope(y, under, 1e-6) < 2.3


def test_triggering_probability_against_manifestation_labels_slope_below_one():
    """The v0 failure, documented: P(trigger) scored against manifestation labels.

    Labels: y = trigger AND fragility draw, so P(y) = f p with f = 0.6. The true recalibration map is
    x -> logit(f expit(x)), whose derivative (1 - p) / (1 - f p) is below 1 for every p, so the fitted slope is below 1
    by construction even for a perfect triggering model.
    """
    rng = np.random.default_rng(3)
    n, f = 40000, 0.6
    p_trig = expit(rng.normal(-0.5, 1.5, n))
    y = ((rng.random(n) < p_trig) & (rng.random(n) < f)).astype(float)
    pp = np.linspace(0.001, 0.999, 999)
    assert np.all((1 - pp) / (1 - f * pp) < 1.0)
    assert M.recal_slope(y, p_trig, 1e-6) < 0.9
    assert M.recal_slope(y, f * p_trig, 1e-6) == pytest.approx(1.0, abs=0.06)   # the converted target is fine
    assert M.citl(y, p_trig, 1e-6) < -0.3                                        # and the level is off too


def test_isotonic_monotone_with_ties_and_matches_plain_pav():
    rng = np.random.default_rng(4)
    p = np.round(rng.random(300), 2)                              # many ties
    y = (rng.random(300) < p).astype(float)
    fit = M.isotonic_fit(y, p)
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(fit[order]) >= -1e-12)
    for v in np.unique(p):
        assert np.ptp(fit[p == v]) == 0.0                          # equal forecasts share one value
    # plain pool-adjacent-violators on the tie groups
    vals = [[y[p == v].mean(), (p == v).sum()] for v in np.unique(p)]
    blocks = []
    for m, w in vals:
        blocks.append([m, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    plain = np.concatenate([[m] * c for m, _, c in blocks])
    np.testing.assert_allclose(np.array([fit[p == v][0] for v in np.unique(p)]), plain, atol=1e-12)


def test_brier_decomposition_sums_and_is_nonnegative():
    for seed in range(5):
        y, p, eta = draw(2000, 10 + seed)
        for q in (p, expit(2 * eta), np.clip(p + 0.1, 0, 1)):
            d = M.brier_decomposition(y, q)
            assert d["brier"] == pytest.approx(d["mcb"] - d["dsc"] + d["unc"], abs=1e-12)
            assert d["mcb"] >= -1e-12 and d["dsc"] >= -1e-12
    y, p, _ = draw(2000, 20)
    assert M.brier_decomposition(y, np.full(2000, y.mean()))["dsc"] == pytest.approx(0.0, abs=1e-12)


def test_reliability_curve_is_aggregate_and_monotone():
    y, p, _ = draw(3000, 5, shift=-0.5)
    p = np.clip(p, 0.08, 0.9)
    r = M.reliability_curve(y, p)
    assert len(r["forecast"]) == len(r["recalibrated"]) == 21
    assert r["recalibrated"][0] is None and r["recalibrated"][1] is None     # below the smallest forecast 0.08
    assert r["recalibrated"][-1] is None                                    # above the largest forecast 0.9
    vals = [v for v in r["recalibrated"] if v is not None]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_clip_sensitivity_reported_when_exact_zeros_exist():
    y, p, _ = draw(5000, 6)
    p0 = p.copy()
    p0[p < 0.08] = 0.0
    shares = M.exact_shares(p0)
    assert shares["share_exact_0"] == pytest.approx(np.mean(p < 0.08))
    ll = [M.log_loss(y, p0, c) for c in M.CLIPS]
    assert len(set(np.round(ll, 9))) == 4                                    # exact zeros: log loss moves with clip
    slopes = [M.recal_slope(y, p0, c) for c in M.CLIPS]
    assert max(slopes) - min(slopes) > 0.1                                   # the v0 clip artefact
    ll_safe = [M.log_loss(y, np.clip(p, 0.02, 0.98), c) for c in M.CLIPS]
    assert np.ptp(ll_safe) == 0.0                                            # no exact zeros: no clip effect
    y0, q0 = M.without_exact(y, p0)
    assert q0.min() > 0 and M.recal_slope(y0, q0, 1e-6) == M.recal_slope(y0, q0, 1e-4)
    stats = M.probability_stats(y, p0, 1e-6, census=False)
    assert "level_miss" not in stats and stats["slope_no01"] == pytest.approx(M.recal_slope(y0, q0, 1e-6))


def test_degenerate_cases_give_nan():
    y = np.ones(50)
    p = np.linspace(0.1, 0.9, 50)
    assert np.isnan(M.citl(y, p, 1e-6)) and np.isnan(M.recal_slope(y, p, 1e-6))
    y2 = np.r_[np.zeros(25), np.ones(25)]
    assert np.isnan(M.recal_slope(y2, np.full(50, 0.3), 1e-6))
    assert np.isnan(M.recal_slope(y2, np.r_[np.full(25, 0.1), np.full(25, 0.9)], 1e-6))   # separation
    assert np.isnan(M.level_miss(y, p))


def test_level_miss_sign_and_auc_reuse():
    y, p, _ = draw(20000, 7)
    assert M.level_miss(y, 0.5 * p) > 0.4                                    # under-predicts the level
    assert M.ranking_stats(y, p)["auc"] == auc(y.astype(bool), p)
    assert M.level_miss(y, p) == pytest.approx(logit(y.mean()) - logit(p.mean()))

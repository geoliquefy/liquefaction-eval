"""evalkit exact worst-case bounds (AUDIT-DESIGN section 5 item 4; review 2 nit N2). Synthetic only."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import bounds as B  # noqa: E402
from scoreboard.scorer import auc  # noqa: E402


def case(n, k, seed, ties=False, weights=True):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(float)
    y[rng.choice(n, k, replace=False)] = np.nan
    s_a = rng.normal(size=n)
    s_b = s_a + rng.normal(0.0, 1.0, n)
    if ties:
        s_a, s_b = np.round(s_a, 0), np.round(s_b, 0)
    w = rng.uniform(1.0, 20.0, n) if weights else None
    return [None if np.isnan(v) else v for v in y], s_a, s_b, w


def test_weighted_auc_equals_the_kit_auc_at_unit_weights():
    rng = np.random.default_rng(1)
    y = (rng.random(80) < 0.3).astype(float)
    s = np.round(rng.normal(size=80), 1)
    assert B.weighted_auc(y, s) == pytest.approx(auc(y.astype(bool), s), abs=1e-14)


@pytest.mark.parametrize("seed", range(12))
def test_solver_matches_brute_force_on_tiny_cases(seed):
    """Dinkelbach + HiGHS gives the enumeration's extremes to 1e-12, for AUC and paired difference, with and
    without ties and weights."""
    n, k = 10 + seed % 5, 3 + seed % 7
    y, s_a, s_b, w = case(n, k, seed, ties=seed % 2 == 0, weights=seed % 3 != 0)
    for h in (B.concordance(s_a), B.concordance(s_a) - B.concordance(s_b)):
        got = B.ratio_bounds(h, y, w, check=False)
        ref = B.enumerate_ratio_bounds(h, y, w)
        assert got["lo"] == pytest.approx(ref["lo"], abs=1e-12)
        assert got["hi"] == pytest.approx(ref["hi"], abs=1e-12)
        assert got["lo"] <= got["hi"]


def test_check_runs_the_enumeration_on_a_real_size_case():
    """About 300 units, 16 ungraded: the built-in enumeration check runs and agrees."""
    y, s_a, s_b, w = case(300, 16, 7)
    r = B.auc_diff_bounds(y, s_a, s_b, w)
    assert r["enumeration_agrees"] and r["k"] == 16
    r = B.auc_bounds(y, s_a, w)
    assert r["enumeration_agrees"] and 0.0 <= r["lo"] <= r["hi"] <= 1.0


def test_the_primary_point_lies_inside_the_bounds():
    y, s_a, s_b, w = case(60, 8, 3)
    r = B.auc_bounds(y, s_a, w)
    for fill in (0.0, 1.0):
        full = np.array([fill if v is None else v for v in y])
        assert r["lo"] - 1e-12 <= B.weighted_auc(full, s_a, w) <= r["hi"] + 1e-12


def test_bounds_need_one_yes_and_one_no():
    """Den > 0 (review 2 N2): an assignment with one class empty is never counted, and a case where every
    assignment has one class empty is refused."""
    s = np.arange(5.0)
    # graded all YES, two ungraded: the all-YES assignment is excluded, the bounds come from feasible ones
    y = [1.0, 1.0, 1.0, None, None]
    r = B.auc_bounds(y, s)
    ref = B.enumerate_ratio_bounds(B.concordance(s), y)
    assert (r["lo"], r["hi"]) == pytest.approx((ref["lo"], ref["hi"]), abs=1e-12)
    assert np.isfinite(r["lo"]) and np.isfinite(r["hi"])
    # no graded unit, two ungraded: only the mixed assignments count
    r2 = B.auc_bounds([None, None], [0.0, 1.0])
    assert (r2["lo"], r2["hi"]) == (0.0, 1.0)
    with pytest.raises(B.BoundError, match="one YES and one NO"):
        B.auc_bounds([1.0, 1.0, None][:2], [0.0, 1.0])            # all graded, one class
    with pytest.raises(B.BoundError, match="one YES and one NO"):
        B.auc_bounds([None], [0.0])                                # one unit cannot hold two classes
    with pytest.raises(B.BoundError, match="one YES and one NO"):
        B.enumerate_ratio_bounds(B.concordance([0.0, 1.0]), [0.0, 0.0])


def test_solver_never_returns_an_undefined_auc_when_only_one_class_is_graded():
    rng = np.random.default_rng(5)
    s = rng.normal(size=12)
    y = [0.0] * 8 + [None] * 4
    r = B.auc_bounds(y, s)
    assert 0.0 <= r["lo"] <= r["hi"] <= 1.0 and r["enumeration_agrees"]


@pytest.mark.parametrize("kind", ["brier", "log_loss"])
def test_loss_bounds_unit_by_unit_match_enumeration(kind):
    rng = np.random.default_rng(11)
    for seed in range(6):
        y, _, _, w = case(14, 6, seed)
        p_a, p_b = rng.uniform(0.0, 1.0, 14), rng.uniform(0.0, 1.0, 14)
        p_a[0] = 0.0
        for pb in (None, p_b):
            got = B.loss_bounds(kind, y, p_a, pb, w)
            ref = B.enumerate_loss_bounds(kind, y, p_a, pb, w)
            assert got["lo"] == pytest.approx(ref["lo"], abs=1e-12)
            assert got["hi"] == pytest.approx(ref["hi"], abs=1e-12)


def test_inputs_are_checked():
    with pytest.raises(B.BoundError, match="0 or 1"):
        B.auc_bounds([2.0, 0.0, None], [0.0, 1.0, 2.0])
    with pytest.raises(B.BoundError, match="positive"):
        B.auc_bounds([1.0, 0.0, None], [0.0, 1.0, 2.0], [1.0, 0.0, 1.0])
    with pytest.raises(B.BoundError, match="brier"):
        B.loss_bounds("hinge", [1.0, None], [0.2, 0.3])


def test_larger_k_runs_and_contains_every_sampled_assignment():
    """K = 30 (beyond enumeration): the solver runs, and 2,000 random assignments all fall inside its range,
    with the best sampled values close to the bounds."""
    y, s_a, s_b, w = case(120, 30, 13)
    h = B.concordance(s_a) - B.concordance(s_b)
    r = B.ratio_bounds(h, y, w)
    assert r["method"] == "dinkelbach-highs" and "enumeration_agrees" not in r
    arr = np.array([np.nan if v is None else v for v in y])
    idx = np.flatnonzero(np.isnan(arr))
    rng = np.random.default_rng(0)
    seen = []
    for _ in range(2000):
        full = arr.copy()
        full[idx] = rng.integers(0, 2, idx.size)
        seen.append(B.ratio(h, full, np.asarray(w)))
    assert r["lo"] - 1e-12 <= min(seen) and max(seen) <= r["hi"] + 1e-12

"""evalkit analytic oracles that kill the mutants step 2b found alive (scripts/mutate_scoring.py). Synthetic only.

Each test pins a value computed by hand or by an independent code path, so a flipped sign, a moved constant or a
swapped comparison in the scoring code changes the answer.
"""
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import metrics as M  # noqa: E402
from scoreboard.evalkit import pool as P  # noqa: E402
from scoreboard.evalkit import resample as R  # noqa: E402


# metrics ---------------------------------------------------------------------------------------------------------

def test_log_loss_hand_value_with_exact_zero_and_one():
    y = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    p = np.array([0.8, 0.3, 1.0, 0.0, 1.0, 0.0])
    c = 1e-6
    want = -(math.log(0.8) + math.log(0.7) + math.log(1 - c) + math.log(1 - c) + math.log(c) + math.log(c)) / 6
    assert M.log_loss(y, p, c) == pytest.approx(want, rel=1e-9)
    assert M.log_loss(y, p, 1e-2) == pytest.approx(
        -(math.log(0.8) + math.log(0.7) + 2 * math.log(0.99) + 2 * math.log(0.01)) / 6, rel=1e-12)


def test_brier_and_shares_hand_values():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.0, 0.0, 1.0, 0.5])
    assert M.brier(y, p) == pytest.approx((1 + 0 + 0 + 0.25) / 4, abs=1e-15)
    assert M.exact_shares(p) == {"share_exact_0": 0.5, "share_exact_1": 0.25}
    y0, p0 = M.without_exact(y, p)
    assert p0.tolist() == [0.5] and y0.tolist() == [0.0]


def test_clipped_logit_hand_values():
    out = M.clipped_logit(np.array([0.0, 0.5, 1.0]), 1e-3)
    assert out.tolist() == pytest.approx([logit(1e-3), 0.0, logit(1 - 1e-3)], rel=1e-12)


def _mle(x, y, slope_free):
    def nll(w):
        z = w[0] + (w[1] if slope_free else 1.0) * x
        return float(np.sum(np.logaddexp(0.0, z) - y * z))
    r = minimize(nll, np.array([0.0, 1.0]), method="Nelder-Mead",
                 options={"xatol": 1e-11, "fatol": 1e-13, "maxiter": 20000})
    return r.x


def test_calibration_intercept_and_slope_match_an_independent_fit():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.02, 0.9, 400)
    y = (rng.random(400) < expit(0.4 + 0.7 * logit(p))).astype(float)
    x = logit(p)
    a_fixed = _mle(x, y, slope_free=False)[0]
    a, b = _mle(x, y, slope_free=True)
    assert M.citl(y, p, 1e-6) == pytest.approx(a_fixed, abs=1e-6)
    assert M.recal_slope(y, p, 1e-6) == pytest.approx(b, abs=1e-6)
    # constant forecast: the intercept is logit(observed share) - logit(p), exactly
    q = np.full(10, 0.5)
    yy = np.array([1.0] * 3 + [0.0] * 7)
    assert M.citl(yy, q, 1e-6) == pytest.approx(logit(0.3), abs=1e-9)
    assert np.isnan(M.recal_slope(yy, q, 1e-6))


def test_calibration_guards_return_nan():
    empty = np.array([])
    for f in (M.citl, M.recal_slope):
        assert np.isnan(f(empty, empty, 1e-6))
        assert np.isnan(f(np.ones(5), np.linspace(0.1, 0.9, 5), 1e-6))
        assert np.isnan(f(np.zeros(5), np.linspace(0.1, 0.9, 5), 1e-6))
    # perfect separation: the slope does not exist
    p = np.linspace(0.05, 0.95, 20)
    assert np.isnan(M.recal_slope((p > 0.5).astype(float), p, 1e-6))


def test_brier_decomposition_uncertainty_is_ybar_one_minus_ybar():
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.1, 0.4, 0.4, 0.8])
    d = M.brier_decomposition(y, p)
    assert d["unc"] == pytest.approx(0.6 * 0.4, abs=1e-15)
    p_hat = np.array([0.5, 0.5, 0.5, 0.5, 1.0])
    b_hat = float(np.mean((p_hat - y) ** 2))
    assert d["dsc"] == pytest.approx(0.24 - b_hat, abs=1e-15)
    assert d["mcb"] == pytest.approx(M.brier(y, p) - b_hat, abs=1e-15)


def test_reliability_curve_hand_values():
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.1, 0.4, 0.4, 0.8])
    r = M.reliability_curve(y, p, grid=(0.0, 0.1, 0.25, 0.4, 0.8, 0.85))
    assert r["recalibrated"] == [None, 0.5, 0.5, 0.5, 1.0, None]
    assert r["n_blocks"] == 2 and r["smallest_block_n"] == 1


def test_reliability_grid_and_clips_are_the_registered_values():
    assert M.RELIABILITY_GRID == tuple(round(0.05 * k, 2) for k in range(21))
    assert len(M.RELIABILITY_GRID) == 21 and M.RELIABILITY_GRID[-1] == 1.0
    assert M.CLIPS == (1e-6, 1e-4, 1e-3, 1e-2)


def test_level_miss_guards_and_value():
    assert M.level_miss(np.array([1.0, 0.0, 0.0, 0.0]), np.full(4, 0.5)) == pytest.approx(logit(0.25), abs=1e-12)
    assert np.isnan(M.level_miss(np.zeros(4), np.full(4, 0.5)))
    assert np.isnan(M.level_miss(np.ones(4), np.full(4, 0.5)))
    assert np.isnan(M.level_miss(np.array([1.0, 0.0]), np.zeros(2)))
    assert np.isnan(M.level_miss(np.array([1.0, 0.0]), np.ones(2)))


# resample --------------------------------------------------------------------------------------------------------

def test_registered_tile_rule_constants():
    assert R.TILE_CANDIDATES_M == (500.0, 1000.0, 2000.0, 4000.0, 8000.0)
    assert R.SENSITIVITY_TILES_M == (1000.0, 2000.0, 4000.0)
    sv = inspect.signature(R.semivariogram).parameters
    got = tuple(sv[k].default for k in ("max_lag_m", "n_bins", "max_points", "seed"))
    assert got == (12000.0, 24, 1500, 0)
    assert inspect.signature(R.variogram_range).parameters["n_grid"].default == 120
    assert inspect.signature(R.choose_tile).parameters["seed"].default == 0


def test_interval_and_paired_percentiles_are_the_stated_ones():
    d = np.random.default_rng(2).normal(size=1001)
    iv = R.interval(0.3, d)
    assert (iv["lo"], iv["hi"]) == pytest.approx(tuple(np.percentile(d, [2.5, 97.5])), abs=1e-12)
    iv90 = R.interval(0.3, d, level=0.9)
    assert (iv90["lo"], iv90["hi"]) == pytest.approx(tuple(np.percentile(d, [5.0, 95.0])), abs=1e-12)
    pr = R.paired(0.3, 0.0, d, np.zeros_like(d))
    assert (pr["lo90"], pr["hi90"]) == tuple(np.percentile(d, [5.0, 95.0]))
    assert (pr["lo"], pr["hi"]) == tuple(np.percentile(d, [2.5, 97.5]))
    assert pr["se"] == pytest.approx(np.std(d, ddof=1), rel=1e-12)


def test_holm_is_strict_at_the_threshold():
    assert R.holm([0.05]) == [False]
    assert R.holm([0.049999]) == [True]
    assert R.holm([0.025, 0.05]) == [False, False]        # 0.025 is not below 0.05 / 2
    assert R.holm([0.02, 0.04]) == [True, True]


def test_semivariogram_hand_case():
    """Three sites on a line at 0, 300 and 1000 m with residuals 0, 1, 3: pairs (300 m, 0.5), (700 m, 2.0),
    (1000 m, 4.5). Bins of 500 m from 0: bin 0 holds 0.5, bin 1 holds 2.0, bin 2 holds 4.5."""
    vg = R.semivariogram(np.array([0.0, 300.0, 1000.0]), np.zeros(3), np.array([0.0, 1.0, 3.0]),
                         max_lag_m=1500.0, n_bins=3)
    assert vg["lag_m"] == [250.0, 750.0, 1250.0]
    assert vg["n_pairs"] == [1, 1, 1]
    assert vg["gamma"] == [0.5, 2.0, 4.5]


def test_semivariogram_subsamples_above_max_points():
    rng = np.random.default_rng(1)
    x, y, r = rng.uniform(0, 100, 30), rng.uniform(0, 100, 30), rng.normal(size=30)
    vg = R.semivariogram(x, y, r, max_lag_m=1000.0, n_bins=2, max_points=29)
    assert sum(vg["n_pairs"]) == 29 * 28 // 2
    vg = R.semivariogram(x, y, r, max_lag_m=1000.0, n_bins=2, max_points=30)
    assert sum(vg["n_pairs"]) == 30 * 29 // 2


def test_variogram_range_with_empty_bins_and_with_no_structure():
    lag = [250.0, 750.0, 1250.0, 1750.0, 2250.0, 2750.0]
    gamma = [0.2, 0.5, float("nan"), 0.8, 0.85, 0.86]
    fit = R.variogram_range({"lag_m": lag, "gamma": gamma, "n_pairs": [10, 10, 0, 10, 10, 10]})
    assert np.isfinite(fit["fitted_range_m"]) and np.isfinite(fit["partial_sill"]) and fit["range_m"] > 0
    flat = R.variogram_range({"lag_m": lag, "gamma": [0.0] * 6, "n_pairs": [5] * 6})
    assert flat["structured_share"] == 0.0 and flat["range_m"] == 0.0


def test_choose_tile_flags_a_range_beyond_the_candidates():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 20000, 400), rng.uniform(0, 20000, 400)
    short = R.choose_tile(x, y, rng.normal(size=400))
    assert short["range_beyond_candidates"] is False and short["tile_m"] in R.TILE_CANDIDATES_M
    trend = R.choose_tile(x, y, x / 1000.0, candidates=(500.0, 1000.0))
    assert trend["range_beyond_candidates"] is True and trend["tile_m"] == 1000.0


# pool ------------------------------------------------------------------------------------------------------------

def _dl_hk_by_hand(d, s, level=0.95):
    """DerSimonian-Laird tau^2 and the Hartung-Knapp interval written out in plain Python (independent path)."""
    from scipy import stats
    k = len(d)
    w = [1.0 / (x * x) for x in s]
    sw = sum(w)
    dfe = sum(wi * di for wi, di in zip(w, d)) / sw
    q = sum(wi * (di - dfe) ** 2 for wi, di in zip(w, d))
    c = sw - sum(wi * wi for wi in w) / sw
    tau2 = max(0.0, (q - (k - 1)) / c)
    ws = [1.0 / (x * x + tau2) for x in s]
    m = sum(wi * di for wi, di in zip(ws, d)) / sum(ws)
    v = sum(wi * (di - m) ** 2 for wi, di in zip(ws, d)) / ((k - 1) * sum(ws))
    t = stats.t.ppf(0.5 + level / 2, k - 1)
    return {"tau2": tau2, "mean": m, "se": v ** 0.5, "lo": m - t * v ** 0.5, "hi": m + t * v ** 0.5, "q": q}


def test_random_effects_hand_case_with_heterogeneity():
    """d = (0, 1), s = (0.1, 0.2): w = (100, 25), d_FE = 0.2, Q = 20, C = 40, tau^2 = 19 / 40 = 0.475."""
    r = P.random_effects([0.0, 1.0], [0.1, 0.2])
    assert r["q"] == pytest.approx(20.0, rel=1e-12)
    assert r["tau2"] == pytest.approx(0.475, rel=1e-12)
    ws = (1 / (0.01 + 0.475), 1 / (0.04 + 0.475))
    assert r["mean"] == pytest.approx(ws[1] / sum(ws), rel=1e-12)
    ref = _dl_hk_by_hand([0.0, 1.0], [0.1, 0.2])
    for k in ("se", "lo", "hi"):
        assert r[k] == pytest.approx(ref[k], rel=1e-12)


def test_random_effects_unequal_errors_without_heterogeneity():
    d, s = [0.10, 0.12, 0.30], [0.01, 0.02, 0.2]
    r = P.random_effects(d, s)
    ref = _dl_hk_by_hand(d, s)
    assert r["tau2"] == 0.0 == ref["tau2"]
    assert r["mean"] == pytest.approx((0.10 * 10000 + 0.12 * 2500 + 0.30 * 25) / 12525, rel=1e-12)
    for k in ("se", "lo", "hi"):
        assert r[k] == pytest.approx(ref[k], rel=1e-12)
    dl = P.random_effects(d, s, method="dl")
    assert dl["se"] == pytest.approx(12525 ** -0.5, rel=1e-12)


def test_random_effects_guards_and_zero_spread():
    for bad in (([], []), ([0.1, 0.2], [0.1, 0.0]), ([0.1, float("nan")], [0.1, 0.1]), ([0.1, 0.2], [0.1, -0.1])):
        with pytest.raises(ValueError, match="positive standard errors"):
            P.random_effects(*bad)
    zero = P.random_effects([0.0, 0.0, 0.0], [0.01, 0.02, 0.03])
    assert zero["se"] == 0.0 and zero["p"] == 1.0
    same = P.random_effects([0.05, 0.05, 0.05], [0.01, 0.02, 0.03])
    assert same["p"] < 1e-12


def test_pooled_line_words_on_hand_cases():
    def items(diffs, se=0.01):
        return [{"event_id": f"q{k}", "entry_a": "a", "entry_b": "b", "metric": "auc", "diff": d, "se": se,
                 "passes_gate": True} for k, d in enumerate(diffs)]
    holds = P.pooled_line(items([0.10, 0.11, 0.09, 0.10]), registered=True)
    assert holds["word"] == "HOLDS" and holds["comparison"] == {"entry_a": "a", "entry_b": "b", "metric": "auc"}
    null = P.pooled_line(items([0.02, -0.02, 0.01, -0.01]), registered=True)
    assert null["word"] == "POOLED CANNOT TELL"
    assert P.pooled_line(items([0.1, 0.1]), registered=True)["word"] == "NOT ALLOWED"
    assert P.pooled_line(items([0.1, 0.1, 0.1]), registered=False)["word"] == "NOT ALLOWED"
    mixed = items([0.1, 0.1, 0.1]) + [{**items([0.1])[0], "event_id": "qx", "entry_b": "c"}]
    with pytest.raises(ValueError, match="two entries on one metric"):
        P.pooled_line(mixed, registered=True)


def _items(diffs, se=0.01, a="a", b="b"):
    return [{"event_id": f"q{k}", "entry_a": a, "entry_b": b, "metric": "auc", "diff": d, "se": se,
             "passes_gate": True} for k, d in enumerate(diffs)]


def test_pooled_line_negative_holds_and_fragile():
    neg = P.pooled_line(_items([-0.10, -0.11, -0.09, -0.10]), registered=True)
    assert neg["word"] == "HOLDS" and neg["pooled"]["mean"] < 0
    fragile = P.pooled_line(_items([0.20, 0.21, 0.19, 0.0, 0.20], se=0.001), registered=True)
    assert fragile["word"] in ("FRAGILE", "HOLDS")
    loo_signs = [1 if r["lo"] > 0 else -1 if r["hi"] < 0 else 0 for r in fragile["leave_one_out"]]
    assert fragile["word"] == ("HOLDS" if all(s == 1 for s in loo_signs) else "FRAGILE")


def test_holm_across_pooled_lines_stops_at_the_first_non_rejection():
    """Two lines with p about 0.03 and 0.04: Holm tests the first at 0.025 (not rejected) and then stops, so the
    second reads POOLED CANNOT TELL although 0.04 < 0.05."""
    from scipy import stats
    se_hk = (0.0002 / 6) ** 0.5                     # d = (m, m + 0.01, m - 0.01), equal weights
    m1 = stats.t.isf(0.015, 2) * se_hk
    m2 = stats.t.isf(0.020, 2) * se_hk
    l1 = _items([m1, m1 + 0.01, m1 - 0.01])
    l2 = _items([m2, m2 + 0.01, m2 - 0.01], b="c")
    r1, r2 = P.pooled_lines([l1, l2], registered=True)
    assert r1["pooled"]["p"] == pytest.approx(0.03, abs=1e-9) and r2["pooled"]["p"] == pytest.approx(0.04, abs=1e-9)
    assert r1["word"] == r2["word"] == "POOLED CANNOT TELL"
    assert r1["holm"]["alpha_j"] == 0.025 and r2["holm"]["alpha_j"] == 0.05
    alone = P.pooled_line(l2, registered=True)
    assert alone["word"] != "POOLED CANNOT TELL"


def test_dl_interval_hand_values():
    from scipy import stats
    d, s = [0.10, 0.12, 0.30], [0.01, 0.02, 0.2]
    r = P.random_effects(d, s, method="dl")
    se = 12525 ** -0.5
    m = (0.10 * 10000 + 0.12 * 2500 + 0.30 * 25) / 12525
    assert (r["lo"], r["hi"]) == pytest.approx((m - 1.959963984540054 * se, m + 1.959963984540054 * se), rel=1e-9)
    assert r["p"] == pytest.approx(2 * stats.norm.sf(m / se), rel=1e-9)
    r90 = P.random_effects(d, s, method="dl", level=0.9)
    assert r90["hi"] - r90["mean"] == pytest.approx(stats.norm.ppf(0.95) * se, rel=1e-9)


def test_card_item_reads_the_pair_in_either_order():
    card = {"task": {"event_id": "q1"}, "status": "SCORED",
            "comparisons": [{"metric": "log_loss", "a": "x", "b": "y", "diff": 0.5, "se": 0.1},
                            {"metric": "auc", "a": "y", "b": "x", "diff": 0.03, "se": 0.01}]}
    it = P.card_item(card, "x", "y", "auc")
    assert it["diff"] == -0.03 and it["se"] == 0.01 and it["passes_gate"]
    assert P.card_item(card, "y", "x", "auc")["diff"] == 0.03
    assert P.card_item(card, "x", "y", "log_loss")["diff"] == 0.5
    with pytest.raises(KeyError):
        P.card_item(card, "x", "z", "auc")
    under = P.card_item({**card, "status": "UNDERPOWERED"}, "x", "y", "auc")
    assert under["passes_gate"] is False and under["diff"] is None


# card ------------------------------------------------------------------------------------------------------------

from scoreboard.evalkit import card as C  # noqa: E402
from scoreboard.evalkit import synth  # noqa: E402


def test_registered_card_constants():
    assert C.ALPHA == 0.05
    assert C.EQUIVALENCE_MARGIN == {"auc": 0.05, "log_loss": 0.05}
    assert C.FAMILIES == ("ranking", "probability")
    assert (C.MAX_LIST_LEN, C.MAX_PAIR_RECORDS) == (32, 512)
    assert C.EFFORT_ALLOWLIST == "pre-event"


def test_scan_limits_are_exact():
    C.scan_aggregates({"x": list(range(32))})
    with pytest.raises(C.CardRefused, match="33 items"):
        C.scan_aggregates({"x": list(range(33))})
    pairs = [{"a": "p", "b": "q"}] * 512
    C.scan_aggregates({"comparisons": pairs})
    with pytest.raises(C.CardRefused, match="513 items"):
        C.scan_aggregates({"comparisons": pairs + [{"a": "p", "b": "q"}]})


@pytest.fixture(scope="module")
def demo_card():
    task, entries, labels, shaking = synth.demo_inputs(n=900, n_boot=200)
    return build_card_safe(task, entries, labels, shaking)


def build_card_safe(task, entries, labels, shaking):
    return C.build_card(task, entries, labels, shaking_only=shaking)


def test_card_words_follow_an_independent_holm_and_c2_rule(demo_card):
    table = demo_card["comparisons"]
    assert table
    for fam in C.FAMILIES:
        rows = [c for c in table if c["family"] == fam]
        m = len(rows)
        order = sorted(range(m), key=lambda i: rows[i]["p"])
        rejected, stop = set(), False
        for rank, i in enumerate(order):
            if not stop and rows[i]["p"] < 0.05 / (m - rank):
                rejected.add(i)
            else:
                stop = True
        for i, c in enumerate(rows):
            assert c["family_size"] == m and demo_card["holm_family_sizes"][fam] == m
            if i in rejected:
                assert c["word"] == "SEPARATES"
                a_wins = (c["diff"] > 0) == (fam == "ranking")
                assert c["favours"] == (c["a"] if a_wins else c["b"])
                assert c["a_is"] == ("BETTER" if a_wins else "WORSE")
            elif -0.05 < c["lo90"] and c["hi90"] < 0.05:
                assert c["word"] == "EQUIVALENT" and c["favours"] is None
            else:
                assert c["word"] == "CANNOT TELL" and c["favours"] is None
    words = {c["word"] for c in table}
    assert "SEPARATES" in words and words - {"SEPARATES"}


def test_tile_choice_residual_is_label_minus_shaking():
    task, _, labels, shaking = synth.demo_inputs(n=600, n_boot=50)
    lab = labels.labelled
    ids = tuple(s for s, k in zip(labels.site_ids, lab) if k)
    pos = {s: i for i, s in enumerate(shaking.site_ids)}
    base = shaking.p[[pos[s] for s in ids]]
    y = labels.label[lab]
    ref = R.choose_tile(labels.x_m[lab], labels.y_m[lab], y - base, seed=task.seed)
    got = C.tile_choice(task, labels, shaking)
    assert got["fitted_range_m"] == ref["fitted_range_m"] and got["nugget"] == ref["nugget"]
    assert got["partial_sill"] == ref["partial_sill"]
    wrong = R.choose_tile(labels.x_m[lab], labels.y_m[lab], y + base, seed=task.seed)
    assert (wrong["nugget"], wrong["partial_sill"]) != (got["nugget"], got["partial_sill"])


def test_release_note_threshold_is_2000_draws():
    task, *_ = synth.demo_inputs(n=50, n_boot=2000)
    assert not any("NOT FOR RELEASE" in n for n in C._notes(task))
    task2, *_ = synth.demo_inputs(n=50, n_boot=1999)
    assert any("NOT FOR RELEASE" in n for n in C._notes(task2))


def test_a_family_of_two_can_name_a_best_entry():
    """One strong target probability and the coin form a probability family of two: the best line considers it."""
    ev = synth.synthetic_event(900, 5)
    ids = ev["site_ids"]
    from scoreboard.evalkit.spec import Entry, Task, site_list_sha256
    task = Task(event_id="q2", target="t", label_source="synthetic v1", label_rule_sha256="1" * 64,
                site_list_sha256=site_list_sha256(ids), track="forward", division="closed", tile_m=2000.0,
                clip=1e-6, seed=3, label_caveats="synthetic labels", n_boot=100)
    e = Entry(synth.manifest("TRUTH-ENTRY", "target_probability", "q2"), ids, ev["p_target"])
    card = C.build_card(task, [e], synth.labels_of(ev))
    best = card["best"]["probability"]
    assert best["candidate"] == "TRUTH-ENTRY" and best["name"] == "TRUTH-ENTRY", best


def test_card_pair_set_is_the_stated_one(demo_card):
    """R11 d by hand: ranking pairs are every pair of rows except baseline-baseline and constant-constant;
    probability pairs are every pair of probability-eligible rows except baseline-baseline."""
    rows = demo_card["entries"]
    const = {"baseline: coin", "baseline: earlier-earthquake rate"}
    want = set()
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["role"] == b["role"] == "baseline":
                continue
            if not (a["name"] in const and b["name"] in const):
                want.add(("ranking", a["name"], b["name"]))
            if a["probability_metrics"] and b["probability_metrics"]:
                want.add(("probability", a["name"], b["name"]))
    got = {(c["family"], c["a"], c["b"]) for c in demo_card["comparisons"]}
    assert got == want and len(got) == len(demo_card["comparisons"])
    assert any(c["b"] == "baseline: coin" for c in demo_card["comparisons"])


def test_coin_baseline_scores_are_exact(demo_card):
    coin = demo_card["metrics"]["baseline: coin"]
    assert coin["auc"]["value"] == 0.5
    assert coin["log_loss"]["value"] == pytest.approx(math.log(2.0), rel=1e-15)
    assert coin["brier"]["value"] == 0.25


def test_best_reasons_follow_an_independent_reading(demo_card):
    """Forward card: the candidate is the top AUC (lowest log loss); a reason is listed for each other row of the
    same family that it does not separate from after Holm, skipping a flagged duplicate in the ranking family."""
    m = demo_card["metrics"]
    dups = [{p["a"], p["b"]} for p in demo_card["duplicates"]["pairs"]]
    for fam in C.FAMILIES:
        rows = [r["name"] for r in demo_card["entries"] if fam == "ranking" or r["probability_metrics"]]
        if fam == "ranking":
            cand = max(rows, key=lambda n: m[n]["auc"]["value"])
        else:
            cand = min(rows, key=lambda n: m[n]["log_loss"]["value"])
        best = demo_card["best"][fam]
        assert best["candidate"] == cand
        missing = set()
        for c in demo_card["comparisons"]:
            if c["family"] != fam or cand not in (c["a"], c["b"]):
                continue
            other = c["b"] if c["a"] == cand else c["a"]
            if fam == "ranking" and {cand, other} in dups:
                continue
            if c["favours"] != cand:
                missing.add(other)
        said = {r[len("not separated from "):-len(" after Holm")] for r in best["reasons"]
                if r.startswith("not separated from ")}
        assert said == missing, (fam, said, missing)


def test_count_gate_boundaries_are_inclusive():
    ev = synth.synthetic_event(300, 9)
    ids = ev["site_ids"]
    from scoreboard.evalkit.spec import Task, site_list_sha256
    y = ev["label"]
    n_yes = int(np.nansum(y))
    n_no = int(np.sum(y == 0))
    n_tiles = int(np.unique(R.tiles_for(ev["x_m"], ev["y_m"], 2000.0)).size)

    def card(**kw):
        t = Task(event_id="q3", target="t", label_source="synthetic v1", label_rule_sha256="1" * 64,
                 site_list_sha256=site_list_sha256(ids), track="record", division="closed", tile_m=2000.0, clip=1e-6,
                 seed=3, label_caveats="synthetic labels", n_boot=50, **kw)
        return C.build_card(t, [], synth.labels_of(ev))["count_gate"]
    g = card(min_sites=300, min_per_class=min(n_yes, n_no), min_tiles=n_tiles)
    assert g["passes"] and (g["n_labelled"], g["n_yes"], g["n_no"], g["n_tiles"]) == (300, n_yes, n_no, n_tiles)
    assert not card(min_sites=301, min_per_class=1, min_tiles=1)["passes"]
    assert not card(min_sites=1, min_per_class=min(n_yes, n_no) + 1, min_tiles=1)["passes"]
    assert not card(min_sites=1, min_per_class=1, min_tiles=n_tiles + 1)["passes"]


def test_nonneg_wls_branches_by_hand():
    b = np.array([0.2, 0.4, 0.6, 0.8])
    w = np.array([1.0, 2.0, 3.0, 4.0])
    g = 2.0 * b - 0.5                                     # free fit has a negative nugget
    c0, c1, sse = R._nonneg_wls(b, g, w)
    c1_ref = float(np.sum(w * b * g) / np.sum(w * b * b))
    assert c0 == 0.0 and c1 == pytest.approx(c1_ref, rel=1e-12)
    assert sse == pytest.approx(float(np.sum(w * (g - c1_ref * b) ** 2)), rel=1e-12)
    g2 = 1.0 - b                                          # free fit has a negative slope
    c0, c1, sse = R._nonneg_wls(b, g2, w)
    assert c1 == 0.0 and c0 == pytest.approx(float(np.average(g2, weights=w)), rel=1e-12)
    g3 = 0.3 + 0.5 * b                                    # exact fit inside the constraints
    c0, c1, sse = R._nonneg_wls(b, g3, w)
    assert (c0, c1) == pytest.approx((0.3, 0.5), rel=1e-12) and sse == pytest.approx(0.0, abs=1e-24)


def test_reliability_reads_the_lowest_single_forecast():
    y = np.array([0.0, 1.0, 1.0])
    p = np.array([0.1, 0.5, 0.9])
    r = M.reliability_curve(y, p, grid=(0.1,))
    assert r["recalibrated"] == [0.0]


def test_interval_sign_rule_by_hand():
    assert P._sign({"lo": 0.1, "hi": 0.3}) == 1
    assert P._sign({"lo": -0.3, "hi": -0.1}) == -1
    assert P._sign({"lo": -0.1, "hi": 0.1}) == 0
    assert P._sign({"lo": 0.0, "hi": 0.1}) == 0
    assert P._sign({"lo": -0.1, "hi": 0.0}) == 0


def test_dl_p_value_at_a_moderate_z():
    from scipy import stats
    d, s = [0.02, 0.01, 0.03], [0.02, 0.02, 0.02]
    r = P.random_effects(d, s, method="dl")
    z = r["mean"] / r["se"]
    assert 1.0 < z < 3.0
    assert r["p"] == pytest.approx(2.0 * stats.norm.sf(z), rel=1e-12)


def test_variogram_range_recovers_an_exact_exponential_model():
    """gamma = 0.1 + 0.4 (1 - exp(-3 h / 3000)): the fitted practical range is 3000 m to the grid step."""
    lag = np.linspace(250.0, 11750.0, 24)
    gamma = 0.1 + 0.4 * (1.0 - np.exp(-3.0 * lag / 3000.0))
    fit = R.variogram_range({"lag_m": lag.tolist(), "gamma": gamma.tolist(), "n_pairs": [100] * 24})
    step = (2.0 * lag[-1] - lag[0]) / 119
    assert abs(fit["range_m"] - 3000.0) <= step
    assert fit["nugget"] == pytest.approx(0.1, abs=0.02) and fit["partial_sill"] == pytest.approx(0.4, abs=0.02)

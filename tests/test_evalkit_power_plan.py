"""C3 power plan (scoreboard/evalkit/lab/power_plan.py): the vectorised pooled line equals the kit's pool.py, sets
take independent earthquakes, and the word rule matches pool.pooled_lines. Synthetic numbers only. No em dashes."""
import numpy as np
import pytest

from scoreboard.evalkit import pool
from scoreboard.evalkit.lab import power_plan as PP


def _rows(seed, M=40, K=5):
    rng = np.random.default_rng(seed)
    d = rng.normal(0.02, 0.03, (M, K))
    s = rng.uniform(0.01, 0.04, (M, K))
    return d, s


@pytest.mark.parametrize("method", ["hk", "equal"])
def test_hk_equals_pool_random_effects(method):
    d, s = _rows(1)
    v = PP.hk(d, s, 0.95, method)
    for m in range(d.shape[0]):
        r = pool.random_effects(d[m], s[m], method=method)
        assert v["mean"][m] == pytest.approx(r["mean"], rel=1e-10, abs=1e-14)
        assert v["lo"][m] == pytest.approx(r["lo"], rel=1e-9, abs=1e-12)
        assert v["hi"][m] == pytest.approx(r["hi"], rel=1e-9, abs=1e-12)
        assert v["p"][m] == pytest.approx(r["p"], rel=1e-8, abs=1e-12)


def test_holds_equals_pooled_lines_word():
    d, s = _rows(2, M=60, K=4)
    d += 0.02
    w = PP.holds(d, s, 0.05, "hk")
    for m in range(d.shape[0]):
        items = [{"event_id": f"e{i}", "entry_a": "a", "entry_b": "b", "metric": "auc", "diff": float(d[m, i]),
                  "se": float(s[m, i]), "passes_gate": True} for i in range(d.shape[1])]
        word = pool.pooled_line(items, registered=True)["word"]
        assert (word == "HOLDS") == bool(w["holds"][m])
        assert (word != "POOLED CANNOT TELL") == bool(w["reject"][m])


def test_draw_sets_use_distinct_worlds_and_gate_passing_slots():
    W, E = 30, 3
    rng = np.random.default_rng(3)
    ok = rng.random((W, E)) < 0.6
    ok[0] = False                                         # a world with no usable earthquake is never drawn
    ident = np.arange(W)[:, None] * 10 + np.arange(E)[None, :]
    b = {"d": ident.astype(float), "s": np.ones((W, E)), "theta": np.zeros((W, E)), "ok": ok}
    d, s, th = PP.draw_sets(b, 8, 200, np.random.default_rng(4))
    worlds = (d // 10).astype(int)
    slots = (d % 10).astype(int)
    for m in range(d.shape[0]):
        assert len(set(worlds[m])) == 8
    assert ok[worlds, slots].all()
    assert PP.draw_sets(b, W, 5, np.random.default_rng(5)) is None


def test_power_curve_null_and_planted():
    W, E = 400, 3
    rng = np.random.default_rng(6)
    ok = np.ones((W, E), bool)
    s = np.full((W, E), 0.02)
    null = {"d": rng.normal(0, 0.02, (W, E)), "s": s, "theta": np.zeros((W, E)) + 1e-9, "ok": ok}
    cur = PP.power_curve(null, "auc", 2000, 7, "plain")
    for row in cur:
        assert row["alpha_0.05"]["reject_right_sign"] + row["alpha_0.05"]["reject_wrong_sign"] < 0.08
    big = {"d": rng.normal(0.03, 0.02, (W, E)), "s": s, "theta": np.full((W, E), 0.03), "ok": ok}
    cur = PP.power_curve(big, "auc", 2000, 8, "plain")
    k = PP.k80(cur)
    assert k is not None and k <= 8
    assert cur[-1]["coverage_models_fixed"] > 0.9


def test_gate_and_subset():
    assert PP.gate(200, 30, 20) and not PP.gate(199, 30, 20) and not PP.gate(300, 29, 20) and not PP.gate(300, 40, 19)
    rng = np.random.default_rng(9)
    assert (PP.subset(100, 0, rng) == np.arange(100)).all()
    sb = PP.subset(1000, 250, rng)
    assert sb.size == 250 and np.all(np.diff(sb) > 0)


def test_conditions_and_seeds():
    assert set(PP.COND_ORDER) == set(PP.CONDITIONS)
    seeds = {PP.world_seed(c, i) for c in PP.COND_ORDER for i in range(1000)}
    assert len(seeds) == len(PP.COND_ORDER) * 1000
    assert PP.family("P8-CALM") == "CAL" and PP.family("P2-A15") == "A"
    assert PP.CONDITIONS["P8-A30"]["rho_m"] == 8000.0 and "rho_m" not in PP.CONDITIONS["P2-A30"]


def test_k80_interp():
    cur = [{"K": k, "alpha_0.05": {"holds_right_sign": v}} for k, v in ((3, 0.5), (4, 0.7), (5, 0.9))]
    assert PP.k80_interp(cur) == pytest.approx(4.5)
    assert PP.k80(cur) == 5
    assert PP.k80_interp(cur[:2]) is None
    assert PP.k80_interp([{"K": 3, "alpha_0.05": {"holds_right_sign": 0.85}}]) == 3.0


# ---- review 1 (2026-11-23): tests that catch the three surviving plants --------------------------------------------
def test_holds_equals_pooled_lines_word_holm_family_of_three():
    """At alpha 0.05 / 3 the leave-one-out level is 1 - alpha_j, not 0.95 (Holm family of 3, this line ranks first)."""
    d, s = _rows(12, M=200, K=6)
    d += 0.025
    a = PP.ALPHA / 3
    w = PP.holds(d, s, a, "hk")
    weak = [[{"event_id": f"w{j}{i}", "entry_a": "a", "entry_b": "b", "metric": "auc", "diff": (-1) ** i * 0.01,
              "se": 0.05, "passes_gate": True} for i in range(6)] for j in range(2)]
    n_first = n_diff = 0
    for m in range(d.shape[0]):
        items = [{"event_id": f"e{i}", "entry_a": "a", "entry_b": "b", "metric": "auc", "diff": float(d[m, i]),
                  "se": float(s[m, i]), "passes_gate": True} for i in range(d.shape[1])]
        line = pool.pooled_lines([items] + weak, registered=True)[0]
        if line["holm"]["rank"] != 1:
            continue
        n_first += 1
        assert line["holm"]["family_size"] == 3 and line["holm"]["alpha_j"] == pytest.approx(a)
        assert (line["word"] == "HOLDS") == bool(w["holds"][m])
        assert (line["word"] != "POOLED CANNOT TELL") == bool(w["reject"][m])
        # rows where a 0.95 leave-one-out level would give another word: the test must see some
        loo95 = np.ones(1, bool)
        for i in range(d.shape[1]):
            keep = np.r_[0:i, i + 1:d.shape[1]]
            r = PP.hk(d[m:m + 1, keep], s[m:m + 1, keep], 0.95)
            loo95 &= (r["lo"] > 0) if w["sign"][m] > 0 else (r["hi"] < 0)
        n_diff += int(bool(w["reject"][m] & loo95[0]) != bool(w["holds"][m]))
    assert n_first > 150 and n_diff >= 5


def test_power_curve_wrong_sign_holds_is_not_power():
    """A line that HOLDS with the opposite sign of the truth adds nothing to power."""
    W, E = 400, 3
    rng = np.random.default_rng(13)
    ok = np.ones((W, E), bool)
    b = {"d": rng.normal(0.03, 0.02, (W, E)), "s": np.full((W, E), 0.02), "theta": np.full((W, E), -0.03), "ok": ok}
    cur = PP.power_curve(b, "auc", 500, 14, "plain")
    assert all(row["alpha_0.05"]["holds_right_sign"] == 0 for row in cur)
    assert all(row["alpha_0.05"]["reject_right_sign"] == 0 for row in cur)
    assert cur[5]["alpha_0.05"]["reject_wrong_sign"] > 0.9
    assert PP.k80(cur) is None


def test_power_curve_truth_is_the_mean_of_theta():
    """The line's truth is the mean of theta over the set, not the median: here the mean is > 0 and the median < 0."""
    W, E = 300, 3
    rng = np.random.default_rng(15)
    ok = np.ones((W, E), bool)
    theta = np.where(rng.random((W, E)) < 0.3, 0.5, -0.01)     # per set of 20: mean > 0, median < 0 (about 95%)
    b = {"d": rng.normal(0.03, 0.005, (W, E)), "s": np.full((W, E), 0.005), "theta": theta, "ok": ok}
    cur = PP.power_curve(b, "auc", 400, 16, "plain")
    row = [r for r in cur if r["K"] == 20][0]
    assert row["alpha_0.05"]["reject_right_sign"] > 0.95


def test_mixed_sets_distinct_worlds_and_sizes():
    W, E = 60, 3
    rng = np.random.default_rng(17)
    banks = []
    for j in range(3):
        ok = rng.random((W, E)) < (0.9, 0.6, 0.2)[j]
        ident = 1000 * j + np.arange(W)[:, None] * 10 + np.arange(E)[None, :]
        banks.append({"d": ident.astype(float), "s": np.ones((W, E)), "theta": np.zeros((W, E)), "ok": ok})
    d, s, th = PP.draw_sets_mixed(banks, (0.45, 0.45, 0.1), 12, 300, np.random.default_rng(18))
    size = (d // 1000).astype(int)
    world = ((d % 1000) // 10).astype(int)
    slot = (d % 10).astype(int)
    for m in range(d.shape[0]):
        assert len(set(world[m])) == 12
        for j, w_, k in zip(size[m], world[m], slot[m]):
            assert banks[j]["ok"][w_, k]
    share = np.bincount(size.ravel(), minlength=3) / size.size
    assert share == pytest.approx([0.45, 0.45, 0.1], abs=0.03)
    assert PP.draw_sets_mixed(banks, (0, 0, 1), 59, 5, np.random.default_rng(19)) is None


def test_seed_recipes():
    """Registered conditions keep recipe 1 (seed + 7); review 1 conditions use a stream disjoint from main seeds."""
    s = PP.world_seed("P2-A15", 3)
    a = PP.subset_rng("P2-A15", s).random(4)
    assert np.allclose(a, np.random.default_rng(PP.world_seed("P2-A15", 10)).random(4))
    s2 = PP.world_seed("P8-A15-E2x130", 3)
    b = PP.subset_rng("P8-A15-E2x130", s2).random(4)
    assert not np.allclose(b, np.random.default_rng(PP.world_seed("P8-A15-E2x130", 10)).random(4))
    assert PP.COND_ORDER[:10] == ("P2-A15", "P2-A30", "P2-A50", "P2-CALP", "P2-CALM",
                                  "P8-A15", "P8-A30", "P8-A50", "P8-CALP", "P8-CALM")
    assert all(k not in PP.LAB_ONLY for c in PP.COND_ORDER[:10] for k in PP.CONDITIONS[c])

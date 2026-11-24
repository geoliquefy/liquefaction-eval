"""C4 certification harness (ruling 47): synthetic checks only. No real data, no lab world. No em dashes."""
import numpy as np
import pytest

from scoreboard.evalkit.lab import c4_certify as C


def test_candidates_are_the_registered_four_in_order():
    assert C.CANDIDATE_ORDER == ("tile4", "block", "cluster", "event2")
    assert C.BLOCK_M == {"auc": 5500.0, "log_loss": 3000.0}
    assert "cluster" in C.NOT_TESTABLE
    assert C.BASE_SEED != 20261123 and C.BASE_SEED != 20261119


def test_pass_rule_needs_the_whole_monte_carlo_range_inside_the_band():
    assert C.coverage_inside({"value": 0.95, "se": 0.005})
    assert not C.coverage_inside({"value": 0.94, "se": 0.006})        # 0.928 lower end
    assert not C.coverage_inside({"value": 0.962, "se": 0.006})       # 0.974 upper end
    assert not C.coverage_inside({"value": None, "se": None})
    assert C.wrong_sign_ok({"value": 0.01, "se": 0.005})
    assert not C.wrong_sign_ok({"value": 0.07, "se": 0.005})


def test_gate_matches_the_card_count_gate():
    assert C.gate_passes(200, 30, 20)
    assert not C.gate_passes(199, 100, 30)
    assert not C.gate_passes(400, 29, 30)
    assert not C.gate_passes(400, 371, 30)
    assert not C.gate_passes(400, 100, 19)


def _toy_event(seed=0, n=600):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 30000, n), rng.uniform(0, 30000, n)
    z = rng.normal(size=n)
    lab = rng.random(n) < 1 / (1 + np.exp(-z))
    ps = [1 / (1 + np.exp(-(z + rng.normal(0, s, n)))) for s in (0.5, 0.8, 0.8)]
    from scoreboard.scorer import grid_clusters
    tiles = grid_clusters(x, y, 4000.0)
    blocks = {m: grid_clusters(x, y, C.BLOCK_M[m]) for m in C.METRICS}
    return lab, ps, tiles, blocks


def test_event_lines_tile4_equals_kit_paired_and_blocks_gate():
    lab, ps, tiles, blocks = _toy_event()
    r = C.event_lines(lab, ps, tiles, blocks, seed=11, n_boot=200)
    assert r["tile4"].shape == (3, 2, 5) and np.all(np.isfinite(r["tile4"]))
    assert np.all(r["tile4"][..., C.LOW] <= r["tile4"][..., C.DIFF] + 1e-12)
    # 30 km square: 36 blocks at 5.5 km and 100 at 3 km, both over the 20-block gate
    assert list(r["n_blocks"]) == [36, 100]
    assert np.all(np.isfinite(r["block"]))
    # point differences agree between methods; only the interval differs
    np.testing.assert_allclose(r["tile4"][..., C.DIFF], r["block"][..., C.DIFF])
    # under the block gate the line is UNDERPOWERED (NaN)
    few = {m: np.zeros_like(blocks[m]) for m in C.METRICS}
    r2 = C.event_lines(lab, ps, tiles, few, seed=11, n_boot=200)
    assert np.all(np.isnan(r2["block"])) and np.all(np.isfinite(r2["tile4"]))


def test_pooled_lines_need_three_gate_passing_earthquakes():
    E = 4
    lines = {m: np.zeros((E, 3, 2, 5)) for m in C.WITHIN}
    for m in C.WITHIN:
        lines[m][..., C.DIFF] = np.arange(E)[:, None, None] * 0.01
        lines[m][..., C.SEB] = 0.01
    draws = np.random.default_rng(0).normal(0, 0.01, (E, 3, 2, 100))
    out = C.pooled_lines(lines, draws, np.array([True, True, False, False]), seed=1)
    assert out["k"] == 2 and all(np.all(np.isnan(out[m])) for m in C.POOLED)
    out = C.pooled_lines(lines, draws, np.array([True, True, True, False]), seed=1)
    assert out["k"] == 3
    for m in C.POOLED:
        assert np.all(np.isfinite(out[m]))
    np.testing.assert_allclose(out["tile4"][..., C.DIFF], 0.01)          # mean of 0, 0.01, 0.02
    assert np.all(out["event2"][..., C.LOW] < out["event2"][..., C.HIGH])


def _cell(ok=True, width=0.05):
    return {"pass": ok, "median_width": width}


def _summary(per, pooled):
    conds = {}
    for c in C.COND_ORDER:
        conds[c] = {"per_earthquake": {m: {met: per(m, c, met) for met in C.METRICS} for m in C.WITHIN},
                    "pooled": {m: {met: pooled(m, c, met) for met in C.METRICS} for m in C.POOLED}}
    return {"conditions": conds}


def test_decide_no_method_certifies():
    out = C.decide(_summary(lambda m, c, met: _cell(False), lambda m, c, met: _cell(False)))
    assert out["per_earthquake"]["certified"] is None and out["pooled"]["certified"] is None
    assert out["per_earthquake"]["candidates"]["cluster"]["verdict"] == "NOT TESTABLE"
    assert out["per_earthquake"]["candidates"]["event2"]["verdict"] == "NOT A CANDIDATE FOR THIS LINE TYPE"


def test_decide_one_failing_cell_fails_the_method():
    out = C.decide(_summary(lambda m, c, met: _cell(not (c == "R-HI" and met == "log_loss")),
                            lambda m, c, met: _cell(False)))
    assert out["per_earthquake"]["certified"] is None
    assert out["per_earthquake"]["candidates"]["tile4"]["failed_cells"] == ["R-HI log_loss"]


def test_decide_narrowest_median_width_wins_ties_to_earlier():
    out = C.decide(_summary(lambda m, c, met: _cell(True, 0.05 if m == "tile4" else 0.04),
                            lambda m, c, met: _cell(True, 0.03)))
    assert out["per_earthquake"]["certified"] == "block"
    assert out["pooled"]["certified"] == "tile4"                           # all equal: the earlier candidate


def test_decide_refuses_with_a_missing_condition():
    s = _summary(lambda m, c, met: _cell(True), lambda m, c, met: _cell(True))
    s["conditions"]["R-HI"] = "NOT RUN"
    out = C.decide(s)
    assert out["conditions_missing"] == ["R-HI"]
    assert out["per_earthquake"]["certified"] is None and out["pooled"]["certified"] is None


@pytest.mark.parametrize("cond", C.COND_ORDER)
def test_world_seeds_are_disjoint_across_conditions(cond):
    seeds = {C.world_seed(cond, i) for i in range(2000)}
    others = {C.world_seed(c, i) for c in C.COND_ORDER if c != cond for i in range(2000)}
    assert not seeds & others


# Review 1 (wave 3): the code that computes the verdict, on a tiny hand-made results dict with known answers.
# Base world set: 400 worlds, 4 earthquakes (sep, feb, jun pass the gate; dec fails it), every check passes.
# Each variant plants one fault and checks that exactly that check fails. The registered run is not rerun.
W_TOY = 400
THETA = np.array([0.02, 0.04, 0.06, 1.0])             # per earthquake; dec (fails the gate) is far off on purpose


def _toy_results():
    W, E = W_TOY, 4
    passes = np.zeros((W, E), bool)
    passes[:, :3] = True
    theta = np.broadcast_to(THETA[None, :, None, None], (W, E, 3, 2)).copy()
    within = np.zeros((W, E, 3, 2, 5))
    within[..., C.DIFF] = theta
    within[..., C.LOW], within[..., C.HIGH] = -0.1, 0.1           # covers 0.02, 0.04 and 0.06
    within[..., C.PVAL], within[..., C.SEB] = 0.5, 0.05
    for w in range(180):                                            # 180 worlds miss one cell, 60 on each slot
        within[w, w % 3, 0, :, C.LOW], within[w, w % 3, 0, :, C.HIGH] = 0.2, 0.3
    within[:, 3, ..., C.LOW], within[:, 3, ..., C.HIGH] = 0.5, 0.6  # dec: never covers, planted wrong-sign
    within[:, 3, ..., C.DIFF], within[:, 3, ..., C.PVAL] = -1.0, 0.001
    pooled = np.zeros((W, 3, 2, 5))
    pooled[..., C.DIFF], pooled[..., C.LOW], pooled[..., C.HIGH] = 0.04, 0.03, 0.05   # truth: mean of sep, feb, jun
    pooled[..., C.PVAL], pooled[..., C.SEB] = 0.5, 0.005
    pooled[:60, 0, :, C.LOW], pooled[:60, 0, :, C.HIGH] = 0.1, 0.2                    # 60 worlds miss one line
    return {"within_tile4": within, "within_block": within.copy(), "pooled_tile4": pooled,
            "pooled_block": pooled.copy(), "pooled_event2": pooled.copy(), "pooled_k": np.full(W, 3),
            "theta_w": theta, "passes": passes, "slot": np.tile(np.arange(E), (W, 1)),
            "n_blocks": np.full((W, E, 2), 30), "r_theta": np.array(50), "seconds": np.array(1.0)}


def test_summary_base_passes_with_the_known_coverage():
    s = C.summarise_condition("B-NULL", _toy_results())
    for m in C.WITHIN:
        for met in C.METRICS:
            o = s["per_earthquake"][m][met]
            assert o["coverage"]["value"] == pytest.approx(0.95)          # 1 - 180 / 400 / 9; dec left out
            assert o["coverage"]["inside"]
            assert set(o["coverage_per_gate_passing_slot"]) == {"sep2010#0", "feb2011#1", "jun2011#2"}
            assert all(v["value"] == pytest.approx(0.95) for v in o["coverage_per_gate_passing_slot"].values())
            assert o["slot_floor_ok"] and o["print_share"] == 1.0 and o["print_share_ok"]
            assert o["wrong_sign_per_comparison"]["value"] == 0.0 and o["wrong_sign_per_comparison"]["ok"]
            assert o["pass"]
    for m in C.POOLED:
        for met in C.METRICS:
            o = s["pooled"][m][met]
            assert o["coverage"]["value"] == pytest.approx(0.95)          # 1 - 60 / 400 / 3
            assert o["coverage"]["inside"] and o["print_share_ok"] and o["wrong_sign_per_comparison"]["ok"]
            assert o["pass"]


def test_summary_slot_floor_fails_one_slot_under_090():
    r = _toy_results()
    r["within_tile4"][:, :3, ..., C.LOW], r["within_tile4"][:, :3, ..., C.HIGH] = -0.1, 0.1
    r["within_tile4"][:180, 2, 0, :, C.LOW], r["within_tile4"][:180, 2, 0, :, C.HIGH] = 0.2, 0.3   # all on jun
    o = C.summarise_within("B-NULL", r, "tile4")["auc"]
    assert o["coverage"]["value"] == pytest.approx(0.95) and o["coverage"]["inside"]
    assert o["coverage_per_gate_passing_slot"]["jun2011#2"]["value"] == pytest.approx(0.85)
    assert not o["slot_floor_ok"] and not o["pass"]


def test_summary_print_share_fails_when_a_method_prints_under_half():
    r = _toy_results()
    r["within_tile4"][:, 1:3, :, 1, C.LOW] = np.nan                  # log loss prints on sep only
    o = C.summarise_within("B-NULL", r, "tile4")["log_loss"]
    assert o["print_share"] == pytest.approx(1 / 3, abs=1e-4) and not o["print_share_ok"]
    assert o["coverage"]["inside"] and not o["pass"]
    # a gate-passing slot with no printed line has no coverage, so it also fails the slot floor
    assert o["coverage_per_gate_passing_slot"]["feb2011#1"]["value"] is None and not o["slot_floor_ok"]
    assert C.summarise_within("B-NULL", r, "tile4")["auc"]["pass"]


def test_summary_counts_a_wrong_sign_separates():
    r = _toy_results()
    r["within_tile4"][:, 0, 0, 0, C.PVAL] = 0.001                    # Holm over 3: 0.003 < 0.05 -> SEPARATES
    r["within_tile4"][:, 0, 0, 0, C.DIFF] = -0.02                    # truth is +0.02
    o = C.summarise_within("B-NULL", r, "tile4")["auc"]
    assert o["wrong_sign_per_comparison"]["value"] == pytest.approx(1 / 9, abs=1e-4)
    assert not o["wrong_sign_per_comparison"]["ok"] and o["coverage"]["inside"] and not o["pass"]
    r["within_tile4"][:, 0, 0, 0, C.DIFF] = 0.02                     # the right sign: no wrong-sign
    assert C.summarise_within("B-NULL", r, "tile4")["auc"]["wrong_sign_per_comparison"]["value"] == 0.0
    p = _toy_results()
    p["pooled_tile4"][:, 0, 0, C.PVAL] = 0.001                       # Holm over 6: 0.006 < 0.05
    p["pooled_tile4"][:, 0, 0, C.DIFF] = -0.04
    o = C.summarise_pooled("B-NULL", p, "tile4")["auc"]
    assert o["wrong_sign_per_comparison"]["value"] == pytest.approx(1 / 3, abs=1e-4) and not o["pass"]


def test_pooled_truth_is_the_mean_over_the_gate_passing_earthquakes_of_each_world():
    r = _toy_results()
    # worlds 200+: jun fails the gate and dec passes; truth there = mean(0.02, 0.04, 1.0)
    r["passes"][200:, 2], r["passes"][200:, 3] = False, True
    t = (0.02 + 0.04 + 1.0) / 3
    r["pooled_tile4"][200:, ..., C.LOW], r["pooled_tile4"][200:, ..., C.HIGH] = t - 0.01, t + 0.01
    o = C.summarise_pooled("B-NULL", r, "tile4")["auc"]
    assert o["coverage"]["value"] == pytest.approx(0.95)
    # an equal mean over all 4 earthquakes (0.28) is covered nowhere
    r["passes"][:] = True
    assert C.summarise_pooled("B-NULL", r, "tile4")["auc"]["coverage"]["value"] == 0.0


def test_pooled_lines_need_k_at_least_three_and_block_with_no_line_fails_print_share():
    r = _toy_results()
    r["pooled_block"][..., C.LOW] = np.nan
    s = C.summarise_pooled("B-NULL", r, "block")["auc"]
    assert s["coverage"]["value"] is None and s["print_share"] == 0.0 and not s["pass"]
    r = _toy_results()
    r["pooled_k"][:] = 2
    s = C.summarise_pooled("B-NULL", r, "tile4")
    assert s["line_share"] == 0.0 and s["auc"]["coverage"]["value"] is None and not s["auc"]["pass"]

"""evalkit pooled line (rule book R1) on synthetic per-earthquake differences."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import resample as R  # noqa: E402
from scoreboard.evalkit import synth  # noqa: E402
from scoreboard.evalkit.pool import pooled_line, pooled_lines, random_effects  # noqa: E402
from scoreboard.scorer import auc  # noqa: E402


def items(diffs, ses, gate=None, a="A"):
    gate = gate or [True] * len(diffs)
    return [{"event_id": f"eq{k}", "entry_a": a, "entry_b": "B", "metric": "auc", "diff": d, "se": s,
             "passes_gate": g} for k, (d, s, g) in enumerate(zip(diffs, ses, gate))]


def test_random_effects_reduces_to_fixed_effect_without_spread():
    r = random_effects([0.05, 0.05, 0.05], [0.02, 0.02, 0.02], method="dl")
    assert r["tau2"] == 0.0 and r["mean"] == pytest.approx(0.05)
    assert r["se"] == pytest.approx(0.02 / np.sqrt(3))


def test_hartung_knapp_matches_its_formula():
    """R1 P1: V = sum w (d - m)^2 / ((k - 1) sum w), t on k - 1 degrees of freedom."""
    d, s = np.array([0.05, 0.03, 0.01, 0.03]), np.full(4, 0.01)
    r = random_effects(d, s)
    w = 1 / (s ** 2 + r["tau2"])
    m = np.sum(w * d) / np.sum(w)
    v = np.sum(w * (d - m) ** 2) / (3 * np.sum(w))
    t = stats.t.ppf(0.975, 3)
    assert r["df"] == 3 and r["mean"] == pytest.approx(m) and r["se"] == pytest.approx(np.sqrt(v))
    assert r["lo"] == pytest.approx(m - t * np.sqrt(v)) and r["hi"] == pytest.approx(m + t * np.sqrt(v))
    assert r["p"] == pytest.approx(2 * stats.t.sf(m / np.sqrt(v), 3))


def test_hartung_knapp_covers_at_nominal_and_normal_dl_does_not():
    """Calibration of P1 on synthetic meta-analyses with 4 earthquakes and between-earthquake spread tau = 0.02."""
    rng = np.random.default_rng(5)
    hk = dl = 0
    n = 2000
    for _ in range(n):
        s = rng.uniform(0.01, 0.03, 4)
        d = rng.normal(0.0, np.sqrt(s ** 2 + 0.02 ** 2))
        a, b = random_effects(d, s), random_effects(d, s, method="dl")
        hk += a["lo"] <= 0 <= a["hi"]
        dl += b["lo"] <= 0 <= b["hi"]
    assert 0.93 <= hk / n <= 0.97
    assert dl / n < 0.92


def test_not_allowed_under_three_earthquakes():
    out = pooled_line(items([0.05, 0.06, 0.07], [0.01] * 3, gate=[True, True, False]), registered=True)
    assert out["word"] == "NOT ALLOWED" and "R1 (c)" in out["reason"]
    assert out["earthquakes_excluded"] == ["eq2"]


def test_not_allowed_unless_registered():
    out = pooled_line(items([0.05] * 4, [0.01] * 4), registered=False)
    assert out["word"] == "NOT ALLOWED" and "R1 (a)" in out["reason"]


def test_holds_when_every_earthquake_agrees():
    out = pooled_line(items([0.05, 0.045, 0.055, 0.05], [0.005] * 4), registered=True)
    assert out["word"] == "HOLDS" and out["label"] == "across these 4 earthquakes"
    assert len(out["leave_one_out"]) == 4 and out["pooled"]["lo"] > 0 and out["pooled"]["method"] == "hk"


def test_fragile_when_one_earthquake_carries_the_result():
    out = pooled_line(items([0.05, 0.03, 0.01, 0.03], [0.01] * 4), registered=True)
    assert out["pooled"]["lo"] > 0
    assert out["word"] == "FRAGILE"
    assert any(r["lo"] <= 0 for r in out["leave_one_out"])


def test_cannot_tell_when_pooled_interval_covers_zero():
    assert pooled_line(items([0.03, -0.02, 0.01], [0.02] * 3), registered=True)["word"] == "POOLED CANNOT TELL"


def test_holm_runs_across_the_pooled_lines_of_a_release():
    """R1: the line with the j-th smallest p is tested at 0.05 / (L - j + 1); leave-one-out at 1 - alpha_j."""
    strong = items([0.05, 0.045, 0.055, 0.05], [0.005] * 4, a="S")
    middling = items([0.05, 0.03, 0.01, 0.03], [0.01] * 4, a="M")        # p about 0.035
    alone = pooled_line(middling, registered=True)
    assert alone["word"] == "FRAGILE"
    lines = pooled_lines([middling, strong, items([0.05] * 2, [0.01] * 2, a="N")], registered=True)
    assert lines[2]["word"] == "NOT ALLOWED"
    assert lines[1]["holm"] == {"family_size": 2, "rank": 1, "alpha_j": 0.025} and lines[1]["word"] == "HOLDS"
    assert lines[0]["holm"]["alpha_j"] == 0.05 and lines[0]["word"] == "FRAGILE"
    three = pooled_lines([middling, strong, items([0.05, 0.03, 0.01, 0.03], [0.01] * 4, a="M2")], registered=True)
    assert three[0]["word"] == "POOLED CANNOT TELL"                    # 0.035 > 0.05 / 2 after the strong line
    loo_level = three[1]["leave_one_out"][0]["level"]
    assert loo_level == pytest.approx(1 - 0.05 / 3)


def test_items_must_share_one_comparison_and_unique_earthquakes():
    bad = items([0.05] * 3, [0.01] * 3)
    bad[1]["metric"] = "log_loss"
    with pytest.raises(ValueError, match="one metric"):
        pooled_line(bad, registered=True)
    dup = items([0.05] * 3, [0.01] * 3)
    dup[2]["event_id"] = "eq0"
    with pytest.raises(ValueError, match="once"):
        pooled_line(dup, registered=True)


def test_all_signal_in_one_synthetic_earthquake_gives_no_pooled_word():
    """R1 check: entry A beats B only in earthquake 0; three earthquakes where they are exchangeable."""
    out_items = []
    for k in range(4):
        ev = synth.synthetic_event(1200, 40 + k)
        rng = np.random.default_rng(90 + k)
        y, tiles = ev["label"].astype(bool), R.tiles_for(ev["x_m"], ev["y_m"], 2000.0)
        weak = expit(ev["intercept"] + ev["slope"] * (ev["s"] + rng.normal(0, 1.0, ev["s"].size)))
        strong = ev["p_target"] if k == 0 else expit(
            ev["intercept"] + ev["slope"] * (ev["s"] + rng.normal(0, 1.0, ev["s"].size)))
        da = R.bootstrap(lambda idx: {"v": auc(y[idx], strong[idx])}, tiles, 300, k)["v"]
        db = R.bootstrap(lambda idx: {"v": auc(y[idx], weak[idx])}, tiles, 300, k)["v"]
        r = R.paired(auc(y, strong), auc(y, weak), da, db)
        out_items.append({"event_id": f"eq{k}", "entry_a": "A", "entry_b": "B", "metric": "auc",
                          "diff": r["diff"], "se": r["se"], "passes_gate": True})
    assert out_items[0]["diff"] - 2 * out_items[0]["se"] > 0      # earthquake 0 alone separates
    assert pooled_line(out_items, registered=True)["word"] in ("FRAGILE", "POOLED CANNOT TELL")


# --- scorecard v2 registration K1: equal weight and leave one group out -------------------------------------------

def test_equal_weight_is_the_one_sample_t_on_the_differences():
    from scoreboard.evalkit.pool import equal_weight
    d = np.array([0.02, 0.05, -0.01, 0.04])
    r = equal_weight(d)
    t = stats.ttest_1samp(d, 0.0)
    assert r["mean"] == pytest.approx(d.mean(), abs=1e-15)
    assert r["se"] == pytest.approx(d.std(ddof=1) / np.sqrt(d.size), rel=1e-12)
    assert r["p"] == pytest.approx(t.pvalue, rel=1e-10)
    ci = t.confidence_interval(0.95)
    assert (r["lo"], r["hi"]) == (pytest.approx(ci.low, rel=1e-10), pytest.approx(ci.high, rel=1e-10))


def test_equal_weight_ignores_the_standard_errors():
    a = pooled_line(items([0.1, 0.0, 0.05], [0.01, 0.5, 0.02]), registered=True, weighting="equal")
    b = pooled_line(items([0.1, 0.0, 0.05], [0.3, 0.01, 0.9]), registered=True, weighting="equal")
    assert a["pooled"]["mean"] == pytest.approx(0.05) == b["pooled"]["mean"]
    assert a["pooled"]["se"] == b["pooled"]["se"]
    assert a["pooled"]["method"] == "equal"


def test_default_weighting_is_unchanged():
    it = items([0.1, 0.0, 0.05], [0.01, 0.5, 0.02])
    assert pooled_line(it, registered=True)["pooled"] == pooled_line(it, registered=True, weighting="re")["pooled"]
    assert pooled_line(it, registered=True)["pooled"]["method"] == "hk"


def test_equal_weight_leave_one_out_uses_equal_weight():
    r = pooled_line(items([0.1, 0.0, 0.05, 0.07], [0.01, 0.5, 0.02, 0.03]), registered=True, weighting="equal")
    assert all(x["method"] == "equal" for x in r["leave_one_out"])
    assert r["leave_one_out"][0]["mean"] == pytest.approx(np.mean([0.0, 0.05, 0.07]))


def test_bad_weighting_refused():
    with pytest.raises(ValueError):
        pooled_line(items([0.1, 0.0, 0.05], [0.01, 0.02, 0.03]), registered=True, weighting="inverse")


def test_leave_one_group_out():
    from scoreboard.evalkit.pool import leave_one_group_out
    it = items([0.1, 0.2, 0.3, 0.4], [0.01] * 4)
    ids = [x["event_id"] for x in it]
    groups = {ids[0]: "R1", ids[1]: "R1", ids[2]: "R2", ids[3]: "R3"}
    r = {x["left_out"]: x for x in leave_one_group_out(it, groups, weighting="equal")}
    assert r["R1"]["mean"] == pytest.approx(0.35)
    assert r["R2"]["mean"] == pytest.approx(np.mean([0.1, 0.2, 0.4]))
    assert set(r) == {"R1", "R2", "R3"}
    r2 = {x["left_out"]: x for x in leave_one_group_out(items([0.1, 0.2], [0.01, 0.01]),
                                                        {"eq0": "A", "eq1": "B"}, weighting="equal")}
    assert r2["A"]["word"] == "NOT ALLOWED"
    with pytest.raises(ValueError):
        leave_one_group_out(it, {ids[0]: "R1"})

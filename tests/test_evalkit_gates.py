"""evalkit scorer gates (rule book R12 a to c) and the duplicate detector. Synthetic only."""
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import gates as G  # noqa: E402


def test_all_gates_pass_and_are_deterministic():
    a = G.run_gates()
    assert a["pass"], {k: v["pass"] for k, v in a["gates"].items()}
    assert set(a["gates"]) == set(G.GATES)
    b = G.run_gates()
    assert a == b and a is not b                            # cached, but each caller gets its own copy


def test_gates_have_teeth():
    res = G.run_gates()["gates"]
    assert res["planted_better"]["auc_diff"]["lo"] > 0
    assert res["null_silent"]["false_separations"] <= G.NULL_MAX_FALSE
    assert res["defect_one_tile"]["auc_clean"] != res["defect_one_tile"]["auc_defect"]
    assert res["defect_join_order"]["auc_by_row_order"] != res["defect_join_order"]["auc_by_id"]
    assert not res["defect_kind"]["eligible"]["score"]


def test_duplicate_detector_joins_monotone_copies_only():
    rng = np.random.default_rng(1)
    s = rng.normal(size=2000)
    a = expit(s + rng.normal(0, 0.3, 2000))
    vals = {"L0": a, "M0": expit(0.9 * np.log(a / (1 - a)) + 0.4), "other": expit(s + rng.normal(0, 0.3, 2000)),
            "rank": np.argsort(np.argsort(a)).astype(float)}
    pairs = G.duplicate_pairs(vals)
    joined = {frozenset((p["a"], p["b"])) for p in pairs}
    assert joined == {frozenset(("L0", "M0")), frozenset(("L0", "rank")), frozenset(("M0", "rank"))}
    assert G.distinct_models(list(vals), pairs) == 2


def test_spearman_handles_ties_and_constants():
    assert G.spearman(np.array([1, 2, 2, 3.0]), np.array([10, 20, 20, 30.0])) == 1.0
    assert np.isnan(G.spearman(np.ones(5), np.arange(5.0)))

"""evalkit effort-only baseline (eval plan step 1c): allowlist, leave one earthquake out, fixed fit, card slot.
Synthetic only. No em dashes."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import effort as E  # noqa: E402
from scoreboard.evalkit import synth  # noqa: E402
from scoreboard.evalkit.card import CardRefused, build_card  # noqa: E402
from scoreboard.evalkit.spec import Entry  # noqa: E402

FEATS = ("builtup_share_250m", "pre_cpt_count_1000m", "tree_share_250m")
EVS = ("e1", "e2", "e3", "e4")


def world(n=600, seed=5):
    rng = np.random.default_rng(seed)
    road = rng.normal(0, 1, n)
    dens = rng.normal(0, 1, n)
    feats, labs = {}, {}
    for i, e in enumerate(EVS):
        after = (rng.random(n) < 0.3 + 0.1 * i).astype(float)
        p = 1 / (1 + np.exp(-(-0.2 - 1.0 * road + 0.8 * dens + 0.5 * after)))
        y = (rng.random(n) < p).astype(int)
        y[rng.random(n) < 0.1] = -1
        feats[e] = {"builtup_share_250m": road, "pre_cpt_count_1000m": dens, "tree_share_250m": after}
        labs[e] = y
    return feats, labs


def test_allowlist_refuses_physics_features():
    for bad in ("pga", "vs30", "water_table_m", "dist_to_river_m", "geology_unit", "lsn", "elevation_m"):
        with pytest.raises(E.EffortRefused, match="allowlist"):
            E.check_features(["builtup_share_250m", bad])
    assert E.check_features(list(E.EFFORT_FEATURES)) == tuple(E.EFFORT_FEATURES)


def test_fit_refuses_a_non_effort_column():
    y = np.array([0, 1] * 20, float)
    with pytest.raises(E.EffortRefused, match="allowlist"):
        E.fit({"builtup_share_250m": np.arange(40.0), "pga": np.arange(40.0)}, y)


def test_loeo_refuses_physics_feature_in_any_event():
    feats, labs = world()
    feats["e1"] = {**feats["e1"], "pga": np.zeros(600)}
    with pytest.raises(E.EffortRefused):
        E.fit_predict_loeo(feats, labs, "e1")
    feats, labs = world()
    feats["e2"] = {**feats["e2"], "pga": np.zeros(600)}
    with pytest.raises(E.EffortRefused):
        E.fit_predict_loeo(feats, labs, "e1")


def test_scored_labels_never_reach_the_fit():
    feats, labs = world()
    p_a, fit_a = E.fit_predict_loeo(feats, labs, "e3")
    for flip in (1 - np.clip(labs["e3"], 0, 1), np.zeros(600, int), np.full(600, -1)):
        labs2 = {**labs, "e3": flip}
        p_b, fit_b = E.fit_predict_loeo(feats, labs2, "e3")
        assert np.array_equal(p_a, p_b)
        assert fit_a.n_train == fit_b.n_train
    # the training count is exactly the other three earthquakes' labelled rows
    assert fit_a.n_train == sum(int((labs[e] >= 0).sum()) for e in EVS if e != "e3")
    # and leaving the scored labels out entirely gives the same answer
    p_c, _ = E.fit_predict_loeo(feats, {e: v for e, v in labs.items() if e != "e3"}, "e3")
    assert np.array_equal(p_a, p_c)


def test_training_labels_do_move_the_fit():
    feats, labs = world()
    p_a, _ = E.fit_predict_loeo(feats, labs, "e3")
    labs2 = {**labs, "e1": np.where(labs["e1"] >= 0, 1 - labs["e1"], -1)}
    p_b, _ = E.fit_predict_loeo(feats, labs2, "e3")
    assert not np.allclose(p_a, p_b)


def test_fit_recovers_signs_and_is_deterministic():
    feats, labs = world(n=3000)
    p1, f1 = E.fit_predict_loeo(feats, labs, "e4")
    p2, f2 = E.fit_predict_loeo(feats, labs, "e4")
    assert np.array_equal(p1, p2)
    c = dict(zip(f1.features, f1.coef))
    assert c["builtup_share_250m"] < 0 < c["pre_cpt_count_1000m"] and c["tree_share_250m"] > 0


def test_zero_variance_feature_is_dropped_and_reported():
    feats, labs = world()
    for e in EVS:
        feats[e] = {**feats[e], "in_aerial_footprint": np.ones(600)}
    _, f = E.fit_predict_loeo(feats, labs, "e2")
    assert f.dropped == ("in_aerial_footprint",)
    assert "in_aerial_footprint" in f.summary()["dropped_zero_variance"]


def test_stratified_auc_removes_a_stratum_effect():
    rng = np.random.default_rng(1)
    strata = np.repeat([0, 1], 500)
    y = np.where(strata == 1, rng.random(1000) < 0.8, rng.random(1000) < 0.2)
    s = strata + rng.normal(0, 0.01, 1000)          # knows only the stratum
    r = E.stratified_auc(y, s, strata)
    assert 0.4 < r["auc"] < 0.6 and r["strata_used"] == 2
    one = E.stratified_auc(np.ones(10, bool) | (np.arange(10) < 0), np.arange(10.0), np.zeros(10))
    assert one["strata_skipped"] == 1 and np.isnan(one["auc"])


def test_card_takes_the_effort_slot_and_refuses_imitations():
    task, entries, labels, shaking = synth.demo_inputs(n=600, n_boot=50)
    ids = labels.site_ids
    p = np.linspace(0.1, 0.9, len(ids))
    good = E.entry(ids, p, task.event_id, "synthetic effort-only baseline")
    card = build_card(task, entries, labels, shaking_only=shaking, effort_only=good)
    names = [r["name"] for r in card["entries"]]
    assert "baseline: effort only" in names
    bad_name = Entry(synth.manifest("baseline: effort", "other_probability", task.event_id), ids, p)
    with pytest.raises(CardRefused, match="effort only"):
        build_card(task, entries, labels, shaking_only=shaking, effort_only=bad_name)
    bad_kind = Entry(synth.manifest(E.NAME, "target_probability", task.event_id), ids, p)
    with pytest.raises(CardRefused, match="other_probability"):
        build_card(task, entries, labels, shaking_only=shaking, effort_only=bad_kind)


def test_post_event_features_are_refused_by_the_baseline_allowlist():
    post = ("cpt_count_250m", "cpt_count_1000m", "cpt_nn_dist_m", "pushed_after_event", "recon_road_dist_m")
    for f in post:
        assert f in E.PROXY_FEATURES and f not in E.EFFORT_FEATURES
        with pytest.raises(E.EffortRefused, match="allowlist"):
            E.check_features(["builtup_share_250m", f])
    assert E.check_features(list(E.PROXY_FEATURES), E.PROXY_FEATURES) == tuple(E.PROXY_FEATURES)
    with pytest.raises(E.EffortRefused, match="EFFORT_FEATURES or PROXY_FEATURES"):
        E.check_features(["pga"], {"pga": "made up"})


def test_proxy_cannot_enter_the_card_and_baseline_marks_place_exposure():
    task, entries, labels, shaking = synth.demo_inputs(n=600, n_boot=50)
    ids = labels.site_ids
    p = np.linspace(0.1, 0.9, len(ids))
    good = E.entry(ids, p, task.event_id, "synthetic effort-only baseline")
    assert good.manifest.exposure[task.event_id] == "seen_place"
    card = build_card(task, entries, labels, shaking_only=shaking, effort_only=good)
    row = [r for r in card["entries"] if r["name"] == E.NAME][0]
    assert row["exposure"] == "SEEN PLACE (OTHER EARTHQUAKES)"
    proxy = Entry(synth.manifest(E.PROXY_NAME, "other_probability", task.event_id), ids, p)
    with pytest.raises(CardRefused, match="effort only"):
        build_card(task, entries, labels, shaking_only=shaking, effort_only=proxy)

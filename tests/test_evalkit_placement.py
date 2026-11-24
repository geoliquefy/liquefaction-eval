"""evalkit placement control (own-work list B7, eval M4). Synthetic and planted data only (correction 1).

No real Canterbury label is read here; the real proxy AUC (0.72) is not recomputed (B8 does that under v3).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import card as C  # noqa: E402
from scoreboard.evalkit import effort as E  # noqa: E402
from scoreboard.evalkit import metrics as M  # noqa: E402
from scoreboard.evalkit import placement as PL  # noqa: E402
from scoreboard.evalkit import synth  # noqa: E402
from scoreboard.evalkit.card import CardRefused, build_card, scan_aggregates  # noqa: E402
from scoreboard.evalkit.spec import Entry, Labels, Task, card_sha256, site_list_sha256  # noqa: E402

EV = "synthetic-placement-1"


# (a) strata rule -------------------------------------------------------------------------------------------------

def test_density_counts_the_disc_and_leaves_out_the_own_record():
    cx = np.array([0.0, 1.0, 500.0, 999.0, 1001.0, 3000.0])
    cy = np.zeros(6)
    d = PL.cpt_density_per_km2(np.array([0.0]), np.array([0.0]), cx, cy, radius_m=1000.0)
    # 0 and 1 m are the site's own record (within 2 m); 500 and 999 m count; 1001 and 3000 m do not
    assert d[0] == pytest.approx(2 / (np.pi * 1.0))
    assert PL.cpt_density_per_km2(np.array([0.0]), np.array([0.0]), cx[:0], cy[:0])[0] == 0.0
    with pytest.raises(PL.PlacementRefused):
        PL.cpt_density_per_km2(np.array([np.nan]), np.array([0.0]), cx, cy)


def test_tertile_strata_use_no_labels_and_merge_equal_cut_points():
    st, cuts = PL.strata_from_density(np.arange(9.0))
    assert st.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2] and len(cuts) == 2
    st, cuts = PL.strata_from_density(np.r_[np.zeros(8), 5.0])        # both cut points are 0: two strata left
    assert cuts == [0.0] and st.tolist() == [0] * 8 + [1]
    st, cuts = PL.strata_from_density(np.zeros(9))                     # all equal: one stratum
    assert st.tolist() == [0] * 9
    st, cuts = PL.strata_from_density(np.r_[np.zeros(4), np.arange(1.0, 6.0)])   # cuts 0 and 2: 0 | 1-2 | 3-5
    assert st.tolist() == [0, 0, 0, 0, 1, 1, 2, 2, 2]
    assert "y" not in PL.build.__code__.co_varnames[:PL.build.__code__.co_argcount]
    with pytest.raises(PL.PlacementRefused):
        PL.strata_from_density(np.arange(5.0), n_strata=1)


def test_pooled_stratified_auc_matches_the_effort_rule_and_weights_sum_to_one():
    rng = np.random.default_rng(3)
    strata = rng.integers(0, 3, 900)
    y = rng.random(900) < np.array([0.2, 0.5, 0.8])[strata]
    s = rng.normal(size=900) + 0.5 * y
    r = PL.pooled_stratified_auc(y, s, strata)
    assert r["auc"] == pytest.approx(E.stratified_auc(y, s, strata)["auc"], abs=1e-12)
    assert sum(p["weight"] for p in r["by_stratum"]) == pytest.approx(1.0)
    manual = sum(p["n_yes"] * p["n_no"] * p["auc"] for p in r["by_stratum"]) / sum(
        p["n_yes"] * p["n_no"] for p in r["by_stratum"])
    assert r["auc"] == pytest.approx(manual)
    one = PL.pooled_stratified_auc(np.r_[np.ones(5), 0, 1].astype(bool), np.arange(7.0), np.r_[np.zeros(5), 1, 1])
    assert one["strata_skipped"] == 1 and one["by_stratum"][0]["weight"] == 0.0


def test_margin_matches_the_card_equivalence_margin():
    assert PL.MARGIN == C.EQUIVALENCE_MARGIN["auc"]
    assert PL.PROXY_NAME == E.PROXY_NAME and PL.PROXY_ALLOWLIST == E.PROXY_ALLOWLIST_NAME
    assert PL.EFFORT_IMAGERY_ALLOWLIST == E.FOOTPRINT_ALLOWLIST_NAME


# Planted worlds ------------------------------------------------------------------------------------------------

def test_planted_placement_on_the_strata_cuts_gives_auc_near_072_and_about_05_inside_strata():
    """Labels step at the same tertile cuts the control uses, so the 0.5 inside is built in (review 1)."""
    over, inside, skill_in = [], [], []
    for seed in range(5):
        w = PL.planted_world(seed, mode="placement_only")
        y = w["label"].astype(bool)
        st, _ = PL.strata_from_density(w["density"])
        over.append(M.auc(y, w["proxy_score"]))
        inside.append(PL.pooled_stratified_auc(y, w["proxy_score"], st)["auc"])
        skill_in.append(PL.pooled_stratified_auc(y, w["skill_score"], st)["auc"])
    assert 0.70 <= np.mean(over) <= 0.75, over
    assert abs(np.mean(inside) - 0.5) < 0.03, inside
    assert np.mean(skill_in) > 0.65                  # a real model keeps its skill inside the strata


def test_planted_placement_off_the_strata_cuts_stays_inside_3_strata_and_goes_inside_5():
    """Review 1: placement alone, stepping at 5 quantile levels (no cut at 1/3 or 2/3). Three strata leave a
    residual (an OPEN BIAS that is placement, not a confound); five strata matching the steps remove it."""
    over, in3, in5 = [], [], []
    for seed in range(5):
        w = PL.planted_world(seed, mode="placement_off_strata")
        y = w["label"].astype(bool)
        st3, _ = PL.strata_from_density(w["density"], 3)
        st5, _ = PL.strata_from_density(w["density"], 5)
        over.append(M.auc(y, w["proxy_score"]))
        in3.append(PL.pooled_stratified_auc(y, w["proxy_score"], st3)["auc"])
        in5.append(PL.pooled_stratified_auc(y, w["proxy_score"], st5)["auc"])
    assert 0.72 <= np.mean(over) <= 0.77, over
    assert 0.56 <= np.mean(in3) <= 0.62, in3
    assert abs(np.mean(in5) - 0.5) < 0.03, in5
    with pytest.raises(PL.PlacementRefused):
        PL.planted_world(0, mode="confound")


def test_planted_continuous_density_still_ranks_inside_strata():
    inside = []
    for seed in range(3):
        w = PL.planted_world(seed, mode="continuous")
        y = w["label"].astype(bool)
        st, _ = PL.strata_from_density(w["density"])
        inside.append(PL.pooled_stratified_auc(y, w["proxy_score"], st)["auc"])
    assert np.mean(inside) > 0.55


def test_bias_word_rules():
    rng = np.random.default_rng(0)
    assert PL.bias_word(0.50, 0.50 + rng.normal(0, 0.01, 500))["word"] == "EXPLAINED BY DENSITY STRATA"
    w = PL.bias_word(0.60, 0.60 + rng.normal(0, 0.01, 500))
    assert w["word"] == "OPEN BIAS" and w["open_bias_size_auc"]["value"] == pytest.approx(0.10)
    assert w["open_bias_size_auc"]["lo"] > 0
    assert PL.bias_word(0.52, 0.52 + rng.normal(0, 0.04, 500))["word"] == "CANNOT TELL"
    assert PL.bias_word(float("nan"), np.array([np.nan]))["word"] == "CANNOT TELL"
    assert PL.bias_word(0.50, 0.50 + rng.normal(0, 0.01, 500))["residual_excludes_0.5"] is False


def test_bias_word_explained_wins_when_both_hold_and_flags_the_residual():
    """Review 1: a narrow interval inside the margin that also excludes 0.5 (large n) stays EXPLAINED, by order."""
    rng = np.random.default_rng(1)
    draws = 0.53 + rng.normal(0, 0.005, 2000)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    lo90, hi90 = np.percentile(draws, [5.0, 95.0])
    assert lo > 0.5 and 0.45 <= lo90 and hi90 <= 0.55             # both conditions hold
    w = PL.bias_word(0.53, draws)
    assert w["word"] == "EXPLAINED BY DENSITY STRATA" and w["residual_excludes_0.5"] is True
    assert w["open_bias_size_auc"]["value"] == pytest.approx(0.03) and w["open_bias_size_auc"]["lo"] > 0
    assert w["rule"].startswith("checked in this order. EXPLAINED")


def test_lift_share_is_not_computed_when_the_lift_is_below_the_margin():
    assert PL.lift_share_removed(0.72, 0.50) == pytest.approx(1.0)
    assert PL.lift_share_removed(0.72, 0.61) == pytest.approx(0.5)
    assert np.isnan(PL.lift_share_removed(0.54, 0.50))
    assert np.isnan(PL.lift_share_removed(0.45, 0.50))


# Card hook -----------------------------------------------------------------------------------------------------

def _world_card_inputs(seed, mode, n_boot=200):
    w = PL.planted_world(seed, mode=mode)
    ids = w["site_ids"]
    labels = Labels(ids, w["label"], w["x_m"], w["y_m"])
    task = Task(event_id=EV, target="planted ejecta", label_source="synthetic planted v1",
                label_rule_sha256="0" * 64, site_list_sha256=site_list_sha256(ids), track="record",
                division="closed", tile_m=2000.0, clip=1e-6, seed=20261123, label_caveats="planted; no survey",
                n_boot=n_boot, census=True, earlier_base_rate=None, earlier_label_sets=0)
    entries = [
        Entry(synth.manifest("Planted real model", "score", EV), ids, w["skill_score"]),
        Entry(synth.manifest("Planted placement rider", "score", EV), ids, w["density"] + 1e-6 * w["s"]),
    ]
    proxy_p = w["proxy_score"] / (w["proxy_score"].max() + 1.0)
    proxy = E.entry(ids, proxy_p, EV, "planted investigation-density proxy", E.PROXY_FEATURES)
    ps = PL.build(ids, w["x_m"], w["y_m"], w["cpt_x"], w["cpt_y"], basis="all_index", proxy=proxy)
    return task, entries, labels, ps


@pytest.fixture(scope="module")
def placement_only_card():
    task, entries, labels, ps = _world_card_inputs(4, "placement_only")
    return build_card(task, entries, labels, placement=ps), (task, entries, labels, ps)


def test_card_placement_block_explains_a_planted_placement_bias(placement_only_card):
    card, (_, _, labels, _) = placement_only_card
    b = card["placement"]
    assert b["status"] == "SCORED" and b["rule"]["basis"] == "all_index" and b["rule"]["n_strata_kept"] == 3
    px = b["proxy"]
    assert 0.69 <= px["auc_overall"]["value"] <= 0.76
    assert abs(px["auc_within_density_strata"]["value"] - 0.5) < 0.04
    assert px["word"] == "EXPLAINED BY DENSITY STRATA", px
    assert px["share_of_lift_removed"] > 0.8
    rider = b["auc"]["Planted placement rider"]
    real = b["auc"]["Planted real model"]
    assert rider["within_minus_overall"]["hi"] < 0                     # the rider loses its lift inside strata
    assert real["auc_within_density_strata"]["value"] > 0.65
    assert "baseline: coin" not in b["auc"]                             # constant rows are left out
    scan_aggregates(card, labels.site_ids)


def test_card_placement_block_prints_an_open_bias_with_its_size():
    task, entries, labels, ps = _world_card_inputs(1, "continuous")
    px = build_card(task, entries, labels, placement=ps)["placement"]["proxy"]
    assert px["word"] == "OPEN BIAS", px
    assert px["open_bias_size_auc"]["value"] > 0.04 and px["open_bias_size_auc"]["lo"] > 0


def test_card_prints_open_bias_for_placement_off_the_strata_cuts_and_says_why():
    """Review 1: placement alone, not lined up with the tertile cuts, prints OPEN BIAS; the card says this can be
    placement left inside a stratum, and that the stratum count is B8's choice."""
    task, entries, labels, ps = _world_card_inputs(2, "placement_off_strata")
    b = build_card(task, entries, labels, placement=ps)["placement"]
    px = b["proxy"]
    assert px["word"] == "OPEN BIAS", px
    assert 0.55 <= px["auc_within_density_strata"]["value"] <= 0.65
    for text in (px["note"], b["rule"]["strata_limit"]):
        assert "placement left inside a stratum" in text and "B8 must register" in text
        assert "pushed_after_event" in text


def test_placement_is_off_by_default_and_leaves_the_card_unchanged(placement_only_card):
    card, (task, entries, labels, _) = placement_only_card
    plain = build_card(task, entries, labels)
    assert "placement" not in plain
    without = {k: v for k, v in card.items() if k != "placement"}
    assert card_sha256(without) == card_sha256(plain)


def test_card_refuses_strata_not_built_by_the_kit_or_changed_after(placement_only_card):
    _, (task, entries, labels, ps) = placement_only_card
    fake = PL.PlacementStrata(ps.site_ids, ps.stratum.copy(), dict(ps.rule), ps.proxy)
    with pytest.raises(CardRefused, match="B7"):
        build_card(task, entries, labels, placement=fake)
    ps2 = PL.build(ps.site_ids, labels.x_m, labels.y_m, np.array([0.0]), np.array([0.0]), basis="pre_event",
                   cpt_dates=["2009-01-01"], first_event_date="2010-09-04")
    ps2.stratum[:] = 0                                   # changed after the build
    with pytest.raises(CardRefused, match="B7"):
        build_card(task, entries, labels, placement=ps2)


def test_pre_event_basis_is_checked_by_push_date_not_claimed():
    """Review 1: pre_event needs push dates; build keeps only CPTs pushed before the first earthquake."""
    ids = ("a", "b")
    sx, sy = np.array([0.0, 5000.0]), np.zeros(2)
    cx = np.array([100.0, 200.0, 300.0, 400.0, 5100.0])
    cy = np.zeros(5)
    dates = ["2009-05-01", "2010-09-03", "2010-09-04", None, "2012-01-01"]
    with pytest.raises(PL.PlacementRefused, match="push date"):
        PL.build(ids, sx, sy, cx, cy, basis="pre_event")
    with pytest.raises(PL.PlacementRefused, match="both"):
        PL.build(ids, sx, sy, cx, cy, basis="all_index", cpt_dates=dates)
    with pytest.raises(PL.PlacementRefused, match="one push date per CPT"):
        PL.build(ids, sx, sy, cx, cy, basis="pre_event", cpt_dates=dates[:3], first_event_date="2010-09-04")
    pre = PL.build(ids, sx, sy, cx, cy, basis="pre_event", cpt_dates=dates, first_event_date="2010-09-04")
    r = pre.rule
    assert r["n_cpts_given"] == 5 and r["n_cpts_counted"] == 2                 # 2009 and the day before only
    assert r["n_cpts_left_out_by_date"] == 2 and r["n_cpts_left_out_undated"] == 1
    assert r["first_event_date"] == "2010-09-04" and "strictly before" in r["date_filter"]
    allx = PL.build(ids, sx, sy, cx, cy, basis="all_index", cpt_dates=dates, first_event_date="2010-09-04")
    assert allx.rule["n_cpts_counted"] == 5 and allx.rule["n_cpts_on_or_after_first_event"] == 2
    assert "n_cpts_left_out_by_date" not in allx.rule
    plain = PL.build(ids, sx, sy, cx, cy, basis="all_index")
    assert plain.rule["n_cpts_counted"] == 5 and "first_event_date" not in plain.rule


def test_build_refuses_unnamed_basis_and_a_proxy_that_is_not_the_proxy():
    ids, x = ("a", "b", "c"), np.array([0.0, 1.0, 2.0])
    with pytest.raises(PL.PlacementRefused, match="basis"):
        PL.build(ids, x, x, x, x, basis="post_event")
    not_proxy = E.entry(ids, np.array([0.1, 0.2, 0.3]), EV, "effort", E.EFFORT_FEATURES)
    with pytest.raises(PL.PlacementRefused, match="proxy"):
        PL.build(ids, x, x, x, x, basis="pre_event", proxy=not_proxy)


def test_underpowered_card_prints_the_rule_only():
    task, entries, labels, ps = _world_card_inputs(0, "placement_only", n_boot=50)
    small = Task(**{**task.to_dict(), "min_sites": 10 ** 6})
    card = build_card(small, entries, labels, placement=ps)
    assert card["status"] == "UNDERPOWERED"
    assert card["placement"]["status"].startswith("UNDERPOWERED") and "proxy" not in card["placement"]


# (b) imagery footprint, known limit N4 ---------------------------------------------------------------------------

def test_footprint_allowlist_swaps_the_class_union_for_the_imagery_extent():
    assert "in_imagery_footprint" in E.FOOTPRINT_FEATURES and "in_aerial_footprint" not in E.FOOTPRINT_FEATURES
    assert set(E.FOOTPRINT_FEATURES) - {"in_imagery_footprint"} == set(E.EFFORT_FEATURES) - {"in_aerial_footprint"}
    with pytest.raises(E.EffortRefused, match="allowlist"):
        E.check_features(["builtup_share_250m", "in_aerial_footprint"], E.FOOTPRINT_FEATURES)
    assert "in_aerial_footprint" in E.EFFORT_FEATURES                   # the v2 allowlist is unchanged


def test_imagery_footprint_feature_refuses_the_class_union_and_classes_outside():
    imagery = shapely.box(0, 0, 1000, 1000)
    classes = [shapely.box(100, 100, 300, 300), shapely.box(600, 600, 900, 900)]
    x = np.array([50.0, 200.0, 950.0, 1500.0])
    y = np.array([50.0, 200.0, 950.0, 500.0])
    f = PL.imagery_footprint_feature(x, y, imagery, classes)
    assert f.tolist() == [1.0, 1.0, 1.0, 0.0]
    with pytest.raises(PL.PlacementRefused, match="N4"):
        PL.imagery_footprint_feature(x, y, shapely.union_all(classes), classes)
    with pytest.raises(PL.PlacementRefused, match="outside"):
        PL.imagery_footprint_feature(x, y, shapely.box(0, 0, 500, 500), classes)
    with pytest.raises(PL.PlacementRefused):
        PL.imagery_footprint_feature(x, y, shapely.Polygon())


def test_planted_n4_leak_union_ranks_the_outcome_but_the_imagery_extent_does_not():
    """Mappers drew classes only around damage: the union flags yes sites; the imagery extent covers all sites."""
    rng = np.random.default_rng(7)
    n = 2000
    x, y = rng.uniform(0, 10000, n), rng.uniform(0, 10000, n)
    lab = rng.random(n) < 0.4
    classes = [shapely.buffer(shapely.Point(a, b), 60.0) for a, b in zip(x[lab][:400], y[lab][:400])]
    union = shapely.contains_xy(shapely.union_all(classes), x, y).astype(float)
    imagery = PL.imagery_footprint_feature(x, y, shapely.box(-10, -10, 10010, 10010), classes)
    assert M.auc(lab, union) > 0.6 and M.auc(lab, imagery) == pytest.approx(0.5)


def test_card_effort_slot_takes_the_imagery_allowlist_only_when_asked():
    task, entries, labels, shaking = synth.demo_inputs(n=600, n_boot=50)
    ids = labels.site_ids
    p = np.linspace(0.1, 0.9, len(ids))
    old = E.entry(ids, p, task.event_id, "synthetic effort-only baseline")
    new = E.entry(ids, p, task.event_id, "synthetic effort-only baseline, imagery footprint", E.FOOTPRINT_FEATURES)
    assert new.manifest.feature_allowlist == "pre-event-imagery"
    with pytest.raises(CardRefused, match="pre-event"):
        build_card(task, entries, labels, shaking_only=shaking, effort_only=new)
    card = build_card(task, entries, labels, shaking_only=shaking, effort_only=new, effort_imagery_footprint=True)
    assert E.NAME in [r["name"] for r in card["entries"]]
    with pytest.raises(CardRefused, match="pre-event-imagery"):
        build_card(task, entries, labels, shaking_only=shaking, effort_only=old, effort_imagery_footprint=True)
    # review 1: the card names the footprint the effort row used; the flag alone is refused; default notes unchanged
    assert PL.EFFORT_IMAGERY_NOTE in card["notes"] and "pre-event-imagery" in PL.EFFORT_IMAGERY_NOTE
    plain = build_card(task, entries, labels, shaking_only=shaking, effort_only=old)
    assert PL.EFFORT_IMAGERY_NOTE not in plain["notes"]
    assert [n for n in card["notes"] if n != PL.EFFORT_IMAGERY_NOTE] == plain["notes"]
    with pytest.raises(CardRefused, match="effort_only"):
        build_card(task, entries, labels, shaking_only=shaking, effort_imagery_footprint=True)

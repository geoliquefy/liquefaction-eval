"""evalkit metamorphic relations on the scoring path (eval plan step 2b; design section 4, V0). Synthetic only.

Each relation transforms the inputs in a way whose effect on the card is known without an oracle:

  site permutation        rows of the label file and of each entry reordered: every score unchanged
  name swap               two entries trade names: their scores trade places, nothing else moves
  rigid shift             coordinates shifted by whole multiples of the largest tile side: tiles and scores unchanged
                          (an arbitrary shift moves the grid: it is anchored at the coordinate origin, see below)
  move toward labels      p -> p + t (y - p), 0 < t < 1: log loss and Brier fall, AUC does not fall
  duplicate earthquake    the same earthquake id twice is refused; an earthquake's sites entered twice (new ids,
                          same places) leave its paired difference, its weight and the pooled word unchanged
  renamed copy            a copy of an entry under a new name gets the identical score and the duplicate detector
                          joins the two
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import gates, pool, resample, synth  # noqa: E402
from scoreboard.evalkit.card import build_card  # noqa: E402
from scoreboard.evalkit.spec import Entry, Labels, Task, site_list_sha256  # noqa: E402

EV = "eq-meta"
TOL = 1e-9
A, B = "ALPHA-ENTRY", "BRAVO-ENTRY"


def make_task(ids, event_id=EV, **kw):
    d = dict(event_id=event_id, target="ejecta seen at the site", label_source="synthetic v1",
             label_rule_sha256="1" * 64, site_list_sha256=site_list_sha256(ids), track="forward", division="closed",
             tile_m=2000.0, clip=1e-6, seed=5, label_caveats="synthetic labels", n_boot=80)
    d.update(kw)
    return Task(**d)


def model(ev, seed, sd):
    rng = np.random.default_rng(seed)
    s = ev["s"] + rng.normal(0.0, sd, ev["s"].size)
    return ev["fragility"] / (1.0 + np.exp(-(ev["intercept"] + ev["slope"] * s)))


def entries_for(ev, event_id=EV, names=(A, B, "CHARLIE-INDEX")):
    ids = ev["site_ids"]
    return [Entry(synth.manifest(names[0], "target_probability", event_id), ids, model(ev, 1, 0.5)),
            Entry(synth.manifest(names[1], "target_probability", event_id), ids, model(ev, 2, 1.0)),
            Entry(synth.manifest(names[2], "score", event_id, exposure="seen_inputs"), ids, ev["s"] + 0.3)]


@pytest.fixture(scope="module")
def ev():
    return synth.synthetic_event(700, 31, unlabelled_share=0.1)


@pytest.fixture(scope="module")
def base_card(ev):
    return build_card(make_task(ev["site_ids"]), entries_for(ev), synth.labels_of(ev))


SKIP_KEYS = {"labels_sha256", "entry_sha256", "variogram_sha256"}


def assert_same(a, b, path="card", skip=SKIP_KEYS):
    """Recursive equality: floats to TOL, everything else exact; keys in `skip` are not compared."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            if k not in skip:
                assert_same(a[k], b[k], f"{path}.{k}", skip)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: lengths"
        for k, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{k}]", skip)
    elif isinstance(a, float) and isinstance(b, float):
        assert math.isclose(a, b, rel_tol=TOL, abs_tol=TOL), f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def rename(obj, mapping):
    """Swap names everywhere: keys, values and inside sentences (names are unique tokens)."""
    if isinstance(obj, dict):
        return {rename(k, mapping): rename(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rename(v, mapping) for v in obj]
    if isinstance(obj, str):
        out = obj
        keys = list(mapping)
        for j, k in enumerate(keys):
            out = out.replace(k, f"\0{j}\0")
        for j, k in enumerate(keys):
            out = out.replace(f"\0{j}\0", mapping[k])
        return out
    return obj


def permuted(ev, perm):
    out = dict(ev)
    for k in ("x_m", "y_m", "s", "field", "p_trigger", "p_target", "label"):
        out[k] = ev[k][perm]
    out["site_ids"] = tuple(ev["site_ids"][i] for i in perm)
    return out


# Site permutation ------------------------------------------------------------------------------------------------

def test_site_permutation_leaves_every_score_unchanged(ev, base_card):
    """Label rows and entry rows reordered (each by its own permutation): the card's numbers and words agree to
    1e-9. The label hash and the entry hashes are hashes of the files in row order (R4, R8), so they are the
    fields allowed to move (step 2b finding: both hashes are row-order bound)."""
    rng = np.random.default_rng(8)
    p_lab = rng.permutation(len(ev["site_ids"]))
    lab_ev = permuted(ev, p_lab)
    ents = entries_for(ev)
    p_ent = rng.permutation(len(ev["site_ids"]))
    ents = [Entry(e.manifest, tuple(e.site_ids[i] for i in p_ent), e.p[p_ent]) for e in ents]
    card = build_card(make_task(lab_ev["site_ids"]), ents, synth.labels_of(lab_ev))
    assert_same(base_card, card)
    assert base_card["label"]["labels_sha256"] != card["label"]["labels_sha256"]


def test_site_permutation_keeps_the_tile_partition_and_the_variogram_choice(ev):
    perm = np.random.default_rng(9).permutation(len(ev["site_ids"]))
    for side in resample.TILE_CANDIDATES_M:
        t0 = resample.tiles_for(ev["x_m"], ev["y_m"], side)
        t1 = resample.tiles_for(ev["x_m"][perm], ev["y_m"][perm], side)
        assert np.array_equal(t0[perm], t1)
    lab = ~np.isnan(ev["label"])
    r = ev["label"][lab] - 0.5
    c0 = resample.choose_tile(ev["x_m"][lab], ev["y_m"][lab], r)
    q = np.random.default_rng(10).permutation(r.size)
    c1 = resample.choose_tile(ev["x_m"][lab][q], ev["y_m"][lab][q], r[q])
    assert c0["tile_m"] == c1["tile_m"] and c0["range_m"] == pytest.approx(c1["range_m"], rel=1e-9)


# Name swap -------------------------------------------------------------------------------------------------------

def test_swapping_two_entry_names_swaps_their_scores_and_nothing_else(ev, base_card):
    card = build_card(make_task(ev["site_ids"]), entries_for(ev, names=(B, A, "CHARLIE-INDEX")),
                      synth.labels_of(ev))
    assert_same(rename(base_card, {A: B, B: A}), card)
    assert card["metrics"][A]["auc"]["value"] == base_card["metrics"][B]["auc"]["value"]
    assert card["metrics"]["CHARLIE-INDEX"] == base_card["metrics"]["CHARLIE-INDEX"]


# Rigid shift -----------------------------------------------------------------------------------------------------

def test_rigid_shift_by_whole_tiles_leaves_tiles_and_scores_unchanged(ev, base_card):
    dx, dy = 3 * 8000.0, -2 * 8000.0          # a multiple of every candidate tile side (500 m to 8 km)
    for side in resample.TILE_CANDIDATES_M:
        assert np.array_equal(resample.tiles_for(ev["x_m"], ev["y_m"], side),
                              resample.tiles_for(ev["x_m"] + dx, ev["y_m"] + dy, side))
    lab = Labels(ev["site_ids"], ev["label"], ev["x_m"] + dx, ev["y_m"] + dy)
    card = build_card(make_task(ev["site_ids"]), entries_for(ev), lab)
    assert_same(base_card, card)


def test_rigid_shift_leaves_the_variogram_tile_choice_unchanged(ev):
    lab = ~np.isnan(ev["label"])
    x, y, r = ev["x_m"][lab], ev["y_m"][lab], ev["label"][lab] - 0.5
    c0 = resample.choose_tile(x, y, r)
    for dx, dy in ((1234.5, -987.25), (5e5, 5e6)):
        c1 = resample.choose_tile(x + dx, y + dy, r)
        assert c0["tile_m"] == c1["tile_m"] and c0["range_m"] == pytest.approx(c1["range_m"], rel=1e-6)


def test_an_arbitrary_shift_moves_the_grid_anchor(ev):
    """Known limit (step 2b finding, not a fix: a fix would move every interval, tier 2). The tile grid is anchored
    at the coordinate origin (floor(x / side)), so a shift that is not a whole multiple of the side changes the
    partition. A card is therefore tied to its projected coordinate system, which the registration must name."""
    t0 = resample.tiles_for(ev["x_m"], ev["y_m"], 2000.0)
    t1 = resample.tiles_for(ev["x_m"] + 700.0, ev["y_m"], 2000.0)
    same = all(len(set(t1[t0 == k])) == 1 for k in np.unique(t0))
    assert not same


# Move toward the labels ------------------------------------------------------------------------------------------

def test_moving_predictions_toward_the_labels_improves_log_loss_and_brier(ev):
    ids = ev["site_ids"]
    p = model(ev, 1, 1.0)
    p[:20] = 0.0                                              # exact zeros, as Level 0 has: the clip must not break it
    y = np.nan_to_num(ev["label"], nan=0.5)                   # unlabelled sites are not scored; any value works
    ts = (0.0, 0.1, 0.3, 0.6, 0.9)
    ents = [Entry(synth.manifest(f"MOVED-{t}", "target_probability", EV), ids, p + t * (y - p)) for t in ts]
    card = build_card(make_task(ids), ents, synth.labels_of(ev))
    m = [card["metrics"][f"MOVED-{t}"] for t in ts]
    ll = [x["log_loss"]["value"] for x in m]
    br = [x["brier"]["value"] for x in m]
    au = [x["auc"]["value"] for x in m]
    assert all(a > b for a, b in zip(ll, ll[1:])), ll
    assert all(a > b for a, b in zip(br, br[1:])), br
    assert all(a <= b + 1e-12 for a, b in zip(au, au[1:])), au
    lab = ~np.isnan(ev["label"])
    for t, b in zip(ts, br):                                  # Brier scales exactly by (1 - t)^2
        assert b == pytest.approx((1 - t) ** 2 * br[0], rel=1e-12)
    for c in card["comparisons"]:
        if (c["family"] == "probability" and c["word"] == "SEPARATES" and c["a"].startswith("MOVED")
                and c["b"].startswith("MOVED")):
            ta, tb = float(c["a"].split("-")[1]), float(c["b"].split("-")[1])
            assert c["favours"] == (c["a"] if ta > tb else c["b"])
    assert lab.sum() > 0


# Duplicate earthquake --------------------------------------------------------------------------------------------

def doubled(ev, entries):
    """Every site entered twice: copies get new ids, same place, same label and same predictions."""
    out = dict(ev)
    for k in ("x_m", "y_m", "s", "field", "p_trigger", "p_target", "label"):
        out[k] = np.concatenate([ev[k], ev[k]])
    out["site_ids"] = tuple(ev["site_ids"]) + tuple(f"{s}-copy" for s in ev["site_ids"])
    ents = [Entry(e.manifest, out["site_ids"], np.concatenate([e.p, e.p])) for e in entries]
    return out, ents


def test_an_earthquake_entered_twice_is_refused_by_the_pooled_line():
    items = [{"event_id": e, "entry_a": A, "entry_b": B, "metric": "auc", "diff": d, "se": 0.02, "passes_gate": True}
             for e, d in (("q1", 0.05), ("q2", 0.04), ("q3", 0.06))]
    with pytest.raises(ValueError, match="once"):
        pool.pooled_line(items + [dict(items[0])], registered=True)


def test_duplicating_an_earthquakes_sites_leaves_its_weight_and_the_pooled_word_unchanged():
    """Weights are per earthquake from the tile bootstrap; copies sit in the same tiles as the originals, so the
    paired difference, its SE (hence the random-effects weight) and the pooled word do not move."""
    cards = {}
    for k, seed in enumerate((41, 42, 43)):
        e = synth.synthetic_event(450, seed)
        eid = f"q{k + 1}"
        ents = entries_for(e, eid)
        cards[eid] = build_card(make_task(e["site_ids"], eid), ents, synth.labels_of(e))
        if k == 0:
            d, dents = doubled(e, ents)
            cards["q1-doubled"] = build_card(make_task(d["site_ids"], eid), dents, synth.labels_of(d))
    it = {k: pool.card_item(c, A, B, "auc") for k, c in cards.items()}
    assert it["q1"]["diff"] == pytest.approx(it["q1-doubled"]["diff"], abs=1e-12)
    assert it["q1"]["se"] == pytest.approx(it["q1-doubled"]["se"], abs=1e-12)
    ll = {k: pool.card_item(c, A, B, "log_loss") for k, c in cards.items()}
    assert ll["q1"]["diff"] == pytest.approx(ll["q1-doubled"]["diff"], abs=1e-12)
    for metric, items in (("auc", it), ("log_loss", ll)):
        one = pool.pooled_line([items["q1"], items["q2"], items["q3"]], registered=True)
        two = pool.pooled_line([items["q1-doubled"], items["q2"], items["q3"]], registered=True)
        assert one["word"] == two["word"], metric
        assert one["pooled"]["mean"] == pytest.approx(two["pooled"]["mean"], abs=1e-12)
        assert one["pooled"]["tau2"] == pytest.approx(two["pooled"]["tau2"], abs=1e-12)


def test_random_effects_weights_do_not_depend_on_item_order():
    rng = np.random.default_rng(3)
    d, s = rng.normal(0.03, 0.02, 5), rng.uniform(0.01, 0.03, 5)
    r0 = pool.random_effects(d, s)
    perm = rng.permutation(5)
    r1 = pool.random_effects(d[perm], s[perm])
    for k in ("mean", "se", "lo", "hi", "p", "tau2"):
        assert r0[k] == pytest.approx(r1[k], rel=1e-12, abs=1e-15)


# Renamed copy ----------------------------------------------------------------------------------------------------

def test_a_renamed_copy_gets_the_identical_score_and_is_joined_as_a_duplicate(ev, base_card):
    ents = entries_for(ev)
    copy = Entry(synth.manifest("ALPHA-COPY", "target_probability", EV), ev["site_ids"], ents[0].p.copy())
    card = build_card(make_task(ev["site_ids"]), ents + [copy], synth.labels_of(ev))
    assert_same(card["metrics"][A], card["metrics"]["ALPHA-COPY"], skip=set())
    assert_same(base_card["metrics"][A], card["metrics"]["ALPHA-COPY"], skip=set())
    pairs = [{p["a"], p["b"]} for p in card["duplicates"]["pairs"]]
    assert {A, "ALPHA-COPY"} in pairs
    assert card["duplicates"]["distinct_models"] == base_card["duplicates"]["distinct_models"]
    for c in card["comparisons"]:
        if {c["a"], c["b"]} == {A, "ALPHA-COPY"}:
            assert c["diff"] == 0.0 and c["word"] != "SEPARATES"
    vals = {A: ents[0].p, "ALPHA-COPY": ents[0].p}
    assert gates.distinct_models(list(vals), gates.duplicate_pairs(vals)) == 1

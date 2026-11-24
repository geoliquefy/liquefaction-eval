"""B8 (eval M4, ruling 48): hash recipe v2 for scorecard v3. Synthetic data only. No em dashes.

Recipe v1 (default) must keep every sealed hash. Recipe v2 must absorb the two container causes of the D1 record
(EVAL-WAVE2 section 6): a refit row whose predictions move by about 1e-12, and a gate value that is 0.0 on one
platform and about 1e-14 on another.
"""
import numpy as np
import pytest

from scoreboard.evalkit import effort as E
from scoreboard.evalkit import synth
from scoreboard.evalkit.card import build_card, hash_recipe, is_refit
from scoreboard.evalkit.shaking import FIT_MARKER
from scoreboard.evalkit.spec import (Entry, EntryManifest, SpecError, Task, card_sha256, entry_sha256, site_list_sha256)

EV = "eq-test"


def test_default_recipe_is_v1_and_unchanged():
    card = {"a": 0.123456789012345, "b": [1e-13, -0.0, 3], "c": {"d": "x", "e": True}}
    assert card_sha256(card) == card_sha256(card, recipe="v1")
    assert card_sha256(card) != card_sha256(card, recipe="v2")


def test_unknown_recipe_refused():
    with pytest.raises(SpecError):
        card_sha256({"a": 1.0}, recipe="v9")


def test_v2_absorbs_relative_gaps_below_1e9():
    a = {"x": 327.951234567, "y": 0.7238158359187}
    b = {"x": 327.951234567 * (1 + 4e-12), "y": 0.7238158359187 * (1 + 7.5e-11)}
    assert card_sha256(a, recipe="v2") == card_sha256(b, recipe="v2")


def test_v2_rounds_near_zero_on_an_absolute_grid():
    # D1 cause 2: G-lsn max_abs_diff is 0.0 on the Mac and 1.4e-14 to 2.8e-14 in the container
    assert card_sha256({"g": 0.0}, recipe="v2") == card_sha256({"g": 2.84e-14}, recipe="v2")
    assert card_sha256({"g": -0.0}, recipe="v2") == card_sha256({"g": 1.4e-14}, recipe="v2")
    # a near-zero value that differs at 1e-12 or more is a real difference
    assert card_sha256({"g": 0.0}, recipe="v2") != card_sha256({"g": 3e-12}, recipe="v2")
    # 9 significant digits above 1e-6
    assert card_sha256({"g": 0.5}, recipe="v2") != card_sha256({"g": 0.500000002}, recipe="v2")


def test_v2_refuses_non_finite():
    with pytest.raises(SpecError):
        card_sha256({"g": float("nan")}, recipe="v2")


def _entry(p, statement="a refit row"):
    ids = tuple(f"s{i}" for i in range(p.size))
    return Entry(synth.manifest("baseline: effort only", "target_probability", EV, statement=statement), ids, p)


def test_refit_entry_hash_absorbs_fit_noise():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.01, 0.99, 3000)
    q = p * (1 + rng.normal(0, 1e-12, p.size))
    a, b = _entry(p), _entry(q)
    assert entry_sha256(a, "v2", refit=True) == entry_sha256(b, "v2", refit=True)
    # recipe v1, and recipe v2 for a fixed row, hash exactly as before
    assert entry_sha256(a) == entry_sha256(a, "v1", refit=True) == entry_sha256(a, "v2", refit=False)


def test_refit_entry_hash_still_binds_the_manifest_and_real_changes():
    p = np.linspace(0.1, 0.9, 50)
    assert entry_sha256(_entry(p), "v2", True) != entry_sha256(_entry(p, "another statement"), "v2", True)
    q = p.copy()
    q[3] += 1e-5
    assert entry_sha256(_entry(p), "v2", True) != entry_sha256(_entry(q), "v2", True)


def test_is_refit_reads_the_fit_markers():
    p = np.linspace(0.1, 0.9, 10)
    assert is_refit(_entry(p, FIT_MARKER + " logistic on earlier earthquakes"))
    assert not is_refit(_entry(p, "pooled yes share per PGA bin"))


def test_hash_recipe_by_card_version():
    assert hash_recipe("v2") == "v1"
    assert hash_recipe("v3") == "v2"


@pytest.fixture(scope="module")
def ev():
    return synth.synthetic_event(600, 13, unlabelled_share=0.1)


def _task(ev, **kw):
    return Task(event_id=EV, target="ejecta seen at the site", label_source="synthetic v1",
                label_rule_sha256="1" * 64, site_list_sha256=site_list_sha256(ev["site_ids"]), track="forward",
                division="closed", tile_m=4000.0, clip=1e-6, seed=3, label_caveats="synthetic labels", n_boot=50, **kw)


def test_v2_card_has_no_recipe_field_and_v3_card_records_it(ev):
    ents = [Entry(synth.manifest("triggering model", "other_probability", EV), ev["site_ids"], ev["p_trigger"])]
    v2 = build_card(_task(ev), ents, synth.labels_of(ev))
    v3 = build_card(_task(ev), ents, synth.labels_of(ev), card_version="v3")
    assert "hash_recipe" not in v2
    assert v3["hash_recipe"]["card"] == "v2"
    assert v3["hash_recipe"]["refit_entries"] == "v2"


def _row_sha(card, name):
    return next(r["entry_sha256"] for r in card["entries"] if r["name"] == name)


def test_card_wires_the_refit_recipe_into_the_effort_and_shaking_only_rows(ev):
    """Review 1 (wave 3): the refit recipe must reach the card rows, not only spec.entry_sha256.

    The two versions of each refit row differ by 2e-12 around a 9-decimal rounding boundary. Recipe v1 (9 decimals)
    splits them; recipe v2 refit (6 decimals) does not. A card.py call that falls back to plain entry_sha256(x) fails
    here.
    """
    ids = ev["site_ids"]
    base = np.round(np.linspace(0.1, 0.9, len(ids)), 6) + 5e-10      # every value sits on a 9-decimal boundary
    lo, hi = base - 1e-12, base + 1e-12
    ents = [Entry(synth.manifest("triggering model", "other_probability", EV), ids, ev["p_trigger"])]
    eff_m = EntryManifest(model=E.NAME, output_kind="other_probability", statement="effort refit",
                          exposure={EV: E.EXPOSURE}, house=False, feature_allowlist="pre-event")
    shk_m = synth.manifest("shaking only", "target_probability", EV,
                           statement=FIT_MARKER + " logistic on earlier earthquakes")
    eff_name, shk_name = "baseline: effort only", "baseline: shaking only"
    shas = {}
    for version in ("v2", "v3"):
        for tag, p in (("lo", lo), ("hi", hi)):
            eff, shk = Entry(eff_m, ids, p), Entry(shk_m, ids, p[::-1].copy())
            card = build_card(_task(ev, earlier_label_sets=1, earlier_base_rate=0.3), ents, synth.labels_of(ev), effort_only=eff, shaking_only=shk,
                              card_version=version)
            shas[version, tag] = (_row_sha(card, eff_name), _row_sha(card, shk_name), eff, shk)
    for i in (0, 1):
        # v3: one hash for both versions, and it is the recipe v2 refit hash of the entry passed in
        assert shas["v3", "lo"][i] == shas["v3", "hi"][i]
        assert shas["v3", "lo"][i] == entry_sha256(shas["v3", "lo"][2 + i], "v2", True)
        # v2: the sealed recipe v1 still splits them, as before
        assert shas["v2", "lo"][i] != shas["v2", "hi"][i]
        assert shas["v2", "lo"][i] == entry_sha256(shas["v2", "lo"][2 + i])

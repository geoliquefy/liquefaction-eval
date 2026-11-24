"""C4 wiring (ruling 47): the v3 card prints the C4 outcome; the v2 path is unchanged. Synthetic data only."""
import numpy as np
import pytest

from scoreboard.evalkit import interval_policy as IP
from scoreboard.evalkit import pool, synth
from scoreboard.evalkit.card import DESCRIPTIVE, build_card
from scoreboard.evalkit.spec import Entry, Task, card_sha256, site_list_sha256

EV = "eq-test"


def make_task(ev, **kw):
    d = dict(event_id=EV, target="ejecta seen at the site", label_source="synthetic v1", label_rule_sha256="1" * 64,
             site_list_sha256=site_list_sha256(ev["site_ids"]), track="forward", division="closed", tile_m=2000.0,
             clip=1e-6, seed=3, label_caveats="synthetic labels", n_boot=100)
    d.update(kw)
    return Task(**d)


def entry(ev, name, kind, values, **kw):
    return Entry(synth.manifest(name, kind, EV, **kw), ev["site_ids"], values)


@pytest.fixture(scope="module")
def ev():
    return synth.synthetic_event(900, 21, unlabelled_share=0.1)


@pytest.fixture(scope="module")
def entries(ev):
    rng = np.random.default_rng(1)
    p1 = ev["fragility"] / (1 + np.exp(-(ev["intercept"] + ev["slope"] * (ev["s"] + rng.normal(0, 0.5, ev["s"].size)))))
    return [entry(ev, "target model", "target_probability", p1, house=True),
            entry(ev, "triggering model", "other_probability", ev["p_trigger"])]


def test_the_registered_outcome_is_printed_on_the_policy():
    assert IP.C4_OUTCOME["truth"] == "models fixed"
    assert IP.REGISTRATION == "docs/evals/C4-CERTIFICATION-REGISTRATION-2026-11-23.md"
    p = IP.policy("v3")
    assert p["record_line"] == IP.C4_OUTCOME["record_line"]
    if IP.C4_OUTCOME["per_earthquake"] is None and IP.C4_OUTCOME["pooled"] is None:
        assert p["record_line"] == "No method certifies; words only on the pooled line"


def test_default_card_is_v2_and_byte_identical(ev, entries):
    a = build_card(make_task(ev), entries, synth.labels_of(ev))
    b = build_card(make_task(ev), entries, synth.labels_of(ev), card_version="v2")
    assert card_sha256(a) == card_sha256(b)
    assert "card_version" not in a and "interval_policy" not in a
    assert any(c["word"] != DESCRIPTIVE for c in a["comparisons"])


def test_unknown_card_version_is_refused(ev, entries):
    with pytest.raises(IP.IntervalPolicyError, match="card_version"):
        build_card(make_task(ev), entries, synth.labels_of(ev), card_version="v4")


@pytest.mark.skipif(IP.C4_OUTCOME["per_earthquake"] is not None, reason="a per-earthquake method certified")
def test_v3_with_no_certified_method_withholds_every_within_word(ev, entries):
    c = build_card(make_task(ev, tile_m=4000.0), entries, synth.labels_of(ev), card_version="v3")
    assert c["card_version"] == "v3" and c["words"] == DESCRIPTIVE
    assert c["comparisons"] and all(x["word"] == DESCRIPTIVE for x in c["comparisons"])
    assert all("p" not in x and "lo" in x and "hi" in x for x in c["comparisons"])
    assert "not certified under models fixed" in c["interval_note"]
    assert IP.RESULTS in c["interval_note"]
    assert c["interval_policy"]["per_earthquake"].startswith("DESCRIPTIVE")
    assert c["tiles"] == {"rule": "fixed", "tile_m": 4000.0}
    assert IP.words_left(c) == []


def test_v3_refuses_any_tile_but_the_fixed_4_km_tile4_interval(ev, entries):
    """Review 1: the v3 note quotes tile4's measured coverage, so a 2 km, 8 km or variogram task is refused."""
    for kw in ({}, {"tile_m": 8000.0}, {"tile_rule": "variogram", "tile_m": 4000.0}):
        with pytest.raises(IP.IntervalPolicyError, match="tile4 interval only"):
            build_card(make_task(ev, **kw), entries, synth.labels_of(ev), card_version="v3")
    build_card(make_task(ev), entries, synth.labels_of(ev), card_version="v2")      # v2 keeps every tile rule


def _line(diffs, ses):
    return [{"event_id": f"e{i}", "entry_a": "A", "entry_b": "B", "metric": "auc", "diff": d, "se": s,
             "passes_gate": True} for i, (d, s) in enumerate(zip(diffs, ses))]


def test_pooled_v2_is_pool_pooled_lines_unchanged():
    lines = [_line([0.05, 0.07, 0.06], [0.01, 0.012, 0.011])]
    a = IP.pooled_lines(lines, registered=True, weighting="equal")
    b = pool.pooled_lines(lines, registered=True, weighting="equal")
    assert a == b


def test_pooled_v3_has_one_interval_and_keeps_the_word():
    lines = [_line([0.05, 0.07, 0.06], [0.01, 0.012, 0.011])]
    with pytest.raises(IP.IntervalPolicyError, match="no interval options"):
        IP.pooled_lines(lines, registered=True, card_version="v3", weighting="equal")
    out = IP.pooled_lines(lines, registered=True, card_version="v3")
    assert out[0]["word"] in ("HOLDS", "FRAGILE", "POOLED CANNOT TELL")
    assert out[0]["pooled"]["method"] == "hk"
    assert out[0]["certified"] is (IP.C4_OUTCOME["pooled"] is not None)
    assert "models fixed" in out[0]["interval_note"]


def test_the_c4_result_is_no_method_and_the_notes_carry_measured_coverage():
    assert IP.C4_OUTCOME["per_earthquake"] is None and IP.C4_OUTCOME["pooled"] is None
    assert IP.C4_OUTCOME["record_line"] == "No method certifies; words only on the pooled line"
    assert "measured coverage 0.87 to 0.94" in IP.per_earthquake_note()
    assert "measured coverage 0.945 to 0.977" in IP.pooled_note()

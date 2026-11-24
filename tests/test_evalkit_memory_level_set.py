"""evalkit memory level "label set" (scorecard v3.1, B8 review 2 finding R2-3). Synthetic only. No em dashes.

The sealed v3 build passed the TASK's label source string for every earlier earthquake, so a set B card whose
earlier earthquakes came from other sources of set B (bench, road25) looked like a same-source memory. Memory level
"label set" keeps each earlier earthquake's real source string, allows a mix within the named label set, and the
card checks the scored source against the task. The default stays the one-source rule of the sealed v3 card.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import geo_null as GN  # noqa: E402
from scoreboard.evalkit import site_memory as SM  # noqa: E402
from scoreboard.evalkit import synth  # noqa: E402
from scoreboard.evalkit.card import CardRefused, build_card  # noqa: E402
from scoreboard.evalkit.spec import Entry, Labels, Task, site_list_sha256  # noqa: E402

EV = "syn-later"
NAN = np.nan
SCORED = "set B: road25"
EARLIER_SRC = ("set B: bench", "set B: bench")


@pytest.fixture(scope="module")
def world():
    ev = synth.synthetic_event(600, 41, unlabelled_share=0.1)
    rng = np.random.default_rng(42)
    n = ev["s"].size
    labs = []
    for icpt in (-1.0, -0.3):
        y = (rng.random(n) < expit(icpt + ev["slope"] * ev["s"] + ev["field"])).astype(float)
        y[rng.random(n) < 0.3] = NAN
        labs.append(Labels(tuple(ev["site_ids"]), y, ev["x_m"] + rng.normal(0, 3, n), ev["y_m"] + rng.normal(0, 3, n)))
    labels = synth.labels_of(ev)
    task = Task(event_id=EV, target="ejecta seen at the site", label_source=SCORED, label_rule_sha256="1" * 64,
                site_list_sha256=site_list_sha256(ev["site_ids"]), track="forward", division="closed", tile_m=2000.0,
                clip=1e-6, seed=3, label_caveats="synthetic labels", n_boot=60, earlier_base_rate=0.3,
                earlier_label_sets=2)
    shaking = Entry(synth.manifest("shaking", "target_probability", EV), ev["site_ids"],
                    np.clip(0.2 + 0.1 * ev["field"], 0.01, 0.9))
    model = Entry(synth.manifest("m", "score", EV), ev["site_ids"], ev["s"] + rng.normal(0, 1.0, n))
    return ev, labs, labels, task, shaking, model


def earlier(labs, sources):
    return [SM.Earlier(f"syn-early-{k}", d, lab, s)
            for k, (d, lab, s) in enumerate(zip(("2020-01-01", "2021-01-01"), labs, sources))]


def test_default_level_is_the_sealed_v3_rule(world):
    ev, labs, labels, task, shaking, model = world
    sm = SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, (SCORED, SCORED)))
    assert "memory_level" not in sm.summary and sm.summary["label_source"] == SCORED
    assert sm.summary["label_persistence"] == SM.PERSISTENCE_NOTE
    with pytest.raises(SM.SiteMemoryRefused, match="one non-empty label source"):
        SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, ("set B: bench", "set B: road25")))
    with pytest.raises(GN.GeoNullRefused, match="one non-empty label source"):
        GN.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, ("set B: bench", "set B: road25")))
    with pytest.raises(SM.SiteMemoryRefused, match="only taken with label_set"):
        SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, (SCORED, SCORED)), scored_label_source=SCORED)


def test_set_level_keeps_real_strings_and_the_same_numbers(world):
    ev, labs, labels, task, shaking, model = world
    old = SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, (SCORED, SCORED)))
    new = SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, EARLIER_SRC), label_set="set B",
                   scored_label_source=SCORED)
    np.testing.assert_array_equal(old.entry.p, new.entry.p)          # the labels decide the numbers, not the strings
    s = new.summary
    assert s["memory_level"] == SM.MEMORY_LEVEL_SET and s["label_set"] == "set B"
    assert s["earlier_label_sources"] == {"syn-early-0": "set B: bench", "syn-early-1": "set B: bench"}
    assert s["label_source"] == "set B: bench" and s["memory_label_source"] == "set B: bench"
    assert s["scored_label_source"] == SCORED and s["same_label_source_as_scored"] is False
    assert s["label_persistence"] == SM.CROSS_SOURCE_NOTE and SM.PERSISTENCE_NOTE not in s["limits"]
    assert SM.CROSS_SOURCE_NOTE in new.entry.manifest.statement
    mixed = SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, ("set B: bench", "set B: road25")),
                     label_set="set B", scored_label_source="set B: aerial")
    assert mixed.summary["label_source"] == "set B: bench + set B: road25"
    assert mixed.summary["memory_label_source"] == "set B: road25"
    same = SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, (SCORED, SCORED)), label_set="set B",
                    scored_label_source=SCORED)
    assert same.summary["same_label_source_as_scored"] is True
    assert same.summary["label_persistence"] == SM.PERSISTENCE_NOTE


def test_set_level_refuses_strings_outside_the_set(world):
    ev, labs, labels, task, shaking, model = world
    for srcs, scored in (((SCORED, "set A: aerial"), SCORED), (EARLIER_SRC, "set A: aerial"),
                         ((SCORED, "set B: "), SCORED), (EARLIER_SRC, None)):
        with pytest.raises(SM.SiteMemoryRefused, match="label set"):
            SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, srcs), label_set="set B",
                     scored_label_source=scored)
        with pytest.raises(GN.GeoNullRefused, match="label set"):
            GN.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, srcs), label_set="set B",
                     scored_label_source=scored)


def test_card_checks_the_scored_source_at_set_level(world):
    ev, labs, labels, task, shaking, model = world
    sm = SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, EARLIER_SRC), label_set="set B",
                  scored_label_source=SCORED)
    gn = GN.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, EARLIER_SRC), label_set="set B",
                  scored_label_source=SCORED)
    card = build_card(task, [model], labels, shaking_only=shaking, site_memory=sm, geo_null=gn)
    assert card["site_memory"]["earlier_label_sources"]["syn-early-0"] == "set B: bench"
    assert card["geographic_null"]["label_source"] == "set B: bench"
    assert card["geographic_null"]["scored_label_source"] == SCORED
    other = Task(**{**task.to_dict(), "label_source": "set B: aerial"})
    with pytest.raises(CardRefused, match="scored label source"):
        build_card(other, [model], labels, shaking_only=shaking, site_memory=sm)
    with pytest.raises(CardRefused, match="scored label source"):
        build_card(other, [model], labels, shaking_only=shaking, geo_null=gn)
    old_gn = GN.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, (SCORED, SCORED)))
    np.testing.assert_array_equal(old_gn.entry.p, gn.entry.p)


def test_card_note_follows_the_memory_summary(world):
    ev, labs, labels, task, shaking, model = world
    cross = SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier(labs, EARLIER_SRC), label_set="set B",
                     scored_label_source=SCORED)
    card = build_card(task, [model], labels, shaking_only=shaking, site_memory=cross)
    assert any(SM.CROSS_SOURCE_NOTE in n for n in card["notes"])
    assert not any(SM.PERSISTENCE_NOTE in n for n in card["notes"])

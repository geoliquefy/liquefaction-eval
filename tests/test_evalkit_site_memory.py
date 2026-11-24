"""evalkit site memory (own-work list B3, eval M4): rule, guards, loader and card slot. Synthetic only."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import site_memory as SM  # noqa: E402
from scoreboard.evalkit import synth  # noqa: E402
from scoreboard.evalkit.card import CardRefused, build_card, scan_aggregates  # noqa: E402
from scoreboard.evalkit.spec import Entry, Labels, SpecError, Task, entry_sha256, site_list_sha256  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "site_memory_history_synthetic.csv"
FIXTURE_DATES = {"syn-eq-a": "2020-01-10", "syn-eq-b": "2020-06-01", "syn-eq-c": "2021-02-01",
                 "syn-eq-d": "2022-03-01"}
NAN = np.nan
SRC = "synthetic v1"          # the label source of every synthetic history and of the card task


def lab(ids, label, x, y):
    return Labels(tuple(ids), np.array(label, float), np.array(x, float), np.array(y, float))


# Step 2: the match -----------------------------------------------------------------------------------------------

def test_recall_same_id_then_nearest_within_20_m_then_none():
    earlier = lab(["a", "b", "c", "d"], [1, 0, NAN, 1], [0, 100, 200, 300], [0, 0, 0, 0])
    sites = SM.Sites(("a", "x1", "x2", "c", "x3"), np.array([500.0, 100.0 + 20.0, 300.0 - 20.01, 205.0, 1000.0]),
                     np.zeros(5))
    mem, how = SM.recall(sites, earlier)
    # a: same id wins even far away; x1: exactly 20 m from b counts; x2: 20.01 m from d does not;
    # c: its own row is unlabelled, so the nearest labelled site within 20 m is used (none here); x3: none
    assert how.tolist() == [SM.SAME_ID, SM.NEAREST, SM.NONE, SM.NONE, SM.NONE]
    assert mem[0] == 1 and mem[1] == 0 and np.isnan(mem[2:]).all()


def test_recall_unlabelled_same_id_falls_back_and_ties_go_to_the_smaller_id():
    earlier = lab(["c", "q", "p"], [NAN, 1, 0], [0, 10, -10], [0, 0, 0])
    sites = SM.Sites(("c",), np.array([0.0]), np.array([0.0]))
    mem, how = SM.recall(sites, earlier)
    assert how[0] == SM.NEAREST and mem[0] == 0          # p and q both 10 m away: p is the smaller id
    with pytest.raises(SM.SiteMemoryRefused):
        SM.recall(sites, earlier, radius_m=0.0)


# Steps 3 and 4: the two versions --------------------------------------------------------------------------------

def test_one_earlier_earthquake_gives_a_ranking_only_score():
    prev = lab(["a", "b", "c", "d"], [1, 0, 1, 1], [0, 100, 200, 300], [0, 0, 0, 0])
    sites = SM.Sites(("a", "b", "z"), np.array([0.0, 100.0, 5000.0]), np.zeros(3))
    sm = SM.build("eq-2", "2021-01-01", sites, [SM.Earlier("eq-1", "2020-01-01", prev, SRC)])
    assert sm.entry.manifest.output_kind == "score" and sm.summary["version"] == "ranking only"
    assert sm.entry.p.tolist() == [1.0, 0.0, 0.75]          # no memory: the memory earthquake's yes share
    c = sm.summary["counts"]
    assert (c["n_sites"], c["n_same_id"], c["n_no_memory"], c["n_memory_yes"], c["n_memory_no"]) == (3, 2, 1, 1, 1)
    assert sm.entry.manifest.exposure == {"eq-2": "seen_place"}
    assert sm.entry.manifest.feature_allowlist == SM.ALLOWLIST_NAME


def test_two_earlier_earthquakes_give_rates_from_the_earlier_pair_only():
    # eq-1 (back) labels: s0..s5 yes, s6..s11 no; s12..s15 absent. eq-2 (memory) labels below.
    back = lab([f"s{k}" for k in range(12)], [1] * 6 + [0] * 6, np.arange(12) * 100.0, np.zeros(12))
    y2 = [1, 1, 1, 1, 0, 0] + [1, 0, 0, 0, 0, 0] + [1, 0, 0, 0]
    prev = lab([f"s{k}" for k in range(16)], y2, np.arange(16) * 100.0, np.zeros(16))
    sites = SM.Sites(("s0", "s7", "s15", "new"), np.array([0.0, 700.0, 1500.0, 9000.0]), np.zeros(4))
    sm = SM.build("eq-3", "2022-01-01", sites, [SM.Earlier("eq-2", "2021-01-01", prev, SRC),
                                                 SM.Earlier("eq-1", "2020-01-01", back, SRC)], min_cell=4)
    r = sm.summary["rates"]
    assert (r["memory_yes"]["n"], r["memory_yes"]["k_yes"]) == (6, 4)
    assert (r["memory_no"]["n"], r["memory_no"]["k_yes"]) == (6, 1)
    assert (r["no_memory"]["n"], r["no_memory"]["k_yes"]) == (4, 1)
    jeff = {k: (v["k_yes"] + 0.5) / (v["n"] + 1) for k, v in r.items()}
    # s0 memory yes, s7 memory no, s15 memory no (its eq-2 label is 0), new has no memory
    assert np.allclose(sm.entry.p, [jeff["memory_yes"], jeff["memory_no"], jeff["memory_no"], jeff["no_memory"]])
    assert sm.entry.manifest.output_kind == "target_probability"
    assert sm.summary["calibration_event"] == "eq-1" and sm.summary["memory_event"] == "eq-2"
    assert 0.0 < sm.entry.p.min() and sm.entry.p.max() < 1.0
    small = SM.build("eq-3", "2022-01-01", sites, [SM.Earlier("eq-2", "2021-01-01", prev, SRC),
                                                    SM.Earlier("eq-1", "2020-01-01", back, SRC)], min_cell=5)
    assert small.summary["rates"]["no_memory"]["p"] == pytest.approx(6.5 / 17)   # 4 < 5: Jeffreys, all 16
    assert small.summary["rates"]["memory_yes"]["p"] == pytest.approx(jeff["memory_yes"])


def test_three_earlier_earthquakes_calibrate_on_the_two_latest_whatever_the_order():
    """Memory = the latest earlier earthquake; calibration = the second latest, read through its memory of it.

    The oldest earthquake has inverted labels, so calibrating on it (or on the wrong pair) changes every cell.
    """
    ids = [f"s{k}" for k in range(12)]
    x, y0 = np.arange(12) * 100.0, np.zeros(12)
    oldest = lab(ids, [0] * 6 + [1] * 6, x, y0)
    middle = lab(ids, [1] * 6 + [0] * 6, x, y0)
    latest = lab(ids, [1, 1, 1, 1, 0, 0] + [1, 0, 0, 0, 0, 0], x, y0)
    e0, e1, e2 = (SM.Earlier("eq-0", "2019-01-01", oldest, SRC), SM.Earlier("eq-1", "2020-01-01", middle, SRC),
                  SM.Earlier("eq-2", "2021-01-01", latest, SRC))
    sites = SM.Sites(("s0", "s4", "s6"), np.array([0.0, 400.0, 600.0]), np.zeros(3))
    shas = set()
    for order in itertools.permutations([e0, e1, e2]):
        sm = SM.build("eq-3", "2022-01-01", sites, list(order), min_cell=1)
        assert sm.summary["memory_event"] == "eq-2" and sm.summary["calibration_event"] == "eq-1"
        r = sm.summary["rates"]
        assert (r["memory_yes"]["n"], r["memory_yes"]["k_yes"]) == (6, 4)      # eq-0 as calibration: (6, 1)
        assert (r["memory_no"]["n"], r["memory_no"]["k_yes"]) == (6, 1)        # eq-0 as calibration: (6, 4)
        assert np.allclose(sm.entry.p, [4.5 / 7, 1.5 / 7, 4.5 / 7])           # s0 yes, s4 no, s6 yes in eq-2
        assert sm.summary["n_earlier"] == 3
        shas.add(entry_sha256(sm.entry))
    assert len(shas) == 1


def test_memory_is_the_latest_earlier_earthquake_whatever_the_list_order():
    a = lab(["s"], [1], [0], [0])
    b = lab(["s"], [0], [0], [0])
    sites = SM.Sites(("s",), np.zeros(1), np.zeros(1))
    e1, e2 = SM.Earlier("eq-1", "2020-01-01", a, SRC), SM.Earlier("eq-2", "2021-01-01", b, SRC)
    one = SM.build("eq-3", "2022-01-01", sites, [e2, e1])
    two = SM.build("eq-3", "2022-01-01", sites, [e1, e2])
    assert one.summary["memory_event"] == two.summary["memory_event"] == "eq-2"
    assert entry_sha256(one.entry) == entry_sha256(two.entry)


# Guards: no scored labels, time order ------------------------------------------------------------------------------

def test_the_builder_refuses_scored_labels_and_bad_history():
    prev = lab(["a"], [1], [0], [0])
    sites = SM.Sites(("a",), np.zeros(1), np.zeros(1))
    e1 = SM.Earlier("eq-1", "2020-01-01", prev, SRC)
    with pytest.raises(SM.SiteMemoryRefused, match="scored labels"):
        SM.build("eq-2", "2021-01-01", prev, [e1])
    with pytest.raises(SM.SiteMemoryRefused, match="strictly before"):
        SM.build("eq-2", "2020-01-01", sites, [e1])
    with pytest.raises(SM.SiteMemoryRefused, match="distinct"):
        SM.build("eq-1", "2021-01-01", sites, [e1])
    with pytest.raises(SM.SiteMemoryRefused, match="share a date"):
        SM.build("eq-3", "2021-01-01", sites, [e1, SM.Earlier("eq-0", "2020-01-01", prev, SRC)])
    with pytest.raises(SM.SiteMemoryRefused, match="at least one"):
        SM.build("eq-2", "2021-01-01", sites, [])
    with pytest.raises(SM.SiteMemoryRefused, match="ISO"):
        SM.build("eq-2", "1/1/2021", sites, [e1])
    with pytest.raises(SM.SiteMemoryRefused, match="no labelled site"):
        SM.build("eq-2", "2021-01-01", sites, [SM.Earlier("eq-1", "2020-01-01", lab(["a"], [NAN], [0], [0]),
                                                           SRC)])
    with pytest.raises(SM.SiteMemoryRefused, match="label source"):
        SM.build("eq-3", "2022-01-01", sites, [e1, SM.Earlier("eq-2", "2021-01-01", prev, "another source")])


# Loader -----------------------------------------------------------------------------------------------------------

def test_loader_drops_scored_labels_and_later_earthquakes():
    sites, earlier = SM.load_history(FIXTURE, FIXTURE_DATES, "syn-eq-c", SRC)
    assert [e.event_id for e in earlier] == ["syn-eq-a", "syn-eq-b"]          # syn-eq-d is later: skipped
    assert isinstance(sites, SM.Sites) and not hasattr(sites, "label")
    assert sites.site_ids == ("syn:c01", "syn:c02", "syn:c03", "syn:c04")
    sm = SM.build("syn-eq-c", FIXTURE_DATES["syn-eq-c"], sites, earlier)
    assert sm.summary["memory_event"] == "syn-eq-b" and sm.summary["counts"]["n_sites"] == 4


def test_loader_never_parses_a_scored_label(tmp_path):
    """Scored rows with a label that would fail the parser still load: the scored label cell is never read."""
    text = FIXTURE.read_text(encoding="utf-8").splitlines()
    out = [r if not r.startswith("syn-eq-c,") else ",".join(r.split(",")[:2] + ["NOT-A-LABEL"] + r.split(",")[3:])
           for r in text]
    p = tmp_path / "h.csv"
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    s1, e1 = SM.load_history(FIXTURE, FIXTURE_DATES, "syn-eq-c", SRC)
    s2, e2 = SM.load_history(p, FIXTURE_DATES, "syn-eq-c", SRC)
    a = SM.build("syn-eq-c", "2021-02-01", s1, e1)
    b = SM.build("syn-eq-c", "2021-02-01", s2, e2)
    assert entry_sha256(a.entry) == entry_sha256(b.entry)


def test_loader_refusals(tmp_path):
    with pytest.raises(SM.SiteMemoryRefused, match="no date"):
        SM.load_history(FIXTURE, {k: v for k, v in FIXTURE_DATES.items() if k != "syn-eq-a"}, "syn-eq-c",
                        SRC)
    with pytest.raises(SM.SiteMemoryRefused, match="scored"):
        SM.load_history(FIXTURE, {**FIXTURE_DATES, "syn-eq-z": "2023-01-01"}, "syn-eq-z", SRC)
    bad = tmp_path / "bad.csv"
    bad.write_text("event,site_id,label,x_m,y_m\n", encoding="utf-8")
    with pytest.raises(SpecError, match="header"):
        SM.load_history(bad, FIXTURE_DATES, "syn-eq-c", SRC)
    dup = tmp_path / "dup.csv"
    dup.write_text(FIXTURE.read_text(encoding="utf-8") + "syn-eq-a,syn:a01,1,0,0\n", encoding="utf-8")
    with pytest.raises(SpecError, match="duplicate"):
        SM.load_history(dup, FIXTURE_DATES, "syn-eq-c", SRC)
    later_dup = tmp_path / "later_dup.csv"                                     # syn-eq-d is later than syn-eq-c
    later_dup.write_text(FIXTURE.read_text(encoding="utf-8") + "syn-eq-d,syn:d02,0,100,0\n", encoding="utf-8")
    with pytest.raises(SpecError, match="duplicate"):
        SM.load_history(later_dup, FIXTURE_DATES, "syn-eq-c", SRC)


# Card slot (R10) --------------------------------------------------------------------------------------------------

EV = "syn-later"


@pytest.fixture(scope="module")
def history():
    """A scored synthetic earthquake and two earlier ones that share its susceptibility and spatial field."""
    ev = synth.synthetic_event(700, 31, unlabelled_share=0.1)
    rng = np.random.default_rng(32)
    n = ev["s"].size
    earlier = []
    for k, (date, icpt) in enumerate((("2020-01-01", -1.0), ("2021-01-01", -0.3))):
        y = (rng.random(n) < expit(icpt + ev["slope"] * ev["s"] + ev["field"])).astype(float)
        y[rng.random(n) < 0.3] = NAN                                   # not every site labelled every time
        jitter = rng.normal(0.0, 4.0, (n, 2))                          # a few metres: within the 20 m radius
        ids = tuple(f"syn:e{k}-{j:05d}" if j % 3 == 0 else sid for j, sid in enumerate(ev["site_ids"]))
        earlier.append(SM.Earlier(f"syn-early-{k}", date, Labels(ids, y, ev["x_m"] + jitter[:, 0],
                                                                  ev["y_m"] + jitter[:, 1]), SRC))
    labels = synth.labels_of(ev)
    task = Task(event_id=EV, target="ejecta seen at the site", label_source="synthetic v1",
                label_rule_sha256="1" * 64, site_list_sha256=site_list_sha256(ev["site_ids"]), track="forward",
                division="closed", tile_m=2000.0, clip=1e-6, seed=3, label_caveats="synthetic labels", n_boot=60,
                earlier_base_rate=0.3, earlier_label_sets=2)
    shaking = Entry(synth.manifest("shaking", "target_probability", EV), ev["site_ids"],
                    np.clip(0.2 + 0.1 * ev["field"], 0.01, 0.9))
    model = Entry(synth.manifest("m", "score", EV), ev["site_ids"], ev["s"] + rng.normal(0, 1.0, n))
    return ev, earlier, labels, task, shaking, model


def test_card_takes_the_site_memory_slot(history):
    ev, earlier, labels, task, shaking, model = history
    sm = SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier)
    assert sm.summary["counts"]["n_no_memory"] < 0.5 * sm.summary["counts"]["n_sites"]
    card = build_card(task, [model], labels, shaking_only=shaking, site_memory=sm)
    meta = {e["name"]: e for e in card["entries"]}
    assert meta[SM.NAME]["role"] == "baseline" and meta[SM.NAME]["exposure"].startswith("SEEN PLACE")
    assert card["site_memory"]["memory_event"] == "syn-early-1" and card["site_memory"]["limits"]
    m = card["metrics"][SM.NAME]
    assert "log_loss" in m and m["auc"]["value"] > 0.55                # shared susceptibility: memory carries signal
    assert not any(c["family"] and {c["a"], c["b"]} <= {SM.NAME, "baseline: coin", "baseline: shaking only",
                                                       "baseline: earlier-earthquake rate"}
                   for c in card["comparisons"])                       # no pair of two baselines (R10)
    scan_aggregates(card, labels.site_ids)
    ranking = SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier[1:])
    card2 = build_card(task, [model], labels, shaking_only=shaking, site_memory=ranking)
    assert "R3" in card2["metrics"][SM.NAME]["probability_metrics"]


def test_card_says_label_persistence_and_the_radius_is_a_new_choice(history):
    """Correction B3 (founder check 2026-11-23): on the same label source the row also measures label persistence
    (aerial specificity 0.47 in Feb 2011), and 20 m is ruling 16's twin-test and display distance, so its use
    here is a new choice for the v3 registration. The card must say both, for both versions."""
    ev, earlier, labels, task, shaking, model = history
    assert "label persistence" in SM.PERSISTENCE_NOTE and "0.47" in SM.PERSISTENCE_NOTE
    assert "Feb 2011" in SM.PERSISTENCE_NOTE and "LABEL-ERROR-RATES-2026-11-19" in SM.PERSISTENCE_NOTE
    for part in ("ruling 16", "twin test", "display", "new choice", "not settled", "v3 registration"):
        assert part in SM.RADIUS_NOTE, part
    for ev_set in (earlier, earlier[1:]):                                 # probability and ranking-only versions
        sm = SM.build(EV, "2022-01-01", SM.sites_of(labels), ev_set)
        card = build_card(task, [model], labels, shaking_only=shaking, site_memory=sm)
        st = {e["name"]: e for e in card["entries"]}[SM.NAME]["statement"]
        assert SM.PERSISTENCE_NOTE in st and "new choice" in st and "v3 registration" in st
        assert SM.PERSISTENCE_NOTE in card["site_memory"]["limits"] and SM.RADIUS_NOTE in card["site_memory"]["limits"]
        assert card["site_memory"]["label_persistence"] == SM.PERSISTENCE_NOTE
        assert card["site_memory"]["radius_status"] == SM.RADIUS_NOTE
        assert any(SM.PERSISTENCE_NOTE in n for n in card["notes"])
        assert any(SM.RADIUS_NOTE in n for n in card["notes"])
    plain = build_card(task, [model], labels, shaking_only=shaking)
    assert not any("site memory" in n for n in plain["notes"])            # no site-memory row, no site-memory note


def test_card_refuses_imitations_and_misuse(history):
    ev, earlier, labels, task, shaking, model = history
    sm = SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier)
    with pytest.raises(CardRefused, match="SiteMemory"):
        build_card(task, [model], labels, shaking_only=shaking, site_memory=sm.entry)
    other = SM.build("syn-other", "2022-01-01", SM.sites_of(labels), earlier)
    with pytest.raises(CardRefused, match="another earthquake"):
        build_card(task, [model], labels, shaking_only=shaking, site_memory=other)
    fake = SM.SiteMemory(Entry(synth.manifest("baseline: site memory", "target_probability", EV), ev["site_ids"],
                               sm.entry.p), sm.summary)
    with pytest.raises(CardRefused, match="provenance"):
        build_card(task, [model], labels, shaking_only=shaking, site_memory=fake)
    # a hand-built object with every checked field right (name, allowlist, kind, exposure, summary) is still refused
    forged = SM.SiteMemory(Entry(sm.entry.manifest, sm.entry.site_ids, sm.entry.p), dict(sm.summary))
    assert forged.entry.manifest.feature_allowlist == SM.ALLOWLIST_NAME
    assert not SM.kit_built(forged) and SM.kit_built(sm)
    with pytest.raises(CardRefused, match="provenance"):
        build_card(task, [model], labels, shaking_only=shaking, site_memory=forged)
    edited = SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier)
    edited.summary["label_source"] = task.label_source + " (edited)"
    with pytest.raises(CardRefused, match="provenance"):
        build_card(task, [model], labels, shaking_only=shaking, site_memory=edited)
    wrong_source = [SM.Earlier(e.event_id, e.date, e.labels, "another source") for e in earlier]
    with pytest.raises(CardRefused, match="label source"):
        build_card(task, [model], labels, shaking_only=shaking,
                   site_memory=SM.build(EV, "2022-01-01", SM.sites_of(labels), wrong_source))
    one_earlier = Task(**{**task.to_dict(), "earlier_label_sets": 1})
    with pytest.raises(CardRefused, match="uses 2 earlier earthquakes"):
        build_card(one_earlier, [model], labels, shaking_only=shaking, site_memory=sm)
    ranking = SM.build(EV, "2022-01-01", SM.sites_of(labels), earlier[1:])
    build_card(one_earlier, [model], labels, shaking_only=shaking, site_memory=ranking)       # ranking only: fine
    no_earlier = Task(**{**task.to_dict(), "earlier_base_rate": None, "earlier_label_sets": 0})
    with pytest.raises(CardRefused, match="no earlier earthquake"):
        build_card(no_earlier, [model], labels, site_memory=sm)

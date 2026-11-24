"""evalkit card: each rule the card enforces, and a test that tries to break it. Synthetic only."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import card as C  # noqa: E402
from scoreboard.evalkit import resample, synth  # noqa: E402
from scoreboard.evalkit.card import CardRefused, build_card, scan_aggregates, write_card  # noqa: E402
from scoreboard.evalkit.pool import card_item  # noqa: E402
from scoreboard.evalkit.spec import Entry, SpecError, Task, site_list_sha256  # noqa: E402

EV = "eq-test"
CONV = "c" * 64


def make_task(ev, **kw):
    d = dict(event_id=EV, target="ejecta seen at the site", label_source="synthetic v1", label_rule_sha256="1" * 64,
             site_list_sha256=site_list_sha256(ev["site_ids"]), track="forward", division="closed", tile_m=2000.0,
             clip=1e-6, seed=3, label_caveats="synthetic labels", n_boot=100)
    d.update(kw)
    return Task(**d)


def entry(ev, name, kind, values, **kw):
    return Entry(synth.manifest(name, kind, EV, **kw), ev["site_ids"], values)


def weak(ev, seed, sd=1.0):
    rng = np.random.default_rng(seed)
    s = ev["s"] + rng.normal(0.0, sd, ev["s"].size)
    return ev["fragility"] * 1 / (1 + np.exp(-(ev["intercept"] + ev["slope"] * s)))


@pytest.fixture(scope="module")
def ev():
    return synth.synthetic_event(900, 21, unlabelled_share=0.1)


@pytest.fixture(scope="module")
def mixed_card(ev):
    """Four kinds of entry on one card."""
    entries = [entry(ev, "target model", "target_probability", weak(ev, 1, 0.5), house=True),
               entry(ev, "triggering model", "other_probability", ev["p_trigger"]),
               entry(ev, "index", "score", ev["s"], exposure="seen_inputs"),
               entry(ev, "converted triggering", "other_probability", 0.6 * ev["p_trigger"], conversion_id=CONV)]
    return build_card(make_task(ev), entries, synth.labels_of(ev))


def metric_block(card, name):
    return card["metrics"][name]


# R3 ------------------------------------------------------------------------------------------------------------

def test_only_target_probabilities_get_probability_metrics(mixed_card):
    for name in ("triggering model", "index"):
        m = metric_block(mixed_card, name)
        assert set(m) == {"auc", "auc_by_tile_m", "probability_metrics"} and "R3" in m["probability_metrics"]
    for name in ("target model", "converted triggering", "baseline: coin"):
        assert "log_loss" in metric_block(mixed_card, name) and "calibration" in metric_block(mixed_card, name)
    prob_members = {n for c in mixed_card["comparisons"] if c["family"] == "probability" for n in (c["a"], c["b"])}
    assert not prob_members & {"triggering model", "index"}
    meta = {e["name"]: e for e in mixed_card["entries"]}
    assert meta["converted triggering"]["conversion_id"] == CONV and meta["converted triggering"]["probability_metrics"]


def test_kind_cannot_be_faked(ev):
    with pytest.raises(SpecError):
        entry(ev, "x", "target_probability", ev["p_target"], conversion_id=CONV)
    with pytest.raises(SpecError):
        entry(ev, "x", "probability", ev["p_target"])
    with pytest.raises(SpecError, match="\\[0, 1\\]"):
        entry(ev, "x", "target_probability", ev["s"])


# R10 -----------------------------------------------------------------------------------------------------------

def test_baselines_are_entries_on_every_card(ev, mixed_card):
    names = [e["name"] for e in mixed_card["entries"]]
    assert "baseline: coin" in names and "R10" in " ".join(mixed_card["notes"])
    shaking = entry(ev, "shaking", "target_probability", np.clip(0.2 + 0.1 * ev["field"], 0.01, 0.9))
    c = build_card(make_task(ev, earlier_base_rate=0.25, earlier_label_sets=1), [entry(ev, "m", "score", ev["s"])],
                   synth.labels_of(ev), shaking_only=shaking)
    assert [e["name"] for e in c["entries"] if e["role"] == "baseline"] == \
        ["baseline: coin", "baseline: earlier-earthquake rate", "baseline: shaking only"]
    assert all("log_loss" in c["metrics"][n] for n in ("baseline: coin", "baseline: shaking only"))
    assert not any("R10" in n for n in c["notes"])
    with pytest.raises(CardRefused, match="baselines"):
        build_card(make_task(ev), [entry(ev, "Baseline: mine", "score", ev["s"])], synth.labels_of(ev))
    with pytest.raises(CardRefused, match="R10"):
        build_card(make_task(ev), [entry(ev, "m", "score", ev["s"])], synth.labels_of(ev),
                   shaking_only=entry(ev, "s", "score", ev["s"]))
    assert not any(c["a"].startswith("baseline:") and c["b"].startswith("baseline:") for c in c["comparisons"])


def test_a_required_baseline_that_is_missing_refuses_the_card(ev):
    """R10: with an earlier earthquake, both earlier baselines are required; a missing one is a scorer fault."""
    shaking = entry(ev, "shaking", "target_probability", np.clip(0.2 + 0.1 * ev["field"], 0.01, 0.9))
    one = [entry(ev, "m", "score", ev["s"])]
    with pytest.raises(CardRefused, match="scorer fault: baseline not built"):
        build_card(make_task(ev, earlier_base_rate=0.25, earlier_label_sets=2), one, synth.labels_of(ev))
    with pytest.raises(CardRefused, match="scorer fault: baseline not built"):
        build_card(make_task(ev, earlier_label_sets=1), one, synth.labels_of(ev), shaking_only=shaking)
    with pytest.raises(CardRefused, match="names no earlier earthquake"):
        build_card(make_task(ev, earlier_base_rate=0.25), one, synth.labels_of(ev))
    none = build_card(make_task(ev), one, synth.labels_of(ev))
    assert any("NOT AVAILABLE" in n for n in none["notes"])


# R11 intervals and count gate ----------------------------------------------------------------------------------

def _interval_leaves(obj, path=""):
    """Every numeric metric leaf outside an interval dict, apart from the allowed descriptive fields."""
    allowed = {"clip", "share_exact_0", "share_exact_1", "reliability"}
    if isinstance(obj, dict):
        if "value" in obj:
            assert {"lo", "hi", "n_draws"} <= set(obj), path
            return []
        return [x for k, v in obj.items() if k not in allowed for x in _interval_leaves(v, f"{path}.{k}")]
    return [path] if isinstance(obj, (int, float)) and not isinstance(obj, bool) else []


def test_every_metric_carries_an_interval(mixed_card):
    assert _interval_leaves(mixed_card["metrics"]) == []
    for c in mixed_card["comparisons"]:
        assert {"lo", "hi", "mdd", "se", "p"} <= set(c)


def test_underpowered_card_prints_counts_only(ev):
    small = synth.synthetic_event(150, 5)
    e = [entry(small, "m", "target_probability", small["p_target"])]
    c = build_card(make_task(small), e, synth.labels_of(small))
    assert c["status"] == "UNDERPOWERED" and not c["count_gate"]["passes"]
    assert "metrics" not in c and c["comparisons"] == [] and "observed_share" not in c
    assert all(b["name"] is None and "not a tie" in b["reasons"][0] for b in c["best"].values())
    assert set(c["count_gate"]) >= {"n_labelled", "n_yes", "n_no", "n_tiles"}
    few_tiles = build_card(make_task(ev, tile_m=8000.0), [entry(ev, "m", "score", ev["s"])], synth.labels_of(ev))
    assert few_tiles["status"] == "UNDERPOWERED" and few_tiles["count_gate"]["n_tiles"] < 20
    rare = synth.synthetic_event(900, 6, fragility=0.03)
    c_rare = build_card(make_task(rare), [entry(rare, "m", "score", rare["s"])], synth.labels_of(rare))
    assert c_rare["status"] == "UNDERPOWERED" and c_rare["count_gate"]["n_yes"] < 30
    lowered = build_card(make_task(rare, min_per_class=5), [entry(rare, "m", "score", rare["s"])],
                         synth.labels_of(rare))
    assert lowered["status"] == "SCORED"                                     # the gate is configurable


def test_words_are_holm_corrected_within_each_family(mixed_card):
    """R11 d: two Holm families per card, ranking (paired AUC) and probability (paired log loss)."""
    comps = mixed_card["comparisons"]
    for fam in C.FAMILIES:
        fc = [c for c in comps if c["family"] == fam]
        assert fc and all(c["family_size"] == len(fc) for c in fc)
        assert mixed_card["holm_family_sizes"][fam] == len(fc)
        rejected = resample.holm([c["p"] for c in fc], C.ALPHA)
        assert [c["word"] == "SEPARATES" for c in fc] == rejected
    for c in comps:
        assert (c["favours"] is None) == (c["word"] != "SEPARATES")
        assert (c["a_is"] in ("BETTER", "WORSE")) == (c["word"] == "SEPARATES")
        assert c["word"] in ("SEPARATES", "EQUIVALENT", "CANNOT TELL")


def test_two_families_are_not_one():
    """A p value that Holm rejects inside a small family must not be lost to a large one (v0 used one family)."""
    pairs = [("ranking", "a", "b")] + [("probability", f"x{k}", f"y{k}") for k in range(9)]
    pts = {n: {"auc": 0.0, "log_loss@1e-06": 0.0} for pr in pairs for n in pr[1:]}
    rng = np.random.default_rng(0)
    draws = {}
    for fam, a, b in pairs:
        shift = 0.03 if fam == "ranking" else 0.0
        draws[a] = {"auc": rng.normal(shift, 0.01, 400), "log_loss@1e-06": rng.normal(0, 0.01, 400)}
        draws[b] = {"auc": np.zeros(400), "log_loss@1e-06": rng.normal(0, 0.01, 400)}
    pts["a"]["auc"] = 0.03
    table = C._word_table(pairs, pts, draws, "log_loss@1e-06")
    assert table[0]["p"] < 0.05 and table[0]["family_size"] == 1 and table[0]["word"] == "SEPARATES"
    assert table[0]["a_is"] == "BETTER"
    assert all(t["family_size"] == 9 for t in table[1:])


def test_equivalent_needs_the_90_interval_inside_the_margin():
    """C2: EQUIVALENT only when not SEPARATES and the 90% paired interval lies inside the R27 margin (0.05)."""
    rng = np.random.default_rng(1)
    pairs = [("ranking", "a", "b"), ("ranking", "c", "d")]
    pts = {n: {"auc": 0.7} for n in "abcd"}
    draws = {"a": {"auc": rng.normal(0.70, 0.004, 500)}, "b": {"auc": rng.normal(0.70, 0.004, 500)},
             "c": {"auc": rng.normal(0.70, 0.04, 500)}, "d": {"auc": rng.normal(0.70, 0.04, 500)}}
    t = C._word_table(pairs, pts, draws, "log_loss@1e-06")
    assert t[0]["word"] == "EQUIVALENT" and -0.05 < t[0]["lo90"] < t[0]["hi90"] < 0.05
    assert t[1]["word"] == "CANNOT TELL"


def test_tile_size_comes_from_the_variogram_before_scoring(ev):
    """R11 c: with tile_rule "variogram" the card recomputes the choice from labels and baselines, refuses a task
    whose tile differs, and records the range and the variogram hash; no entry enters the choice."""
    labels = synth.labels_of(ev)
    base = make_task(ev, tile_rule="variogram")
    choice = C.tile_choice(base, labels)
    assert choice["residual"].startswith("label minus 0.5") and len(choice["variogram_sha256"]) == 64
    assert choice["tile_m"] in resample.TILE_CANDIDATES_M
    one = [entry(ev, "m", "score", ev["s"])]
    card = build_card(make_task(ev, tile_rule="variogram", tile_m=choice["tile_m"]), one, labels)
    assert card["tiles"]["tile_m"] == choice["tile_m"] and card["tiles"]["variogram_sha256"] == choice["variogram_sha256"]
    wrong = 8000.0 if choice["tile_m"] != 8000.0 else 500.0
    with pytest.raises(CardRefused, match="R11 c"):
        build_card(make_task(ev, tile_rule="variogram", tile_m=wrong), one, labels)
    shaking = entry(ev, "shaking", "target_probability", np.clip(0.2 + 0.1 * ev["field"], 0.01, 0.9))
    with_base = C.tile_choice(make_task(ev, earlier_base_rate=0.25, earlier_label_sets=1), labels, shaking)
    assert with_base["residual"] == "label minus the shaking-only p"
    rate = C.tile_choice(make_task(ev, earlier_base_rate=0.25, earlier_label_sets=1), labels)
    assert rate["residual"] == "label minus the earlier-earthquake rate"


def test_sensitivity_flags_match_the_words_beside(mixed_card):
    for c in mixed_card["comparisons"]:
        tile_moves = len(set(c["words_by_tile_m"].values()) | {c["word"]}) > 1
        assert tile_moves <= ("TILE-SENSITIVE" in c["flags"])
        if c["family"] == "probability":
            assert set(c["words_by_clip"]) == {"1e-06", "0.0001", "0.001", "0.01"}
            clip_moves = len(set(c["words_by_clip"].values()) | {c["word"]}) > 1
            assert clip_moves <= ("CLIP-SENSITIVE" in c["flags"])
        else:
            assert "words_by_clip" not in c and "CLIP-SENSITIVE" not in c["flags"]


# R5, R6, R16: who can be named ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def strong_case():
    ev = synth.synthetic_event(2500, 31)
    return ev, synth.labels_of(ev)


def _strong_card(strong_case, track="forward", **kw):
    ev, labels = strong_case
    entries = [entry(ev, "oracle", "target_probability", ev["p_target"], **kw),
               entry(ev, "weak", "target_probability", weak(ev, 2, 1.5))]
    return build_card(make_task(ev, track=track, n_boot=150), entries, labels)


def test_forward_card_names_a_best_entry_only_when_separated(strong_case):
    c = _strong_card(strong_case)
    assert c["track"] == "FORWARD"
    for fam in ("ranking", "probability"):
        b = c["best"][fam]
        assert b["candidate"] == "oracle"
        assert b["name"] == "oracle", b["reasons"]
        for comp in c["comparisons"]:
            if comp["family"] == fam and "oracle" in (comp["a"], comp["b"]):
                assert comp["word"] == "SEPARATES" and comp["favours"] == "oracle"


def test_a_monotone_copy_does_not_block_the_ranking_name(strong_case):
    ev, labels = strong_case
    p = ev["p_target"]
    entries = [entry(ev, "oracle", "target_probability", p),
               entry(ev, "oracle copy", "score", np.log(p / (1 - p))),
               entry(ev, "weak", "target_probability", weak(ev, 2, 1.5))]
    c = build_card(make_task(ev, n_boot=150), entries, labels)
    assert {frozenset((d["a"], d["b"])) for d in c["duplicates"]["pairs"]} == {frozenset(("oracle", "oracle copy"))}
    assert c["best"]["ranking"]["name"] in ("oracle", "oracle copy"), c["best"]["ranking"]["reasons"]


def test_a_baseline_is_never_named_best(strong_case):
    """R5 point 4 and R10: a baseline that tops a family on a forward card is never named, and the reason cites R10."""
    ev, labels = strong_case
    anti = entry(ev, "anti", "target_probability", 1.0 - ev["p_target"])
    c = build_card(make_task(ev, n_boot=150), [anti], labels)
    for fam in ("ranking", "probability"):
        b = c["best"][fam]
        assert b["candidate"] == "baseline: coin" and b["name"] is None, (fam, b)
        assert any("R10" in r for r in b["reasons"]), b["reasons"]


def test_record_card_never_names_one(strong_case):
    c = _strong_card(strong_case, track="record")
    assert c["track"] == "RETROSPECTIVE"
    assert all(b["name"] is None and "R5" in b["reasons"][0] for b in c["best"].values())


def test_house_and_exposed_entries_are_not_named(strong_case):
    house = _strong_card(strong_case, house=True)
    assert house["best"]["ranking"]["name"] is None and "R16 c" in " ".join(house["best"]["ranking"]["reasons"])
    unknown = _strong_card(strong_case, exposure="unknown")
    assert unknown["best"]["ranking"]["name"] is None
    assert "NOT CHECKED" in " ".join(unknown["best"]["ranking"]["reasons"])


def test_house_marked_first_and_exposure_printed(mixed_card):
    meta = mixed_card["entries"]
    assert meta[0]["name"] == "target model" and meta[0]["mark"] == "HOUSE" and meta[0]["house"]
    assert [m["role"] for m in meta][-1] == "baseline"
    marks = {m["name"]: m["exposure"] for m in meta}
    assert marks["index"] == "SEEN INPUTS" and marks["triggering model"] == "UNSEEN"
    assert mixed_card["scored_by"] == C.SCORED_BY
    losses = [c for c in mixed_card["comparisons"] if c["favours"] and "target model" in (c["a"], c["b"])
              and c["favours"] != "target model"]
    assert losses and mixed_card["comparisons"][:len(losses)] == losses     # house losses print first (R16 b)


# R2, R6, R7, R12, R15: inputs ----------------------------------------------------------------------------------

def test_task_and_site_list_hashes_are_checked(ev):
    t = make_task(ev)
    e = [entry(ev, "m", "score", ev["s"])]
    with pytest.raises(CardRefused, match="R2: the task hash"):
        build_card(t, e, synth.labels_of(ev), registered_task_sha256="0" * 64)
    build_card(t, e, synth.labels_of(ev), registered_task_sha256=t.sha256())
    with pytest.raises(CardRefused, match="site list"):
        build_card(make_task(ev, site_list_sha256="2" * 64), e, synth.labels_of(ev))


def test_exposure_and_division_are_required(ev):
    other = Entry(synth.manifest("m", "score", "another-quake"), ev["site_ids"], ev["s"])
    with pytest.raises(CardRefused, match="R6"):
        build_card(make_task(ev), [other], synth.labels_of(ev))
    m = synth.manifest("m", "score", EV)
    opened = Entry(type(m)(**{**m.to_dict(), "division": "open"}), ev["site_ids"], ev["s"])
    with pytest.raises(CardRefused, match="R15"):
        build_card(make_task(ev), [opened], synth.labels_of(ev))
    with pytest.raises(CardRefused, match="own model name"):
        build_card(make_task(ev), [entry(ev, "m", "score", ev["s"]), entry(ev, "m", "score", ev["s"])],
                   synth.labels_of(ev))


def test_join_is_by_site_id_only(ev):
    perm = np.random.default_rng(0).permutation(len(ev["site_ids"]))
    p = weak(ev, 4)
    straight = entry(ev, "m", "target_probability", p)
    shuffled = Entry(straight.manifest, tuple(np.array(ev["site_ids"])[perm]), p[perm])
    a = build_card(make_task(ev), [straight], synth.labels_of(ev))
    b = build_card(make_task(ev), [shuffled], synth.labels_of(ev))
    assert a["metrics"] == b["metrics"]
    short = Entry(straight.manifest, ev["site_ids"][:-50], p[:-50])
    with pytest.raises(SpecError, match="no value"):
        build_card(make_task(ev), [short], synth.labels_of(ev))


def test_failed_gates_block_every_card(ev, monkeypatch):
    monkeypatch.setattr(C.gates, "run_gates", lambda: {"pass": False, "seed": 0, "gates": {}})
    with pytest.raises(CardRefused, match="R12"):
        build_card(make_task(ev), [entry(ev, "m", "score", ev["s"])], synth.labels_of(ev))


# R8, R9 extras -------------------------------------------------------------------------------------------------

def test_label_error_and_ceiling(ev, mixed_card):
    assert mixed_card["label"]["error"].startswith("NOT MEASURED")
    c = build_card(make_task(ev, label_error={"sensitivity": 0.9, "specificity": 0.95}),
                   [entry(ev, "m", "score", ev["s"])], synth.labels_of(ev))
    assert 0.5 < c["label"]["auc_ceiling"] < 1.0 and c["label"]["labels_sha256"] == mixed_card["label"]["labels_sha256"]


def test_level_miss_only_on_a_census(ev, mixed_card):
    assert "NOT SHOWN" in metric_block(mixed_card, "target model")["level_miss"]
    c = build_card(make_task(ev, census=True), [entry(ev, "m", "target_probability", ev["p_target"])],
                   synth.labels_of(ev))
    lm = c["metrics"]["m"]["level_miss"]
    assert lm["lo"] <= 0.0 <= lm["hi"]                                       # the true probability has no level miss


def test_duplicates_are_counted_as_one_model(ev):
    p = weak(ev, 7)
    copy = 1 / (1 + np.exp(-3 * np.log(p / (1 - p))))
    c = build_card(make_task(ev), [entry(ev, "L0", "target_probability", p), entry(ev, "M0", "target_probability",
                                                                                     copy)], synth.labels_of(ev))
    assert [(d["a"], d["b"]) for d in c["duplicates"]["pairs"]] == [("L0", "M0")]
    assert c["duplicates"]["distinct_models"] == 1                          # coin and constants are not counted


# R17: aggregates only ------------------------------------------------------------------------------------------

def test_scan_refuses_per_site_content(mixed_card, ev, tmp_path):
    scan_aggregates(mixed_card, ev["site_ids"])
    for bad in ({"values": [0.1] * 100}, {"note": "worst site nzgd:CPT_1234"}, {"note": ev["site_ids"][7]},
                {"values": np.zeros(3)}, {ev["site_ids"][3]: 1.0}, {"x": object()}):
        with pytest.raises(CardRefused, match="R17"):
            scan_aggregates({**mixed_card, "extra": bad}, ev["site_ids"])
    with pytest.raises(CardRefused, match="R17"):
        write_card({**mixed_card, "extra": {"usgs:abc": 1}}, tmp_path / "c.json")
    assert not (tmp_path / "c.json").exists()
    pairs = [{"a": f"m{k}", "b": "coin", "diff": 0.1} for k in range(45)]
    scan_aggregates({**mixed_card, "comparisons": pairs}, ev["site_ids"])        # 9 rows give 45 pair records
    for bad in ([0.1] * 45, [{"v": 0.1}] * 45, pairs[:44] + [{"a": ev["site_ids"][2], "b": "coin"}]):
        with pytest.raises(CardRefused, match="R17"):
            scan_aggregates({**mixed_card, "comparisons": bad}, ev["site_ids"])
    with pytest.raises(CardRefused, match="R17"):
        scan_aggregates({**mixed_card, "other": pairs}, ev["site_ids"])        # only under `comparisons`


def test_site_id_shaped_names_cannot_reach_a_card(ev):
    with pytest.raises(CardRefused, match="R17"):
        build_card(make_task(ev), [entry(ev, "rer:Site-9", "score", ev["s"])], synth.labels_of(ev))
    with pytest.raises(CardRefused, match="R17"):
        build_card(make_task(ev), [entry(ev, ev["site_ids"][0], "score", ev["s"])], synth.labels_of(ev))


# Output form and pooling ---------------------------------------------------------------------------------------

def test_card_is_plain_json_and_deterministic(ev, mixed_card, tmp_path):
    text = json.dumps(mixed_card, allow_nan=False, sort_keys=True)
    write_card(mixed_card, tmp_path / "c.json", ev["site_ids"])
    assert json.loads((tmp_path / "c.json").read_text()) == json.loads(text)
    entries = [entry(ev, "target model", "target_probability", weak(ev, 1, 0.5), house=True),
               entry(ev, "triggering model", "other_probability", ev["p_trigger"]),
               entry(ev, "index", "score", ev["s"], exposure="seen_inputs"),
               entry(ev, "converted triggering", "other_probability", 0.6 * ev["p_trigger"], conversion_id=CONV)]
    again = build_card(make_task(ev), entries, synth.labels_of(ev))
    assert json.dumps(again, sort_keys=True) == text


def test_card_item_reads_a_paired_difference(mixed_card):
    a = card_item(mixed_card, "target model", "index", "auc")
    b = card_item(mixed_card, "index", "target model", "auc")
    assert a["passes_gate"] and a["diff"] == -b["diff"] and a["se"] == b["se"] > 0
    with pytest.raises(KeyError):
        card_item(mixed_card, "index", "triggering model", "log_loss")


# --- scorecard v2 registration K2: a descriptive card withholds every word ------------------------------------------

def test_descriptive_card_withholds_words_and_keeps_numbers(mixed_card):
    note = "not certified; lab coverage 0.80 to 0.96 depending on an unidentified range"
    d = C.descriptive(mixed_card, note)
    assert d["metrics"] == mixed_card["metrics"]
    assert d["words"] == C.DESCRIPTIVE and d["interval_note"] == note
    assert len(d["comparisons"]) == len(mixed_card["comparisons"])
    keep = ("diff", "lo", "hi", "lo90", "hi90", "se", "n_draws")
    before = {(c["family"], c["a"], c["b"]): c for c in mixed_card["comparisons"]}
    for c in d["comparisons"]:
        assert c["word"] == C.DESCRIPTIVE
        for k in ("a_is", "favours", "words_by_tile_m", "words_by_clip", "flags", "p", "mdd"):
            assert k not in c
        for k in keep:
            assert c[k] == before[(c["family"], c["a"], c["b"])][k]
    order = [(c["family"], c["a"], c["b"]) for c in d["comparisons"]]
    assert order == sorted(order)
    assert all(v["name"] is None and v["reasons"] == [C.DESCRIPTIVE] for v in d["best"].values())
    text = json.dumps(d)
    for w in ("SEPARATES", "EQUIVALENT", "CANNOT TELL", "BETTER", "WORSE", "TILE-SENSITIVE", "CLIP-SENSITIVE"):
        assert w not in json.dumps(d["comparisons"]) and w not in json.dumps(d["best"])
    assert "holm_family_sizes" not in d
    assert mixed_card["comparisons"][0].get("word") != C.DESCRIPTIVE      # input not changed
    assert text


def test_descriptive_needs_a_note(mixed_card):
    with pytest.raises(ValueError):
        C.descriptive(mixed_card, " ")

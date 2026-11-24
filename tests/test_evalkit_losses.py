"""B1 (eval M4): the house declares its probability and prints its losses first. Synthetic only. No em dashes."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import losses, synth  # noqa: E402
from scoreboard.evalkit.card import CardRefused, build_card, descriptive  # noqa: E402
from scoreboard.evalkit.cli import main  # noqa: E402
from scoreboard.evalkit.spec import (Entry, EntryManifest, SpecError, Task, card_sha256, entry_sha256,  # noqa: E402
                                     probability_eligible, site_list_sha256)

EV = "eq-test"
# Recorded on the base commit e82afc3 (eval/wave1-base-2026-11-23) before B1: synth.demo_inputs(n=600, n_boot=50).
BASE_DEMO_CARD = "58cd053954bf2390a3cf003f3ab898ac6d8cb475a717cc63849c0cdc2b4ce7a2"
BASE_DEMO_DESCRIPTIVE = "ef685483e480e027b56b19cd6a169c9c5005867c1bb873270dd2e60dd11194ea"
BASE_DEMO_ENTRIES = ("df483c8428cf", "69db65c2c180", "cd03a9640128", "b9d5b041ecbb")


def task_for(ev, **kw):
    d = dict(event_id=EV, target="ejecta seen at the site", label_source="synthetic v1", label_rule_sha256="1" * 64,
             site_list_sha256=site_list_sha256(ev["site_ids"]), track="record", division="closed", tile_m=2000.0,
             clip=1e-3, seed=3, label_caveats="synthetic labels", n_boot=60)
    d.update(kw)
    return Task(**d)


def man(name, kind="target_probability", house=True, **kw):
    return EntryManifest(model=name, output_kind=kind, statement=f"synthetic {kind}", exposure={EV: "unseen"},
                         house=house, **kw)


@pytest.fixture(scope="module")
def ev():
    return synth.synthetic_event(900, 21, unlabelled_share=0.1)


@pytest.fixture(scope="module")
def demo():
    return synth.demo_inputs(n=600, n_boot=50)


def card_of(ev, entries, **kw):
    return build_card(task_for(ev), entries, synth.labels_of(ev), losses_first=True, **kw)


# (1) Declaration ----------------------------------------------------------------------------------------------

def test_declared_house_probability_gets_log_loss_and_calibration(ev):
    """Level 0 and M0 shape: a house triggering probability. Undeclared it is ranking only; declared it is scored."""
    plain = Entry(man("level0 plain", "other_probability"), ev["site_ids"], ev["p_trigger"])
    decl = Entry(man("level0 declared", "other_probability", declared_target=True), ev["site_ids"], ev["p_trigger"])
    assert not probability_eligible(plain.manifest) and probability_eligible(decl.manifest)
    card = card_of(ev, [plain, decl])
    assert "R3" in card["metrics"]["level0 plain"]["probability_metrics"]
    m = card["metrics"]["level0 declared"]
    assert {"log_loss", "brier", "calibration", "reliability"} <= set(m)
    meta = {e["name"]: e for e in card["entries"]}
    assert meta["level0 declared"]["declaration"] == losses.DECLARED_MARK
    assert meta["level0 declared"]["output_kind"] == "other_probability"       # the kind is not rewritten
    assert "declared_target" not in meta["level0 plain"]
    assert card["losses"]["no_proper_score"] == ["level0 plain"]
    probs = {n for c in card["comparisons"] if c["family"] == "probability" for n in (c["a"], c["b"])}
    assert "level0 declared" in probs and "level0 plain" not in probs


@pytest.mark.parametrize("kw", [dict(kind="other_probability", house=False), dict(kind="score"),
                                dict(kind="target_probability"),
                                dict(kind="other_probability", conversion_id="c" * 64)])
def test_only_a_house_unconverted_other_probability_may_declare(kw):
    with pytest.raises(SpecError, match="declared_target"):
        man("x", declared_target=True, **kw)


def test_declaration_must_be_a_boolean():
    with pytest.raises(SpecError, match="declared_target"):
        EntryManifest.from_dict({"model": "x", "output_kind": "other_probability", "statement": "s",
                                 "exposure": {EV: "unseen"}, "house": True, "declared_target": "yes"})


def test_declaration_is_bound_in_the_entry_hash_and_round_trips(ev):
    base = man("m", "other_probability")
    decl = man("m", "other_probability", declared_target=True)
    assert "declared_target" not in base.to_dict() and decl.to_dict()["declared_target"] is True
    assert man("m", "other_probability", declared_target=False) == base
    assert EntryManifest.from_dict(decl.to_dict()) == decl
    p = ev["p_trigger"]
    assert entry_sha256(Entry(base, ev["site_ids"], p)) != entry_sha256(Entry(decl, ev["site_ids"], p))


# (2) LOSS block -------------------------------------------------------------------------------------------------

def test_house_row_worse_than_coin_prints_loss_first(ev):
    bad = np.clip(1.0 - ev["p_target"], 0.02, 0.98)              # confident and backwards
    card = card_of(ev, [Entry(man("house bad"), ev["site_ids"], bad)])
    assert next(iter(card)) == "losses"
    block = card["losses"]
    assert block["status"] == "SCORED" and block["n_lines"] == 1
    line = block["lines"][0]
    assert line["line"] == "LOSS" and line["house"] == "house bad"
    coin = next(i for i in line["lost_to"] if i["baseline"] == "baseline: coin")
    hm, cm = card["metrics"]["house bad"], card["metrics"]["baseline: coin"]
    assert coin["log_loss"] == {"house": hm["log_loss"]["value"], "baseline": cm["log_loss"]["value"]}
    assert coin["brier"] == {"house": hm["brier"]["value"], "baseline": cm["brier"]["value"]}
    assert coin["worse_on"] == ["log loss", "Brier"] and coin["text"].startswith("LOSS: house loss: house bad")
    assert "worse" not in coin["text"].lower() and "point estimates, not a word" in coin["text"]
    assert line["kind"] == losses.HOUSE_KIND and line["mark"] == "HOUSE"
    pr = coin["paired_log_loss_house_minus_baseline"]
    assert pr["diff"] == pytest.approx(coin["log_loss"]["house"] - coin["log_loss"]["baseline"])
    assert pr["lo"] <= pr["diff"] <= pr["hi"]
    assert block["clip"] == 1e-3


def test_house_row_better_than_every_baseline_prints_nothing(ev):
    card = card_of(ev, [Entry(man("house good"), ev["site_ids"], ev["p_target"])])
    assert next(iter(card)) == "losses"
    assert card["losses"]["lines"] == [] and card["losses"]["n_lines"] == 0


def test_only_house_rows_get_loss_lines(ev):
    bad = np.clip(1.0 - ev["p_target"], 0.02, 0.98)
    card = card_of(ev, [Entry(man("outside bad", house=False), ev["site_ids"], bad),
                        Entry(man("house good"), ev["site_ids"], ev["p_target"])])
    assert card["losses"]["lines"] == []


def test_a_declared_row_that_overstates_loses_to_the_coin(ev):
    """Labels are manifestation (triggering x 0.6); a triggering probability declared as manifestation runs high."""
    hot = np.clip(ev["p_trigger"] * 0.2 + 0.75, 0.0, 1.0)
    card = card_of(ev, [Entry(man("declared hot", "other_probability", declared_target=True), ev["site_ids"], hot)])
    line = card["losses"]["lines"][0]
    assert line["house"] == "declared hot" and line["declared_target"] is True
    assert "baseline: coin" in {i["baseline"] for i in line["lost_to"]}


def test_every_baseline_is_checked_not_only_the_coin(ev):
    """A row better than the coin but worse than the earlier-earthquake rate still prints a LOSS line."""
    rate = float(np.nanmean(ev["label"]))
    shaky = Entry(EntryManifest(model="baseline: shaking only", output_kind="target_probability", statement="s",
                                exposure={EV: "unseen"}, house=False), ev["site_ids"], np.full(len(ev["site_ids"]), rate))
    mid = np.full(len(ev["site_ids"]), 0.4)
    card = build_card(task_for(ev, earlier_label_sets=1, earlier_base_rate=rate),
                      [Entry(man("house flat"), ev["site_ids"], mid)], synth.labels_of(ev), shaking_only=shaky,
                      losses_first=True)
    lost = {i["baseline"] for i in card["losses"]["lines"][0]["lost_to"]}
    assert lost == {"baseline: earlier-earthquake rate", "baseline: shaking only"}


def test_underpowered_card_says_not_scored(ev):
    card = build_card(task_for(ev, min_sites=10_000), [Entry(man("h"), ev["site_ids"], ev["p_target"])],
                      synth.labels_of(ev), losses_first=True)
    assert card["status"] == "UNDERPOWERED" and next(iter(card)) == "losses"
    assert card["losses"]["status"] == "NOT SCORED" and card["losses"]["lines"] == []


def test_loss_block_survives_the_descriptive_card_and_reads_old_cards(ev):
    bad = np.clip(1.0 - ev["p_target"], 0.02, 0.98)
    entries = [Entry(man("house bad"), ev["site_ids"], bad)]
    plain = build_card(task_for(ev), entries, synth.labels_of(ev))
    first = build_card(task_for(ev), entries, synth.labels_of(ev), losses_first=True)
    assert losses.loss_block(plain) == first["losses"]                    # reads a card built without the flag
    d = descriptive(first, "note")
    assert next(iter(d)) == "losses" and d["losses"] == first["losses"]
    assert {k: v for k, v in first.items() if k != "losses"} == plain


def test_cli_summary_prints_losses_first(tmp_path, capsys):
    assert main(["demo", "--n-boot", "50", "--losses-first", "--out", str(tmp_path / "c.json")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert next(iter(out)) == "losses" and all(t.startswith("LOSS: ") for t in out["losses"])
    assert next(iter(json.loads((tmp_path / "c.json").read_text()))) == "losses"


# Proper-score gate (review 1) -----------------------------------------------------------------------------------

def test_require_house_proper_score_refuses_an_undeclared_house_probability(ev):
    plain = Entry(man("level0 plain", "other_probability"), ev["site_ids"], ev["p_trigger"])
    with pytest.raises(CardRefused, match="level0 plain"):
        card_of(ev, [plain], require_house_proper_score=True)
    assert card_of(ev, [plain])["losses"]["no_proper_score"] == ["level0 plain"]       # off by default: built


def test_require_house_proper_score_passes_declared_converted_and_outside_rows(ev):
    p = ev["p_trigger"]
    entries = [Entry(man("level0 declared", "other_probability", declared_target=True), ev["site_ids"], p),
               Entry(man("house converted", "other_probability", conversion_id="c" * 64), ev["site_ids"], p),
               Entry(man("outside trigger", "other_probability", house=False), ev["site_ids"], p),
               Entry(man("house target"), ev["site_ids"], ev["p_target"])]
    card = card_of(ev, entries, require_house_proper_score=True)
    assert card["losses"]["no_proper_score"] == []
    assert losses.house_rows_without_proper_score(e.manifest for e in entries) == []


def test_cli_require_house_proper_score_refuses(tmp_path):
    ev = synth.synthetic_event(300, 5)
    d = tmp_path / "entries"
    d.mkdir()
    (d / "lvl0.json").write_text(json.dumps(man("lvl0", "other_probability").to_dict()))
    (d / "lvl0.csv").write_text("site_id,p\n" + "".join(f"{s},{v}\n" for s, v in zip(ev["site_ids"], ev["p_trigger"])))
    lab = synth.labels_of(ev)
    (tmp_path / "labels.csv").write_text("site_id,label,x_m,y_m\n" + "".join(
        f"{s},{'' if not k else int(v)},{x},{y}\n" for s, v, k, x, y in zip(lab.site_ids, lab.label, lab.labelled,
                                                                            lab.x_m, lab.y_m)))
    (tmp_path / "task.json").write_text(json.dumps(task_for(ev).to_dict()))
    args = ["score", "--task", str(tmp_path / "task.json"), "--entries", str(d), "--labels",
            str(tmp_path / "labels.csv"), "--out", str(tmp_path / "c.json")]
    assert main(args + ["--require-house-proper-score"]) != 0 and not (tmp_path / "c.json").exists()
    assert main(args) == 0 and (tmp_path / "c.json").exists()                   # off: the same inputs build


# Correction 8: both LSN fragility rows are our run of one published model ----------------------------------------

@pytest.mark.parametrize("name", ["LSN + global fragility [authors-maurer-2026-11-19]",
                                  "LSN + CES fragility [authors-maurer-2026-11-19]",
                                  "LSN + global fragility [default-2026-11]", "LSN + CES fragility [default-2026-11]",
                                  "LSN + CES fragility [authors-new-chain]", "LSN + global fragility"])
def test_both_lsn_fragility_rows_under_any_chain_id_are_published_ours(name):
    assert losses.is_published_ours(name)


@pytest.mark.parametrize("name", ["Level 0", "M0 (Level 0 + earthquake band)", "LSN + local fragility [x]",
                                  "my LSN + CES fragility", "LSN + CES fragility [x] extra"])
def test_other_rows_are_not_published_ours(name):
    assert not losses.is_published_ours(name)


def test_both_lsn_rows_get_the_same_mark_and_a_published_model_loss_line(ev):
    """Two converted house rows named like the two LSN rows, one source: both lose, both read the same way."""
    bad = np.clip(1.0 - ev["p_target"], 0.02, 0.98)
    names = ("LSN + global fragility [authors-test]", "LSN + CES fragility [authors-test]")
    entries = [Entry(man(n, "score", conversion_id="c" * 64), ev["site_ids"], bad + 0.001 * i)
               for i, n in enumerate(names)]
    entries.append(Entry(man("house bad"), ev["site_ids"], bad))
    card = card_of(ev, entries)
    by = {ln["house"]: ln for ln in card["losses"]["lines"]}
    assert set(by) == {*names, "house bad"}
    a, b = by[names[0]], by[names[1]]
    assert a["kind"] == b["kind"] == losses.PUBLISHED_OURS_KIND
    assert a["mark"] == b["mark"] == losses.PUBLISHED_OURS_MARK
    assert a["conversion_id"] == b["conversion_id"] == "c" * 64
    for ln in (a, b):
        for item in ln["lost_to"]:
            assert item["text"].startswith(f"LOSS: loss of our run of a published model (Maurer, engrXiv 121, "
                                           f"Table 3): {ln['house']}")
            assert "house loss" not in item["text"]
    assert by["house bad"]["kind"] == losses.HOUSE_KIND
    assert by["house bad"]["lost_to"][0]["text"].startswith("LOSS: house loss: house bad")


# Hand-made card dicts: orientation flip, tie, NOT RUN rows ------------------------------------------------------

def _hand_card(h, b, comparisons=(), extra_entry=None):
    entries = [{"name": "H", "role": "entry", "house": True, "mark": "HOUSE", "probability_metrics": True},
               {"name": "baseline: coin", "role": "baseline", "house": False, "probability_metrics": True}]
    if extra_entry:
        entries.append(extra_entry)
    return {"task": {"clip": 1e-3}, "status": "SCORED", "entries": entries, "comparisons": list(comparisons),
            "metrics": {"H": {"log_loss": {"value": h[0]}, "brier": {"value": h[1]}},
                        "baseline: coin": {"log_loss": {"value": b[0]}, "brier": {"value": b[1]}}}}


def test_paired_interval_flips_when_the_house_is_comparison_b():
    comp = {"family": "probability", "a": "baseline: coin", "b": "H", "diff": -0.2, "lo": -0.3, "hi": -0.1}
    item = losses.loss_block(_hand_card((0.9, 0.3), (0.7, 0.25), [comp]))["lines"][0]["lost_to"][0]
    assert item["paired_log_loss_house_minus_baseline"] == {"diff": 0.2, "lo": pytest.approx(0.1),
                                                            "hi": pytest.approx(0.3)}
    same = {"family": "probability", "a": "H", "b": "baseline: coin", "diff": 0.2, "lo": 0.1, "hi": 0.3}
    item = losses.loss_block(_hand_card((0.9, 0.3), (0.7, 0.25), [same]))["lines"][0]["lost_to"][0]
    assert item["paired_log_loss_house_minus_baseline"] == {"diff": 0.2, "lo": 0.1, "hi": 0.3}
    none = dict(same, diff=None)
    item = losses.loss_block(_hand_card((0.9, 0.3), (0.7, 0.25), [none]))["lines"][0]["lost_to"][0]
    assert "paired_log_loss_house_minus_baseline" not in item


def test_a_tie_prints_no_loss_line():
    assert losses.loss_block(_hand_card((0.7, 0.25), (0.7, 0.25)))["lines"] == []
    one = losses.loss_block(_hand_card((0.7, 0.26), (0.7, 0.25)))["lines"][0]["lost_to"][0]
    assert one["worse_on"] == ["Brier"]


def test_a_not_run_row_is_listed_apart_and_gets_no_line():
    nr = {"name": "LSN + CES fragility [authors-x]", "role": "entry", "house": True, "probability_metrics": False,
          "author_confirmed": False}
    block = losses.loss_block(_hand_card((0.6, 0.2), (0.7, 0.25), extra_entry=nr))
    assert block["not_run_by_author_settings"] == [nr["name"]] and block["no_proper_score"] == []
    assert block["lines"] == []


# Defaults keep every older card bit for bit ---------------------------------------------------------------------

def test_defaults_reproduce_the_base_kit_card_bit_for_bit(demo):
    task, entries, labels, shaking = demo
    assert tuple(entry_sha256(e)[:12] for e in entries) == BASE_DEMO_ENTRIES
    card = build_card(task, entries, labels, shaking_only=shaking)
    assert "losses" not in card and all("declared_target" not in e for e in card["entries"])
    assert card_sha256(card) == BASE_DEMO_CARD
    assert card_sha256(descriptive(card, "note")) == BASE_DEMO_DESCRIPTIVE
    explicit = build_card(task, entries, labels, shaking_only=shaking, losses_first=False)
    assert json.dumps(explicit) == json.dumps(card)

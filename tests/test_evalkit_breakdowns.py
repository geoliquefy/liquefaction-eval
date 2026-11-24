"""B6 (eval M4): breakdowns by PGA bin, Ic class, crust thickness and severity class. Synthetic and planted data
only (ruling 45, correction 1). No em dashes."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import breakdowns as BD  # noqa: E402
from scoreboard.evalkit import synth  # noqa: E402
from scoreboard.evalkit.card import build_card, descriptive, scan_aggregates  # noqa: E402
from scoreboard.evalkit.spec import Entry, EntryManifest, Task, card_sha256, site_list_sha256  # noqa: E402

EV = "eq-test"
# The same pinned demo card as tests/test_evalkit_losses.py (base commit, before B1 and B6).
BASE_DEMO_CARD = "58cd053954bf2390a3cf003f3ab898ac6d8cb475a717cc63849c0cdc2b4ce7a2"
BASE_DEMO_DESCRIPTIVE = "ef685483e480e027b56b19cd6a169c9c5005867c1bb873270dd2e60dd11194ea"
N = 1200


def task_for(ids, **kw):
    d = dict(event_id=EV, target="ejecta seen at the site", label_source="synthetic v1", label_rule_sha256="1" * 64,
             site_list_sha256=site_list_sha256(ids), track="record", division="closed", tile_m=2000.0,
             clip=1e-3, seed=5, label_caveats="synthetic labels", n_boot=50)
    d.update(kw)
    return Task(**d)


def man(name, kind="target_probability", house=False):
    return EntryManifest(model=name, output_kind=kind, statement=f"synthetic {kind}", exposure={EV: "unseen"},
                         house=house)


@pytest.fixture(scope="module")
def planted():
    """A planted world. PGA, Ic and crust are drawn per site. The 'shaking-led' entry knows the label only at
    PGA 0.30 g and above; below that it is noise. The 'severe-led' entry ranks moderate-or-worse sites first but
    cannot tell minor ejecta from none."""
    rng = np.random.default_rng(20261123)
    ev = synth.synthetic_event(N, 11, field_sd=0.3)
    ids = ev["site_ids"]
    pga = rng.uniform(0.05, 0.6, N)
    pga[:40] = np.nan                       # 40 sites with no PGA value
    ic = rng.uniform(1.2, 3.8, N)
    crust = rng.uniform(0.0, 7.0, N)
    y = ev["label"]
    high = np.nan_to_num(pga, nan=0.0) >= 0.3
    p_shake = np.where(high, np.where(y == 1, 0.8, 0.2), rng.uniform(0.0, 1.0, N))
    sev = np.where(y == 1, rng.integers(1, 4, N), 0).astype(float)   # 1 minor, 2 moderate, 3 severe
    p_sev = np.where(sev >= 2, 0.9, 0.3) + rng.uniform(-0.05, 0.05, N)
    entries = [Entry(man("shaking-led", house=True), ids, p_shake),
               Entry(man("severe-led"), ids, p_sev),
               Entry(man("an index", "score"), ids, ev["s"])]
    vals = (BD.SiteValues("pga", "synthetic PGA, g", ids, pga), BD.SiteValues("ic", "synthetic site Ic", ids, ic),
            BD.SiteValues("crust", "synthetic crust, m", ids, crust),
            BD.SiteValues("severity", "synthetic severity class 0 to 3", ids, sev))
    return {"ev": ev, "ids": ids, "entries": entries, "values": vals, "pga": pga, "sev": sev, "y": y}


@pytest.fixture(scope="module")
def card(planted):
    p = planted
    return build_card(task_for(p["ids"]), p["entries"], synth.labels_of(p["ev"]),
                      breakdowns=BD.Breakdowns(p["values"]))


def test_defaults_reproduce_the_base_card_bit_for_bit():
    task, entries, labels, shaking = synth.demo_inputs(n=600, n_boot=50)
    card = build_card(task, entries, labels, shaking_only=shaking)
    assert "breakdowns" not in card
    assert card_sha256(card) == BASE_DEMO_CARD
    assert card_sha256(descriptive(card, "note")) == BASE_DEMO_DESCRIPTIVE
    assert json.dumps(build_card(task, entries, labels, shaking_only=shaking, breakdowns=None)) == json.dumps(card)


def test_all_four_tables_print_with_cells_named(card):
    b = card["breakdowns"]
    assert {"pga", "ic", "crust", "severity"} <= set(b)
    assert tuple(b["pga"]["cells"]) == BD.cell_names("pga")
    assert tuple(b["ic"]["cells"])[:6] == ("below 1.31", "1.31 to 2.05", "2.05 to 2.60", "2.60 to 2.95",
                                           "2.95 to 3.60", "3.60 and above")
    assert tuple(b["severity"]["cells"]) == ("any ejecta", "moderate or worse")
    assert "Robertson" in b["ic"]["edges_source"]
    assert BD.NO_WINNER in b["notes"]


def test_n_per_cell_adds_up_and_blank_values_go_to_no_value(card, planted):
    labelled = int(np.isfinite(planted["y"]).sum())
    for kind in ("pga", "ic", "crust"):
        assert sum(c["n"] for c in card["breakdowns"][kind]["cells"].values()) == labelled
    assert card["breakdowns"]["pga"]["cells"][BD.NO_VALUE]["n"] == 40
    assert card["breakdowns"]["ic"]["cells"][BD.NO_VALUE]["n"] == 0


def test_cells_under_the_floor_print_too_few_and_nothing_else(card):
    pga = card["breakdowns"]["pga"]["cells"]
    assert BD.is_too_few(card["breakdowns"]["ic"]["cells"][BD.NO_VALUE])
    shown = [c for c in pga.values() if c["status"] == "SHOWN"]
    assert shown and all(c["n"] >= 30 and min(c["n_yes"], c["n_no"]) >= 5 for c in shown)


def test_floor_is_a_setting(planted):
    p = planted
    vals = (p["values"][0],)
    hi = build_card(task_for(p["ids"]), p["entries"], synth.labels_of(p["ev"]),
                    breakdowns=BD.Breakdowns(vals, floor_n=10_000))
    assert all(BD.is_too_few(c) for c in hi["breakdowns"]["pga"]["cells"].values())
    assert hi["breakdowns"]["floor_n"] == 10_000


def test_planted_pga_effect_shows_in_the_right_cells(card):
    cells = card["breakdowns"]["pga"]["cells"]
    for name in ("0.30 to 0.40 g", "0.40 to 0.50 g", "0.50 g and above"):
        assert cells[name]["rows"]["shaking-led"]["auc"]["value"] > 0.95
    for name in ("0.10 to 0.20 g", "0.20 to 0.30 g"):
        a = cells[name]["rows"]["shaking-led"]["auc"]
        assert abs(a["value"] - 0.5) < 0.15 and a["lo"] < 0.5 < a["hi"]


def test_planted_severity_effect_and_ranking_only(card):
    sev = card["breakdowns"]["severity"]["cells"]
    assert sev["moderate or worse"]["rows"]["severe-led"]["auc"]["value"] > 0.95
    assert sev["any ejecta"]["rows"]["severe-led"]["auc"]["value"] < sev["moderate or worse"]["rows"]["severe-led"][
        "auc"]["value"]
    for c in sev.values():
        for m in c["rows"].values():
            assert set(m) == {"auc"}                 # R3: no proper score on a severity relabel
    assert BD.SEVERITY_NOTE in card["breakdowns"]["notes"]


def test_probability_rows_get_proper_scores_and_score_rows_do_not(card):
    c = next(c for c in card["breakdowns"]["pga"]["cells"].values() if c["status"] == "SHOWN")
    assert {"auc", "log_loss@0.001", "brier", "mean_p"} == set(c["rows"]["shaking-led"])
    assert "R3" in c["rows"]["an index"]["probability_metrics"] and "brier" not in c["rows"]["an index"]


def test_no_winner_rows_keep_card_order_and_no_best_key(card):
    order = [e["name"] for e in card["entries"]]
    for kind in ("pga", "ic", "crust", "severity"):
        for c in card["breakdowns"][kind]["cells"].values():
            if c["status"] == "SHOWN":
                assert list(c["rows"]) == [n for n in order if n in c["rows"]]
    BD.check_no_winner(card["breakdowns"])
    with pytest.raises(BD.BreakdownError, match="names no winner"):
        BD.check_no_winner({"pga": {"cells": {"x": {"best": "shaking-led"}}}})


def test_breakdowns_hold_aggregates_only(card, planted):
    scan_aggregates(card["breakdowns"], planted["ids"])
    text = json.dumps(card["breakdowns"])
    assert "syn:" not in text


def test_join_by_site_id_not_by_position(planted):
    p = planted
    ids = p["ids"]
    perm = np.random.default_rng(1).permutation(len(ids))
    shuffled = BD.SiteValues("pga", "synthetic PGA, g", tuple(ids[k] for k in perm), p["pga"][perm])
    a = build_card(task_for(ids), p["entries"], synth.labels_of(p["ev"]), breakdowns=BD.Breakdowns((p["values"][0],)))
    b = build_card(task_for(ids), p["entries"], synth.labels_of(p["ev"]), breakdowns=BD.Breakdowns((shuffled,)))
    assert a["breakdowns"] == b["breakdowns"]


def test_inputs_refused(planted):
    ids = planted["ids"]
    with pytest.raises(BD.BreakdownError, match="kind"):
        BD.SiteValues("pgv", "x", ids, np.zeros(len(ids)))
    with pytest.raises(BD.BreakdownError, match="statement"):
        BD.SiteValues("pga", " ", ids, np.zeros(len(ids)))
    with pytest.raises(BD.BreakdownError, match="whole numbers"):
        BD.SiteValues("severity", "s", ids, np.full(len(ids), 1.5))
    with pytest.raises(BD.BreakdownError, match="negative"):
        BD.SiteValues("crust", "c", ids, np.full(len(ids), -1.0))
    with pytest.raises(BD.BreakdownError, match="at most once"):
        BD.Breakdowns((planted["values"][0], planted["values"][0]))
    with pytest.raises(BD.BreakdownError, match="moderate_at"):
        BD.Breakdowns((planted["values"][0],), moderate_at=1)
    short = BD.SiteValues("pga", "p", ids[:-5], np.zeros(len(ids) - 5))
    with pytest.raises(BD.BreakdownError, match="no entry for 5"):
        build_card(task_for(ids), planted["entries"], synth.labels_of(planted["ev"]),
                   breakdowns=BD.Breakdowns((short,)))


def test_bin_edges_are_half_open():
    v = np.array([0.0999, 0.1, 0.2999, 0.3, 0.5, np.nan])
    assert BD.assign("pga", v).tolist() == [0, 1, 2, 3, 5, 6]


# ---------------------------------------------------------------- review 1 fixes (synthetic and planted data only)

def _shown_rows(card):
    """Every row name printed in any breakdowns cell."""
    names = set()
    for kind in ("pga", "ic", "crust", "severity"):
        for c in card["breakdowns"].get(kind, {}).get("cells", {}).values():
            names |= set(c.get("rows", {}))
    return names


def test_underpowered_card_prints_counts_only_in_every_cell():
    """R11 (b): 150 planted sites fail the count gate; no breakdowns cell may carry a score."""
    ev = synth.synthetic_event(150, 3, field_sd=0.3)
    ids = ev["site_ids"]
    p_house = 1.0 / (1.0 + np.exp(-ev["s"]))
    entries = [Entry(man("house p", house=True), ids, p_house), Entry(man("an index", "score"), ids, ev["s"])]
    pga = np.full(150, 0.35)                       # every site in one PGA cell, the case the review reproduced
    sev = np.where(ev["label"] == 1, 2.0, 0.0)
    vals = (BD.SiteValues("pga", "synthetic PGA, g", ids, pga),
            BD.SiteValues("severity", "synthetic severity", ids, sev))
    c = build_card(task_for(ids), entries, synth.labels_of(ev), breakdowns=BD.Breakdowns(vals, floor_tiles=1))
    assert c["status"] == "UNDERPOWERED" and "metrics" not in c
    assert BD.UNDERPOWERED in c["breakdowns"]["notes"]
    assert _shown_rows(c) == set()
    for kind in ("pga", "severity"):
        for cell in c["breakdowns"][kind]["cells"].values():
            assert BD.is_counts_only(cell) and cell["status"] == BD.UNDERPOWERED and "rows" not in cell
    assert c["breakdowns"]["pga"]["cells"]["0.30 to 0.40 g"]["n"] == 150


def test_one_tile_cell_prints_too_few_tiles(planted):
    """Every site of the 0.30 to 0.40 g cell at one point: one tile, so no zero-width interval is printed."""
    p = planted
    ev = dict(p["ev"])
    pga = np.nan_to_num(p["pga"], nan=-1.0)
    one = (pga >= 0.3) & (pga < 0.4)
    ev["x_m"], ev["y_m"] = ev["x_m"].copy(), ev["y_m"].copy()
    ev["x_m"][one], ev["y_m"][one] = 100.0, 100.0
    c = build_card(task_for(p["ids"]), p["entries"], synth.labels_of(ev), breakdowns=BD.Breakdowns((p["values"][0],)))
    cell = c["breakdowns"]["pga"]["cells"]["0.30 to 0.40 g"]
    assert cell == {"n": int(one.sum()), "status": BD.TOO_FEW_TILES, "n_tiles": 1}
    assert BD.is_counts_only(cell) and not BD.is_too_few(cell)
    assert c["breakdowns"]["floor_tiles"] == 20      # default: the task min_tiles
    other = c["breakdowns"]["pga"]["cells"]["0.40 to 0.50 g"]
    assert other["status"] == "SHOWN" and other["n_tiles"] >= 20
    with pytest.raises(BD.BreakdownError, match="floor_tiles"):
        BD.Breakdowns((p["values"][0],), floor_tiles=0)


def test_per_class_floor_prints_too_few(planted):
    """A cell with 40 no sites and 3 yes sites clears floor_n (30) but not floor_per_class (5)."""
    p = planted
    y = p["y"]
    pga = np.full(len(y), 0.25)
    no_k, yes_k = np.flatnonzero(y == 0)[:40], np.flatnonzero(y == 1)[:3]
    pga[no_k], pga[yes_k] = 0.05, 0.05
    sv = BD.SiteValues("pga", "synthetic PGA, g", p["ids"], pga)
    c = build_card(task_for(p["ids"]), p["entries"], synth.labels_of(p["ev"]),
                   breakdowns=BD.Breakdowns((sv,), floor_tiles=1))
    cell = c["breakdowns"]["pga"]["cells"]["below 0.10 g"]
    assert cell["n"] == 43 and BD.is_too_few(cell)
    assert c["breakdowns"]["pga"]["cells"]["0.20 to 0.30 g"]["status"] == "SHOWN"


def test_unconfirmed_outside_row_is_absent_from_every_breakdowns_cell(planted):
    """B2: a row its authors have not confirmed gets no breakdowns score, and the author gate is the backstop."""
    from scoreboard.evalkit import outside
    p = planted
    c = build_card(task_for(p["ids"]), p["entries"], synth.labels_of(p["ev"]),
                   breakdowns=BD.Breakdowns(p["values"]), author_confirmed={"severe-led": False, "an index": True})
    names = _shown_rows(c)
    assert "severe-led" not in names
    assert {"shaking-led", "an index"} <= names
    leak = json.loads(json.dumps(c))
    cell = next(x for x in leak["breakdowns"]["pga"]["cells"].values() if x["status"] == "SHOWN")
    cell["rows"]["severe-led"] = {"auc": {"value": 0.9}}
    with pytest.raises(outside.AuthorGateError, match="breakdowns cell"):
        outside.check_card(leak)


def test_model_names_under_rows_are_not_structure_keys():
    """A model named best or rank is a row name, not a winner key; a winner key inside a row is still refused."""
    BD.check_no_winner({"pga": {"cells": {"x": {"rows": {"best": {"auc": 0.6}, "rank": {"auc": 0.5}}}}}})
    with pytest.raises(BD.BreakdownError, match="names no winner"):
        BD.check_no_winner({"pga": {"cells": {"x": {"rows": {"m": {"auc": 0.6, "diff": 0.1}}}}}})


def test_severity_table_counts_label_disagreements(planted):
    p = planted
    y, sev = p["y"], p["sev"].copy()
    sev[np.flatnonzero(y == 1)[:7]] = 0.0
    sev[np.flatnonzero(y == 0)[:4]] = 2.0
    sv = BD.SiteValues("severity", "synthetic severity class 0 to 3", p["ids"], sev)
    c = build_card(task_for(p["ids"]), p["entries"], synth.labels_of(p["ev"]), breakdowns=BD.Breakdowns((sv,)))
    assert c["breakdowns"]["severity"]["label_disagreements"] == {"label_yes_class_0": 7,
                                                                  "label_no_class_1_or_worse": 4}


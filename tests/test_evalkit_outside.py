"""evalkit B2: outside-model rows print NOT RUN BY AUTHOR SETTINGS until their authors confirm; a gate refuses a card
that prints a score for an unconfirmed one; Zhu empty cells are off-grid and dropped, never 0. Synthetic only."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import outside, synth, zhu_cells  # noqa: E402
from scoreboard.evalkit.card import CardRefused, build_card, descriptive, write_card  # noqa: E402
from scoreboard.evalkit.spec import Entry, Task, card_sha256, site_list_sha256  # noqa: E402

EV = "eq-test"
CONV = "c" * 64
ZHU, GEM = "USGS Zhu 2017", "GEM TS22"
CES_AUTH = "LSN + CES fragility [authors-maurer-2026-11-19]"
CES_DEF = "LSN + CES fragility [default-2026-11]"
GLO_AUTH = "LSN + global fragility [authors-maurer-2026-11-19]"
GLO_DEF = "LSN + global fragility [default-2026-11]"


def make_task(ev, **kw):
    d = dict(event_id=EV, target="ejecta seen at the site", label_source="synthetic v1", label_rule_sha256="1" * 64,
             site_list_sha256=site_list_sha256(ev["site_ids"]), track="forward", division="closed", tile_m=2000.0,
             clip=1e-6, seed=3, label_caveats="synthetic labels", n_boot=60)
    d.update(kw)
    return Task(**d)


def entry(ev, name, kind, values, **kw):
    return Entry(synth.manifest(name, kind, EV, **kw), ev["site_ids"], values)


@pytest.fixture(scope="module")
def ev():
    return synth.synthetic_event(700, 31, unlabelled_share=0.1)


@pytest.fixture(scope="module")
def entries(ev):
    rng = np.random.default_rng(5)
    noisy = ev["s"] + rng.normal(0, 0.7, ev["s"].size)
    return [entry(ev, "house model", "target_probability", ev["p_target"], house=True),
            entry(ev, CES_AUTH, "score", noisy, house=True, conversion_id=CONV),
            entry(ev, CES_DEF, "score", noisy + 0.1, house=True, conversion_id=CONV),
            entry(ev, ZHU, "other_probability", ev["p_trigger"]),
            entry(ev, GEM, "score", -noisy)]


@pytest.fixture(scope="module")
def plain(ev, entries):
    return build_card(make_task(ev), entries, synth.labels_of(ev))


@pytest.fixture(scope="module")
def flagged(ev, entries):
    return build_card(make_task(ev), entries, synth.labels_of(ev), author_confirmed={ZHU: True})


def meta(card):
    return {e["name"]: e for e in card["entries"]}


# Which rows are outside rows --------------------------------------------------------------------------------------

def test_outside_rows_are_non_house_entries_and_author_id_lsn_rows():
    assert outside.is_outside(ZHU, "entry", False) and outside.is_outside(GEM, "entry", False)
    assert outside.is_outside(CES_AUTH, "entry", True)
    assert outside.is_outside(GLO_AUTH, "entry", True)
    assert not outside.is_outside(CES_DEF, "entry", True)
    assert not outside.is_outside(GLO_DEF, "entry", True)
    assert not outside.is_outside("house model", "entry", True)
    assert not outside.is_outside("baseline: coin", "baseline", False)


# Default keeps the old card ---------------------------------------------------------------------------------------

def test_default_card_has_no_author_marks_and_scores_every_row(plain):
    assert "author_rule" not in plain
    assert all("author_confirmed" not in e for e in plain["entries"])
    assert {ZHU, GEM, CES_AUTH} <= set(plain["metrics"])
    assert outside.NOT_RUN not in str(plain)


def test_default_card_hash_does_not_depend_on_the_new_module(ev, entries, plain, monkeypatch):
    """With the option off, the card is the same even if the B2 hooks are made no-ops (sealed v2 reproducible)."""
    import scoreboard.evalkit.card as C
    monkeypatch.setattr(C.outside, "split_rows", lambda *a, **k: pytest.fail("split_rows ran with the option off"))
    monkeypatch.setattr(C.outside, "mark_card", lambda *a, **k: pytest.fail("mark_card ran with the option off"))
    again = build_card(make_task(ev), entries, synth.labels_of(ev))
    assert card_sha256(again) == card_sha256(plain)


# card_sha256 of the `plain` fixture card, built by the kit at e82afc3 (before B2) and at cf2af7b (B2). A change to
# the always-on lines (entry_marks, author_gate) that alters the default card fails here. If a deliberate kit change
# moves this hash, the new value needs its own note: the sealed v2 card must still rebuild from its frozen commit.
PRE_B2_DEFAULT_CARD_SHA256 = "c6665f56349199048a27c0df1e5620124075680b8f281ed47c1d20e3a840b146"


def test_default_card_hash_equals_the_pre_b2_card(plain):
    assert card_sha256(plain) == PRE_B2_DEFAULT_CARD_SHA256


# Option on ---------------------------------------------------------------------------------------------------------

def test_unconfirmed_outside_rows_print_not_run_instead_of_scores(flagged):
    m = meta(flagged)
    for n in (GEM, CES_AUTH):
        assert m[n]["author_confirmed"] is False and outside.NOT_RUN in m[n]["mark"]
        assert flagged["metrics"][n] == outside.NOT_RUN
    assert m[ZHU]["author_confirmed"] is True and isinstance(flagged["metrics"][ZHU], dict)
    for n in ("house model", CES_DEF, "baseline: coin"):
        assert "author_confirmed" not in m[n] and isinstance(flagged["metrics"][n], dict)
    assert flagged["author_rule"]["not_run"] == [CES_AUTH, GEM]
    assert flagged["author_rule"]["confirmed"] == [ZHU]


def test_unconfirmed_rows_enter_no_comparison_best_or_duplicate(flagged):
    bad = {GEM, CES_AUTH}
    assert flagged["comparisons"]
    assert not any({c["a"], c["b"]} & bad for c in flagged["comparisons"])
    for b in flagged["best"].values():
        assert not ({b.get("candidate"), b.get("name")} & bad)
    assert not any({d["a"], d["b"]} & bad for d in flagged["duplicates"]["pairs"])


def test_entry_order_is_kept(plain, flagged):
    assert [e["name"] for e in plain["entries"]] == [e["name"] for e in flagged["entries"]]


def test_missing_name_counts_as_unconfirmed_and_a_stray_name_is_refused(ev, entries):
    card = build_card(make_task(ev), entries, synth.labels_of(ev), author_confirmed={})
    assert set(card["author_rule"]["not_run"]) == {ZHU, GEM, CES_AUTH}
    with pytest.raises(ValueError, match="not outside-model rows"):
        build_card(make_task(ev), entries, synth.labels_of(ev), author_confirmed={"house model": True})
    with pytest.raises(TypeError):
        build_card(make_task(ev), entries, synth.labels_of(ev), author_confirmed={ZHU: "yes"})


def test_underpowered_card_still_marks_not_run(ev, entries):
    card = build_card(make_task(ev, min_sites=10_000), entries, synth.labels_of(ev), author_confirmed={})
    assert card["status"] == "UNDERPOWERED"
    assert meta(card)[ZHU]["mark"] == outside.NOT_RUN and "metrics" not in card


def test_descriptive_keeps_the_marks(flagged):
    d = descriptive(flagged, "interval note")
    assert d["metrics"][GEM] == outside.NOT_RUN and d["author_rule"] == flagged["author_rule"]


# The gate ----------------------------------------------------------------------------------------------------------

def test_gate_refuses_a_card_that_prints_a_score_for_an_unconfirmed_row(flagged, tmp_path):
    outside.check_card(flagged)
    forged = copy.deepcopy(flagged)
    forged["metrics"][GEM] = copy.deepcopy(flagged["metrics"][ZHU])
    with pytest.raises(outside.AuthorGateError, match="metrics block"):
        outside.check_card(forged)
    with pytest.raises(CardRefused, match="B2"):
        write_card(forged, tmp_path / "card.json")
    with pytest.raises(CardRefused, match="B2"):
        descriptive(forged, "interval note")
    assert not (tmp_path / "card.json").exists()


@pytest.mark.parametrize("where", ["comparison", "best", "duplicate", "flag_only"])
def test_gate_refuses_every_other_way_a_score_can_print(flagged, where):
    forged = copy.deepcopy(flagged)
    if where == "comparison":
        forged["comparisons"].append({"family": "ranking", "a": CES_AUTH, "b": "baseline: coin", "diff": 0.1})
    elif where == "best":
        forged["best"]["ranking"] = {"candidate": GEM, "name": GEM, "reasons": []}
    elif where == "duplicate":
        forged["duplicates"]["pairs"].append({"a": GEM, "b": "house model", "rho": 0.99})
    else:   # the author block was stripped of the name but the entry still says false
        forged["author_rule"]["not_run"] = []
        forged["metrics"][GEM] = {"auc": {"point": 0.7}}
    with pytest.raises(outside.AuthorGateError):
        outside.check_card(forged)


def test_card_without_author_block_or_flags_passes(plain):
    outside.check_card(plain)


@pytest.mark.parametrize("path", ["check_card", "write_card"])
def test_gate_refuses_a_flagged_row_score_after_the_author_block_is_deleted(flagged, tmp_path, path):
    """Review 1: deleting author_rule must not open the gate while an entry still says author_confirmed false."""
    forged = copy.deepcopy(flagged)
    del forged["author_rule"]
    forged["metrics"][GEM] = copy.deepcopy(flagged["metrics"][ZHU])
    if path == "check_card":
        with pytest.raises(outside.AuthorGateError, match="metrics block"):
            outside.check_card(forged)
    else:
        with pytest.raises(CardRefused, match="B2"):
            write_card(forged, tmp_path / "card.json")
        assert not (tmp_path / "card.json").exists()


def test_cell_input_is_the_three_steps_in_one_call():
    gx, gy, g = zhu_grid()
    lon = np.array([135.0, 135.2, 135.21, 135.4, 135.46, 135.1])
    lat = np.array([34.5, 34.6, 34.61, 34.8, 34.8, 34.7])
    y = np.array([1, 1, 0, 0, 1, 1])
    c = zhu_cells.cell_input(gx, gy, g, lon, lat, y)
    r = zhu_cells.read_cells(gx, gy, g, lon, lat)
    keep, rec = zhu_cells.drop_off_grid(r, y)
    assert c.rec == rec and c.note == zhu_cells.drop_note(rec)
    assert c.label.tolist() == y[keep].tolist() and np.array_equal(c.values, r.values[keep])
    assert np.isfinite(c.values).all() and c.values.size == 3


# Zhu empty cells -------------------------------------------------------------------------------------------------

def zhu_grid():
    gx = np.linspace(135.0, 135.4, 5)          # 0.1 degree cells
    gy = np.linspace(34.5, 34.8, 4)
    g = np.arange(20, dtype=float).reshape(4, 5) / 40.0
    g[1, 2] = np.nan                            # one empty cell
    return gx, gy, g


def test_empty_cell_is_off_grid_not_zero():
    gx, gy, g = zhu_grid()
    lon = np.array([135.0, 135.2, 135.21, 135.4, 135.46, 134.0])
    lat = np.array([34.5, 34.6, 34.61, 34.8, 34.8, 34.6])
    r = zhu_cells.read_cells(gx, gy, g, lon, lat)
    assert r.on_grid.tolist() == [True, False, False, True, False, False]
    assert r.empty_cell.tolist() == [False, True, True, False, False, False]
    assert r.outside_extent.tolist() == [False, False, False, False, True, True]
    assert np.isnan(r.values[~r.on_grid]).all()
    assert r.values[0] == g[0, 0] and r.values[3] == g[3, 4]
    assert not np.any(r.values[~r.on_grid] == 0.0)


def test_on_grid_values_equal_the_v2_nearest_cell_reader():
    gx, gy, g = zhu_grid()
    rng = np.random.default_rng(2)
    lon, lat = rng.uniform(134.96, 135.44, 200), rng.uniform(34.46, 34.84, 200)
    r = zhu_cells.read_cells(gx, gy, g, lon, lat)
    ix = np.abs(gx[None, :] - lon[:, None]).argmin(axis=1)
    iy = np.abs(gy[None, :] - lat[:, None]).argmin(axis=1)
    v2 = np.nan_to_num(g[iy, ix], nan=0.0)
    assert np.array_equal(r.values[r.on_grid], v2[r.on_grid])
    assert r.on_grid.sum() < 200 and r.n_off_grid == int(np.isnan(g[iy, ix]).sum())


def test_drop_off_grid_drops_and_counts():
    gx, gy, g = zhu_grid()
    lon = np.array([135.0, 135.2, 135.21, 135.4, 135.46, 135.1])
    lat = np.array([34.5, 34.6, 34.61, 34.8, 34.8, 34.7])
    y = np.array([1, 1, 0, 0, 1, 1])
    r = zhu_cells.read_cells(gx, gy, g, lon, lat)
    keep, rec = zhu_cells.drop_off_grid(r, y)
    assert keep.tolist() == [True, False, False, True, False, True]
    assert rec == {"n_sites": 6, "n_kept": 3, "n_off_grid": 3, "n_empty_cell": 2, "n_outside_extent": 1,
                   "n_yes_dropped": 2, "n_no_dropped": 1}
    note = zhu_cells.drop_note(rec)
    assert note.startswith("3 of 6 sites off the model grid, dropped from this cell") and "2 yes, 1 no" in note
    assert np.isfinite(r.values[keep]).all()


def test_no_off_grid_prints_zero_count():
    gx, gy, g = zhu_grid()
    r = zhu_cells.read_cells(gx, gy, g, [135.0], [34.5])
    keep, rec = zhu_cells.drop_off_grid(r)
    assert keep.all() and zhu_cells.drop_note(rec) == "0 of 1 sites off the model grid"


def test_reader_refuses_bad_shapes():
    gx, gy, g = zhu_grid()
    with pytest.raises(ValueError):
        zhu_cells.read_cells(gx, gy, g.T, [135.0], [34.5])
    with pytest.raises(ValueError):
        zhu_cells.read_cells(gx, gy, g, [135.0, 135.1], [34.5])
    with pytest.raises(ValueError):
        zhu_cells.read_cells(gx, gy, g, [np.nan], [34.5])


# Correction 8: both LSN fragility rows always get the same mark ----------------------------------------------------
# Both fragilities come from one published table (Maurer, engrXiv 121, Table 3), and our chain fails parity with it.

@pytest.mark.parametrize("chain", ["[authors-maurer-2026-11-19]", "[authors-maurer-2026-12-01]", "[authors-x]",
                                   "[default-2026-11]", "[default-2026-12]", "", " [authors-maurer-2026-11-19] x"])
@pytest.mark.parametrize("house", [True, False])
def test_both_lsn_rows_match_the_outside_rule_alike_under_any_chain_id(chain, house):
    ces, glo = f"LSN + CES fragility{chain}", f"LSN + global fragility{chain}"
    assert outside.fragility_twin(ces) == glo and outside.fragility_twin(glo) == ces
    assert outside.is_outside(ces, "entry", house) == outside.is_outside(glo, "entry", house)
    assert not outside.is_outside(ces, "baseline", house) and not outside.is_outside(glo, "baseline", house)


def test_fragility_twin_is_none_for_other_rows():
    for n in (ZHU, GEM, "house model", "LSN + other fragility [authors-x]", "baseline: coin"):
        assert outside.fragility_twin(n) is None


@pytest.fixture(scope="module")
def four_lsn(ev):
    rng = np.random.default_rng(9)
    noisy = ev["s"] + rng.normal(0, 0.7, ev["s"].size)
    return [entry(ev, "house model", "target_probability", ev["p_target"], house=True),
            entry(ev, CES_AUTH, "score", noisy, house=True, conversion_id=CONV),
            entry(ev, GLO_AUTH, "score", noisy + 0.2, house=True, conversion_id=CONV),
            entry(ev, CES_DEF, "score", noisy + 0.1, house=True, conversion_id=CONV),
            entry(ev, GLO_DEF, "score", noisy + 0.3, house=True, conversion_id=CONV),
            entry(ev, ZHU, "other_probability", ev["p_trigger"])]


def lsn_mark(card, name):
    m = meta(card)[name]
    return (m.get("author_confirmed"), m.get("mark"), m.get("published_run"),
            card["metrics"][name] if isinstance(card.get("metrics", {}).get(name), str) else "scored")


@pytest.mark.parametrize("confirmed", [{}, {ZHU: True}, {CES_AUTH: True, GLO_AUTH: True},
                                       {CES_AUTH: False, GLO_AUTH: False, ZHU: True}])
def test_both_lsn_rows_print_the_same_mark_on_a_card(ev, four_lsn, confirmed):
    card = build_card(make_task(ev), four_lsn, synth.labels_of(ev), author_confirmed=confirmed)
    for chain in ("[authors-maurer-2026-11-19]", "[default-2026-11]"):
        ces, glo = f"LSN + CES fragility {chain}", f"LSN + global fragility {chain}"
        assert lsn_mark(card, ces) == lsn_mark(card, glo)
    if not confirmed.get(CES_AUTH):
        for n in (CES_AUTH, GLO_AUTH):
            assert card["metrics"][n] == outside.NOT_RUN and outside.NOT_RUN in meta(card)[n]["mark"]
            assert meta(card)[n]["published_run"].startswith("our run of a published model")
        assert {CES_AUTH, GLO_AUTH} <= set(card["author_rule"]["not_run"])
        assert not any({c["a"], c["b"]} & {CES_AUTH, GLO_AUTH} for c in card["comparisons"])
    else:
        assert isinstance(card["metrics"][CES_AUTH], dict) and isinstance(card["metrics"][GLO_AUTH], dict)
    for n in (CES_DEF, GLO_DEF):    # the default chain id claims no author settings: house, scored, unmarked
        assert "author_confirmed" not in meta(card)[n] and isinstance(card["metrics"][n], dict)


def test_both_lsn_rows_are_unmarked_with_the_option_off(ev, four_lsn):
    card = build_card(make_task(ev), four_lsn, synth.labels_of(ev))
    for n in (CES_AUTH, GLO_AUTH, CES_DEF, GLO_DEF):
        assert "author_confirmed" not in meta(card)[n] and "published_run" not in meta(card)[n]
        assert isinstance(card["metrics"][n], dict)
    assert "author_rule" not in card and outside.NOT_RUN not in str(card)


@pytest.mark.parametrize("confirmed", [{CES_AUTH: True}, {GLO_AUTH: True}, {CES_AUTH: True, GLO_AUTH: False},
                                       {CES_AUTH: False, GLO_AUTH: True}])
def test_confirming_one_lsn_row_without_its_twin_is_refused(ev, four_lsn, confirmed):
    """One published table, so the two rows are confirmed together or not at all: they can never split."""
    with pytest.raises(ValueError, match="same value"):
        build_card(make_task(ev), four_lsn, synth.labels_of(ev), author_confirmed=confirmed)


def test_gate_refuses_a_score_forged_onto_one_lsn_row(ev, four_lsn):
    card = build_card(make_task(ev), four_lsn, synth.labels_of(ev), author_confirmed={ZHU: True})
    forged = copy.deepcopy(card)
    forged["metrics"][GLO_AUTH] = copy.deepcopy(card["metrics"][ZHU])
    with pytest.raises(outside.AuthorGateError, match="metrics block"):
        outside.check_card(forged)

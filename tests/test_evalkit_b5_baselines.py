"""evalkit B5 (eval M4): PGV baselines from ShakeMap grids and the geographic null. Synthetic and planted only.

Loaders run on the synthetic fixtures in tests/fixtures; no real grid and no real label is read here.
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import geo_null as GN  # noqa: E402
from scoreboard.evalkit import pgv as PGV  # noqa: E402
from scoreboard.evalkit import site_memory as SM  # noqa: E402
from scoreboard.evalkit import synth  # noqa: E402
from scoreboard.evalkit.card import CardRefused, scan_aggregates  # noqa: E402
from scoreboard.evalkit.card import build_card as _build_card  # noqa: E402
from scoreboard.evalkit.spec import Entry, Labels, Task, card_sha256, site_list_sha256  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"
GRID = FIX / "shakemap_grid_synthetic.xml"
GRID_NOPGV = FIX / "shakemap_grid_nopgv_synthetic.xml"
NAN = np.nan
SRC = "synthetic v1"
EV = "syn-b5"
EV_DATE = "2022-01-01"
_SHAKING = {}


def build_card(task, entries, labels, **kw):
    """The kit card with the fixture's R10 shaking-only baseline (the task names earlier earthquakes)."""
    return _build_card(task, entries, labels, shaking_only=_SHAKING["entry"], **kw)


# Loader on fixtures -----------------------------------------------------------------------------------------------

def test_read_grid_fixture_nodes_bilinear_and_outside():
    g = PGV.read_grid(GRID)
    assert (g.event_id, g.version, g.magnitude, g.nlon, g.nlat) == ("syn-grid-1", "3", 6.2, 4, 3)
    assert g.units["PGV"] == "cm/s" and g.has("PGA") and len(g.sha256) == 64
    # PGV = 10 + 1000 (lon - 172) + 500 (lat + 43.60): linear, so bilinear is exact between nodes
    lon = np.array([172.00, 172.03, 172.015, 172.025, 171.99, 172.01])
    lat = np.array([-43.60, -43.58, -43.595, -43.585, -43.59, -43.57])
    v = PGV.sample(g, "PGV", lon, lat)
    want = 10 + 1000 * (lon - 172.0) + 500 * (lat + 43.60)
    assert np.allclose(v[:4], want[:4], atol=1e-9)
    assert np.isnan(v[4:]).all()                                      # west of the grid; north of the grid


def test_read_grid_from_zip_matches_xml(tmp_path):
    z = tmp_path / "grid.xml.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.write(GRID, "grid.xml")
    a, b = PGV.read_grid(GRID), PGV.read_grid(z)
    assert np.array_equal(a.fields["PGV"], b.fields["PGV"]) and a.sha256 != b.sha256   # hash of the stored file


def test_missing_pgv_is_said_never_filled():
    lon, lat = np.array([172.01, 172.02]), np.array([-43.59, -43.59])
    nopgv = PGV.read_grid(GRID_NOPGV)
    v, why = PGV.pgv_at(nopgv, lon, lat)
    assert why == "PGV NOT IN GRID" and np.isnan(v).all()
    v, why = PGV.pgv_at(None, lon, lat)
    assert why == "NO SHAKEMAP GRID" and np.isnan(v).all()
    g = PGV.read_grid(GRID)
    v, why = PGV.pgv_at(g, np.array([172.01, 180.0, 172.02]), np.array([-43.59, -43.59, -43.59]))
    assert why == "PGV MISSING AT 1 OF 3 SITES" and np.isnan(v[1]) and np.isfinite(v[[0, 2]]).all()


def test_an_empty_node_poisons_its_cells_only():
    text = GRID.read_text().replace("172.0100 -43.5900 6.0 21.0000 25.0000", "172.0100 -43.5900 6.0 21.0000 nan")
    assert "nan" in text
    g = PGV.parse_grid(text)
    v = PGV.sample(g, "PGV", np.array([172.005, 172.025]), np.array([-43.595, -43.595]))
    assert np.isnan(v[0]) and np.isfinite(v[1])                       # the cell next to the empty node only


def test_grid_refusals(tmp_path):
    text = GRID.read_text()
    with pytest.raises(PGV.PgvRefused, match="Kaikoura"):
        PGV.parse_grid(text.replace("syn-grid-1", "us1000778i"))
    with pytest.raises(PGV.PgvRefused, match="expected 5 x 3"):
        PGV.parse_grid(text.replace('nlon="4"', 'nlon="5"'))
    with pytest.raises(PGV.PgvRefused, match="not a ShakeMap"):
        PGV.parse_grid("<xml/>")
    with pytest.raises(PGV.PgvRefused, match="units"):
        PGV.pgv_at(PGV.parse_grid(text.replace('units="cm/s"', 'units="m/s"')), [172.01], [-43.59])
    z = tmp_path / "two.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.write(GRID, "a/grid.xml")
        f.write(GRID, "b/grid.xml")
    with pytest.raises(PGV.PgvRefused, match="exactly one"):
        PGV.read_grid(z)


# The PGV rows ------------------------------------------------------------------------------------------------------

def test_score_row_and_not_available_record():
    g = PGV.read_grid(GRID)
    ids = ("syn:a", "syn:b")
    b = PGV.score_baseline("syn-grid-1", ids, [172.01, 172.02], [-43.59, -43.59], g)
    assert b.entry.manifest.model == PGV.NAME_SCORE and b.entry.manifest.output_kind == "score"
    assert np.allclose(b.entry.p, [25.0, 35.0]) and b.summary["status"] == PGV.STATUS_BUILT
    assert b.summary["grid"]["sha256"] == g.sha256 and PGV.kit_built(b)
    na = PGV.score_baseline("syn-grid-1", ids, [172.01, 999.0], [-43.59, -43.59], g)
    assert na.entry is None and na.summary["status"].startswith("NOT AVAILABLE: PGV MISSING AT 1 OF 2 SITES")
    assert "no fill" in na.summary["status"] and PGV.kit_built(na)
    with pytest.raises(PGV.PgvRefused, match="not the scored earthquake"):
        PGV.score_baseline("other", ids, [172.01, 172.02], [-43.59, -43.59], g)


def test_fit_guards_and_earlier_only():
    rng = np.random.default_rng(5)
    pgv = np.exp(rng.normal(3, 0.6, 400))
    y = (rng.random(400) < expit(-6 + 1.6 * np.log(pgv))).astype(float)
    f = PGV.fit([PGV.Earlier("e1", pgv, y)], "scored")
    assert f["b"] > 0 and f["converged"] and f["fitted_on"] == ["e1"]
    with pytest.raises(PGV.PgvRefused, match="earlier ones only"):
        PGV.fit([PGV.Earlier("scored", pgv, y)], "scored")
    with pytest.raises(PGV.PgvRefused, match="separated"):
        PGV.fit([PGV.Earlier("e1", pgv, (pgv > 20).astype(float))], "scored")
    with pytest.raises(PGV.PgvRefused, match="not above 0"):
        PGV.fit([PGV.Earlier("e1", pgv, (rng.random(400) < expit(6 - 1.6 * np.log(pgv))).astype(float))], "s")
    gap = pgv.copy()
    gap[3] = NAN
    with pytest.raises(PGV.PgvRefused, match="PGV MISSING IN EARLIER EARTHQUAKE e1"):
        PGV.fit([PGV.Earlier("e1", gap, y)], "scored")


def test_task_id_and_shakemap_id_may_differ():
    """A real task id is a name ('feb2011'); a real grid carries a ComCat id. Synthetic grid, the id is only a name."""
    g = PGV.read_grid(GRID)
    named = PGV.Grid("usp000huvq", g.magnitude, g.version, g.lon0, g.lat0, g.dlon, g.dlat, g.nlon, g.nlat,
                     g.fields, g.units, g.sha256)
    ids, lon, lat = ("syn:a", "syn:b"), [172.01, 172.02], [-43.59, -43.59]
    with pytest.raises(PGV.PgvRefused, match="pass shakemap_id"):
        PGV.score_baseline("feb2011", ids, lon, lat, named)
    with pytest.raises(PGV.PgvRefused, match="expected grid id 'usp000other'"):
        PGV.score_baseline("feb2011", ids, lon, lat, named, shakemap_id="usp000other")
    with pytest.raises(PGV.PgvRefused, match="Kaikoura"):
        PGV.score_baseline("kaikoura", ids, lon, lat, named, shakemap_id="us1000778i")
    b = PGV.score_baseline("feb2011", ids, lon, lat, named, shakemap_id="usp000huvq")
    assert b.summary["scored_event"] == "feb2011" and b.summary["shakemap_id"] == "usp000huvq"
    assert b.summary["grid"]["event_id"] == "usp000huvq" and b.entry.manifest.exposure == {"feb2011": "unseen"}
    assert "feb2011" in b.entry.manifest.statement and "usp000huvq" in b.entry.manifest.statement
    assert PGV.card_problem(b, "feb2011", 0) is None
    assert "another earthquake" in PGV.card_problem(b, "usp000huvq", 0)
    na = PGV.score_baseline("feb2011", ids, [172.01, 999.0], lat, named, shakemap_id="usp000huvq")
    assert na.entry is None and na.summary["shakemap_id"] == "usp000huvq"
    rng = np.random.default_rng(5)
    pgv = np.exp(rng.normal(3.0, 0.6, 300))
    y = (rng.random(300) < expit(-6 + 1.8 * np.log(pgv))).astype(float)
    with pytest.raises(PGV.PgvRefused, match="in the fit set"):
        PGV.probability_baseline("feb2011", ids, lon, lat, named, [PGV.Earlier("usp000huvq", pgv, y)],
                                 shakemap_id="usp000huvq")
    pr = PGV.probability_baseline("feb2011", ids, lon, lat, named, [PGV.Earlier("sep2010", pgv, y)],
                                  shakemap_id="usp000huvq")
    assert pr.summary["shakemap_id"] == "usp000huvq" and "scored earthquake feb2011" in pr.entry.manifest.statement


# Card fixture: a synthetic earthquake whose labels follow PGV, a site term and a spatial field ----------------------

def _grid_for(ev_id: str, scale: float, rng) -> PGV.Grid:
    """A synthetic grid on 'lon' = x km, 'lat' = y km, 1 km nodes, smooth ln PGV."""
    n = 22
    xs = np.arange(n, dtype=float)
    gx, gy = np.meshgrid(xs, xs)
    lnpgv = 3.0 + 0.6 * np.sin(gx / 4.0) + 0.5 * np.cos(gy / 5.0) + rng.normal(0, 0.05, (n, n))
    return PGV.Grid(ev_id, 6.5, "1", 0.0, 0.0, 1.0, 1.0, n, n, {"PGV": scale * np.exp(lnpgv)}, {"PGV": "cm/s"},
                    "f" * 64)


@pytest.fixture(scope="module")
def world():
    ev = synth.synthetic_event(700, 41, unlabelled_share=0.1)
    rng = np.random.default_rng(42)
    n = ev["s"].size
    lon, lat = ev["x_m"] / 1000.0, ev["y_m"] / 1000.0
    grids = {k: _grid_for(k, sc, rng) for k, sc in (("syn-early-0", 0.7), ("syn-early-1", 1.0), (EV, 1.2))}

    def draw(k, icpt):
        lp = np.log(PGV.sample(grids[k], "PGV", lon, lat))
        return (rng.random(n) < expit(icpt + 1.5 * (lp - 3.0) + ev["slope"] * ev["s"] + ev["field"])).astype(float)

    earlier_geo, earlier_pgv = [], []
    for k, (date, icpt) in enumerate((("2020-01-01", -1.0), ("2021-01-01", -0.6))):
        eid = f"syn-early-{k}"
        y = draw(eid, icpt)
        lab = rng.random(n) < 0.7
        yl = np.where(lab, y, NAN)
        earlier_geo.append(SM.Earlier(eid, date, Labels(ev["site_ids"], yl, ev["x_m"], ev["y_m"]), SRC))
        earlier_pgv.append(PGV.Earlier(eid, PGV.sample(grids[eid], "PGV", lon[lab], lat[lab]), y[lab]))
    y = draw(EV, -0.8)
    y[rng.random(n) < 0.1] = NAN
    labels = Labels(ev["site_ids"], y, ev["x_m"], ev["y_m"])
    task = Task(event_id=EV, target="ejecta seen at the site", label_source=SRC, label_rule_sha256="1" * 64,
                site_list_sha256=site_list_sha256(ev["site_ids"]), track="forward", division="closed",
                tile_m=2000.0, clip=1e-6, seed=3, label_caveats="synthetic labels", n_boot=60,
                earlier_base_rate=0.3, earlier_label_sets=2)
    model = Entry(synth.manifest("m", "score", EV), ev["site_ids"], ev["s"] + rng.normal(0, 1.0, n))
    _SHAKING["entry"] = Entry(synth.manifest("shaking", "target_probability", EV), ev["site_ids"],
                              np.clip(0.2 + 0.1 * ev["field"], 0.01, 0.9))
    return ev, lon, lat, grids, earlier_geo, earlier_pgv, labels, task, model


def test_default_card_is_unchanged_when_b5_is_off(world):
    ev, lon, lat, grids, eg, ep, labels, task, model = world
    plain = build_card(task, [model], labels)
    off = build_card(task, [model], labels, pgv=None, geo_null=None)
    assert card_sha256(plain) == card_sha256(off)
    assert "pgv" not in plain and "geographic_null" not in plain
    assert not any("B5" in n for n in plain["notes"])


def test_card_takes_pgv_rows_and_planted_pgv_signal_shows(world):
    ev, lon, lat, grids, eg, ep, labels, task, model = world
    sc = PGV.score_baseline(EV, ev["site_ids"], lon, lat, grids[EV])
    card = build_card(task, [model], labels, pgv=sc)
    meta = {e["name"]: e for e in card["entries"]}
    assert meta[PGV.NAME_SCORE]["role"] == "baseline" and meta[PGV.NAME_SCORE]["exposure"] == "UNSEEN"
    assert "R3" in card["metrics"][PGV.NAME_SCORE]["probability_metrics"]
    assert card["metrics"][PGV.NAME_SCORE]["auc"]["value"] > 0.6           # labels follow PGV by construction
    assert card["pgv"]["status"] == PGV.STATUS_BUILT and card["pgv"]["grid"]["sha256"] == "f" * 64
    pr = PGV.probability_baseline(EV, ev["site_ids"], lon, lat, grids[EV], ep)
    card2 = build_card(task, [model], labels, pgv=pr)
    assert "log_loss" in card2["metrics"][PGV.NAME_PROB] and card2["pgv"]["fit"]["fitted_on"] == [e.event_id for e in ep]
    assert not any({c["a"], c["b"]} <= {PGV.NAME_PROB, "baseline: coin", "baseline: earlier-earthquake rate"}
                   for c in card2["comparisons"])                          # no pair of two baselines (R10)
    scan_aggregates(card2, labels.site_ids)


def test_card_prints_not_available_and_adds_no_row(world):
    ev, lon, lat, grids, eg, ep, labels, task, model = world
    lon2 = lon.copy()
    lon2[:3] = 500.0                                                        # three sites off the grid
    na = PGV.score_baseline(EV, ev["site_ids"], lon2, lat, grids[EV])
    card = build_card(task, [model], labels, pgv=na)
    assert PGV.NAME_SCORE not in card["metrics"] and PGV.NAME_SCORE not in {e["name"] for e in card["entries"]}
    assert card["pgv"]["status"].startswith("NOT AVAILABLE: PGV MISSING AT 3 OF 700 SITES")
    assert any("NOT AVAILABLE" in n and "no fill" in n for n in card["notes"])
    gap = [PGV.Earlier(e.event_id, np.where(np.arange(e.pgv.size) == 0, NAN, e.pgv), e.y) for e in ep]
    na2 = PGV.probability_baseline(EV, ev["site_ids"], lon, lat, grids[EV], gap)
    assert na2.entry is None and "PGV MISSING IN EARLIER EARTHQUAKE" in na2.summary["status"]
    no_grid = PGV.score_baseline(EV, ev["site_ids"], lon, lat, None)
    assert build_card(task, [model], labels, pgv=no_grid)["pgv"]["reason"] == "NO SHAKEMAP GRID"


def test_card_refuses_pgv_imitations(world):
    ev, lon, lat, grids, eg, ep, labels, task, model = world
    sc = PGV.score_baseline(EV, ev["site_ids"], lon, lat, grids[EV])
    with pytest.raises(CardRefused, match="PgvBaseline"):
        build_card(task, [model], labels, pgv=sc.entry)
    forged = PGV.PgvBaseline(Entry(sc.entry.manifest, sc.entry.site_ids, sc.entry.p), dict(sc.summary))
    with pytest.raises(CardRefused, match="provenance"):
        build_card(task, [model], labels, pgv=forged)
    edited = PGV.score_baseline(EV, ev["site_ids"], lon, lat, grids[EV])
    edited.summary["status"] = "edited"
    with pytest.raises(CardRefused, match="provenance"):
        build_card(task, [model], labels, pgv=edited)
    pr = PGV.probability_baseline(EV, ev["site_ids"], lon, lat, grids[EV], ep)
    with pytest.raises(CardRefused, match="names 1"):
        build_card(Task(**{**task.to_dict(), "earlier_label_sets": 1}), [model], labels, pgv=pr)
    other = PGV.score_baseline("syn-grid-x", ev["site_ids"], lon, lat,
                               PGV.Grid("syn-grid-x", 6.0, "1", 0, 0, 1, 1, 22, 22, grids[EV].fields,
                                        {"PGV": "cm/s"}, "e" * 64))
    with pytest.raises(CardRefused, match="another earthquake"):
        build_card(task, [model], labels, pgv=other)


# The geographic null ----------------------------------------------------------------------------------------------

def test_geo_null_rule_on_a_hand_case():
    early = Labels(("k1", "k2", "k3"), np.array([1.0, 0.0, NAN]), np.array([0.0, 50.0, 100.0]), np.zeros(3))
    sites = SM.Sites(("a", "b", "c"), np.array([0.0, 30.0, 400.0]), np.array([0.0, 40.0, 0.0]))
    g = GN.build("eq-2", "2021-01-01", sites, [SM.Earlier("eq-1", "2020-01-01", early, SRC)])
    assert np.allclose(g.entry.p, [0.0, -50.0, -400.0])                    # only the labelled yes k1 is known
    assert g.entry.manifest.output_kind == "score" and g.entry.manifest.exposure == {"eq-2": "seen_place"}
    assert g.summary["n_known_yes"] == 1 and "Hijmans 2012" in g.summary["rule_quote"]
    with pytest.raises(GN.GeoNullRefused, match="scored labels never enter"):
        GN.build("eq-2", "2021-01-01", early, [SM.Earlier("eq-1", "2020-01-01", early, SRC)])
    with pytest.raises(GN.GeoNullRefused, match="no labelled yes"):
        GN.build("eq-2", "2021-01-01", sites, [SM.Earlier("eq-1", "2020-01-01",
                                            Labels(("k1",), np.array([0.0]), np.zeros(1), np.zeros(1)), SRC)])
    with pytest.raises(GN.GeoNullRefused, match="none equal"):
        GN.build("eq-1", "2021-01-01", sites, [SM.Earlier("eq-1", "2020-01-01", early, SRC)])
    assert g.summary["scored_date"] == "2021-01-01" and g.summary["earlier_dates"] == {"eq-1": "2020-01-01"}


def test_geo_null_refuses_a_later_earthquake():
    """A later earthquake's yes labels must never become known liquefaction (the reviewer's feb2011 / jun2011 case)."""
    early = Labels(("k1",), np.array([1.0]), np.zeros(1), np.zeros(1))
    sites = SM.Sites(("a",), np.zeros(1), np.zeros(1))
    for later in ("2011-06-13", "2011-02-22"):                              # after, and on the same day
        with pytest.raises(GN.GeoNullRefused, match="strictly before"):
            GN.build("feb2011", "2011-02-22", sites, [SM.Earlier("jun2011", later, early, SRC)])
    with pytest.raises(GN.GeoNullRefused, match="ISO"):
        GN.build("feb2011", "22 Feb 2011", sites, [SM.Earlier("sep2010", "2010-09-04", early, SRC)])
    with pytest.raises(GN.GeoNullRefused, match="ISO"):
        GN.build("feb2011", "2011-02-22", sites, [SM.Earlier("sep2010", "Sep 2010", early, SRC)])
    ok = GN.build("feb2011", "2011-02-22", sites, [SM.Earlier("sep2010", "2010-09-04", early, SRC)])
    assert ok.summary["earlier_dates"] == {"sep2010": "2010-09-04"}


def test_matched_sites_pairs_by_distance_within_tolerance():
    early = Labels(("k",), np.array([1.0]), np.zeros(1), np.zeros(1))
    # yes at 100 m and 1000 m; no at 110, 105, 5000 m. tr 0.33: y1 pairs with n2 (105, closest), y2 finds none
    lab = Labels(("y1", "y2", "n1", "n2", "n3"), np.array([1, 1, 0, 0, 0], float),
                 np.array([100.0, 1000.0, 110.0, 105.0, 5000.0]), np.zeros(5))
    m = GN.matched_sites(lab, [SM.Earlier("eq-1", "2020-01-01", early, SRC)])
    assert m["site_ids"] == ("n2", "y1") and m["n_pairs"] == 1 and (m["n_yes_in"], m["n_no_in"]) == (2, 3)
    wide = GN.matched_sites(lab, [SM.Earlier("eq-1", "2020-01-01", early, SRC)], tr=5.0)
    assert wide["n_pairs"] == 2 and set(wide["site_ids"]) == {"y1", "y2", "n2", "n1"}


def test_card_geo_null_and_planted_copy_gets_no_calibrated_credit(world):
    ev, lon, lat, grids, eg, ep, labels, task, model = world
    g = GN.build(EV, EV_DATE, SM.sites_of(labels), eg)
    copy = Entry(synth.manifest("copy of the null", "score", EV), ev["site_ids"], 3.0 * g.entry.p + 1.0)
    card = build_card(task, [model, copy], labels, geo_null=g)
    meta = {e["name"]: e for e in card["entries"]}
    assert meta[GN.NAME]["role"] == "baseline" and meta[GN.NAME]["exposure"].startswith("SEEN PLACE")
    assert card["metrics"][GN.NAME]["auc"]["value"] > 0.55                 # shared spatial field: the null has skill
    cal = card["geographic_null"]["calibrated_auc"]
    assert set(cal) == {"m", "copy of the null"}                           # entries only, never a baseline
    assert cal["copy of the null"]["value"] == pytest.approx(0.5, abs=1e-12)   # planted copy: no credit
    assert cal["m"]["value"] == pytest.approx(card["metrics"]["m"]["auc"]["value"]
                                              - card["metrics"][GN.NAME]["auc"]["value"] + 0.5)
    assert cal["m"]["lo"] <= cal["m"]["value"] <= cal["m"]["hi"]
    assert card["geographic_null"]["rule_quote"] == GN.RULE_QUOTE
    assert any(GN.INTERPRETATION_NOTE in n for n in card["notes"])
    scan_aggregates(card, labels.site_ids)


def test_card_refuses_geo_null_misuse(world):
    ev, lon, lat, grids, eg, ep, labels, task, model = world
    g = GN.build(EV, EV_DATE, SM.sites_of(labels), eg)
    with pytest.raises(CardRefused, match="GeoNull"):
        build_card(task, [model], labels, geo_null=g.entry)
    forged = GN.GeoNull(Entry(g.entry.manifest, g.entry.site_ids, g.entry.p), dict(g.summary))
    with pytest.raises(CardRefused, match="provenance"):
        build_card(task, [model], labels, geo_null=forged)
    wrong = [SM.Earlier(e.event_id, e.date, e.labels, "another source") for e in eg]
    with pytest.raises(CardRefused, match="label source"):
        build_card(task, [model], labels, geo_null=GN.build(EV, EV_DATE, SM.sites_of(labels), wrong))
    with pytest.raises(CardRefused, match="names 1"):
        build_card(Task(**{**task.to_dict(), "earlier_label_sets": 1}), [model], labels, geo_null=g)
    with pytest.raises(CardRefused, match="no earlier earthquake"):
        build_card(Task(**{**task.to_dict(), "earlier_base_rate": None, "earlier_label_sets": 0}), [model], labels,
                   geo_null=g)
    with pytest.raises(CardRefused, match="another earthquake"):
        build_card(task, [model], labels, geo_null=GN.build("syn-x", EV_DATE, SM.sites_of(labels), eg))

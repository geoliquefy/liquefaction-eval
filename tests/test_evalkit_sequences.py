"""evalkit sequences (rule R25, item D6, eval M1): window, shock list, task kind, mark, per-shock combine.

Synthetic catalogues and synthetic sites, plus a fixture cut from the public ComCat and GeoNet catalogues
(tests/fixtures/sequence_*; scripts/cut_sequence_fixture.py). No label is read. The fixture sites are a synthetic
7 x 7 grid over the Christchurch urban box, not CPT or label sites. No em dashes.
"""
import ast
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoreboard.evalkit import sequences as S  # noqa: E402
from scoreboard.evalkit.card import scan_aggregates  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
T0 = datetime(2030, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
SITE_LAT = np.array([-40.00, -40.01, -40.02, -40.30])
SITE_LON = np.array([175.00, 175.01, 175.02, 175.00])


def ev(i, minutes, mag, mag_type="mww", lat=-40.0, lon=175.0, depth=8.0, cat="comcat", etype="earthquake"):
    return S.Event(cat, f"{cat[0]}{i}", T0 + timedelta(minutes=minutes), lat, lon, depth, mag, mag_type, etype)


def window(days_after=3):
    return S.attribution_window(T0 + timedelta(days=days_after), T0)


# ------------------------------------------------------------------------------------------------ geometry

def test_radius_follows_d6():
    assert S.radius_km(6.0) == pytest.approx(30.0)
    assert S.radius_km(5.0) == pytest.approx(30 * 10 ** -0.5)
    assert S.radius_km(9.0) == 200.0


def test_hypocentral_distance_adds_depth():
    e = ev(1, 0, 6.0, depth=10.0)
    assert S.hypocentral_km(e, [-40.0], [175.0])[0] == pytest.approx(10.0)
    one_deg = 2 * math.pi * S.EARTH_RADIUS_KM / 360
    assert S.hypocentral_km(ev(1, 0, 6.0, depth=0.0), [-41.0], [175.0])[0] == pytest.approx(one_deg, rel=1e-9)


# ------------------------------------------------------------------------------------------------ window

def test_window_falls_back_to_thirty_days_before_the_origin():
    w = window()
    assert w.start == T0 - timedelta(days=30) and w.end == T0 + timedelta(days=3)
    assert "30 days" in w.rule


def test_window_starts_at_the_previous_survey_end():
    w = S.attribution_window(T0 + timedelta(days=3), T0, previous_survey_end=T0 - timedelta(days=90))
    assert w.start == T0 - timedelta(days=90) and "previous survey" in w.rule


def test_window_edges_are_start_exclusive_end_inclusive():
    w = window()
    assert not w.contains(w.start) and w.contains(w.end)


def test_bad_windows_are_refused():
    with pytest.raises(S.SequenceError):
        S.attribution_window(T0 - timedelta(days=1), T0)
    with pytest.raises(S.SequenceError):
        S.attribution_window(T0 + timedelta(days=1), T0, previous_survey_end=T0)


# ------------------------------------------------------------------------------------------------ shock list and kind

def test_one_shock_is_a_single_task():
    cat = [ev(1, 0, 6.0), ev(2, 60, 4.9)]
    plans = S.plan_tasks(S.shock_list(cat, window(), SITE_LAT, SITE_LON))
    assert len(plans) == 1 and plans[0].kind == S.SINGLE and plans[0].shock_ids == ("c1",)
    assert plans[0].origin == T0


def test_two_shocks_make_a_sequence_with_the_first_origin():
    cat = [ev(2, 80, 6.0), ev(1, 0, 5.3)]           # caller order is not time order
    plans = S.plan_tasks(S.shock_list(cat, window(), SITE_LAT, SITE_LON), task_target="yes = ejecta")
    assert len(plans) == 1
    p = plans[0]
    assert p.kind == S.SEQUENCE and p.origin == T0 and p.shock_ids == ("c1", "c2")
    assert p.target == S.SEQUENCE_TARGET and p.summary()["card_word"] == "SEQUENCE"


def test_attributing_source_gives_one_task_per_shock():
    cat = [ev(1, 0, 5.3), ev(2, 80, 6.0)]
    plans = S.plan_tasks(S.shock_list(cat, window(), SITE_LAT, SITE_LON), attributes_per_shock=True)
    assert [p.kind for p in plans] == [S.PER_SHOCK, S.PER_SHOCK]
    assert [p.shock_ids for p in plans] == [("c1",), ("c2",)]
    assert plans[1].origin == T0 + timedelta(minutes=80)


def test_unsorted_input_gives_time_ordered_shocks_and_the_earliest_origin():
    cat = [ev(3, 80, 6.0), ev(2, 0, 5.3), ev(1, 0, 5.1)]      # a tie on time is broken by event id
    sl = S.shock_list(cat, window(), SITE_LAT, SITE_LON)
    assert [s.event.event_id for s in sl.shocks] == ["c1", "c2", "c3"]
    p = S.plan_tasks(sl)[0]
    assert p.origin == T0 and p.shock_ids == ("c1", "c2", "c3")
    per = S.plan_tasks(sl, attributes_per_shock=True)
    assert [q.origin for q in per] == [T0, T0, T0 + timedelta(minutes=80)]
    hand_built = S.ShockList(sl.window, tuple(reversed(sl.shocks)), (), sl.n_sites)   # plan_tasks sorts too
    assert S.plan_tasks(hand_built)[0].shock_ids == ("c1", "c2", "c3")


def test_fallback_window_is_taken_to_its_fixed_point():
    # Main shock at T0; shocks 29 and 31 days before. From T0 the window holds c1 and c2. Rebuilt from the origin
    # of c1 (T0 - 29 d) it also holds c0. The kit refuses the first window; the fixed point holds all three.
    cat = [ev(0, -60 * 24 * 31, 5.5), ev(1, -60 * 24 * 29, 5.5), ev(2, 0, 6.0)]
    sl = S.shock_list(cat, window(), SITE_LAT, SITE_LON)
    assert [s.event.event_id for s in sl.shocks] == ["c1", "c2"]
    with pytest.raises(S.SequenceError, match="anchor"):
        S.plan_tasks(sl)
    fixed = S.sequence_shock_list(cat, T0 + timedelta(days=3), T0, SITE_LAT, SITE_LON)
    assert [s.event.event_id for s in fixed.shocks] == ["c0", "c1", "c2"]
    assert fixed.window.anchor == T0 - timedelta(days=31)
    p = S.plan_tasks(fixed)[0]
    assert p.kind == S.SEQUENCE and p.origin == T0 - timedelta(days=31)
    # A previous survey end fixes the window: one pass, no re-anchoring, and plan_tasks does not refuse.
    prev = S.sequence_shock_list(cat, T0 + timedelta(days=3), T0, SITE_LAT, SITE_LON,
                                 previous_survey_end=T0 - timedelta(days=30))
    assert [s.event.event_id for s in prev.shocks] == ["c1", "c2"] and prev.window.anchor is None
    assert S.plan_tasks(prev)[0].origin == T0 - timedelta(days=29)
    # A long run of shocks 20 days apart walks back; max_iter bounds it.
    run = [ev(i, -60 * 24 * 20 * i, 5.5) for i in range(8)]
    with pytest.raises(S.SequenceError, match="did not settle"):
        S.sequence_shock_list(run, T0 + timedelta(days=3), T0, SITE_LAT, SITE_LON, max_iter=3)
    assert len(S.sequence_shock_list(run, T0 + timedelta(days=3), T0, SITE_LAT, SITE_LON).shocks) == 8


def test_shocks_out_of_reach_out_of_window_or_not_earthquakes_do_not_count():
    cat = [ev(1, 0, 6.0),
           ev(2, 30, 5.0, lat=-41.0),                                  # 111 km away, R 9.5 km
           ev(3, -60 * 24 * 40, 6.5),                                  # 40 days before the origin
           ev(4, 60 * 24 * 4, 6.5),                                    # after the survey end
           ev(5, 90, 5.5, etype="quarry blast")]
    sl = S.shock_list(cat, window(), SITE_LAT, SITE_LON)
    assert [s.event.event_id for s in sl.shocks] == ["c1"]


def test_coverage_counts_only_sites_inside_the_radius():
    sl = S.shock_list([ev(1, 0, 5.0, depth=5.0)], window(), SITE_LAT, SITE_LON)
    assert sl.shocks[0].n_sites_covered == 3            # the fourth site is 33 km south


def test_a_non_mw_magnitude_of_five_or_more_is_unresolved_and_refused():
    cat = [ev(1, 0, 6.0), ev(2, 10, 5.2, mag_type="mb")]
    sl = S.shock_list(cat, window(), SITE_LAT, SITE_LON)
    assert not sl.complete and [e.event_id for e in sl.unresolved] == ["c2"]
    with pytest.raises(S.SequenceError, match="UNRESOLVED"):
        S.plan_tasks(sl)
    resolved = S.shock_list(cat, window(), SITE_LAT, SITE_LON, census_magnitudes={"c2": 5.1})
    assert S.plan_tasks(resolved)[0].kind == S.SEQUENCE
    assert resolved.shocks[1].magnitude_source == S.FROM_CALLER
    not_census = S.shock_list(cat, window(), SITE_LAT, SITE_LON, census_magnitudes={"c2": None})
    assert S.plan_tasks(not_census)[0].kind == S.SINGLE


def test_a_non_mw_magnitude_below_five_does_not_block_by_default():
    cat = [ev(1, 0, 6.0), ev(2, 10, 4.8, mag_type="mb"), ev(3, 20, 4.5, mag_type="mwr")]   # 4.5 is in the band
    sl = S.shock_list(cat, window(), SITE_LAT, SITE_LON)
    assert sl.complete and S.plan_tasks(sl)[0].kind == S.SINGLE
    # With a broader catalogue, rows in [unresolved_from, 5.0) without a detail magnitude become UNRESOLVED.
    wide = S.shock_list(cat, window(), SITE_LAT, SITE_LON, unresolved_from=4.5)
    assert [e.event_id for e in wide.unresolved] == ["c2", "c3"]
    with pytest.raises(S.SequenceError, match="UNRESOLVED"):
        S.plan_tasks(wide)
    fixed = S.shock_list(cat, window(), SITE_LAT, SITE_LON, unresolved_from=4.5,
                         detail_magnitudes={"c2": 4.9}, census_magnitudes={"c3": 5.0})
    assert fixed.complete and [s.event.event_id for s in fixed.shocks] == ["c1", "c3"]


def test_summary_magnitude_is_marked_a_lower_bound_and_detail_wins():
    cat = [ev(1, 0, 4.95, mag_type="mwr")]
    assert not S.shock_list(cat, window(), SITE_LAT, SITE_LON).shocks
    sl = S.shock_list(cat, window(), SITE_LAT, SITE_LON, detail_magnitudes={"c1": 5.02})
    assert sl.shocks[0].census_mw == 5.02 and sl.shocks[0].magnitude_source == S.FROM_DETAIL
    sl2 = S.shock_list([ev(1, 0, 5.3, mag_type="mwr")], window(), SITE_LAT, SITE_LON)
    assert sl2.shocks[0].magnitude_source == S.FROM_SUMMARY


def test_an_empty_window_is_refused():
    with pytest.raises(S.SequenceError, match="no census shock"):
        S.plan_tasks(S.shock_list([ev(1, 0, 4.0)], window(), SITE_LAT, SITE_LON))


def test_geonet_rows_cannot_build_the_shock_list():
    with pytest.raises(S.SequenceError, match="ComCat"):
        S.shock_list([ev(1, 0, 6.0, cat="geonet")], window(), SITE_LAT, SITE_LON)


def test_census_from_detail_takes_the_largest_mw_family_and_skips_mwp():
    d = {"properties": {"mag": 5.9, "magType": "mww", "products": {
        "origin": [{"properties": {"magnitude-type": "mw", "magnitude": "5.97"}},
                   {"properties": {"magnitude-type": "mwp", "magnitude": "6.4"}},
                   {"properties": {"magnitude-type": "me", "magnitude": "6.33"}}],
        "moment-tensor": [{"properties": {"derived-magnitude-type": "Mwc", "derived-magnitude": "5.95"}}]}}}
    assert S.census_from_detail(d) == pytest.approx(5.97)
    assert S.census_from_detail({"properties": {"mag": 5.0, "magType": "mb", "products": {}}}) is None


# ------------------------------------------------------------------------------------------------ mark

def test_mark_after_an_earlier_shock():
    cat = [ev(1, 0, 5.3), ev(2, 80, 6.0)]
    sl = S.shock_list(cat, window(), SITE_LAT, SITE_LON)
    seq = S.plan_tasks(sl)[0]
    assert not S.after_earlier_shock(seq, T0 - timedelta(days=2))
    assert S.after_earlier_shock(seq, T0 + timedelta(minutes=30))
    assert S.after_earlier_shock(seq, T0 - timedelta(days=2), [T0 + timedelta(minutes=5)])
    first, second = S.plan_tasks(sl, attributes_per_shock=True)
    assert not S.after_earlier_shock(first, T0 + timedelta(minutes=30))
    assert S.after_earlier_shock(second, T0 + timedelta(minutes=30))
    assert not S.after_earlier_shock(second, T0 - timedelta(minutes=1))
    single = S.plan_tasks(S.shock_list([ev(1, 0, 6.0)], window(), SITE_LAT, SITE_LON))[0]
    assert not S.after_earlier_shock(single, T0 + timedelta(days=1))
    assert not S.after_earlier_shock(second, T0)                    # exactly at the first shock: not after it


def test_mark_counts_from_the_first_earlier_shock_with_three_shocks():
    cat = [ev(1, 0, 5.3), ev(2, 60, 5.4), ev(3, 120, 6.0)]
    third = S.plan_tasks(S.shock_list(cat, window(), SITE_LAT, SITE_LON), attributes_per_shock=True)[2]
    assert S.after_earlier_shock(third, T0 + timedelta(minutes=30))  # between the first and the second shock
    assert not S.after_earlier_shock(third, T0 - timedelta(minutes=1))


# ------------------------------------------------------------------------------------------------ combine

def test_combine_runs_takes_the_largest_over_runs_that_include_the_site():
    runs = {"a": np.array([0.1, 0.5, np.nan, np.nan]), "b": np.array([0.3, 0.2, 0.4, np.nan])}
    out = S.combine_runs(runs)
    np.testing.assert_allclose(out[:3], [0.3, 0.5, 0.4])
    assert np.isnan(out[3])
    masked = S.combine_runs(runs, include={"b": np.array([False, True, True, True])})
    np.testing.assert_allclose(masked[:3], [0.1, 0.5, 0.4])
    with pytest.raises(S.SequenceError):
        S.combine_runs({})
    with pytest.raises(S.SequenceError):
        S.combine_runs(runs, include={"z": np.ones(4, bool)})


def _script_tree():
    return ast.parse((ROOT / "scripts" / "run_g6b_p3v2.py").read_text())


def test_p_max_matches_the_old_two_shock_script_exactly():
    """The moved function equals scripts/run_g6b_p3v2.py p_of, read from the script's own source."""
    fn = next(n for n in _script_tree().body if isinstance(n, ast.FunctionDef) and n.name == "p_of")
    rng = np.random.default_rng(7)
    mu, sig_r = -0.4, 0.55
    ns = {"np": np, "ndtr": ndtr, "SIG_R": sig_r, "MU": mu}
    exec(compile(ast.Module([fn], []), "p_of", "exec"), ns)
    lks = [rng.normal(-0.5, 0.8, 300), rng.normal(-0.2, 0.8, 300)]
    se = rng.uniform(0.1, 0.4, 300)
    np.testing.assert_array_equal(S.p_max_over_shocks(lks, se, mu, sig_r), ns["p_of"](lks, se))
    np.testing.assert_allclose(S.combine_runs({str(k): S.p_max_over_shocks([lk], se, mu, sig_r)
                                               for k, lk in enumerate(lks)}), ns["p_of"](lks, se))


def test_legacy_two_shock_map_is_the_old_fore_map():
    node = next(n for n in _script_tree().body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "FORE")
    assert ast.literal_eval(node.value) == S.LEGACY_TWO_SHOCK


# ------------------------------------------------------------------------------------------------ GeoNet check

def test_cross_check_matches_and_lists_geonet_only_rows():
    cc = [ev(1, 0, 6.0), ev(2, 80, 5.5)]
    gn = [ev(11, 0.2, 6.1, mag_type="Mw", lat=-40.02, cat="geonet"),
          ev(12, 200, 5.4, mag_type="ML", cat="geonet"),                # no ComCat row
          ev(13, 300, 5.6, mag_type="ML", lat=-42.0, cat="geonet")]     # out of reach
    sl = S.shock_list(cc, window(), SITE_LAT, SITE_LON)
    x = S.cross_check(sl, cc, gn, SITE_LAT, SITE_LON)
    assert [m[:2] for m in x.matched] == [("c1", "g11")]
    assert x.comcat_only == ("c2",) and x.geonet_only == ("g12",)


def test_cross_check_does_not_match_a_geonet_row_beyond_max_km():
    cc = [ev(1, 0, 6.0)]
    gn = [ev(11, 0.1, 6.0, mag_type="Mw", lat=-40.5, cat="geonet")]   # 6 s apart, about 56 km south
    x = S.cross_check(S.shock_list(cc, window(), SITE_LAT, SITE_LON), cc, gn, SITE_LAT, SITE_LON)
    assert x.matched == () and x.comcat_only == ("c1",) and x.geonet_only == ("g11",)


# ------------------------------------------------------------------------------------------------ readers and output

def test_readers_refuse_duplicate_ids(tmp_path):
    head = "time,latitude,longitude,depth,mag,magType,id,type\n"
    row = "2030-01-10T12:00:00.000Z,-40,175,8,6,mww,x1,earthquake\n"
    p = tmp_path / "c.csv"
    p.write_text(head + row + row)
    with pytest.raises(S.SequenceError):
        S.read_comcat_csv([p])
    g = tmp_path / "g.txt"
    line = "1|2030-01-10T12:00:00|-40|175|8|GNS|GNS|GNS|1|Mw|6.0|GNS|x|earthquake\n"
    g.write_text("#h\n" + line + line)
    with pytest.raises(S.SequenceError):
        S.read_geonet_text([g])


def test_plan_summary_passes_the_aggregate_scan():
    plan = S.plan_tasks(S.shock_list([ev(1, 0, 5.3), ev(2, 80, 6.0)], window(), SITE_LAT, SITE_LON))[0]
    scan_aggregates({"sequence": plan.summary()})
    assert set(plan.summary()) == {"kind", "origin_utc", "shocks", "window_shocks", "target", "card_word"}


# ------------------------------------------------------------------------------------------------ public fixture

def _fixture():
    cc = S.read_comcat_csv([FIX / "sequence_comcat_canterbury_2010_2011.csv"])
    gn = S.read_geonet_text([FIX / "sequence_geonet_canterbury_2010_2011.txt"])
    det = json.loads((FIX / "sequence_comcat_detail_census.json").read_text())["census_mw"]
    la, lo = np.meshgrid(np.linspace(-43.60, -43.48, 7), np.linspace(172.55, 172.78, 7))
    return cc, gn, det, la.ravel(), lo.ravel()


def _plan(origin, census=None):
    cc, gn, det, la, lo = _fixture()
    org = S._utc(origin)
    sl = S.sequence_shock_list(cc, org + timedelta(days=3), org, la, lo,
                               census_magnitudes=census, detail_magnitudes=det)
    return sl, S.cross_check(sl, cc, gn, la, lo)


# GeoNet-only M >= 5 rows per window. Each ComCat shock already has its own GeoNet row within 1 s; these are other
# GeoNet events 34 to 75 s later (likely coda aftershocks that the ComCat M >= 5 list lacks), not second solutions.
GEONET_ONLY = {"sep2010": ("3669651",), "feb2011": ("3510134",), "jun2011": ("3586848",), "dec2011": ()}


def test_jun_2011_is_a_sequence_credited_to_foreshock_and_main_shock():
    sl, x = _plan("2011-06-13T02:20:49Z")
    assert [s.event.event_id for s in sl.shocks] == ["usp000j338", "usp000j33b"]
    assert [round(s.census_mw, 2) for s in sl.shocks] == [5.2, 5.97]
    assert sl.shocks[0].n_sites_covered < sl.n_sites == sl.shocks[1].n_sites_covered
    p = S.plan_tasks(sl)[0]
    assert p.kind == S.SEQUENCE
    assert p.origin == datetime(2011, 6, 13, 1, 1, 0, 280000, tzinfo=timezone.utc)
    assert sl.shocks[1].event.time.strftime("%H:%M:%S") == "02:20:49"     # R25: the larger at 02:20:49 UTC
    assert [(m[0], m[1], m[2]) for m in x.matched] == [("usp000j338", "3528810", 5.3), ("usp000j33b", "3528839", 6.0)]
    per = S.plan_tasks(sl, attributes_per_shock=True)
    assert S.after_earlier_shock(per[1], "2011-06-13T02:00:00Z") and not S.after_earlier_shock(per[0], "2011-06-13T02:00:00Z")
    assert x.geonet_only == GEONET_ONLY["jun2011"]
    assert sl.window.anchor == p.origin                                   # re-anchored on the foreshock


def test_dec_2011_has_more_shocks_than_the_old_two_shock_map():
    sl, x = _plan("2011-12-23T02:18:03Z")
    assert [e.event_id for e in sl.unresolved] == ["usp000jch1"]          # mb 5.0 in the summary
    with pytest.raises(S.SequenceError, match="UNRESOLVED"):
        S.plan_tasks(sl)
    # Resolved either way (GeoNet lists it as Mw 5.0), it is out of reach of the grid (R 9.5 km): three shocks.
    for census in ({"usp000jch1": None}, {"usp000jch1": 5.0}):
        sl2, _ = _plan("2011-12-23T02:18:03Z", census)
        p = S.plan_tasks(sl2)[0]
        assert p.kind == S.SEQUENCE and p.shock_ids == ("usp000jch0", "usp000jch2", "usp000jch9")
        assert sl2.shocks[1].n_sites_covered < sl2.n_sites
    assert x.geonet_only == GEONET_ONLY["dec2011"]


def test_feb_2011_main_shock_leads_its_own_sequence():
    sl, x = _plan("2011-02-21T23:51:42Z")
    assert x.geonet_only == GEONET_ONLY["feb2011"]
    p = S.plan_tasks(sl)[0]
    assert p.kind == S.SEQUENCE and p.shock_ids[0] == "usp000huvq" and len(p.shock_ids) == 3
    assert p.origin.strftime("%Y-%m-%dT%H:%M:%S") == "2011-02-21T23:51:42"


def test_sep_2010_needs_its_mb_aftershocks_resolved():
    sl, x = _plan("2010-09-03T16:35:47Z")
    assert [s.event.event_id for s in sl.shocks] == ["usp000hk46"] and sl.shocks[0].census_mw == 7.0
    assert len(sl.unresolved) == 3
    with pytest.raises(S.SequenceError):
        S.plan_tasks(sl)
    assert x.matched[0][1] == "3366146"
    assert x.geonet_only == GEONET_ONLY["sep2010"]

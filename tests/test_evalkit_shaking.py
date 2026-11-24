"""evalkit shaking baselines (B4, eval M4): continuous PGA / MSF(M) score, earlier-only fit, old binned row beside.

Synthetic only. No Canterbury data is read or scored here.
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard import g5_entry  # noqa: E402
from scoreboard.evalkit import metrics, shaking as S, synth  # noqa: E402
from scoreboard.evalkit.card import CardRefused, build_card  # noqa: E402
from scoreboard.evalkit.lab import rules  # noqa: E402
from scoreboard.evalkit.spec import Entry, Task, site_list_sha256  # noqa: E402

EV = "eq-test"


def _earlier(event_id, mw, n, seed, a=1.0, b=2.0, msfmax=S.MSFMAX):
    """Labels drawn from the true model p = expit(a + b ln(PGA / MSF(M)))."""
    rng = np.random.default_rng(seed)
    pga = np.exp(rng.uniform(np.log(0.05), np.log(0.8), n))
    y = (rng.random(n) < expit(a + b * np.log(pga / S.msf(mw, msfmax)))).astype(float)
    return S.Earlier(event_id, mw, pga, y)


# ---------------------------------------------------------------------------------------------------------------
# MSF (form MSF-M-ONLY-1.3: B&I 2014 eq. 2.19, page 14, MSFmax fixed at 1.3)
# ---------------------------------------------------------------------------------------------------------------

def test_msf_is_one_at_m75_and_matches_the_formula():
    assert S.msf(7.5) == pytest.approx(1.0, abs=1e-4)
    for m in (5.9, 6.0, 6.2, 7.1, 7.8):
        want = 1.0 + (1.3 - 1.0) * (8.64 * np.exp(-m / 4.0) - 1.325)
        assert S.msf(m) == pytest.approx(want, rel=1e-12)
    assert S.msf(6.2) == pytest.approx(1.1526, abs=1e-4)          # worked by hand: 1 + 0.3 (1.8338 - 1.325)


def test_msf_falls_with_magnitude_and_grows_with_msfmax():
    ms = np.linspace(5.0, 8.5, 15)
    v = np.array([S.msf(m) for m in ms])
    assert np.all(np.diff(v) < 0)
    assert S.msf(6.0, 2.2) > S.msf(6.0, 1.3) > S.msf(6.0, 1.09) > 1.0


def test_msf_shares_the_house_g5_value():
    assert S.MSFMAX == g5_entry.MSFMAX
    for m in (5.9, 6.2, 7.1):
        assert S.msf(m) == pytest.approx(float(g5_entry.msf(m)), rel=1e-12)


@pytest.mark.parametrize("bad", [1.0, 2.5, -1.0])
def test_msfmax_outside_eq_2_20_is_refused(bad):
    with pytest.raises(ValueError, match="MSFmax"):
        S.msf(6.5, bad)


@pytest.mark.parametrize("bad", [float("nan"), 3.0, 10.0, True, "7"])
def test_bad_magnitude_is_refused(bad):
    with pytest.raises(ValueError, match="magnitude"):
        S.msf(bad)


def test_the_msf_form_is_named_cpt_free_and_cited_to_eq_2_19():
    """Correction B4 (founder check 2026-11-23): the B&I 2014 MSF needs qc1Ncs (eq. 2.20); the baseline uses a
    stated CPT-free form, MSF-M-ONLY-1.3, and cites eq. 2.19 on page 14 (not eq. 2.17, which is Idriss 1999)."""
    assert S.MSF_FORM == "MSF-M-ONLY-1.3"
    for text in (S.MSF_SOURCE, S.msf_form()):
        assert "MSF-M-ONLY-1.3" in text and "eq. 2.19" in text and "page 14" in text and "1.3" in text
        assert "eq. 2.17" not in text
    assert "VARIANT" in S.msf_form(1.8) and "1.8" in S.msf_form(1.8)
    assert "VARIANT" not in S.msf_form(1.3)
    # eq. 2.19 property: MSF = MSFmax at M 5.25 (eq. 2.18 is built so), whatever the fixed MSFmax
    for mx in (1.09, 1.3, 1.8, 2.2):
        assert S.msf(5.25, mx) == pytest.approx(mx, abs=2e-3)
    # the fixed 1.3 is what eq. 2.20 gives at qc1Ncs about 107; no site qc1Ncs is read
    assert 1.09 + (107.0 / 180.0) ** 3 == pytest.approx(S.MSFMAX, abs=1e-3)


def test_no_function_takes_a_cpt_value():
    """Shaking only: no MSF or row builder has a parameter for qc1Ncs, Ic, FC or any other CPT or site value."""
    cpt_like = {"qc1ncs", "qc1n", "qc", "qt", "fs", "ic", "fc", "cpt", "site", "msfmax_per_site", "n1_60cs"}
    for fn in (S.msf, S.score, S.fit, S.probability, S.score_entry, S.probability_entry):
        names = {n.lower() for n in inspect.signature(fn).parameters}
        assert not names & cpt_like, fn.__name__


@pytest.mark.parametrize("bad", [np.array([1.3, 1.5]), [1.3], np.array([1.3]), "1.3", True])
def test_a_per_site_msfmax_is_refused(bad):
    """MSFmax is one number for every site; an array (a value per site, as eq. 2.20 would give) is refused."""
    with pytest.raises(ValueError, match="MSFmax"):
        S.msf(6.5, bad)
    with pytest.raises(ValueError, match="MSFmax"):
        S.score([0.1, 0.2], 6.5, bad)


def test_row_statements_and_fit_record_name_the_form():
    earlier = [_earlier("eq-a", 7.1, 3000, 1), _earlier("eq-b", 6.0, 3000, 2)]
    e, f = S.probability_entry(("syn:1", "syn:2"), [0.1, 0.3], 6.2, EV, earlier)
    sc = S.score_entry(("syn:1", "syn:2"), [0.1, 0.3], 6.2, EV)
    for st in (e.manifest.statement, sc.manifest.statement):
        assert "MSF-M-ONLY-1.3" in st and "eq. 2.19" in st and "no CPT input" in st
        assert "eq. 2.17" not in st
    assert f["msf_form"] == S.msf_form() and f["msf_source"] == S.MSF_SOURCE
    v = S.score_entry(("syn:1",), [0.1], 6.2, EV, msfmax=1.8)
    assert "VARIANT of MSF-M-ONLY-1.3" in v.manifest.statement


def test_integer_and_numpy_magnitudes_are_accepted():
    assert S.msf(np.int64(7)) == pytest.approx(S.msf(7.0), rel=1e-15)
    assert S.msf(7) == pytest.approx(S.msf(np.float32(7.0)), rel=1e-7)


# ---------------------------------------------------------------------------------------------------------------
# The score: continuous, no ties of its own
# ---------------------------------------------------------------------------------------------------------------

def test_score_is_strictly_monotone_and_adds_no_ties():
    rng = np.random.default_rng(5)
    pga = rng.uniform(0.02, 0.9, 3000)
    s = S.score(pga, 6.2)
    order = np.argsort(pga)
    assert np.all(np.diff(s[order]) > 0)
    t = S.tie_count(s)
    assert t == {"n": 3000, "n_distinct": 3000, "n_rows_tied": 0}


def test_binned_rule_ties_where_the_score_does_not():
    rng = np.random.default_rng(6)
    e = S.Earlier("eq-a", 7.1, rng.uniform(0.05, 0.8, 2000), (rng.random(2000) < 0.3).astype(float))
    pga = rng.uniform(0.05, 0.8, 1000)
    binned = S.binned_p(S.binned_rates(e.pga, e.y), pga)
    assert S.tie_count(binned)["n_distinct"] <= len(S.PGA_EDGES) - 1
    assert S.tie_count(S.score(pga, 6.2))["n_rows_tied"] == 0


def test_tie_count_counts_only_equal_inputs():
    t = S.tie_count(S.score([0.1, 0.2, 0.2, 0.3, 0.3, 0.3], 6.0))
    assert t == {"n": 6, "n_distinct": 3, "n_rows_tied": 5}


def test_score_auc_does_not_depend_on_msfmax_inside_one_earthquake():
    rng = np.random.default_rng(7)
    pga = rng.uniform(0.05, 0.8, 800)
    y = (rng.random(800) < expit(-1 + 3 * pga)).astype(bool)
    base = metrics.auc(y, pga)
    for mx in (1.09, 1.3, 2.2):
        assert metrics.auc(y, S.score(pga, 6.0, mx)) == pytest.approx(base, abs=1e-12)


def test_score_refuses_bad_pga():
    with pytest.raises(ValueError, match="finite"):
        S.score([0.1, np.nan], 6.0)
    with pytest.raises(ValueError, match="negative"):
        S.score([0.1, -0.01], 6.0)


# ---------------------------------------------------------------------------------------------------------------
# The probability: fitted on earlier earthquakes only
# ---------------------------------------------------------------------------------------------------------------

def test_fit_recovers_the_true_model_across_magnitudes():
    earlier = [_earlier("eq-a", 7.1, 20000, 1), _earlier("eq-b", 6.0, 20000, 2)]
    f = S.fit(earlier, EV)
    assert f["a"] == pytest.approx(1.0, abs=0.1)
    assert f["b"] == pytest.approx(2.0, abs=0.1)
    assert f["slope_positive"] and f["fitted_on"] == ["eq-a", "eq-b"] and f["fitted_on_mw"] == [7.1, 6.0]
    assert f["n_rows"] == 40000


def test_msf_is_what_pools_two_magnitudes():
    """With labels from the MSF model, a fit on raw ln PGA (MSFmax 1.0 would be) is worse than the MSF fit."""
    earlier = [_earlier("eq-a", 7.1, 20000, 1, msfmax=2.2), _earlier("eq-b", 5.9, 20000, 2, msfmax=2.2)]
    good = S.fit(earlier, EV, msfmax=2.2)
    assert good["b"] == pytest.approx(2.0, abs=0.1)
    weak = S.fit(earlier, EV, msfmax=1.09)
    assert abs(weak["b"] - 2.0) > abs(good["b"] - 2.0)


def test_fit_refuses_the_scored_earthquake():
    earlier = [_earlier("eq-a", 7.1, 500, 1), _earlier(EV, 6.2, 500, 2)]
    with pytest.raises(ValueError, match="earlier ones only"):
        S.fit(earlier, EV)
    with pytest.raises(ValueError, match="earlier set"):
        S.binned_entry(("syn:1",), [0.2], EV, earlier)


def test_fit_refuses_no_earlier_and_a_repeat():
    with pytest.raises(ValueError, match="at least one"):
        S.fit([], EV)
    e = _earlier("eq-a", 7.1, 500, 1)
    with pytest.raises(ValueError, match="once"):
        S.fit([e, e], EV)


def test_probability_is_the_fit_read_at_the_scored_magnitude():
    """Review 1: p must use the scored earthquake M, not 7.5 and not an earlier M."""
    earlier = [_earlier("eq-a", 7.1, 3000, 1), _earlier("eq-b", 6.0, 3000, 2)]
    rng = np.random.default_rng(9)
    ids = tuple(f"syn:{k:04d}" for k in range(400))
    pga = rng.uniform(0.05, 0.7, 400)
    e, f = S.probability_entry(ids, pga, 6.2, EV, earlier)
    m = 1.0 + (S.MSFMAX - 1.0) * (8.64 * np.exp(-6.2 / 4.0) - 1.325)        # eq. 2.19 written out again
    assert np.allclose(e.p, expit(f["a"] + f["b"] * np.log(pga / m)), rtol=1e-12, atol=0)
    for other in (7.5, 7.1, 6.0, 5.5):
        e2, f2 = S.probability_entry(ids, pga, other, EV, earlier)
        assert f2["a"] == f["a"] and f2["b"] == f["b"]                    # the fit does not see the scored M
        assert np.min(np.abs(e2.p - e.p)) > 1e-6                          # but p does
    lower, _ = S.probability_entry(ids, pga, 5.5, EV, earlier)
    assert np.all(lower.p < e.p)                                          # smaller M, larger MSF, weaker demand
    assert e.manifest.output_kind == "target_probability" and e.manifest.exposure == {EV: "unseen"}
    assert np.all(np.diff(e.p[np.argsort(pga)]) > 0)                      # b > 0: same order as the score
    assert S.tie_count(e.p)["n_rows_tied"] == 0


def test_the_api_has_no_place_for_scored_labels_and_earlier_labels_do_move_p():
    """Review 1: the old self-equal test could not fail. Check the signature, and that p follows the fit set."""
    for fn in (S.probability_entry, S.probability, S.score_entry):
        names = set(inspect.signature(fn).parameters)
        assert not names & {"y", "labels", "label", "scored_labels"}, fn.__name__
    rng = np.random.default_rng(10)
    ids = tuple(f"syn:{k:04d}" for k in range(200))
    pga = rng.uniform(0.05, 0.7, 200)
    a, _ = S.probability_entry(ids, pga, 6.2, EV, [_earlier("eq-a", 7.1, 3000, 1, a=1.0, b=2.0)])
    b, _ = S.probability_entry(ids, pga, 6.2, EV, [_earlier("eq-a", 7.1, 3000, 1, a=-0.5, b=2.0)])
    assert np.min(np.abs(a.p - b.p)) > 1e-3


def test_statement_carries_the_fit():
    earlier = [_earlier("eq-a", 7.1, 3000, 1), _earlier("eq-b", 6.0, 3000, 2)]
    e, f = S.probability_entry(("syn:1", "syn:2"), [0.1, 0.3], 6.2, EV, earlier)
    st = e.manifest.statement
    assert st.startswith(S.FIT_MARKER)
    for part in (f"a = {f['a']:.4f}", f"b = {f['b']:.4f}", "MSFmax 1.3", "eq-a (M 7.1)", "eq-b (M 6)",
                 f"n_rows {f['n_rows']}", f"n_yes {f['n_yes']}", "at M 6.2", "never on this one"):
        assert part in st, part
    assert f["converged"] and f["n_iter"] >= 1 and f["separated"] is False


# ---------------------------------------------------------------------------------------------------------------
# Fit guards (review 1): refuse, never keep, a fit that has no stable finite estimate
# ---------------------------------------------------------------------------------------------------------------

def test_fit_refuses_a_separated_earlier_set():
    rng = np.random.default_rng(20)
    pga = np.exp(rng.uniform(np.log(0.05), np.log(0.8), 1000))
    with pytest.raises(ValueError, match="separated"):
        S.fit([S.Earlier("eq-a", 7.1, pga, (pga > 0.3).astype(float))], EV)
    edge = np.concatenate([pga, [0.3, 0.3]])                             # quasi-complete: yes and no share 0.3 g
    y = np.concatenate([(pga > 0.3).astype(float), [0.0, 1.0]])
    with pytest.raises(ValueError, match="separated"):
        S.fit([S.Earlier("eq-a", 7.1, edge, y)], EV)
    with pytest.raises(ValueError, match="separated"):                   # reversed: yes only at weak shaking
        S.fit([S.Earlier("eq-a", 7.1, pga, (pga < 0.3).astype(float))], EV)


def test_fit_refuses_a_negative_slope():
    earlier = [_earlier("eq-a", 7.1, 3000, 21, a=-1.0, b=-1.5)]
    with pytest.raises(ValueError, match="not above 0"):
        S.fit(earlier, EV)


@pytest.mark.parametrize("n_yes", [0, 1, S.MIN_CLASS - 1])
def test_fit_refuses_too_few_yes(n_yes):
    rng = np.random.default_rng(22)
    pga = rng.uniform(0.05, 0.8, 500)
    y = np.zeros(500)
    y[np.argsort(pga)[:n_yes]] = 1.0                                      # yes at the lowest PGA, as in the review
    with pytest.raises(ValueError, match="at least"):
        S.fit([S.Earlier("eq-a", 7.1, pga, y)], EV)


def test_fit_refuses_too_few_no():
    rng = np.random.default_rng(23)
    pga = rng.uniform(0.05, 0.8, 500)
    y = np.ones(500)
    y[:S.MIN_CLASS - 1] = 0.0
    with pytest.raises(ValueError, match="at least"):
        S.fit([S.Earlier("eq-a", 7.1, pga, y)], EV)


def test_fit_refuses_no_convergence(monkeypatch):
    earlier = [_earlier("eq-a", 7.1, 3000, 1)]
    monkeypatch.setattr(S, "MAX_ITER", 1)
    with pytest.raises(ValueError, match="did not converge"):
        S.fit(earlier, EV)
    with pytest.raises(ValueError, match="did not converge"):
        S.probability_entry(("syn:1",), [0.2], 6.2, EV, earlier)


def test_earlier_rows_need_positive_pga_and_binary_labels():
    with pytest.raises(ValueError, match="above 0"):
        S.Earlier("eq-a", 7.1, [0.0, 0.2], [0, 1])
    with pytest.raises(ValueError, match="0 or 1"):
        S.Earlier("eq-a", 7.1, [0.1, 0.2], [0, 0.5])


# ---------------------------------------------------------------------------------------------------------------
# The old binned row is the scorecard v1/v2 rule, unchanged
# ---------------------------------------------------------------------------------------------------------------

def test_binned_rates_equal_the_v1_rule_copy_in_the_lab():
    rng = np.random.default_rng(11)
    for n in (40, 200, 5000):
        pga = rng.uniform(0.0, 0.9, n) ** 1.5
        y = (rng.random(n) < 0.3).astype(float)
        assert S.binned_rates(pga, y) == rules.shaking_rates(pga, y)
        bins = S.binned_rates(pga, y)
        want = rules.shaking_only_p([(pga, y)], pga)
        assert np.allclose(S.binned_p(bins, pga), want)


def test_bins_record_is_card_safe():
    rng = np.random.default_rng(12)
    rec = S.bins_record(S.binned_rates(rng.uniform(0, 0.9, 500), (rng.random(500) < 0.3).astype(float)))
    assert rec[-1]["pga_to_g"] is None and all(set(r) == {"pga_from_g", "pga_to_g", "n", "yes"} for r in rec)


# ---------------------------------------------------------------------------------------------------------------
# Card: the continuous rows and the old row beside
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case():
    ev = synth.synthetic_event(900, 41)
    rng = np.random.default_rng(42)
    pga = np.exp(-1.6 + 0.4 * ev["field"] + rng.normal(0.0, 0.3, ev["s"].size))
    earlier = [_earlier("eq-a", 7.1, 1500, 3, a=0.5, b=1.5), _earlier("eq-b", 6.0, 1500, 4, a=0.5, b=1.5)]
    return ev, synth.labels_of(ev), pga, earlier


def _task(ev, **kw):
    d = dict(event_id=EV, target="ejecta seen at the site", label_source="synthetic v1", label_rule_sha256="1" * 64,
             site_list_sha256=site_list_sha256(ev["site_ids"]), track="record", division="open", tile_m=2000.0,
             clip=1e-6, seed=3, label_caveats="synthetic labels", n_boot=60, earlier_base_rate=0.3,
             earlier_label_sets=2)
    d.update(kw)
    return Task(**d)


def _house(ev):
    return Entry(synth.manifest("house", "target_probability", EV, house=True), ev["site_ids"], ev["p_target"])


def test_card_carries_score_probability_and_old_row(case):
    ev, labels, pga, earlier = case
    ids = ev["site_ids"]
    prob, fit = S.probability_entry(ids, pga, 6.2, EV, earlier)
    old, _ = S.binned_entry(ids, pga, EV, earlier)
    sc = S.score_entry(ids, pga, 6.2, EV)
    c = build_card(_task(ev), [_house(ev)], labels, shaking_only=prob, shaking_score=sc, shaking_binned=old)
    names = [e["name"] for e in c["entries"]]
    assert names[-3:] == [S.NAME_PROB, S.NAME_SCORE, S.NAME_BINNED]
    meta = {e["name"]: e for e in c["entries"]}
    assert meta[S.NAME_SCORE]["probability_metrics"] is False and meta[S.NAME_SCORE]["exposure"] == "UNSEEN"
    assert meta[S.NAME_BINNED]["probability_metrics"] is True
    assert "OLD RULE" in meta[S.NAME_BINNED]["statement"]
    assert "NOT APPLICABLE" in c["metrics"][S.NAME_SCORE]["probability_metrics"]
    assert c["metrics"][S.NAME_SCORE]["auc"]["value"] == pytest.approx(
        metrics.auc(labels.label.astype(bool), pga), abs=1e-12)
    assert c["metrics"][S.NAME_PROB]["auc"]["value"] == pytest.approx(
        c["metrics"][S.NAME_SCORE]["auc"]["value"], abs=1e-12)        # b > 0: same ranking
    assert not any({x["a"], x["b"]} <= {S.NAME_PROB, S.NAME_SCORE, S.NAME_BINNED} for x in c["comparisons"])
    assert fit["slope_positive"]


def test_score_row_works_with_no_earlier_earthquake(case):
    ev, labels, pga, _ = case
    sc = S.score_entry(ev["site_ids"], pga, 7.1, EV)
    c = build_card(_task(ev, earlier_base_rate=None, earlier_label_sets=0), [_house(ev)], labels, shaking_score=sc)
    assert S.NAME_SCORE in c["metrics"]


def test_card_refuses_a_misnamed_or_misdeclared_shaking_row(case):
    ev, labels, pga, earlier = case
    ids = ev["site_ids"]
    prob, _ = S.probability_entry(ids, pga, 6.2, EV, earlier)
    old, _ = S.binned_entry(ids, pga, EV, earlier)
    as_prob = Entry(synth.manifest(S.NAME_SCORE, "target_probability", EV), ids, np.clip(pga, 0, 1))
    with pytest.raises(CardRefused, match="output kind score"):
        build_card(_task(ev), [_house(ev)], labels, shaking_only=prob, shaking_score=as_prob)
    wrong = Entry(synth.manifest("baseline: pga", "score", EV), ids, pga)
    with pytest.raises(CardRefused, match="must be named"):
        build_card(_task(ev), [_house(ev)], labels, shaking_only=prob, shaking_score=wrong)
    with pytest.raises(CardRefused, match="beside the shaking-only p"):
        build_card(_task(ev, earlier_base_rate=None, earlier_label_sets=0), [_house(ev)], labels,
                   shaking_binned=old)


def test_card_refuses_the_old_rule_in_the_shaking_only_slot(case):
    """Review 1: with the old row beside, the shaking-only slot must hold the continuous fit."""
    ev, labels, pga, earlier = case
    ids = ev["site_ids"]
    old, _ = S.binned_entry(ids, pga, EV, earlier)
    renamed = Entry(synth.manifest(S.NAME_PROB, "target_probability", EV), ids, old.p)
    with pytest.raises(CardRefused, match="continuous fit"):
        build_card(_task(ev), [_house(ev)], labels, shaking_only=old, shaking_binned=old)
    with pytest.raises(CardRefused, match="continuous fit"):
        build_card(_task(ev), [_house(ev)], labels, shaking_only=renamed, shaking_binned=old)
    forged = Entry(synth.manifest(S.NAME_PROB, "target_probability", EV, statement=S.FIT_MARKER + " forged"),
                   ids, old.p)
    with pytest.raises(CardRefused, match="same values"):
        build_card(_task(ev), [_house(ev)], labels, shaking_only=forged, shaking_binned=old)

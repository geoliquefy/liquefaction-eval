"""D8 scoring (eval M5, ruling 55b): registration 8.2 point 3 tests and amendment 1 items, on PLANTED labels only.
No test reads the sealed D8 label folder: an audit hook fails any open under it. No em dashes."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scoreboard.evalkit import card as C
from scoreboard.evalkit import d8_score as D
from scoreboard.evalkit import interval_policy as IP
from scoreboard.evalkit import world_inputs as W

ROOT = Path(__file__).resolve().parents[1]
SEALED_TOKEN = "world-inputs-v2-2026-11-23/labels-sealed"


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, Path)) and SEALED_TOKEN in str(args[0]):
        raise PermissionError("a test touched the sealed D8 label folder")


sys.addaudithook(_hook)


def _script():
    spec = importlib.util.spec_from_file_location("d8_score_script", ROOT / "scripts" / "d8_score_2026_11_24.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------------------------
# Planted world: groups of NGL and PRJ-6544 cases, label free; labels planted beside
# ---------------------------------------------------------------------------------------------------------------

def _world(n_ngl=4, n_prj=2, per=30, sites_per=None, seed=1, names=None, held=True):
    rng = np.random.default_rng(seed)
    rows = []
    names = names or [f"Planted {k}" for k in range(n_ngl + n_prj)]
    for k, eq in enumerate(names):
        src = D.NGL if k < n_ngl else D.PRJ
        ns = sites_per or per
        for i in range(per):
            s = i % ns
            rows.append({"case_id": W.id_hash("t", f"{k}|{i}"), "source": src, "earthquake": eq,
                         "event_id": f"e{k}", "site_id_hash": W.id_hash("s", f"{k}|{s}"),
                         "cpt_ref_hash": W.id_hash("c", f"{k}|{i}"),
                         "lat": 10.0 + k + 0.02 * (s % 7), "lon": 20.0 + 0.02 * (s // 7),
                         "pga_primary_source": "PRJ-4022 Parent_MEAN",
                         "kit_exp_bi14": "fitted_on" if k == 0 else "unseen", "kit_exp_shaking": "unseen",
                         "kit_exp_zhu2017": "unknown"})
    if held:
        rows[0]["pga_primary_source"] = W.WITHHELD_SOURCE
        rows[0]["earthquake"] = "Elbistan"
        rows[0]["event_id"] = "e0"
        for r in rows:
            if r["event_id"] == "e0":
                r["earthquake"] = "Elbistan"
    cases = pd.DataFrame(rows)
    y = rng.random(len(cases)) < 0.45
    preds = {}
    for row, v, _ in D.FILES.values():
        noise = rng.normal(0, 1, len(cases))
        val = np.where(y, 0.6, 0.4) + 0.3 * noise
        if row == D.LEVEL0:
            val = 1 / (1 + np.exp(-3 * (val - 0.5)))
        if row != D.SHAKING and not row.startswith("USGS"):
            val = np.where(cases["source"] == D.NGL, val, np.nan)
        preds.setdefault(row, {})[v] = dict(zip(cases["case_id"], val))
    reasons = {c: ("ON-GRID VALUE" if i % 5 else "OFF-GRID") for i, c in enumerate(cases["case_id"])}
    labels = []
    for c, src, yy in zip(cases["case_id"], cases["source"], y):
        if src == D.NGL:
            labels.append({"case_id": c, "fldm_json": json.dumps([{"FLDM_SFEV": int(yy)}]), "observation_raw": None})
        else:
            labels.append({"case_id": c, "fldm_json": None, "observation_raw": "1" if yy else "0"})
    return cases, preds, reasons, {}, labels


def _score(world, **kw):
    cases, preds, reasons, split, labels = world
    return D.score(cases, preds, reasons, split, labels, n_boot=kw.pop("n_boot", 40), **kw)


# ---------------------------------------------------------------------------------------------------------------
# 8.2 point 3: the registered tests
# ---------------------------------------------------------------------------------------------------------------

def test_changed_label_sha_stops_before_any_parse(tmp_path, monkeypatch):
    S = _script()
    f = tmp_path / "planted-labels.parquet"
    pd.DataFrame({"case_id": ["a"], "fldm_json": [None], "observation_raw": ["1"]}).to_parquet(f)
    called = []
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: called.append(1))
    run_txt: list[str] = []
    with pytest.raises(D.D8ScoreStop, match="sha256 mismatch"):
        S.read_labels(f, "0" * 64, tmp_path, run_txt)
    assert not called
    txt = (tmp_path / "RUN.txt").read_text()
    assert "STOPPED" in txt and "0" * 64 in txt and W.sha_file(f) in txt
    assert not (tmp_path / "D8-SCORES-2026-11-24.json").exists()


def test_held_back_case_is_out_and_must_be_exactly_one():
    world = _world()
    out = _score(world)
    assert out["scope"]["cases_in_scope"] == out["scope"]["cases_in_inputs"] - 1
    assert out["scope"]["held_back"] == D.HELD_BACK_TEXT
    cases = world[0].copy()
    cases["pga_primary_source"] = "none"
    with pytest.raises(D.D8ScoreStop, match="exactly 1 held-back"):
        D.scope(cases)
    cases = world[0].copy()
    cases.loc[[0, 1], "pga_primary_source"] = W.WITHHELD_SOURCE
    with pytest.raises(D.D8ScoreStop):
        D.scope(cases)


def test_kaikoura_and_canterbury_are_refused():
    cases = _world()[0]
    k = cases.copy()
    k.loc[5, "earthquake"] = "Kaikoura"
    with pytest.raises(W.D8Refused):
        D.scope(k)
    c = cases.copy()
    c.loc[5, "earthquake"] = "Darfield"
    with pytest.raises(D.D8ScoreStop):
        D.scope(c)


@pytest.mark.parametrize("rows,want", [
    ([{"FLDM_SFEV": 1}], ("YES", False)), ([{"FLDM_SFEV": "1"}], ("YES", False)),
    ([{"FLDM_SFEV": 1.0}, {"FLDM_SFEV": 0}], ("YES", True)), ([{"FLDM_SFEV": True}], ("YES", False)),
    ([{"FLDM_SFEV": 0}], ("NO", False)), ([{"FLDM_SFEV": False}, {"FLDM_SFEV": 2}], ("NO", False)),
    ([{"FLDM_SFEV": 2}], (None, False)), ([{"FLDM_SFEV": None}], (None, False)), ([], (None, False)),
    ([{"FLDM_SFEV": "yes"}], (None, False)), ([{"FLDM_SFEV": 1.5}], (None, False)),
    ([{"FLDM_SFEV": 0.9}], (None, False)),     # precheck N1: never truncated to 0
])
def test_ngl_class_rule(rows, want):
    assert D.classify_ngl(json.dumps(rows)) == want


def test_ngl_class_rule_bad_json():
    assert D.classify_ngl(None) == (None, False)
    assert D.classify_ngl("not json") == (None, False)


@pytest.mark.parametrize("raw,want", [("1", "YES"), (" Yes ", "YES"), ("Liquefaction", "YES"), (1, "YES"),
                                      ("0.0", "NO"), ("Non-liquefaction", "NO"), ("not liquefied", "NO"), (0, "NO"),
                                      ("2", None), ("maybe", None), (None, None), (float("nan"), None)])
def test_prj_class_rule(raw, want):
    assert D.classify_prj(raw) == want


def test_prj_one_percent_rule_gives_counts_only():
    world = list(_world(n_ngl=2, n_prj=2))
    labels = [dict(r) for r in world[4]]
    prj = [r for r in labels if r["observation_raw"] is not None]
    for r in prj[: max(1, len(prj) // 20)]:
        r["observation_raw"] = "probable"
    world[4] = labels
    out = _score(world)
    assert "FAILED" in out["scope"]["prj6544_coding_check"]
    for g in out["groups"].values():
        if g["source"] == "PRJ-6544":
            assert g["read"].startswith("COUNTS ONLY") and "variants" not in g
            assert isinstance(g["labels"]["distinct raw values"], int)


def test_duplicate_case_id_stops():
    world = list(_world())
    world[4] = world[4] + [dict(world[4][3])]
    with pytest.raises(D.D8ScoreStop, match="duplicate"):
        _score(world)


def test_no_word_outside_f1():
    out = _score(_world())
    known = set(_world()[0]["case_id"])
    clean = D.check_output(out, known)
    assert IP.words_left({k: v for k, v in clean.items() if k != "f1"}) == []
    bad = json.loads(json.dumps(clean))
    next(iter(bad["groups"].values()))["read"] = "SEPARATES"
    with pytest.raises(D.D8ScoreStop, match="verdict word"):
        D.check_output(bad, known)


def _items(n, sites=250, yes=60, no=60, tiles=25, eqs=None, diff=0.05, se=0.01, fitted=False):
    eqs = eqs or [f"Q{k}" for k in range(n)]
    return {f"{e} [NGL]": {"earthquake": e, "counts": {"cases": sites, "cases_yes": yes, "cases_no": no,
                                                       "sites": sites, "yes_sites": yes, "no_sites": no,
                                                       "tiles": tiles},
                           "diff": diff + 0.001 * k, "se": se, "fitted": fitted} for k, e in enumerate(eqs)}


def test_not_allowed_for_each_r1_condition():
    # (b) and (e): under the R27 gate on the pair intersection (precheck S1: sites, not cases)
    few = {ln: _items(4, sites=93) for ln in D.F1_LINES}
    r = D.f1_lines(few, r1c_shown=True)
    assert all(x["word"] == "NOT ALLOWED" and any("R1 (b) and (e)" in s for s in x["reasons"]) for x in r.values())
    # (c): not shown for this registration (A1-3), even with a full set
    full = {ln: _items(4) for ln in D.F1_LINES}
    r = D.f1_lines(full)
    assert all(x["word"] == "NOT ALLOWED" and any(s.startswith("R1 (c)") for s in x["reasons"]) for x in r.values())
    # (d): a set earthquake in the 9.3 list removes the whole line
    d = {ln: _items(4, eqs=["Loma Prieta", "Q1", "Q2", "Q3"]) for ln in D.F1_LINES}
    r = D.f1_lines(d, r1c_shown=True)
    assert all(x["word"] == "NOT ALLOWED" and any(s.startswith("R1 (d)") for s in x["reasons"]) for x in r.values())
    # (e): a set earthquake with no finite positive s_i
    e = {ln: _items(4, se=float("nan")) for ln in D.F1_LINES}
    r = D.f1_lines(e, r1c_shown=True)
    assert all(any("s_i" in s for s in x["reasons"]) for x in r.values())


def test_lines_not_allowed_leave_the_holm_family():
    """Precheck S2: R1 (b), (d), (e) apply before the kit call; only lines left enter Holm."""
    pi = {"L1": _items(4), "L2": _items(4, eqs=["Northridge", "A", "B", "C"]), "L3": _items(2)}
    r = D.f1_lines(pi, r1c_shown=True)
    assert r["L2"]["word"] == "NOT ALLOWED" and r["L3"]["word"] == "NOT ALLOWED"
    assert r["L1"]["word"] in ("HOLDS", "FRAGILE", "POOLED CANNOT TELL")
    assert r["L1"]["holm"]["family_size"] == 1
    assert all("R1 (c): " not in s or "not shown" in s for x in r.values() for s in x.get("reasons", []))


def test_every_pooled_line_carries_the_world_interval_note():
    """Precheck S5."""
    out = _score(_world())
    for blk in ("f1", "descriptive_lines"):
        for x in out[blk].values():
            assert D.NOTE_INTERVAL in x["notes"] and IP.pooled_note() in x["notes"]


def test_missing_and_out_of_scope_label_rows_are_counted():
    """Precheck S4."""
    world = list(_world())
    labels = [dict(r) for r in world[4]]
    dropped = labels.pop(7)
    labels.append({"case_id": "not-in-scope", "fldm_json": None, "observation_raw": "1"})
    world[4] = labels
    out = _score(world)
    assert out["scope"]["cases_with_no_label_row"] == 1
    assert out["scope"]["label_rows_out_of_scope_ignored"] == 1
    assert sum(g["labels"]["NOT LABELLED, no row"] for g in out["groups"].values()) == 1
    assert dropped["case_id"] not in json.dumps(out)


def test_existing_output_with_a_metric_refuses_the_run(tmp_path):
    S = _script()
    out = tmp_path / "planted-out"
    out.mkdir()
    (out / "D8-SCORES-2026-11-24.json").write_text("{}")
    lab = tmp_path / "planted.parquet"
    pd.DataFrame({"case_id": ["a"]}).to_parquet(lab)
    a = S.main.__globals__["argparse"].Namespace(
        pillar=str(tmp_path), out=str(out), registration="reg.md", registration_sha256="x", amendment="amd.md",
        amendment_sha256="y", commit=None, labels=str(lab), label_sha256="z", heavy_label="", n_boot=10)
    (tmp_path / "reg.md").write_text("r")
    (tmp_path / "amd.md").write_text("a")
    a.registration_sha256 = W.sha_file(tmp_path / "reg.md")
    a.amendment_sha256 = W.sha_file(tmp_path / "amd.md")
    S.D.REGISTRATION_SHA256, keep = a.registration_sha256, S.D.REGISTRATION_SHA256
    S.check_commit, keep_cc = (lambda c: "f" * 40), S.check_commit
    try:
        with pytest.raises(D.D8ScoreStop, match="output JSON exists"):
            S.run(a, planted=True)
    finally:
        S.D.REGISTRATION_SHA256, S.check_commit = keep, keep_cc


def test_output_scan_refuses_case_id_site_hash_and_coordinate():
    world = _world()
    out = _score(world)
    cases = world[0]
    known = set(cases["case_id"]) | set(cases["site_id_hash"])
    D.check_output(out, known)
    for planted in (cases["case_id"].iloc[3], cases["site_id_hash"].iloc[3]):
        bad = json.loads(json.dumps(D.check_output(out, known)))
        bad["notes"].append(planted)
        with pytest.raises(D.D8ScoreStop, match="aggregate scan"):
            D.check_output(bad, known)
    bad = json.loads(json.dumps(D.check_output(out, known)))
    bad["scope"]["lat"] = 1.0
    with pytest.raises(D.D8ScoreStop, match="coordinate"):
        D.check_output(bad, known)


# ---------------------------------------------------------------------------------------------------------------
# Amendment 1 items
# ---------------------------------------------------------------------------------------------------------------

def test_kit_loads_from_this_worktree(monkeypatch):
    """Precheck S6."""
    S = _script()
    assert Path(S.check_kit_origin()).is_relative_to(ROOT)
    assert Path(D.__file__).resolve().is_relative_to(ROOT)
    with pytest.raises(D.D8ScoreStop, match="not from the worktree"):
        S.check_kit_origin(Path("/nonexistent-worktree"))


def test_sixty_seven_groups_pass_the_scan():
    """Precheck S7: per-group results keyed by name, so 67 groups pass card.scan_aggregates."""
    world = _world(n_ngl=46, n_prj=21, per=12, seed=3)
    out = _score(world, n_boot=20)
    assert len(out["groups"]) == 67
    cases = world[0]
    clean = D.check_output(out, set(cases["case_id"]) | set(cases["site_id_hash"]))
    C.scan_aggregates(clean, frozenset(cases["case_id"]))
    for v in clean["descriptive_lines"].values():
        assert isinstance(v["per_earthquake"], dict)
        for k in ("equal_weight", "random_effects_hk"):
            if k in v:
                assert isinstance(v[k]["leave_one_out"], dict) and "earthquakes_pooled" not in v[k]
    assert max(v["ngl_groups_passing_record_gate"] for v in clean["descriptive_lines"].values()) > C.MAX_LIST_LEN


def test_registered_run_f1_is_not_allowed_and_descriptive_lines_have_no_word():
    out = _score(_world(n_ngl=5, n_prj=1, per=40, seed=7))
    assert all(x["word"] == "NOT ALLOWED" for x in out["f1"].values())
    for x in out["descriptive_lines"].values():
        assert x["word"] in ("DESCRIPTIVE", "NOT ALLOWED")
        for k in ("equal_weight", "random_effects_hk"):
            if k in x:
                assert "holm" not in x[k] and "p" not in x[k]["pooled"]


def test_sites_not_cases_for_gates():
    """V1: several cases on one site count once."""
    world = _world(n_ngl=2, n_prj=1, per=30, sites_per=2)
    out = _score(world)
    for g in out["groups"].values():
        for v in (g.get("variants") or {}).values():
            for r in v["rows"].values():
                assert r["counts"]["sites"] <= 2 and "auc" not in r


def test_same_draws_reproduce():
    a = D.check_output(_score(_world()), set())
    b = D.check_output(_score(_world()), set())
    assert a == b


def test_render_md_has_no_dashes_and_marks_planted():
    j = D.check_output(_score(_world()), set())
    j.update(planted=True, kit_commit="c" * 40)
    md = D.render_md(j)
    assert "PLANTED" in md and chr(0x2014) not in md and chr(0x2013) not in md


def test_module_has_no_em_dashes():
    for p in (ROOT / "scoreboard/evalkit/d8_score.py", ROOT / "scripts/d8_score_2026_11_24.py", Path(__file__)):
        t = p.read_text()
        assert chr(0x2014) not in t and chr(0x2013) not in t

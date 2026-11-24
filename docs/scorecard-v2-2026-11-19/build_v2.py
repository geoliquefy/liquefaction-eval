"""Liquefaction model scorecard v2, record track (G6+G1+G5, eval milestone M4, plan step 3d).

Registration: docs/scorecard-v2-2026-11-19/REGISTRATION.md (pillar 0c736f2), committed alone before this file.
Run from the evalkit worktree (repos/liquefaction-model-evalkit, branch evalkit-2026-11-15) with the model venv:
  <repos/liquefaction-model/.venv/bin/python> -I <this file> <out dir>
Reuses scorecard v1's loaders (build_v1.py, imported through docs/evals/effort_baseline.py, main() of neither run).
Every number comes from the kit (card.build_card + card.descriptive, metrics, resample, pool, effort,
chain_settings). Aggregates only (NZGD clause 6.2): the JSON passes card.scan_aggregates against every CPT name,
row id and world case id. The bench v2 test split is never read. No synthetic data. One heavy job; runs alone.
No em dashes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
KIT = Path.cwd()
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "scripts"))
PILLAR = KIT.parents[1]
HERE = PILLAR / "docs" / "scorecard-v2-2026-11-19"
OUTD = Path(sys.argv[1])
REGISTRATION = "docs/scorecard-v2-2026-11-19/REGISTRATION.md (pillar 0c736f2)"
LEDGER = Path.home() / ".geoliquefy-runs" / "exposure-ledger-2026-11-19" / "site_ledger.csv"
LEDGER_SHA = "b20c02d30375ae82b71d8a644428bde31b993b92c78e8ea68c8368acee99f9ce"
LER = PILLAR / "docs" / "evals" / "LABEL-ERROR-RATES-2026-11-19.json"
WORLD = PILLAR / "data" / "world-case-histories-2026-11-19"
WORLD_SHA = {"cases.parquet": "c42d2e0d1d239fb4d5df75079404de6c9f1fb74abef33c3fd753c11854e1b931",
             "traces.npz": "2b5e3fcd79042b0904ecf2fbcfa2eaa706bbae8eb9f923a4b85a25c0a7c29faf"}
ZHU_GRIDS = {"Hyogoken-Nambu": "usp0006rew/ground-failure-preferred/us-p0006rew-v1-1597402383402/zhu_2017_general.hdf5",
             "Chi-Chi": "usp0009eq0/ground-failure-preferred/us-p0009eq0-v1-1597409918955/zhu_2017_general.hdf5"}
ZHU_ROOT = PILLAR / "data" / "comcat-shakemap-nonus-2026-11-07" / "raw"
TILE_M, MIN_SITES, MIN_CLASS, MIN_TILES = 4000.0, 200, 30, 10
W_TILE_M, W_MIN_CLASS, W_MIN_TILES = 1000.0, 3, 5
NOTE = "not certified; lab coverage 0.80 to 0.96 depending on an unidentified range"
EVDATE = {"sep2010": "2010-09-04", "feb2011": "2011-02-22", "jun2011": "2011-06-13", "dec2011": "2011-12-23"}


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


# v1 code path (hash-checked inputs, rows, p0, pu, LSN, GEM, fragility, baselines) via the step 1c script
EB = _load("effort_baseline", PILLAR / "docs" / "evals" / "effort_baseline.py")
B = EB.B
from scoreboard import label_sets as LS  # noqa: E402
from scoreboard.evalkit import card as C, effort as EF, metrics as M, pool as PL, resample as RS  # noqa: E402
from scoreboard.evalkit import chain_settings as CS, spec as SP  # noqa: E402
from scoreboard.evalkit.spec import Entry, EntryManifest, Labels, Task, site_list_sha256  # noqa: E402

rows, EVS, SEED, N_BOOT, CLIP = B.rows, B.EVS, B.SEED, B.N_BOOT, B.CLIP
d_rows = B.d
g6 = _load("g6", KIT / "scripts" / "run_g6_jun2011.py")
g1c = _load("g1c", KIT / "scripts" / "run_g1_curated.py")
V1J = json.loads((PILLAR / "docs" / "scorecard-v1-2026-11-15" / "scorecard.json").read_text())
LEDGER_NAME = {"Level 0": "Level 0", "M0 (Level 0 + earthquake band)": "M0 (Level 0 + earthquake band)",
               "LSN + global fragility [authors-maurer-2026-11-19]": "LSN + global fragility",
               "LSN + CES fragility [authors-maurer-2026-11-19]": "LSN + CES fragility",
               "LSN + global fragility [default-2026-11]": "LSN + global fragility",
               "LSN + CES fragility [default-2026-11]": "LSN + CES fragility",
               "USGS Zhu 2017": "USGS Zhu 2017", "GEM TS22": "GEM TS22"}
V1_NAME = {"Level 0": "Level 0", "M0 (Level 0 + earthquake band)": "M0 (Level 0 + earthquake band)",
           "LSN + global fragility [default-2026-11]": "LSN + global fragility",
           "LSN + CES fragility [default-2026-11]": "LSN + CES fragility",
           "USGS Zhu 2017": "USGS Zhu 2017", "GEM TS22": "GEM TS22"}
log("v1 code path loaded", len(rows))


# -------------------------------------------------------------------------------------------------------------------
# LSN under both chain ids (registration 4.1), gate G-lsn
# -------------------------------------------------------------------------------------------------------------------
def lsn_both_ids():
    mod = g6.bench_module()
    P, RC, Cq, ld = mod._pc()
    drv = RC._load_drivers()
    tr = RC._load_traces()
    pre = RC._load_predrill(drv["scpg_id"])
    corpus = np.load(os.path.join(os.path.dirname(P.__file__), "..", "canterbury", "build", "corpus.npz"),
                     allow_pickle=True)
    R = np.load(B.ROWS_NPZ, allow_pickle=True)
    bb2 = mod.load_bench_v2()
    bb2 = bb2[bb2["split"].isin(["train", "val"])].reset_index(drop=True)
    LS.check_no_test(bb2)
    feb = bb2[bb2["event"] == "2011-Christchurch"].sort_values("cpt").reset_index(drop=True)
    if not np.array_equal(feb["cpt"].to_numpy(), R["cpt"]):
        raise SystemExit("G-lsn: bench Feb rows are not the rows npz order")
    i_feb = feb["row_id"].to_numpy() - 1
    u2 = np.where(corpus["valid"][i_feb, :, 2], corpus["values"][i_feb, :, 2], np.nan).astype(np.float64)
    gw = pd.read_parquet(g6.GWT)
    out, gate = {}, {}
    for e in EVS:
        s = gw[gw["event"] == g6.GWT_EV[e]].drop_duplicates("cpt_id").set_index("cpt_id")
        mid = s["gwt_depth_mid_m"].reindex(feb["cpt"]).to_numpy(float)
        gwt = np.clip(np.where(np.isfinite(mid), mid, feb["gwt_store_m"].to_numpy(float)), 0, None)
        n = len(feb)
        for sid in (CS.DEFAULT_ID, CS.AUTHORS_ID):
            v = np.full(n, np.nan)
            for a in range(0, n, 1500):
                j = slice(a, min(a + 1500, n))
                r = CS.run(P, Cq, tr["z"], tr["qc"][i_feb[j]], tr["fs"][i_feb[j]], gwt[j], R[f"pga_{e}"][j],
                           np.full(j.stop - j.start, g6.MW[e]), sid, u2=u2[j], predrill_m=pre[i_feb[j]])
                v[j] = r["lsn"]
            out[(sid, e)] = v
        ref = B.LSN[f"lsn_{e}"]
        mine = out[(CS.DEFAULT_ID, e)]
        both = np.isfinite(ref) & np.isfinite(mine)
        gate[e] = {"max_abs_diff": float(np.max(np.abs(ref[both] - mine[both]))),
                   "finite_pattern_equal": bool(np.array_equal(np.isfinite(ref), np.isfinite(mine))),
                   "authors_finite": int(np.isfinite(out[(CS.AUTHORS_ID, e)]).sum()),
                   "authors_over_default_median_where_default_pos": float(np.median(
                       out[(CS.AUTHORS_ID, e)][mine > 0] / mine[mine > 0]))}
        log("lsn", e, gate[e])
    ok = all(g["max_abs_diff"] <= 1e-9 and g["finite_pattern_equal"] for g in gate.values())
    if not ok:
        raise SystemExit(f"G-lsn failed: {gate}")
    for (sid, e), v in out.items():
        if not np.all(np.isfinite(v)):
            raise SystemExit(f"LSN {sid} {e}: non-finite values")
    return out, {"pass": ok, "per_earthquake": gate}, push_dates_of(feb)


def push_dates_of(feb):
    return pd.to_datetime(feb["push_date"]).to_numpy()


# -------------------------------------------------------------------------------------------------------------------
# Canterbury cards
# -------------------------------------------------------------------------------------------------------------------
def entries_for(e, msk, ids, lsn):
    base = {en.manifest.model: en for en in B.entries_for(e, msk, ids)}
    out = [base["Level 0"], base["M0 (Level 0 + earthquake band)"]]
    for sid in (CS.AUTHORS_ID, CS.DEFAULT_ID):
        v = lsn[(sid, e)][msk]
        for short, key in (("global", "C-GLOBAL"), ("CES", "C-CES")):
            name = f"LSN + {short} fragility [{sid}]"
            stmt = (f"BI14 LSN under chain setting {sid} (evalkit.chain_settings) through the frozen conversion "
                    f"{key} (x_m {B.CONV[key]['x_m']}, beta {B.CONV[key]['beta']})")
            m = EntryManifest(model=name, output_kind="score", statement=stmt,
                              exposure={e: B.EXPO[f"LSN + {short} fragility"][e]}, house=True,
                              conversion_id=B.canon_sha({**B.CONV[key], "chain_setting": sid}), division="open")
            out.append(Entry(m, ids, B.fragility(v, B.CONV[key])))
    out += [base["USGS Zhu 2017"], base["GEM TS22"]]
    return out


def baseline_parts(earlier, e, msk, ids):
    if not earlier:
        return None, None, None
    ey = np.concatenate([x[0] for x in earlier])
    ep = np.concatenate([x[1] for x in earlier])
    rate = float(ey.sum() / ey.size)
    bins = B.shaking_rates(ep, ey)
    sm = EntryManifest(model="baseline: shaking only", output_kind="target_probability",
                       statement="pooled yes share per PGA bin over earlier label sets of this set",
                       exposure={e: "unseen"}, house=False)
    return Entry(sm, ids, B.read_bins(bins, d_rows["pga_" + e][msk])), rate, bins


def one_card(sname, e, S, msk, lsn, effort_p, earlier, tag):
    lab = S[e][0]
    ids = tuple(B.SID[msk].tolist())
    y = lab[msk].astype(float)
    n_yes = int(y.sum())
    out = {"population": tag, "n": int(y.size), "n_yes": n_yes}
    if y.size == 0:
        return {**out, "status": "NOT AVAILABLE", "why": "no site in this population"}
    labels = Labels(ids, y, rows.x[msk], rows.y[msk])
    ents = entries_for(e, msk, ids, lsn)
    shaking, rate, bins = baseline_parts(earlier, e, msk, ids)
    src = LS.SETS[sname][e]
    task = Task(event_id=e, target="liquefaction ejecta seen at the surface at the CPT (any severity)",
                label_source=f"set {sname}: {src}", label_rule_sha256=B.label_rule_sha(sname, e),
                site_list_sha256=site_list_sha256(ids), track="record", division="open", tile_m=TILE_M, clip=CLIP,
                seed=SEED, label_caveats=f"{B.CAVEAT[src]}. {B.COMMON_CAVEAT}", n_boot=N_BOOT,
                min_sites=MIN_SITES, min_per_class=MIN_CLASS, min_tiles=MIN_TILES, earlier_base_rate=rate,
                earlier_label_sets=len(earlier), tile_rule="fixed")
    eff = EF.entry(ids, effort_p[msk], e, "pre-event effort baseline (evalkit.effort, allowlist pre-event), "
                                          "fitted leave one earthquake out within this label set")
    card = C.build_card(task, ents, labels, shaking_only=shaking, effort_only=eff)
    card = C.descriptive(card, NOTE)
    card["shaking_bins"] = ([{"pga_from_g": b[0], "pga_to_g": (None if np.isinf(b[1]) else b[1]), "n": b[2],
                              "yes": b[3]} for b in bins] if bins else "NOT AVAILABLE")
    card["earthquake"] = B.EVNAME[e]
    card["population"] = tag
    card["tiles_4km"] = int(np.unique(RS.tiles_for(rows.x[msk], rows.y[msk], TILE_M)).size)
    return card


def exposure_classes(ledger, name, e):
    col = ledger[f"{LEDGER_NAME[name]}|{e}"].to_numpy()
    others = [ledger[f"{LEDGER_NAME[name]}|{o}"].to_numpy() for o in EVS if o != e]
    other_seen = np.zeros(col.size, bool)
    for o in others:
        other_seen |= (o == "SEEN") | (o == "POSSIBLE")
    cls = np.where(col == "SEEN", "SEEN", np.where(col == "POSSIBLE", "POSSIBLE",
                                                     np.where(col == "NOT CHECKED", "NOT CHECKED",
                                                              np.where(other_seen, "PLACE", "UNSEEN"))))
    return cls


def unseen_read(card, sname, e, S, msk_full, lsn, effort_p, earlier, ledger):
    """Registration section 5: each entry and the baselines on that entry's UNSEEN sites (same tile rule)."""
    lab = S[e][0]
    ids_full = B.SID[msk_full]
    out = {}
    ents_full = {en.manifest.model: np.asarray(en.p, float) for en in entries_for(e, msk_full, tuple(ids_full), lsn)}
    shaking, rate, _ = baseline_parts(earlier, e, msk_full, tuple(ids_full))
    base = {"baseline: effort only": effort_p[msk_full]}
    if shaking is not None:
        base["baseline: shaking only"] = np.asarray(shaking.p, float)
    for name, v in ents_full.items():
        cls = exposure_classes(ledger, name, e)[msk_full]
        counts = {k: int((cls == k).sum()) for k in ("SEEN", "POSSIBLE", "PLACE", "UNSEEN", "NOT CHECKED")}
        k = cls == "UNSEEN"
        y = lab[msk_full][k].astype(bool)
        xx, yy = rows.x[msk_full][k], rows.y[msk_full][k]
        nt = int(np.unique(RS.tiles_for(xx, yy, TILE_M)).size) if k.any() else 0
        rec = {"exposure_counts_labelled_sites": counts, "unseen_n": int(k.sum()), "unseen_yes": int(y.sum()),
               "unseen_tiles_4km": nt}
        if not (k.sum() >= MIN_SITES and y.sum() >= MIN_CLASS and (~y).sum() >= MIN_CLASS and nt >= MIN_TILES):
            rec["read"] = "COUNTS ONLY (under the count gate)"
            out[name] = rec
            continue
        series = {name: v[k], **{b: bv[k] for b, bv in base.items()}}
        tiles = RS.tiles_for(xx, yy, TILE_M)

        def stat(idx, series=series, y=y):
            yi = y[idx]
            if yi.all() or not yi.any():
                return {s: np.nan for s in series}
            return {s: M.auc(yi, sv[idx]) for s, sv in series.items()}

        pt = stat(np.arange(y.size))
        dr = RS.bootstrap(stat, tiles, N_BOOT, SEED)
        rec["auc"] = {s: RS.interval(pt[s], dr[s]) for s in series}
        rec["minus"] = {b: {kk: vv for kk, vv in RS.paired(pt[name], pt[b], dr[name], dr[b]).items()
                            if kk not in ("p", "mdd")} for b in base}
        rec["read"] = "DESCRIPTIVE"
        out[name] = rec
    return out


def proxy_auc(sname, e, S, proxy_p, msk):
    y = S[e][0][msk].astype(bool)
    tiles = RS.tiles_for(rows.x[msk], rows.y[msk], TILE_M)
    v = proxy_p[msk]

    def stat(idx):
        yi = y[idx]
        return {"a": np.nan} if (yi.all() or not yi.any()) else {"a": M.auc(yi, v[idx])}
    return RS.interval(stat(np.arange(y.size))["a"], RS.bootstrap(stat, tiles, N_BOOT, SEED)["a"])


def canterbury(lsn, push):
    feats, raw, desc = EB.build_features()
    ledger = pd.read_csv(LEDGER).set_index("cpt").reindex(rows.cpt)
    if ledger.isna().all(axis=1).any():
        raise SystemExit("ledger misses a site")
    out = {"cards": {}, "pre_event_cpt_cards": {}, "unseen_reads": {}, "diagnostic_proxy_auc": {},
           "effort_fits": {}, "v1_auc_gate": {}}
    worst = 0.0
    for sname in ("B", "A"):
        S = LS.build_label_set(sname, rows)
        labels_all = {k: v[0] for k, v in S.items()}
        out["cards"][sname], out["pre_event_cpt_cards"][sname], out["unseen_reads"][sname] = {}, {}, {}
        out["diagnostic_proxy_auc"][sname], out["effort_fits"][sname] = {}, {}
        earlier = []
        for e in EVS:
            lab, msk = S[e]
            eff_p, f = EF.fit_predict_loeo(feats["baseline"], labels_all, e, EF.EFFORT_FEATURES)
            prox_p, _ = EF.fit_predict_loeo(feats["proxy"], labels_all, e, EF.PROXY_FEATURES)
            out["effort_fits"][sname][e] = f.summary()
            card = one_card(sname, e, S, msk, lsn, eff_p, earlier, "all labelled sites")
            out["cards"][sname][e] = card
            v1 = V1J["cards"][sname][e]
            if v1.get("status") == "SCORED" and card.get("status") == "SCORED":
                for mine, theirs in V1_NAME.items():
                    worst = max(worst, abs(card["metrics"][mine]["auc"]["value"] -
                                           v1["metrics"][theirs]["auc"]["value"]))
            pre = msk & (push < np.datetime64(EVDATE[e]))
            out["pre_event_cpt_cards"][sname][e] = one_card(sname, e, S, pre, lsn, eff_p, earlier,
                                                            f"CPTs pushed before {EVDATE[e]}")
            out["unseen_reads"][sname][e] = unseen_read(card, sname, e, S, msk, lsn, eff_p, earlier, ledger)
            out["diagnostic_proxy_auc"][sname][e] = proxy_auc(sname, e, S, prox_p, msk)
            earlier.append((lab[msk].astype(float), d_rows["pga_" + e][msk]))
            log(sname, e, card.get("status"), "pre-event", out["pre_event_cpt_cards"][sname][e].get("status"),
                out["pre_event_cpt_cards"][sname][e].get("n"))
    out["v1_auc_gate"] = {"pass": worst <= 1e-12, "max_abs_diff": worst,
                          "how": "kit AUC of the v1 entries (default id LSN rows) on every v1 card that printed AUC"}
    if not out["v1_auc_gate"]["pass"]:
        raise SystemExit(f"G-v1 failed: {worst}")
    out["effort_feature_summary"] = desc
    return out


def ceilings():
    c = json.loads(LER.read_text())["cells"]
    src = {"B": {"sep2010": "bench", "feb2011": "bench", "jun2011": "road25", "dec2011": "aerial"},
           "A": {e: "aerial" for e in EVS}}
    out = {}
    for s, m in src.items():
        out[s] = {}
        for e, so in m.items():
            hits = {k.split("|")[2] or "measured": {"C": v["C"], "ci95": v["ci95"]["C"]}
                    for k, v in c.items() if k.startswith(f"{e}|{so}|")}
            out[s][e] = {"source": so, "menon_C": hits,
                         "status": {"sep2010": "ASSUMPTION-DRIVEN (bench as reference)", "feb2011": "IDENTIFIED",
                                    "jun2011": "SCENARIOS ONLY", "dec2011": "SCENARIOS ONLY"}[e]}
    return out


def pooled_canterbury(cards_b):
    lines = {"C1": ("Level 0", "baseline: shaking only", ("feb2011", "jun2011", "dec2011")),
             "C2": ("Level 0", "baseline: effort only", EVS)}
    items = [[PL.card_item(cards_b[e], a, b, "auc") for e in evs] for a, b, evs in lines.values()]
    re_ = PL.pooled_lines(items, registered=True)
    eq_ = PL.pooled_lines(items, registered=True, weighting="equal")
    out = {}
    for (lid, (a, b, evs)), r, q in zip(lines.items(), re_, eq_):
        for x in (r, q):
            x["word"] = "DESCRIPTIVE"
            x.pop("holm", None)
        out[lid] = {"population": "Canterbury, set B", "a": a, "b": b, "metric": "auc", "earthquakes": list(evs),
                    "primary_kit_random_effects": r, "beside_equal_weight": q,
                    "why_descriptive": "Certificate 4 was certified at a 2 km model-error range; the real "
                                       "model-difference range is not identified (LAB-CERTIFICATES 11.4, 12.1)"}
    return out


def sign_agreement(cards):
    out = {}
    for e in ("sep2010", "dec2011"):
        a, b = cards["A"][e], cards["B"][e]
        if a.get("status") != "SCORED" or b.get("status") != "SCORED":
            out[e] = "NOT AVAILABLE"
            continue
        ca = {(c["family"], c["a"], c["b"]): c["diff"] for c in a["comparisons"]}
        agree = dis = 0
        for c in b["comparisons"]:
            k = (c["family"], c["a"], c["b"])
            if k in ca and c["diff"] != 0 and ca[k] != 0:
                agree += int(np.sign(c["diff"]) == np.sign(ca[k]))
                dis += int(np.sign(c["diff"]) != np.sign(ca[k]))
        out[e] = {"same_sign": agree, "opposite_sign": dis,
                  "note": "Dec 2011 sets A and B use the same aerial labels" if e == "dec2011" else ""}
    return out


# -------------------------------------------------------------------------------------------------------------------
# World case histories (registration 3.2, 4.2)
# -------------------------------------------------------------------------------------------------------------------
def world():
    import h5py
    for f, h in WORLD_SHA.items():
        if sha_file(WORLD / f) != h:
            raise SystemExit(f"G-world: {f} hash mismatch")
    cs = pd.read_parquet(WORLD / "cases.parquet")
    tr = np.load(WORLD / "traces.npz")
    mod = g6.bench_module()
    P, RC, Cq, _ = mod._pc()
    Z = g1c.Z
    use = cs[cs["pop_union"] | cs["pop_g1_source_rule"]].copy()
    grids, keep = [], []
    off = tr["offsets"]
    for _, r in use.iterrows():
        a, b = int(off[r["row"]]), int(off[r["row"] + 1])
        g = g1c.to_grid(tr["depth_m"][a:b], tr["qc_kpa"][a:b], tr["fs_kpa"][a:b])
        if g is None:
            raise SystemExit("a scorable world case has fewer than 20 readings")
        grids.append(g)
    qc = np.stack([g[0] for g in grids])
    fs = np.stack([g[1] for g in grids])
    gwt = np.where(use["pop_union"].to_numpy() | use["pop_g1_source_rule"].to_numpy(),
                   use["gwt_m"].to_numpy(float), np.nan)
    pga, mw = use["pga_g"].to_numpy(float), use["mw"].to_numpy(float)
    if not np.all(np.isfinite(gwt) & np.isfinite(pga) & np.isfinite(mw)):
        raise SystemExit("world: a population case lacks PGA, water table or Mw")
    ii = np.arange(len(use))
    lnk = g6.lnk_star(P, Z, qc, fs, ii, np.clip(gwt, 0, None), pga, mw)
    use["level0"] = B.g5.p_level0(lnk, np.zeros(lnk.size))
    r = CS.run(P, Cq, Z, qc, fs, np.clip(gwt, 0, None), pga, mw, CS.DEFAULT_ID)
    use["lsn"], use["lpi"] = r["lsn"], r["lpi"]
    # cross-check against the registered G1 curated path (grid_values: lpi_block + RC.indices, 15 m)
    chk = []
    for k in range(len(use)):
        gv = g1c.grid_values({"qc": qc[k], "fs": fs[k], "gwt": float(gwt[k]), "pga": float(pga[k]),
                              "mw": float(mw[k])}, mod, P, RC, Cq)
        chk.append((gv["BI14-LPI"], gv["BI14-LSN"]))
    chk = np.array(chk)
    xcheck = {"lpi_max_abs_diff_vs_g1_curated": float(np.nanmax(np.abs(chk[:, 0] - use["lpi"].to_numpy()))),
              "lsn_max_abs_diff_vs_g1_curated": float(np.nanmax(np.abs(chk[:, 1] - use["lsn"].to_numpy()))),
              "nan_pattern_equal": bool(np.array_equal(np.isfinite(chk), np.isfinite(
                  np.column_stack([use["lpi"], use["lsn"]]))))}
    use["zhu"] = np.nan
    zhu_info = {}
    for eq, rel in ZHU_GRIDS.items():
        f = h5py.File(ZHU_ROOT / rel, "r")
        gx, gy, gm = f["x"][()], f["y"][()], f["model"][()]
        k = (use["earthquake"] == eq).to_numpy()
        ix = np.abs(gx[None, :] - use.loc[k, "lon"].to_numpy()[:, None]).argmin(axis=1)
        iy = np.abs(gy[None, :] - use.loc[k, "lat"].to_numpy()[:, None]).argmin(axis=1)
        v = gm[iy, ix].astype(float)
        zhu_info[eq] = {"cases": int(k.sum()), "nan_cells_set_to_0": int(np.isnan(v).sum()),
                        "grid_sha256": sha_file(ZHU_ROOT / rel)}
        use.loc[k, "zhu"] = np.nan_to_num(v, nan=0.0)
    out = {"cross_check": xcheck, "zhu_grids": zhu_info, "populations": {}}
    for pop, col in (("P1", "pop_g1_source_rule"), ("P2", "pop_union")):
        out["populations"][pop] = world_population(use[use[col]].copy())
    log("world done", {p: len(v["earthquakes"]) for p, v in out["populations"].items()})
    return out, use


def world_population(df):
    res = {"earthquakes": {}}
    rows_ = ["Level 0", "BI14-LSN (both LSN fragility rows)", "BI14-LPI", "USGS Zhu 2017", "baseline: shaking only"]
    col = {"Level 0": "level0", "BI14-LSN (both LSN fragility rows)": "lsn", "BI14-LPI": "lpi",
           "USGS Zhu 2017": "zhu", "baseline: shaking only": "pga_g"}
    for eq, g in df.groupby("earthquake", sort=True):
        y = (g["label"].to_numpy(int) > 0)
        lat, lon = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
        x = 6371000.0 * np.cos(np.deg2rad(lat.mean())) * np.deg2rad(lon - lon.mean())
        yy = 6371000.0 * np.deg2rad(lat - lat.mean())
        tiles = RS.tiles_for(x, yy, W_TILE_M)
        nt = int(np.unique(tiles).size)
        rec = {"region": str(g["region"].iloc[0]), "n": int(y.size), "n_yes": int(y.sum()), "tiles_1km": nt,
               "exposure": {"B&I chain (Level 0, M0, LSN, LPI)": g["exp_bi14"].value_counts().to_dict(),
                            "LSN global fragility": g["exp_lsn_global_fragility"].value_counts().to_dict(),
                            "USGS Zhu 2017": g["exp_zhu2017"].value_counts().to_dict()},
               "sources": g["source"].value_counts().to_dict()}
        passes = y.sum() >= W_MIN_CLASS and (~y).sum() >= W_MIN_CLASS and nt >= W_MIN_TILES
        rec["passes_record_gate"] = bool(passes)
        if not passes:
            rec["read"] = "COUNTS ONLY (under the record gate)"
            res["earthquakes"][eq] = rec
            continue
        have = [r for r in rows_ if np.all(np.isfinite(g[col[r]].to_numpy(float)))]
        if "USGS Zhu 2017" in have and eq not in ZHU_GRIDS:
            have.remove("USGS Zhu 2017")
        series = {r: g[col[r]].to_numpy(float) for r in have}

        def stat(idx, series=series, y=y):
            yi = y[idx]
            if yi.all() or not yi.any():
                return {s: np.nan for s in series}
            return {s: M.auc(yi, v[idx]) for s, v in series.items()}

        pt = stat(np.arange(y.size))
        dr = RS.bootstrap(stat, tiles, N_BOOT, SEED)
        rec["auc"] = {s: RS.interval(pt[s], dr[s]) for s in series}
        rec["not_run"] = [r for r in rows_ if r not in have] + ["GEM TS22 (inputs not on disk)"]
        rec["paired"] = {}
        for a, b in (("Level 0", "baseline: shaking only"), ("Level 0", "USGS Zhu 2017"),
                     ("BI14-LSN (both LSN fragility rows)", "baseline: shaking only"),
                     ("BI14-LPI", "baseline: shaking only")):
            if a in series and b in series:
                rec["paired"][f"{a} minus {b}"] = {k: v for k, v in RS.paired(pt[a], pt[b], dr[a], dr[b]).items()
                                                   if k not in ("p", "mdd")}
                rec["paired"][f"{a} minus {b}"]["word"] = "DESCRIPTIVE"
        rec["read"] = "DESCRIPTIVE"
        res["earthquakes"][eq] = rec
    return res


def pooled_world(pops):
    out = {}
    for pop in ("P1", "P2"):
        eqs = pops[pop]["earthquakes"]
        groups = {e: v["region"] for e, v in eqs.items()}
        for lid, a, b in (("W1", "Level 0", "baseline: shaking only"), ("W2", "Level 0", "USGS Zhu 2017")):
            items = []
            for e, v in eqs.items():
                key = f"{a} minus {b}"
                ok = v.get("passes_record_gate") and key in v.get("paired", {})
                if v.get("passes_record_gate") is False or (lid == "W2" and not ok):
                    continue
                pr = v["paired"][key]
                items.append({"event_id": e, "entry_a": a, "entry_b": b, "metric": "auc", "diff": pr["diff"],
                              "se": pr["se"], "passes_gate": True})
            eq_ = PL.pooled_line(items, registered=True, weighting="equal") if items else {
                "word": "NOT ALLOWED", "reason": "no earthquake has both rows"}
            re_ = PL.pooled_line(items, registered=True) if items else None
            rec = {"population": f"World {pop}", "a": a, "b": b, "metric": "auc",
                   "primary_equal_weight": eq_, "beside_kit_weights": re_}
            for x in (eq_, re_):
                if x is not None and x.get("word") != "NOT ALLOWED":
                    x["word"] = "DESCRIPTIVE"
                    x.pop("holm", None)
            if eq_.get("word") == "DESCRIPTIVE":
                rec["leave_one_region_out"] = PL.leave_one_group_out(items, groups, weighting="equal")
            rec["why_descriptive"] = "no certificate covers equal weighting or world earthquakes"
            out[f"{lid} {pop}"] = rec
            if pop == "P2":
                rec["note"] = "beside, no word (registration 3.2)"
    return out


def main():
    kg = C.gates.run_gates()
    if not kg["pass"]:
        raise SystemExit("G-k failed")
    lsn, g_lsn, push = lsn_both_ids()
    cant = canterbury(lsn, push)
    pooled_c = pooled_canterbury(cant["cards"]["B"])
    wld, use = world()
    pooled_w = pooled_world(wld["populations"])
    kit_commit = subprocess.run(["git", "-C", str(KIT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    res = {"registration": REGISTRATION, "track": "RECORD TRACK, RETROSPECTIVE", "rulebook": "draft, not adopted",
           "statements": ["every label was read before", "no skill test", "no winner named",
                          "external parity NOT ACHIEVED (2 registered attempts DISAGREE: PARITY-M3-2026-11-19.md)",
                          "audit (step 1d) not graded", "rule book draft, not adopted",
                          "no word within an earthquake; every pooled line DESCRIPTIVE",
                          "founder ruling on the coverage truth (models fixed or averaged) pending; a models-fixed "
                          "ruling could allow per-earthquake words only under a NEW registration"],
           "interval_note": NOTE,
           "kit": {"branch": "evalkit-2026-11-15", "commit": kit_commit},
           "seed": SEED, "n_boot": N_BOOT, "tile_m": TILE_M, "tile_grid": "NZTM2000, anchored at the grid origin",
           "count_gate": {"sites": MIN_SITES, "per_class": MIN_CLASS, "tiles_4km": MIN_TILES},
           "world_record_gate": {"per_class": W_MIN_CLASS, "tiles_1km": W_MIN_TILES},
           "gates": {"G-k kit self-checks": {"pass": kg["pass"]}, "G-lsn default id reproduces v1 LSN": g_lsn,
                     "G-v1 kit AUC equals v1 card AUC": cant["v1_auc_gate"],
                     "G-world dataset hashes": {"pass": True, **WORLD_SHA},
                     "G-t test split unread": {"pass": True, "how": "label_sets.check_no_test on every bench read"}},
           "inputs_sha256": {"site_ledger.csv": sha_file(LEDGER), **{p.name: h for p, h in B.EXPECT.items()}},
           "label_ceilings_menon_C": ceilings(),
           "canterbury": {k: v for k, v in cant.items() if k != "v1_auc_gate"},
           "pooled_lines": {**pooled_c, **pooled_w},
           "label_source_sign_agreement": sign_agreement(cant["cards"]),
           "world": wld,
           "attribution": ["PRJ-3012: Geyin and Maurer, global CPT liquefaction case histories, DesignSafe, "
                           "doi:10.17603/ds2-wftt-mv37, ODC-By", "NGL: Next Generation Liquefaction database "
                           "(nextgenerationliquefaction.org)", "NZGD (Canterbury CPTs), terms clause 6.2: aggregates "
                           "only", "PRJ-2937: Geyin, Maurer, Bradley, Green and van Ballegooy 2020, ODC-By"]}
    if LEDGER_SHA != res["inputs_sha256"]["site_ledger.csv"]:
        raise SystemExit("ledger hash mismatch")
    res = C._json_safe(res)
    ids = set(str(c) for c in rows.cpt) | set(B.SID.tolist())
    for c in ("case_id", "cpt_id", "site_id"):
        ids |= set(use[c].dropna().astype(str))
    ids = {i for i in ids if i.strip() and i not in ("nan", "None", "<NA>")}
    C.scan_aggregates(res, sorted(ids))
    res["gates"]["G-c no per-site output"] = {"pass": True, "how": "card.scan_aggregates on the whole file"}
    res["card_sha256"] = SP.card_sha256({k: v for k, v in res.items() if k != "card_sha256"})
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "scorecard.json").write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    (OUTD / "scan_ids.sha256").write_text(hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() + "\n")
    log("wrote", OUTD / "scorecard.json", res["card_sha256"])


if __name__ == "__main__":
    main()

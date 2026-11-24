"""Aggregate lab runs into the protocol's performance measures (section P) and decision rules. No em dashes.

  <venv python> -m scoreboard.evalkit.lab.aggregate <run_dir> <out.json>

Reads <run_dir>/<condition>.npz written by lab.run. Writes aggregates only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from .. import pool, resample
from .run import COMPS, RUN_ORDER, SIZES

RULES = ("variogram", "trend-variogram-v2", "pairdiff-variogram-v2", "fixed-1km", "fixed-2km", "fixed-4km",
         "fixed-8km")
METRICS = ("auc", "log_loss")
EVENTS = ("sep2010", "feb2011", "jun2011", "dec2011")
LO, HI = 0.93, 0.97
D, L, H, P, SE, BELOW = range(6)


def load(run_dir: Path, cond: str) -> dict | None:
    f = run_dir / f"{cond}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def chosen(r: dict, rule: str) -> np.ndarray:
    """Size index per (world, event, comp, metric)."""
    W, E = r["m1b"].shape
    if rule == "variogram":
        idx = np.broadcast_to(r["m1a"][:, :, None, None], (W, E, 3, 2))
    elif rule == "trend-variogram-v2":
        idx = np.broadcast_to(r["m1b"][:, :, None, None], (W, E, 3, 2))
    elif rule == "pairdiff-variogram-v2":
        idx = r["m1c"]
    else:
        side = float(rule.split("-")[1].rstrip("km")) * 1000.0
        idx = np.full((W, E, 3, 2), SIZES.index(side))
    return np.asarray(idx, int)


def pick(r: dict, idx: np.ndarray) -> np.ndarray:
    """res at the chosen size: (W, E, 3, 2, 6); invalid (-1) -> NaN."""
    res = r["res"]                                   # W, E, S, 3, 2, 6
    safe = np.clip(idx, 0, len(SIZES) - 1)
    out = np.take_along_axis(res, safe[:, :, None, :, :, None], axis=2)[:, :, 0]
    out[idx < 0] = np.nan
    return out


def thetas(cond: str, r: dict) -> tuple[np.ndarray, np.ndarray]:
    """theta per (event position, comp, metric) and a mask of exact-zero (null) cells."""
    diff = r["res"][:, :, 0, :, :, D]                # point differences do not depend on the size
    W, E = diff.shape[:2]
    th = np.nanmean(diff, axis=0)
    exact = np.zeros((E, 3, 2), bool)
    if cond == "B-NULL":
        exact[:] = True
    else:
        exact[:, 2, :] = True                         # E2 - E3 exchangeable
    if cond == "B-CAL":
        exact[:, 0, :] = True                         # E1, E2 exchangeable (f1 = 1)
        exact[:, 1, 0] = True                         # E1 vs shifted E2: AUC same as E1 vs E2
        exact[:, 2, 1] = False                        # log loss of the shifted copy differs
    th = np.where(exact, 0.0, th)
    return th, exact


def degenerate_mask(cond: str, r: dict) -> np.ndarray:
    """(E, 3, 2) cells left out of coverage: B-CAL E2 - E3 AUC is identical by construction."""
    E = r["res"].shape[1]
    m = np.zeros((E, 3, 2), bool)
    if cond == "B-CAL":
        m[:, 2, 0] = True
    return m


def gate(r: dict, idx: np.ndarray) -> np.ndarray:
    """Count gate per (W, E, 3, 2) at the chosen size: 200 sites, 30 per class, 20 tiles."""
    n, npos = r["n"], r["n_pos"]
    tiles = np.take_along_axis(r["n_tiles"][:, :, :, None, None].repeat(3, 3).repeat(2, 4),
                               np.clip(idx, 0, len(SIZES) - 1)[:, :, None], axis=2)[:, :, 0]
    ok = (n >= 200) & (np.minimum(npos, n - npos) >= 30)
    return ok[:, :, None, None] & (tiles >= 20), tiles


def cover_stats(cov: np.ndarray) -> dict:
    """cov: (W, k) with NaN for left-out cells; coverage with a world-clustered Monte Carlo SE."""
    per_w = np.nanmean(cov, axis=1)
    per_w = per_w[np.isfinite(per_w)]
    if per_w.size < 2:
        return {"coverage": None, "mcse": None, "n_worlds": int(per_w.size), "status": "NOT RUN"}
    c, se = float(per_w.mean()), float(per_w.std(ddof=1) / np.sqrt(per_w.size))
    lo, hi = c - 1.96 * se, c + 1.96 * se
    status = "PASS" if (hi >= LO and lo <= HI) else ("UNDER" if hi < LO else "OVER")
    return {"coverage": round(c, 4), "mcse": round(se, 4), "n_worlds": int(per_w.size),
            "n_intervals": int(np.isfinite(cov).sum()), "status": status}


def coverage_cells(cond: str, r: dict, rule: str, gate_only: bool = False) -> dict:
    idx = chosen(r, rule)
    x = pick(r, idx)
    th, _ = thetas(cond, r)
    deg = degenerate_mask(cond, r)
    inside = (x[..., L] <= th[None]) & (th[None] <= x[..., H])
    cov = np.where(np.isfinite(x[..., L]) & ~deg[None], inside.astype(float), np.nan)
    if gate_only:
        g, _ = gate(r, idx)
        cov = np.where(g, cov, np.nan)
    E = r["res"].shape[1]
    out = {}
    for e in range(E):
        for mi, m in enumerate(METRICS):
            name = f"{EVENTS[int(r['slot'][0, e])]}#{e}/{m}"
            out[name] = cover_stats(cov[:, e, :, mi])
    for mi, m in enumerate(METRICS):
        out[f"all/{m}"] = cover_stats(cov[:, :, :, mi].reshape(cov.shape[0], -1))
    return out


def widths(r: dict, rule: str) -> dict:
    x = pick(r, chosen(r, rule))
    w = x[..., H] - x[..., L]
    return {m: float(np.nanmedian(w[..., mi])) for mi, m in enumerate(METRICS)}


def holm_words(cond: str, r: dict, rule: str) -> dict:
    """Holm within each event family (3 comps, one metric). Null rate on exact-zero comps; power on the others."""
    x = pick(r, chosen(r, rule))
    th, exact = thetas(cond, r)
    deg = degenerate_mask(cond, r)
    W, E = x.shape[:2]
    rej = np.zeros((W, E, 3, 2), bool)
    for w in range(W):
        for e in range(E):
            for mi in range(2):
                p = x[w, e, :, mi, P]
                if np.all(np.isfinite(p)):
                    rej[w, e, :, mi] = resample.holm(list(p))
    finite = np.isfinite(x[..., P])
    nullm = exact[None] & finite & ~deg[None]
    out = {"null_rate_holm": None, "null_rate_unadjusted": None, "fwer_event_family": None, "power": {}}
    if nullm.any():
        k = int(nullm.sum())
        rate = float(rej[nullm].mean())
        out["null_rate_holm"] = {"rate": round(rate, 4), "n": k,
                                 "upper95": round(float(stats.beta.ppf(0.975, rej[nullm].sum() + 1, k - rej[nullm].sum())), 4)}
        un = (x[..., P] < 0.05)[nullm]
        out["null_rate_unadjusted"] = {"rate": round(float(un.mean()), 4), "n": k}
    if cond == "B-NULL":
        fam = rej.any(axis=2)                          # W, E, 2
        out["fwer_event_family"] = {m: round(float(fam[..., mi].mean()), 4) for mi, m in enumerate(METRICS)}
    eff = ~exact[None] & finite & ~deg[None]
    if eff.any():
        sign_ok = np.sign(x[..., D]) == np.sign(th[None])
        for mi, m in enumerate(METRICS):
            sel = eff[..., mi]
            if sel.any():
                hit = (rej[..., mi] & sign_ok[..., mi])[sel]
                out["power"][m] = {"rate": round(float(hit.mean()), 4), "n": int(sel.sum()),
                                   "mean_abs_theta": round(float(np.mean(np.abs(np.broadcast_to(th[None], x.shape[:4])[..., mi][sel]))), 4),
                                   "median_mdd": round(float(np.nanmedian(2.8016 * x[..., mi, SE][sel])), 4)}
    return out


def hk(cond: str, r: dict, rule: str, slots_used: tuple[int, ...] | None = None) -> dict:
    """Pooled HK line per world, comp and metric over the events that pass the count gate (at least 3)."""
    idx = chosen(r, rule)
    x = pick(r, idx)
    g, _ = gate(r, idx)
    th, exact = thetas(cond, r)
    deg = degenerate_mask(cond, r)
    W, E = x.shape[:2]
    ev = list(range(E)) if slots_used is None else list(slots_used)
    res = {}
    for mi, m in enumerate(METRICS):
        cov, rej_null, allowed, n_lines = [], [], 0, 0
        for w in range(W):
            row_cov = []
            for c in range(3):
                if deg[0, c, mi]:
                    continue
                n_lines += 1
                use = [e for e in ev if g[w, e, c, mi] and np.isfinite(x[w, e, c, mi, SE]) and x[w, e, c, mi, SE] > 0]
                if len(use) < pool.MIN_EVENTS:
                    continue
                allowed += 1
                rr = pool.random_effects([x[w, e, c, mi, D] for e in use], [x[w, e, c, mi, SE] for e in use])
                mu = float(np.mean([th[e, c, mi] for e in use]))
                row_cov.append(float(rr["lo"] <= mu <= rr["hi"]))
                if all(exact[e, c, mi] for e in use):
                    rej_null.append(rr["p"] < 0.05)
            cov.append(np.mean(row_cov) if row_cov else np.nan)
        res[m] = {**cover_stats(np.array(cov)[:, None]), "allowed_share": round(allowed / max(n_lines, 1), 4),
                  "null_rejection": (round(float(np.mean(rej_null)), 4) if rej_null else None),
                  "null_lines": len(rej_null)}
    return res


def count_gate_bands(cond: str, r: dict, rule: str) -> dict:
    idx = chosen(r, rule)
    x = pick(r, idx)
    g, tiles = gate(r, idx)
    th, _ = thetas(cond, r)
    deg = degenerate_mask(cond, r)
    inside = ((x[..., L] <= th[None]) & (th[None] <= x[..., H])).astype(float)
    ok = np.isfinite(x[..., L]) & ~deg[None]
    out = {}
    for name, sel in (("gate_pass", g & ok), ("gate_fail", ~g & ok)):
        cv = np.where(sel, inside, np.nan).reshape(x.shape[0], -1)
        out[name] = cover_stats(cv)
    for lo_t, hi_t in ((0, 10), (10, 20), (20, 40), (40, 10 ** 9)):
        sel = ok & (tiles >= lo_t) & (tiles < hi_t)
        cv = np.where(sel, inside, np.nan).reshape(x.shape[0], -1)
        out[f"tiles_{lo_t}_{hi_t if hi_t < 10 ** 9 else 'up'}"] = cover_stats(cv)
    return out


def tile_choice(r: dict) -> dict:
    E = r["m1b"].shape[1]
    out = {}
    for e in range(E):
        name = f"{EVENTS[int(r['slot'][0, e])]}#{e}"
        d = {}
        for rule, key in (("variogram", "m1a"), ("trend-variogram-v2", "m1b")):
            v = r[key][:, e]
            d[rule] = {f"{SIZES[k] / 1000:g}km": round(float(np.mean(v == k)), 3) for k in range(len(SIZES))}
        ident = r["m1b_identified"][:, e].astype(float)
        d["trend-variogram-v2 range not identified"] = round(float(np.nanmean(ident == 0)), 3)
        alt = r["m1a_alt"][:, e]
        if np.any(alt >= 0):
            ok = alt >= 0
            d["variogram same tile with another earlier-label history"] = round(float(np.mean(alt[ok] == r["m1a"][ok, e])), 3)
        d["tiles_at_sizes_median"] = np.median(r["n_tiles"][:, e], axis=0).astype(int).tolist()
        out[name] = d
    return out


def rank_hist(cond: str, r: dict, rule: str) -> dict:
    x = pick(r, chosen(r, rule))
    _, exact = thetas(cond, r)
    deg = degenerate_mask(cond, r)
    out = {}
    for mi, m in enumerate(METRICS):
        b = x[..., mi, BELOW][exact[None, ..., mi] & ~deg[None, ..., mi] & np.isfinite(x[..., mi, BELOW])]
        if b.size < 50:
            continue
        h = np.histogram(b, bins=20, range=(0, 1))[0]
        chi = stats.chisquare(h)
        out[m] = {"counts": h.tolist(), "chi2_p": round(float(chi.pvalue), 4), "n": int(b.size)}
    return out


def explore_small_cluster_t(cond: str, r: dict) -> dict:
    """EXPLORATORY, added after the B-NULL result (not registered, not a certificate): interval
    diff +/- t_{G-1, 0.975} SE sqrt(G / (G - 1)) with G = tiles at a fixed side and SE = the tile-bootstrap SE; and
    the ratio of the across-world sd of the point difference to the root mean bootstrap SE (null-exact cells)."""
    th, exact = thetas(cond, r)
    deg = degenerate_mask(cond, r)
    out = {}
    for si, s in enumerate(SIZES):
        x = r["res"][:, :, si]                       # W, E, 3, 2, 6
        G = r["n_tiles"][:, :, si][:, :, None, None].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            half = stats.t.ppf(0.975, np.maximum(G - 1, 1)) * x[..., SE] * np.sqrt(G / np.maximum(G - 1, 1))
            pt = 2 * stats.t.sf(np.abs(x[..., D]) / (x[..., SE] * np.sqrt(G / np.maximum(G - 1, 1))),
                                np.maximum(G - 1, 1))
        inside = (np.abs(x[..., D] - th[None]) <= half).astype(float)
        cov = np.where(np.isfinite(half) & ~deg[None], inside, np.nan)
        d = {m: cover_stats(cov[..., mi].reshape(cov.shape[0], -1)) for mi, m in enumerate(METRICS)}
        nm = exact[None] & ~deg[None] & np.isfinite(pt)
        if nm.any():
            d["null_rate_unadjusted_t"] = round(float((pt[nm] < 0.05).mean()), 4)
            dd, ss = np.where(nm, x[..., D], np.nan), np.where(nm, x[..., SE], np.nan)
            d["sd_over_se"] = round(float(np.sqrt(np.nanmean(dd ** 2)) / np.sqrt(np.nanmean(ss ** 2))), 3)
        out[f"{s / 1000:g}km"] = d
    return out


def main(run_dir: str, out_path: str) -> None:
    rd = Path(run_dir)
    summary = {"protocol": "docs/evals/LAB-PROTOCOL-2026-11-19.md", "n_boot": 400, "conditions": {}}
    for cond in RUN_ORDER:
        r = load(rd, cond)
        if r is None:
            summary["conditions"][cond] = "NOT RUN"
            continue
        th, _ = thetas(cond, r)
        c = {"n_worlds": int(r["res"].shape[0]), "n_events": int(r["res"].shape[1]),
             "theta": {f"{EVENTS[int(r['slot'][0, e])]}#{e}": {f"{COMPS[k][0] + 1}-{COMPS[k][1] + 1}/{m}": round(float(th[e, k, mi]), 4)
                                                            for k in range(3) for mi, m in enumerate(METRICS)}
                       for e in range(th.shape[0])},
             "sites_median": np.median(r["n"], axis=0).astype(int).tolist(),
             "prevalence_mean": np.round(np.mean(r["n_pos"] / np.maximum(r["n"], 1), axis=0), 3).tolist(),
             "tile_choice": tile_choice(r), "rules": {},
             "exploratory_small_cluster_t": explore_small_cluster_t(cond, r)}
        for rule in RULES:
            rr = {"coverage": coverage_cells(cond, r, rule), "coverage_gate_pass_only": coverage_cells(cond, r, rule, True),
                  "median_width": widths(r, rule), "words": holm_words(cond, r, rule)}
            if cond in ("B-NULL", "B-MED", "V-EV8", "X-RM-LONG", "X-LONGRAND"):
                rr["hk"] = {f"k={r['res'].shape[1]}": hk(cond, r, rule)}
                if cond in ("B-NULL", "B-MED"):
                    rr["hk"]["k=3"] = hk(cond, r, rule, (0, 1, 2))
            rr["count_gate"] = count_gate_bands(cond, r, rule)
            if cond in ("B-NULL", "B-MED"):
                rr["rank_hist_null_comps"] = rank_hist(cond, r, rule)
            c["rules"][rule] = rr
        summary["conditions"][cond] = c
        print("aggregated", cond, flush=True)
    summary["decision"] = decide(summary)
    Path(out_path).write_text(json.dumps(summary, indent=1, default=float))


def decide(summary: dict) -> dict:
    """Protocol decision rule for the tile rule (aim 1)."""
    base = [summary["conditions"].get(c) for c in ("B-NULL", "B-MED")]
    if any(not isinstance(b, dict) for b in base):
        return {"status": "base conditions not run"}
    table = {}
    for rule in RULES:
        cells = []
        for b in base:
            cov = b["rules"][rule]["coverage"]
            cells += [v for k, v in cov.items() if not k.startswith("all/")]
        under = sum(v["status"] == "UNDER" for v in cells)
        over = sum(v["status"] == "OVER" for v in cells)
        table[rule] = {"under_cells": under, "over_cells": over, "cells": len(cells),
                       "min_coverage": min(v["coverage"] for v in cells),
                       "eligible": under == 0, "fully_in_band": under == 0 and over == 0,
                       "b_med_auc_width": base[1]["rules"][rule]["median_width"]["auc"]}
    elig = [k for k, v in table.items() if v["eligible"]]
    order = ["trend-variogram-v2", "fixed-1km", "fixed-2km", "fixed-4km", "fixed-8km", "pairdiff-variogram-v2",
             "variogram"]
    if elig:
        full = [k for k in elig if table[k]["fully_in_band"]] or elig
        wmin = min(table[k]["b_med_auc_width"] for k in full)
        tied = [k for k in full if table[k]["b_med_auc_width"] <= 1.05 * wmin]
        win = sorted(tied, key=order.index)[0]
        verdict = "eligible"
    else:
        win = max(table, key=lambda k: table[k]["min_coverage"])
        verdict = "no rule eligible: highest minimum coverage named"
    return {"table": table, "winner": win, "how": verdict}


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

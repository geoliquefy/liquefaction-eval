"""Amendment 1 certification (docs/evals/LAB-PROTOCOL-2026-11-19-AMENDMENT-1.md, sections A4 to A6). No em dashes.

  <venv python> -m scoreboard.evalkit.lab.certify <run_dir> <out.json>

Calibrates k_m for tile4-inflated on C-NULL and C-MED only, then scores every candidate on the fresh R-* worlds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from .. import pool, resample
from .aggregate import METRICS, D, SE, cover_stats, load
from .run import COMPS

I4, I8 = 3, 4
GATE8 = 8
EVENTS = ("sep2010", "feb2011", "jun2011", "dec2011")
RULES = ("tile4-inflated", "tile8-t", "jackknife8-t")


def thetas(cond: str, r: dict):
    diff = r["res"][:, :, I4, :, :, D]
    th = np.nanmean(diff, axis=0)
    exact = np.zeros(th.shape, bool)
    if "NULL" in cond:
        exact[:] = True
    else:
        exact[:, 2, :] = True
    return np.where(exact, 0.0, th), exact


def calibrate(rd: Path) -> dict:
    num, den = {m: [] for m in METRICS}, {m: [] for m in METRICS}
    for cond in ("C-NULL", "C-MED"):
        r = load(rd, cond)
        _, exact = thetas(cond, r)
        x = r["res"][:, :, I4]
        for mi, m in enumerate(METRICS):
            sel = exact[None, :, :, mi] & np.isfinite(x[..., mi, SE])
            num[m].append(x[..., mi, D][sel] ** 2)
            den[m].append(x[..., mi, SE][sel] ** 2)
    return {m: round(float(np.sqrt(np.mean(np.concatenate(num[m]))) / np.sqrt(np.mean(np.concatenate(den[m])))), 4)
            for m in METRICS}


def intervals(r: dict, rule: str, k: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """diff, half-width, p, gate (W, E, 3, 2)."""
    n, npos = r["n"], r["n_pos"]
    base_gate = (n >= 200) & (np.minimum(npos, n - npos) >= 30)
    if rule == "tile4-inflated":
        x = r["res"][:, :, I4]
        kk = np.array([k[m] for m in METRICS])
        se = x[..., SE] * kk
        half = 1.959963984540054 * se
        with np.errstate(invalid="ignore", divide="ignore"):
            p = 2 * stats.norm.sf(np.abs(x[..., D]) / se)
        gate = base_gate & (r["n_tiles"][:, :, I4] >= 20)
        return x[..., D], half, p, np.broadcast_to(gate[:, :, None, None], half.shape)
    G = r["n_tiles"][:, :, I8].astype(float)[:, :, None, None]
    df = np.maximum(G - 1, 1)
    if rule == "tile8-t":
        x = r["res"][:, :, I8]
        se = x[..., SE] * np.sqrt(G / df)
        d = x[..., D]
    else:
        d, se = r["jack"][..., 0], r["jack"][..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        half = stats.t.ppf(0.975, df) * se
        p = 2 * stats.t.sf(np.abs(d) / se, df)
    gate = base_gate & (r["n_tiles"][:, :, I8] >= GATE8)
    return d, half, p, np.broadcast_to(gate[:, :, None, None], half.shape)


def holm_fwer(p: np.ndarray, gate: np.ndarray, exact: np.ndarray, across: bool) -> float:
    """Share of card families (W, E, metric) or cards (W, E) with any null-exact comparison SEPARATES."""
    W, E = p.shape[:2]
    hits, total = 0, 0
    for w in range(W):
        for e in range(E):
            groups = [list(range(6))] if across else [[0, 2, 4], [1, 3, 5]]
            flat_p = p[w, e].reshape(-1)          # order (comp, metric)
            flat_g = gate[w, e].reshape(-1)
            flat_x = np.broadcast_to(exact[e], (3, 2)).reshape(-1)
            for grp in groups:
                g = [i for i in grp if flat_g[i] and np.isfinite(flat_p[i])]
                if not g:
                    continue
                rej = resample.holm([flat_p[i] for i in g])
                total += 1
                hits += any(rj and flat_x[i] for rj, i in zip(rej, g))
    return round(hits / max(total, 1), 4), total


def within(cond: str, r: dict, rule: str, k: dict) -> dict:
    th, exact = thetas(cond, r)
    d, half, p, gate = intervals(r, rule, k)
    cover = (np.abs(d - th[None]) <= half).astype(float)
    ok = np.isfinite(half)
    out = {"cells": {}, "cells_all_events": {}}
    for e in range(d.shape[1]):
        name = f"{EVENTS[int(r['slot'][0, e])]}#{e}"
        for mi, m in enumerate(METRICS):
            cv = np.where(ok[:, e, :, mi] & gate[:, e, :, mi], cover[:, e, :, mi], np.nan)
            out["cells"][f"{name}/{m}"] = cover_stats(cv) if np.isfinite(cv).any() else {"status": "GATED (no words)"}
            out["cells_all_events"][f"{name}/{m}"] = cover_stats(np.where(ok[:, e, :, mi], cover[:, e, :, mi], np.nan))
    for mi, m in enumerate(METRICS):
        cv = np.where(ok[..., mi] & gate[..., mi], cover[..., mi], np.nan).reshape(d.shape[0], -1)
        out["cells"][f"all/{m}"] = cover_stats(cv)
    out["gate_pass_share"] = round(float(gate[..., 0, 0].mean()), 3)
    fw = holm_fwer(p, gate, exact, across=False)
    fa = holm_fwer(p, gate, exact, across=True)
    out["fwer_family"] = {"rate": fw[0], "families": fw[1]}
    out["fwer_card_6"] = {"rate": fa[0], "cards": fa[1]}
    if not exact.all():
        sign = np.sign(d) == np.sign(th[None])
        pw = {}
        for mi, m in enumerate(METRICS):
            rej = np.zeros(p.shape[:3], bool)
            for w in range(p.shape[0]):
                for e in range(p.shape[1]):
                    if gate[w, e, 0, mi] and np.all(np.isfinite(p[w, e, :, mi])):
                        rej[w, e] = resample.holm(list(p[w, e, :, mi]))
            sel = ~exact[None, :, :, mi] & gate[..., mi]
            pw[m] = round(float((rej & sign[..., mi])[sel].mean()), 4) if sel.any() else None
        out["power"] = pw
    out["median_half_width"] = {m: round(float(np.nanmedian(np.where(gate[..., mi], half[..., mi], np.nan))), 4)
                                for mi, m in enumerate(METRICS)}
    return out


def event_level(cond: str, r: dict, slots) -> dict:
    th, exact = thetas(cond, r)
    x = r["res"][:, :, I4]
    W = x.shape[0]
    out = {}
    for mi, m in enumerate(METRICS):
        cov, hits, fams, pw_hits, pw_n = [], 0, 0, 0, 0
        for w in range(W):
            row, ps, xs, signs = [], [], [], []
            for c in range(3):
                use = [e for e in slots if np.isfinite(x[w, e, c, mi, SE]) and x[w, e, c, mi, SE] > 0]
                if len(use) < pool.MIN_EVENTS:
                    continue
                rr = pool.random_effects([x[w, e, c, mi, D] for e in use], [x[w, e, c, mi, SE] for e in use])
                mu = float(np.mean([th[e, c, mi] for e in use]))
                row.append(float(rr["lo"] <= mu <= rr["hi"]))
                ps.append(rr["p"])
                xs.append(all(exact[e, c, mi] for e in use))
                signs.append(np.sign(rr["mean"]) == np.sign(mu))
            cov.append(np.mean(row) if row else np.nan)
            if ps:
                rej = resample.holm(ps)
                fams += 1
                hits += any(rj and xx for rj, xx in zip(rej, xs))
                for rj, xx, sg in zip(rej, xs, signs):
                    if not xx:
                        pw_n += 1
                        pw_hits += rj and sg
        out[m] = {**cover_stats(np.array(cov)[:, None]), "fwer_3_lines": round(hits / max(fams, 1), 4),
                  "power": round(pw_hits / pw_n, 4) if pw_n else None}
    return out


def main(run_dir: str, out_path: str) -> None:
    rd = Path(run_dir)
    k = calibrate(rd)
    res = {"amendment": "docs/evals/LAB-PROTOCOL-2026-11-19-AMENDMENT-1.md", "k_m": k, "conditions": {}}
    for cond in ("R-NULL", "R-MED", "R-HI", "R-EV8"):
        r = load(rd, cond)
        if r is None:
            res["conditions"][cond] = "NOT RUN"
            continue
        th, _ = thetas(cond, r)
        c = {"n_worlds": int(r["res"].shape[0]), "theta_E1_E2": {f"{EVENTS[int(r['slot'][0, e])]}#{e}":
             [round(float(th[e, 0, 0]), 4), round(float(th[e, 0, 1]), 4)] for e in range(th.shape[0])}}
        for rule in RULES:
            c[rule] = within(cond, r, rule, k)
        E = r["res"].shape[1]
        c["event-level"] = {f"k={E}": event_level(cond, r, list(range(E)))}
        if E == 4:
            c["event-level"]["k=3"] = event_level(cond, r, [0, 1, 2])
        res["conditions"][cond] = c
        print("certified", cond, flush=True)
    Path(out_path).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

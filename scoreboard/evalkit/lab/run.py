"""Run lab conditions (protocol sections D3, M, P) and keep per-world aggregates in memory-light arrays.

  <venv python> -m scoreboard.evalkit.lab.run <fit.json> <out_dir> <condition> [n_worlds] [workers]

Per world and earthquake slot it stores, for each candidate tile side, comparison (E1-E2, E1-E3, E2-E3) and metric
(AUC, log loss): the paired difference, the 95% interval, the bootstrap p and SE (kit resample.paired), the share of
draws below 0; and the tile side each rule picks. Nothing per site is stored. One condition per call; at most 3
worker processes. No em dashes.
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ...scorer import auc, grid_clusters
from .. import metrics, resample
from . import engine, fit as F, rules
from .world import World

SIZES = rules.CANDIDATES_M                     # 500, 1000, 2000, 4000, 8000 m
COMPS = ((0, 1), (0, 2), (1, 2))
N_BOOT = 400
CLIP = 1e-6
PAIRDIFF_SITES = 2000
BASE_SEED = 20261119
RUN_ORDER = ("B-NULL", "B-MED", "X-SMALL", "X-RM-LONG", "X-LONGRAND", "X-PREV-HI", "V-EV8", "B-CAL", "B-SMALL",
             "X-TREND2", "X-RS-LONG", "X-RS-SHORT", "X-ANISO", "X-TREND0", "X-PREV-LO", "X-RM-SHORT", "X-LABEL",
             "C-NULL", "C-MED", "R-NULL", "R-MED", "R-HI", "R-EV8")          # amendment 1 conditions last
# amendment 1 (docs/evals/LAB-PROTOCOL-2026-11-19-AMENDMENT-1.md): model error refitted to the real paired differences
REFIT = {"rho_m": 8000.0, "sd_n": 0.05}
LIGHT = ("C-NULL", "C-MED", "R-NULL", "R-MED", "R-HI", "R-EV8")   # no tile-rule fits; bootstrap at 4 and 8 km only
LIGHT_SIZES = (3, 4)
JACK_SIZE = 4                                                     # index of 8 km
F1_SMALL, F1_MED = 0.7, 0.35
CONDITIONS = {
    "B-NULL": {"f1": 1.0}, "B-MED": {"f1": F1_MED}, "B-SMALL": {"f1": F1_SMALL},
    "B-CAL": {"f1": 1.0, "cal_shift": 0.5},
    "V-EV8": {"f1": F1_MED, "slots": (0, 1, 2, 3, 0, 1, 2, 3)},
    "X-RS-SHORT": {"f1": F1_MED, "rho_s": 300.0}, "X-RS-LONG": {"f1": F1_MED, "rho_s": 12000.0},
    "X-ANISO": {"f1": F1_MED, "aniso": 3.0}, "X-TREND0": {"f1": F1_MED, "trend_mult": 0.0},
    "X-TREND2": {"f1": F1_MED, "trend_mult": 2.0}, "X-LONGRAND": {"f1": F1_MED, "longrand_sd": 0.5},
    "X-PREV-HI": {"f1": F1_MED, "prevalence": 0.88}, "X-PREV-LO": {"f1": F1_MED, "prevalence": 0.10},
    "X-RM-SHORT": {"f1": F1_MED, "rho_m": 500.0}, "X-RM-LONG": {"f1": F1_MED, "rho_m": 8000.0},
    "X-LABEL": {"f1": F1_MED, "miss_m": 0.4, "lam": 0.8},
    "X-SMALL": {"f1": F1_MED, "box_sides_m": (6000.0, 8000.0, 10000.0, 14000.0)},
    "C-NULL": {"f1": 1.0, **REFIT}, "C-MED": {"f1": F1_MED, **REFIT},
    "R-NULL": {"f1": 1.0, **REFIT}, "R-MED": {"f1": F1_MED, **REFIT},
    "R-HI": {"f1": F1_MED, **REFIT, "rho_m": 20000.0},
    "R-EV8": {"f1": F1_MED, **REFIT, "slots": (0, 1, 2, 3, 0, 1, 2, 3)},
}

_G: dict = {}


def _init(fit_path: str, cond: str) -> None:
    fitd = json.loads(Path(fit_path).read_text())
    real = F.load_real()
    geom = F.geometry(real, fitd["glm"]["quad"])
    params = {"sigma_s": fitd["chosen"]["sigma_s"], "rho_s": fitd["chosen"]["rho_s"], **CONDITIONS[cond]}
    world = World(geom, params, fitd["glm"]["b"], fitd["prevalence"])
    small = bool(params.get("box_sides_m"))
    ctx = {}
    if not small:
        for e, m in geom.masks.items():
            x, y = geom.x[m], geom.y[m]
            sub = np.sort(np.random.default_rng(BASE_SEED).choice(m.sum(), min(PAIRDIFF_SITES, m.sum()), replace=False))
            ctx[e] = {"pairs": rules.Pairs(x, y), "sub": sub, "pairs_sub": rules.Pairs(x[sub], y[sub]),
                      "tiles": {s: grid_clusters(x, y, s) for s in SIZES}, "counts": rules.tile_counts(x, y)}
    _G.update(geom=geom, world=world, ctx=ctx, small=small, cond=cond)


def _event_ctx(ev: dict) -> dict:
    g = _G["geom"]
    if not _G["small"]:
        return _G["ctx"][ev["event"]]
    x, y = g.x[ev["idx"]], g.y[ev["idx"]]
    n = x.size
    sub = np.arange(n) if n <= PAIRDIFF_SITES else np.sort(
        np.random.default_rng(BASE_SEED).choice(n, PAIRDIFF_SITES, replace=False))
    pairs = rules.Pairs(x, y) if n >= 2 else None
    return {"pairs": pairs, "sub": sub, "pairs_sub": rules.Pairs(x[sub], y[sub]) if n >= 2 else None,
            "tiles": {s: grid_clusters(x, y, s) for s in SIZES}, "counts": rules.tile_counts(x, y)}


def _size_index(tile_m: float) -> int:
    return SIZES.index(float(tile_m))


def analyse_event(ev: dict, earlier: list, seed: int) -> dict:
    g = _G["geom"]
    yb = ev["y"].astype(bool)
    yf = yb.astype(float)
    x, yy = g.x[ev["idx"]], g.y[ev["idx"]]
    n, npos = yb.size, int(yb.sum())
    out = {"n": n, "n_pos": npos, "n_tiles": np.zeros(len(SIZES), int),
           "res": np.full((len(SIZES), 3, 2, 6), np.nan)}
    if npos < 5 or n - npos < 5:
        out["degenerate"] = True
        return out
    ctx = _event_ctx(ev)
    out["n_tiles"] = np.array([ctx["counts"][float(s)] for s in SIZES])
    ps = ev["p"]
    pa = [auc(yb, p) for p in ps]
    pl = [metrics.log_loss(yf, p, CLIP) for p in ps]
    ranked = [engine.Ranked(yb, p) for p in ps]
    sl = [engine.site_log_loss(yf, p, CLIP) for p in ps]
    light = _G["cond"] in LIGHT
    for si, s in enumerate(SIZES):
        if light and si not in LIGHT_SIZES:
            continue
        w = engine.weights(ctx["tiles"][s], N_BOOT, seed + si)
        da = [r.auc(w) for r in ranked]
        dl = [engine.mean_loss(w, l) for l in sl]
        for ci, (a, b) in enumerate(COMPS):
            for mi, (pt, dr) in enumerate(((pa, da), (pl, dl))):
                r = resample.paired(pt[a], pt[b], dr[a], dr[b])
                d = dr[a] - dr[b]
                d = d[np.isfinite(d)]
                below = float(np.mean(d < 0) + 0.5 * np.mean(d == 0)) if d.size else np.nan
                out["res"][si, ci, mi] = (r["diff"], r["lo"], r["hi"], r["p"], r["se"], below)
    # delete-one-tile jackknife at 8 km (amendment 1, rule jackknife8-t): diff and SE_J per comparison and metric
    inv, G = engine.tile_index(ctx["tiles"][SIZES[JACK_SIZE]])
    wj = np.ones((G, n))
    wj[inv, np.arange(n)] = 0.0
    keep = (wj * yb).sum(1) > 0
    keep &= (wj * (~yb)).sum(1) > 0
    ja = [r.auc(wj) for r in ranked]
    jl = [engine.mean_loss(wj, l) for l in sl]
    out["jack"] = np.full((3, 2, 2), np.nan)
    for ci, (a, b) in enumerate(COMPS):
        for mi, (pt, dr) in enumerate(((pa, ja), (pl, jl))):
            d = (dr[a] - dr[b])[keep]
            if d.size >= 2:
                out["jack"][ci, mi] = (pt[a] - pt[b], float(np.sqrt((d.size - 1) / d.size * np.sum((d - d.mean()) ** 2))))
    if light:
        return out
    # M1a: the kit's current rule
    base = rules.shaking_only_p(earlier, ev["pga_med"])
    resid = yf - (base if base is not None else 0.5)
    ca = resample.choose_tile(x, yy, resid, seed=seed)
    out["m1a"] = _size_index(ca["tile_m"])
    out["m1a_range"] = ca["fitted_range_m"]
    if len(earlier) >= 2:          # the Dec A / Dec B case: same labels, a different earlier history
        alt = rules.shaking_only_p(earlier[-1:], ev["pga_med"])
        out["m1a_alt"] = _size_index(resample.choose_tile(x, yy, yf - alt, seed=seed)["tile_m"])
    # M1b: trend-variogram-v2
    rb = rules.trend_residual(yf, x, yy, np.log(ev["pga_med"]), g.centre)
    fb = rules.fit_range_in_window(ctx["pairs"].variogram(rb))
    cb = rules.choose_v2(fb, ctx["counts"])
    out["m1b"] = _size_index(cb["tile_m"])
    out["m1b_identified"] = fb["identified"]
    out["m1b_range"] = fb["range_m"]
    # M1c: pairdiff-variogram-v2, per comparison and metric, on the fixed subset
    sub = ctx["sub"]
    pv = [engine.placement_values(yb, p) for p in ps]
    m1c = np.zeros((3, 2), int)
    for ci, (a, b) in enumerate(COMPS):
        for mi, contrib in enumerate((pv[a] - pv[b], sl[a] - sl[b])):
            fc = rules.fit_range_in_window(ctx["pairs_sub"].variogram(contrib[sub]))
            m1c[ci, mi] = _size_index(rules.choose_v2(fc, ctx["counts"])["tile_m"])
    out["m1c"] = m1c
    return out


def run_world(i: int) -> list[dict]:
    seed = BASE_SEED * 1000 + RUN_ORDER.index(_G["cond"]) * 100000 + i
    evs = _G["world"].draw(seed)
    earlier, out = [], []
    for k, ev in enumerate(evs):
        r = analyse_event(ev, earlier, seed * 16 + k * 7)
        r["slot"] = ev["slot"]
        r["pos"] = k
        out.append(r)
        earlier.append((ev["pga_med"], ev["y"].astype(float)))
    return out


def main(fit_path: str, out_dir: str, cond: str, n_worlds: int, workers: int = 3) -> None:
    t0 = time.time()
    workers = min(workers, 3)
    with Pool(workers, initializer=_init, initargs=(fit_path, cond)) as pool:
        worlds = []
        for k, w in enumerate(pool.imap(run_world, range(n_worlds), chunksize=2)):
            worlds.append(w)
            if (k + 1) % 50 == 0:
                print(f"{cond}: {k + 1}/{n_worlds} worlds, {time.time() - t0:.0f} s", flush=True)
    keys = ("n", "n_pos", "n_tiles", "res", "jack", "m1a", "m1a_range", "m1a_alt", "m1b", "m1b_identified", "m1b_range",
            "m1c", "slot", "pos")
    arr = {}
    n_ev = len(worlds[0])
    for k in keys:
        vals = []
        for w in worlds:
            row = []
            for r in w:
                v = r.get(k, np.nan)
                if k in ("res",) and r.get("degenerate"):
                    v = r["res"]
                if k == "jack" and "jack" not in r:
                    v = np.full((3, 2, 2), np.nan)
                if k == "m1c" and "m1c" not in r:
                    v = np.full((3, 2), -1)
                if k in ("m1a", "m1b", "m1a_alt") and k not in r:
                    v = -1
                row.append(v)
            vals.append(row)
        arr[k] = np.array(vals)
    arr["n_events"] = np.array(n_ev)
    np.savez_compressed(Path(out_dir) / f"{cond}.npz", **arr)
    print(f"{cond}: done {n_worlds} worlds in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 200, int(a[4]) if len(a) > 4 else 3)

"""C4 (ruling 47): certify one interval method for the "models fixed" truth, or record that none certifies. No em dashes.

  <venv python> -m scoreboard.evalkit.lab.c4_certify run <fit.json> <out_dir> <condition> <n_worlds> [r_theta] [workers]
  <venv python> -m scoreboard.evalkit.lab.c4_certify summarise <out_dir> <out.json>

Registration: pillar docs/evals/C4-CERTIFICATION-REGISTRATION-2026-11-23.md (committed alone before any run). The
truth is C2's truth (i), "models fixed" (lab/two_truths.py): theta_w = the mean paired difference over r_theta
redraws of shaking, outcomes and labels, with this world's ground and three model-error fields held. The interval
replicate is World.draw(seed) bit for bit (two_truths.draw_fixed then draw_events).

Candidates, fixed in this order (registration section 3); no other method is built here:
  (0) "tile4"         the kit's plain 4 km tile bootstrap, percentile 95% (resample.paired). Pooled line: the kit's
                      default pooled line, Hartung-Knapp (pool.random_effects) on the per-earthquake tile SEs.
  (1) "block"         C1 (a) spatial block bootstrap at the measured paired-difference range, percentile 95%: AUC on
                      5.5 km blocks, log loss on 3.0 km blocks (intervals.MEASURED_PAIRDIFF_RANGE_M, block_side).
                      Under 20 blocks the line is UNDERPOWERED (no interval). Pooled line: Hartung-Knapp on the
                      per-earthquake block SEs.
  (2) "cluster"       C1 (b) cluster bootstrap on survey units: NOT TESTABLE. The lab worlds and the kit's real
                      inputs carry no survey-unit id; any stand-in would be a new method (correction 4).
  (3) "event2"        C1 (c) event-level resampling, two stage (the only new form of (c)), percentile 95%, 2,000 draws,
                      within draws = the 4 km tile bootstrap paired draws. Pooled lines only.
Pooled lines pool the earthquakes that pass the kit count gate at 4 km tiles in this replicate (200 sites, 30 per
class, 20 tiles); fewer than 3 gives no line (R1 c). Pooled truth = the equal-weight mean of theta_w over the pooled
earthquakes. Pooled words: Holm across the 6 pooled lines of a world (3 comparisons x 2 metrics), as
pool.pooled_lines does for one release. Nothing per site is stored or written.
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ...scorer import auc, grid_clusters
from .. import intervals as IV
from .. import metrics, pool as P, resample
from . import engine, two_truths as TT
from .run import CLIP, COMPS, N_BOOT
from .world import EVENTS

BASE_SEED = 20261124                        # fresh worlds: not C2's 20261123 nor the lab's 20261119 streams
COND_ORDER = TT.COND_ORDER                  # R-NULL, R-MED, B-NULL, B-MED, R-HI
LO, HI = 0.93, 0.97
WRONG_SIGN_BOUND = 0.075                    # lab bound per comparison (LAB-CERTIFICATES section 4)
SLOT_FLOOR = 0.90                           # registration 5.1 guard: no gate-passing earthquake below this point
PRINT_SHARE_MIN = 0.5                       # registration 5.1: a method must print on half the gate-passing lines
METRICS = ("auc", "log_loss")
WITHIN = ("tile4", "block")
POOLED = ("tile4", "block", "event2")
NOT_TESTABLE = {"cluster": "no survey-unit id exists in the lab worlds or the kit's real inputs; a stand-in would "
                           "be a new method (correction 4)"}
CANDIDATE_ORDER = ("tile4", "block", "cluster", "event2")
BLOCK_M = {m: IV.block_side(IV.MEASURED_PAIRDIFF_RANGE_M[m]) for m in METRICS}    # 5500, 3000
EVENT_BOOT = 2000
DIFF, LOW, HIGH, PVAL, SEB = range(5)
MIN_EVENTS = P.MIN_EVENTS


def world_seed(cond: str, i: int) -> int:
    return BASE_SEED * 1000 + COND_ORDER.index(cond) * 100000 + i


# ---------------------------------------------------------------------------------------------------------------
# Per-earthquake intervals for every candidate
# ---------------------------------------------------------------------------------------------------------------
def gate_passes(n: int, n_pos: int, n_tiles: int) -> bool:
    return n >= 200 and min(n_pos, n - n_pos) >= 30 and n_tiles >= 20


def event_lines(y: np.ndarray, ps: list, tiles: np.ndarray, blocks: dict, seed: int, n_boot: int = N_BOOT) -> dict:
    """Per-earthquake lines. Returns {"tile4": (3, 2, 5), "block": (3, 2, 5), "n_blocks": (2,),
    "tile_diff_draws": (3, 2, n_boot)}. A line under the block count gate is NaN (UNDERPOWERED)."""
    yb = np.asarray(y, bool)
    out = {"tile4": np.full((3, 2, 5), np.nan), "block": np.full((3, 2, 5), np.nan),
           "n_blocks": np.array([int(np.unique(blocks[m]).size) for m in METRICS]),
           "tile_diff_draws": np.full((3, 2, n_boot), np.nan)}
    if yb.sum() < 5 or (~yb).sum() < 5:
        return out
    yf = yb.astype(float)
    pa = [auc(yb, p) for p in ps]
    pl = [metrics.log_loss(yf, p, CLIP) for p in ps]
    sl = [engine.site_log_loss(yf, p, CLIP) for p in ps]
    rk = [engine.Ranked(yb, p) for p in ps]
    w4 = engine.weights(tiles, n_boot, seed)
    d4 = ([r.auc(w4) for r in rk], [engine.mean_loss(w4, s) for s in sl])
    wa = engine.weights(blocks["auc"], n_boot, seed + 1)
    wl = engine.weights(blocks["log_loss"], n_boot, seed + 2)
    db = ([r.auc(wa) for r in rk], [engine.mean_loss(wl, s) for s in sl])
    pts = (pa, pl)
    for ci, (a, b) in enumerate(COMPS):
        for mi in range(2):
            r = resample.paired(pts[mi][a], pts[mi][b], d4[mi][a], d4[mi][b])
            out["tile4"][ci, mi] = (r["diff"], r["lo"], r["hi"], r["p"], r["se"])
            out["tile_diff_draws"][ci, mi] = np.asarray(d4[mi][a]) - np.asarray(d4[mi][b])
            if out["n_blocks"][mi] >= IV.MIN_BLOCKS:
                r = IV.paired(pts[mi][a], pts[mi][b], db[mi][a], db[mi][b], form="percentile")
                out["block"][ci, mi] = (r["diff"], r["lo"], r["hi"], r["p"], r["se"])
    return out


def pooled_lines(lines: dict, tile_draws: np.ndarray, passes: np.ndarray, seed: int) -> dict:
    """Pooled lines over the gate-passing earthquakes. lines: method -> (E, 3, 2, 5); tile_draws (E, 3, 2, B).
    Returns method -> (3, 2, 5) (mean, lo, hi, p, se) and "k"; NaN when fewer than MIN_EVENTS pass."""
    use = np.flatnonzero(passes)
    out = {m: np.full((3, 2, 5), np.nan) for m in POOLED}
    out["k"] = int(use.size)
    if use.size < MIN_EVENTS:
        return out
    for ci in range(3):
        for mi in range(2):
            for m in ("tile4", "block"):
                d = lines[m][use, ci, mi, DIFF]
                s = lines[m][use, ci, mi, SEB]
                if np.all(np.isfinite(d)) and np.all(np.isfinite(s)) and np.all(s > 0):
                    r = P.random_effects(d, s)
                    out[m][ci, mi] = (r["mean"], r["lo"], r["hi"], r["p"], r["se"])
            d = lines["tile4"][use, ci, mi, DIFF]
            if np.all(np.isfinite(d)):
                r = IV.event_bootstrap(d, within=[tile_draws[e, ci, mi] for e in use], n_boot=EVENT_BOOT,
                                       seed=seed + ci * 2 + mi, form="percentile")
                out["event2"][ci, mi] = (r["mean"], r["lo"], r["hi"], r["p"], r["se"])
    return out


# ---------------------------------------------------------------------------------------------------------------
# Runner (one condition per call, at most 3 workers)
# ---------------------------------------------------------------------------------------------------------------
_G: dict = {}


def _init(fit_path: str, cond: str, r_theta: int) -> None:
    TT._init(fit_path, cond, r_theta)
    g = TT._G["world"].g
    _G.update(TT._G)
    _G["blocks"] = {e: {m: grid_clusters(g.x[mask], g.y[mask], BLOCK_M[m]) for m in METRICS}
                    for e, mask in g.masks.items()}


def run_world(i: int) -> dict:
    world, cond = _G["world"], _G["cond"]
    seed = world_seed(cond, i)
    rng = np.random.default_rng(seed)
    world._reset()
    fixed = TT.draw_fixed(world, rng)
    evs = TT.draw_events(world, fixed, rng)            # = World.draw(seed): the interval replicate
    E = len(evs)
    lines = {m: np.full((E, 3, 2, 5), np.nan) for m in WITHIN}
    tdraw = np.full((E, 3, 2, N_BOOT), np.nan)
    n, npos, ntiles = np.zeros(E, int), np.zeros(E, int), np.zeros(E, int)
    nblocks = np.zeros((E, 2), int)
    for k, ev in enumerate(evs):
        t = _G["tiles"][ev["event"]]
        r = event_lines(ev["y"], ev["p"], t, _G["blocks"][ev["event"]], seed * 16 + k * 7)
        for m in WITHIN:
            lines[m][k] = r[m]
        tdraw[k] = r["tile_diff_draws"]
        nblocks[k] = r["n_blocks"]
        n[k], npos[k], ntiles[k] = ev["y"].size, int(ev["y"].sum()), int(np.unique(t).size)
    passes = np.array([gate_passes(n[k], npos[k], ntiles[k]) for k in range(E)])
    pooled = pooled_lines(lines, tdraw, passes, seed * 16 + 900)
    reps = np.full((_G["r_theta"], E, 3, 2), np.nan)
    for r in range(_G["r_theta"]):
        TT.reset_redrawn(world)
        for k, ev in enumerate(TT.draw_events(world, fixed, rng)):
            reps[r, k] = TT.point_diffs(ev["y"], ev["p"])
    return {"lines": lines, "pooled": pooled, "reps": reps, "n": n, "n_pos": npos, "n_tiles": ntiles,
            "n_blocks": nblocks, "passes": passes, "slot": np.array([ev["slot"] for ev in evs])}


def main_run(fit_path: str, out_dir: str, cond: str, n_worlds: int, r_theta: int = 50, workers: int = 3) -> None:
    t0 = time.time()
    workers = min(workers, 3)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with Pool(workers, initializer=_init, initargs=(fit_path, cond, r_theta)) as pl:
        worlds = []
        for k, w in enumerate(pl.imap(run_world, range(n_worlds), chunksize=1)):
            worlds.append(w)
            if (k + 1) % 50 == 0:
                print(f"{cond}: {k + 1}/{n_worlds} worlds, {time.time() - t0:.0f} s", flush=True)
    reps = np.stack([w["reps"] for w in worlds])
    arr = {f"within_{m}": np.stack([w["lines"][m] for w in worlds]) for m in WITHIN}
    arr.update({f"pooled_{m}": np.stack([w["pooled"][m] for w in worlds]) for m in POOLED})
    arr.update(pooled_k=np.array([w["pooled"]["k"] for w in worlds]),
               theta_w=np.nanmean(reps, axis=1), theta_w_sd=np.nanstd(reps, axis=1, ddof=1),
               n=np.stack([w["n"] for w in worlds]), n_pos=np.stack([w["n_pos"] for w in worlds]),
               n_tiles=np.stack([w["n_tiles"] for w in worlds]), n_blocks=np.stack([w["n_blocks"] for w in worlds]),
               passes=np.stack([w["passes"] for w in worlds]), slot=np.stack([w["slot"] for w in worlds]),
               r_theta=np.array(r_theta), seconds=np.array(time.time() - t0))
    np.savez_compressed(Path(out_dir) / f"{cond}.npz", **arr)
    print(f"{cond}: done {n_worlds} worlds (r_theta {r_theta}) in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Summary and the registered decision
# ---------------------------------------------------------------------------------------------------------------
def mc_range(c: dict) -> tuple[float, float] | None:
    if c["value"] is None:
        return None
    return c["value"] - 1.96 * c["se"], c["value"] + 1.96 * c["se"]


def coverage_inside(c: dict) -> bool:
    """Registration 5.1: the whole 95% Monte Carlo range inside 0.93 to 0.97."""
    r = mc_range(c)
    return r is not None and r[0] >= LO and r[1] <= HI


def wrong_sign_ok(c: dict) -> bool:
    """Registration 5.1: wrong-sign SEPARATES per comparison, upper end of the 95% Monte Carlo range <= 0.075."""
    r = mc_range(c)
    return r is not None and r[1] <= WRONG_SIGN_BOUND


def _holm_within(p: np.ndarray) -> np.ndarray:
    """p (W, E, 3, 2) -> Holm over the 3 comparisons of one metric and earthquake."""
    return TT.holm_separates(p)


def _holm_pooled(p: np.ndarray) -> np.ndarray:
    """p (W, 3, 2) -> Holm across the 6 pooled lines of one world."""
    out = np.zeros(p.shape, bool)
    for w in range(p.shape[0]):
        pv = p[w].reshape(-1)
        if np.all(np.isfinite(pv)):
            out[w] = np.array(resample.holm(list(pv))).reshape(3, 2)
    return out


def _cov(res: np.ndarray, t: np.ndarray, judged: np.ndarray) -> np.ndarray:
    hit = ((res[..., LOW] <= t) & (t <= res[..., HIGH])).astype(float)
    return np.where(judged, hit, np.nan)


def summarise_within(cond: str, r: dict, m: str) -> dict:
    res = r[f"within_{m}"]                                  # W, E, 3, 2, 5
    t = r["theta_w"]
    gate = r["passes"][..., None, None] & np.isfinite(t)    # W, E, 1, 1 -> broadcast
    gate = np.broadcast_to(gate, t.shape)
    printed = np.isfinite(res[..., LOW]) & np.isfinite(res[..., HIGH])
    judged = gate & printed
    cover = _cov(res, t, judged)
    sep = _holm_within(res[..., PVAL]) & printed
    wrong = sep & (np.sign(res[..., DIFF]) != np.sign(t))
    width = res[..., HIGH] - res[..., LOW]
    slots = [EVENTS[int(s)] for s in r["slot"][0]]
    out = {}
    for mi, met in enumerate(METRICS):
        c = TT.clustered(cover[..., mi])
        per_slot = {}
        for e in range(t.shape[1]):
            if r["passes"][:, e].mean() >= 0.5:
                per_slot[f"{slots[e]}#{e}"] = TT.clustered(cover[:, e, :, mi])
        ws = TT.clustered(np.where(judged[..., mi], wrong[..., mi].astype(float), np.nan))
        share = float(judged[..., mi].sum() / max(gate[..., mi].sum(), 1))
        w_med = float(np.nanmedian(np.where(judged[..., mi], width[..., mi], np.nan))) if judged[..., mi].any() else None
        floor_ok = all(v["value"] is not None and v["value"] >= SLOT_FLOOR for v in per_slot.values())
        out[met] = {"coverage": {**c, "mc_range": mc_range(c), "inside": coverage_inside(c)},
                    "coverage_per_gate_passing_slot": per_slot, "slot_floor_ok": floor_ok,
                    "wrong_sign_per_comparison": {**ws, "mc_range": mc_range(ws), "ok": wrong_sign_ok(ws)},
                    "print_share": round(share, 4), "print_share_ok": share >= PRINT_SHARE_MIN,
                    "median_width": None if w_med is None else round(w_med, 5),
                    "separates_rate": TT.clustered(np.where(judged[..., mi], sep[..., mi].astype(float), np.nan))}
        o = out[met]
        o["pass"] = bool(o["coverage"]["inside"] and o["slot_floor_ok"] and o["wrong_sign_per_comparison"]["ok"]
                         and o["print_share_ok"])
    return out


def summarise_pooled(cond: str, r: dict, m: str) -> dict:
    res = r[f"pooled_{m}"]                                   # W, 3, 2, 5
    passes = r["passes"]
    tw = r["theta_w"]                                        # W, E, 3, 2
    wts = passes[..., None, None].astype(float)
    t = np.where(wts.sum(axis=1) > 0, (np.nan_to_num(tw) * wts).sum(axis=1) / np.maximum(wts.sum(axis=1), 1), np.nan)
    has = (r["pooled_k"] >= MIN_EVENTS)[:, None, None] & np.isfinite(t)
    printed = np.isfinite(res[..., LOW]) & np.isfinite(res[..., HIGH])
    judged = has & printed
    cover = _cov(res, t, judged)
    sep = _holm_pooled(res[..., PVAL]) & printed
    wrong = sep & (np.sign(res[..., DIFF]) != np.sign(t))
    width = res[..., HIGH] - res[..., LOW]
    out = {"line_share": round(float((r["pooled_k"] >= MIN_EVENTS).mean()), 4),
           "k_counts": {str(k): int((r["pooled_k"] == k).sum()) for k in np.unique(r["pooled_k"])}}
    for mi, met in enumerate(METRICS):
        c = TT.clustered(cover[..., mi])
        ws = TT.clustered(np.where(judged[..., mi], wrong[..., mi].astype(float), np.nan))
        share = float(judged[..., mi].sum() / max(has[..., mi].sum(), 1))
        w_med = float(np.nanmedian(np.where(judged[..., mi], width[..., mi], np.nan))) if judged[..., mi].any() else None
        o = {"coverage": {**c, "mc_range": mc_range(c), "inside": coverage_inside(c)},
             "wrong_sign_per_comparison": {**ws, "mc_range": mc_range(ws), "ok": wrong_sign_ok(ws)},
             "print_share": round(share, 4), "print_share_ok": share >= PRINT_SHARE_MIN,
             "median_width": None if w_med is None else round(w_med, 5),
             "separates_rate": TT.clustered(np.where(judged[..., mi], sep[..., mi].astype(float), np.nan))}
        o["pass"] = bool(o["coverage"]["inside"] and o["wrong_sign_per_comparison"]["ok"] and o["print_share_ok"])
        out[met] = o
    return out


def decide(summary: dict) -> dict:
    """Registration sections 5 and 6: per line type, a candidate passes only if every metric passes in every
    condition; among passing candidates the narrowest median width wins (median over conditions and metrics of the
    per-cell medians), ties to the earlier candidate. Returns the outcome dict the card wiring reads."""
    conds = summary["conditions"]
    missing = [c for c in COND_ORDER if not isinstance(conds.get(c), dict)]
    outcome = {"conditions_missing": missing}
    for kind, methods in (("per_earthquake", WITHIN), ("pooled", POOLED)):
        rows = {}
        for m in CANDIDATE_ORDER:
            if m in NOT_TESTABLE:
                rows[m] = {"verdict": "NOT TESTABLE", "reason": NOT_TESTABLE[m]}
                continue
            if m not in methods:
                rows[m] = {"verdict": "NOT A CANDIDATE FOR THIS LINE TYPE"}
                continue
            if missing:
                rows[m] = {"verdict": "NOT RUN", "reason": f"conditions missing: {missing}"}
                continue
            fails = [f"{c} {met}" for c in COND_ORDER for met in METRICS if not conds[c][kind][m][met]["pass"]]
            widths = [conds[c][kind][m][met]["median_width"] for c in COND_ORDER for met in METRICS]
            widths = [w for w in widths if w is not None]
            rows[m] = {"verdict": "PASS" if not fails else "FAIL", "failed_cells": fails,
                       "median_width": float(np.median(widths)) if widths else None}
        passing = [m for m in CANDIDATE_ORDER if rows[m].get("verdict") == "PASS"]
        chosen = min(passing, key=lambda m: (rows[m]["median_width"], CANDIDATE_ORDER.index(m))) if passing else None
        outcome[kind] = {"candidates": rows, "certified": chosen}
    return outcome


def load(out_dir: Path, cond: str) -> dict | None:
    f = Path(out_dir) / f"{cond}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {k: z[k] for k in z.files}


def summarise_condition(cond: str, r: dict) -> dict:
    out = {"n_worlds": int(r["theta_w"].shape[0]), "r_theta": int(r["r_theta"]),
           "seconds": round(float(r["seconds"]), 1), "slots": [EVENTS[int(s)] for s in r["slot"][0]],
           "gate_pass_share_by_slot": {f"{EVENTS[int(s)]}#{e}": round(float(r["passes"][:, e].mean()), 3)
                                       for e, s in enumerate(r["slot"][0])},
           "median_blocks_by_slot": {f"{EVENTS[int(s)]}#{e}": {met: int(np.median(r["n_blocks"][:, e, mi]))
                                                               for mi, met in enumerate(METRICS)}
                                     for e, s in enumerate(r["slot"][0])},
           "per_earthquake": {m: summarise_within(cond, r, m) for m in WITHIN},
           "pooled": {m: summarise_pooled(cond, r, m) for m in POOLED}}
    return out


def main_summarise(out_dir: str, out_path: str) -> None:
    res = {"item": "C4 (ruling 47), registration docs/evals/C4-CERTIFICATION-REGISTRATION-2026-11-23.md",
           "truth": "models fixed (C2 truth (i))", "target_band": [LO, HI], "wrong_sign_bound": WRONG_SIGN_BOUND,
           "n_boot_within": N_BOOT, "n_boot_event": EVENT_BOOT, "block_m": BLOCK_M, "conditions": {}}
    for cond in COND_ORDER:
        r = load(Path(out_dir), cond)
        res["conditions"][cond] = summarise_condition(cond, r) if r is not None else "NOT RUN"
    res["outcome"] = decide(res)
    Path(out_path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["outcome"], indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run":
        main_run(a[1], a[2], a[3], int(a[4]), int(a[5]) if len(a) > 5 else 50, int(a[6]) if len(a) > 6 else 3)
    elif a[0] == "summarise":
        main_summarise(a[1], a[2])
    else:
        raise SystemExit(__doc__)

"""C2 step 1 (ruling 45, correction 4): coverage of the PLAIN 4 km tile bootstrap under two truths. No em dashes.

  <venv python> -m scoreboard.evalkit.lab.two_truths run <fit.json> <out_dir> <condition> <n_worlds> [r_theta] [workers]
  <venv python> -m scoreboard.evalkit.lab.two_truths summarise <out_dir> <out.json>

The method is the kit's own: percentile interval and bootstrap p from resample.paired on fixed 4 km NZTM2000 tiles
(scorer.grid_clusters), no inflation (k = 1), the word SEPARATES by Holm over the 3 comparisons of one metric on one
earthquake. No new interval method is built here.

Two truths for the same interval (LAB-CERTIFICATES-2026-11-19.md section 12.2):
  (i)  models fixed: theta_w = the expected paired difference given this world's ground (site field, trend, density)
       and this world's three model-error fields; shaking (event term, within-event field), true outcomes and label
       errors are redrawn. Estimated per world as the mean of r_theta replicate point differences that are separate
       from the replicate that carries the interval.
  (ii) models averaged (the marginal truth of the lab certificates): theta = the mean over the condition's worlds,
       exact 0 where the entries are exchangeable (every comparison in a *-NULL condition; E2 - E3 elsewhere).

Per outer world the first replicate continues the world's own random stream, so its draw is World.draw(seed) bit for
bit: the interval replicate is a fresh lab world, as in the certificates. Nothing per site is stored or written.
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import expit, ndtr

from ...scorer import auc, grid_clusters
from .. import metrics, resample
from . import engine, fit as F
from .run import CLIP, COMPS, CONDITIONS, N_BOOT
from .world import ATTENUATION_C, EVENTS, SE as SENS, SP as SPEC, World

TILE_M = 4000.0
BASE_SEED = 20261123                      # fresh worlds: not the lab's 20261119 streams
COND_ORDER = ("R-NULL", "R-MED", "B-NULL", "B-MED", "R-HI")
LO, HI = 0.93, 0.97
METRICS = ("auc", "log_loss")
DIFF, LOW, HIGH, PVAL, SEB = range(5)


# ---------------------------------------------------------------------------------------------------------------
# Splitting World.draw into the fixed part (ground, models) and the redrawn part (shaking, outcomes, labels)
# ---------------------------------------------------------------------------------------------------------------
def draw_fixed(world: World, rng: np.random.Generator) -> dict:
    """Ground part, the three model-error fields and the box mask, in World.draw's random order."""
    p, g = world.p, world.g
    part = world._site_part(rng)
    k_s = p["sigma_s"]
    mod = [k_s * (p["sd_m"] * world.s_model.draw(rng) + rng.normal(0.0, p["sd_n"], g.x.size)) for _ in range(3)]
    keep = np.ones(g.x.size, bool)
    if p["box_sides_m"]:
        side = float(rng.choice(p["box_sides_m"]))
        anyl = np.flatnonzero(np.logical_or.reduce([g.masks[e] for e in EVENTS]))
        c = anyl[rng.integers(anyl.size)]
        keep = (np.abs(g.x - g.x[c]) <= side / 2) & (np.abs(g.y - g.y[c]) <= side / 2)
    return {"part": part, "mod": mod, "keep": keep}


def draw_events(world: World, fixed: dict, rng: np.random.Generator) -> list[dict]:
    """Shaking, true outcomes and observed labels for every slot, given the fixed part (World.draw's loop)."""
    g, p = world.g, world.p
    part, mod, keep = fixed["part"], fixed["mod"], fixed["keep"]
    f = (p["f1"], 1.0, 1.0)
    out = []
    for slot in p["slots"]:
        e = EVENTS[slot]
        idx = np.flatnonzero(g.masks[e] & keep)
        ev = rng.normal(0.0, p["event_sd"])
        w = g.sd_lnpga[e] * world.s_pga.draw(rng)
        eta = world.a[e] + world.b[e] * (g.ln_pga[e] + ev + w) + part
        z = rng.random(g.x.size) < expit(eta)
        se_i = SENS[e] * (1.0 - p["miss_m"] * g.access_u)
        u = ndtr(world.s_err.draw(rng)) if world.s_err is not None else rng.random(g.x.size)
        yobs = np.where(z, u < se_i, u < 1.0 - SPEC[e])
        known = world.a[e] + world.b[e] * g.ln_pga[e] + part
        c = 1.0 / np.sqrt(1.0 + ATTENUATION_C * world.b[e] ** 2 * (g.sd_lnpga[e] ** 2 + p["event_sd"] ** 2))
        ps = [expit(c * (known + f[k] * mod[k])) for k in range(3)]
        if p["cal_shift"]:
            ps[2] = expit(c * (known + f[2] * mod[2]) + p["cal_shift"])
        out.append({"slot": slot, "event": e, "idx": idx, "y": yobs[idx], "z": z[idx],
                    "p": [q[idx] for q in ps], "pga_med": np.exp(g.ln_pga[e][idx])})
    return out


def reset_redrawn(world: World) -> None:
    """Forget spare draws of the samplers that a replicate redraws (shaking, label errors)."""
    for s in (world.s_pga, world.s_err):
        if s is not None:
            s.reset()


# ---------------------------------------------------------------------------------------------------------------
# Per-event analysis
# ---------------------------------------------------------------------------------------------------------------
def point_diffs(y: np.ndarray, ps: list) -> np.ndarray:
    """(3, 2) paired point differences (AUC, log loss) for COMPS; NaN when a class has fewer than 5 sites."""
    yb = np.asarray(y, bool)
    out = np.full((3, 2), np.nan)
    if yb.sum() < 5 or (~yb).sum() < 5:
        return out
    pa = [auc(yb, p) for p in ps]
    pl = [metrics.log_loss(yb.astype(float), p, CLIP) for p in ps]
    for ci, (a, b) in enumerate(COMPS):
        out[ci] = (pa[a] - pa[b], pl[a] - pl[b])
    return out


def interval_event(y: np.ndarray, ps: list, tiles: np.ndarray, seed: int, n_boot: int = N_BOOT) -> np.ndarray:
    """(3, 2, 5): diff, percentile lo, hi, bootstrap p, SE of the plain tile bootstrap (kit resample.paired)."""
    yb = np.asarray(y, bool)
    out = np.full((3, 2, 5), np.nan)
    if yb.sum() < 5 or (~yb).sum() < 5:
        return out
    yf = yb.astype(float)
    pa = [auc(yb, p) for p in ps]
    pl = [metrics.log_loss(yf, p, CLIP) for p in ps]
    w = engine.weights(tiles, n_boot, seed)
    da = [engine.Ranked(yb, p).auc(w) for p in ps]
    dl = [engine.mean_loss(w, engine.site_log_loss(yf, p, CLIP)) for p in ps]
    for ci, (a, b) in enumerate(COMPS):
        for mi, (pt, dr) in enumerate(((pa, da), (pl, dl))):
            r = resample.paired(pt[a], pt[b], dr[a], dr[b])
            out[ci, mi] = (r["diff"], r["lo"], r["hi"], r["p"], r["se"])
    return out


# ---------------------------------------------------------------------------------------------------------------
# Runner (one condition per call, at most 3 workers)
# ---------------------------------------------------------------------------------------------------------------
_G: dict = {}


def _init(fit_path: str, cond: str, r_theta: int) -> None:
    fitd = json.loads(Path(fit_path).read_text())
    _rebase_paths()
    real = F.load_real()
    geom = F.geometry(real, fitd["glm"]["quad"])
    params = {"sigma_s": fitd["chosen"]["sigma_s"], "rho_s": fitd["chosen"]["rho_s"], **CONDITIONS[cond]}
    if params.get("box_sides_m"):
        raise ValueError("box conditions are not part of C2 step 1")
    world = World(geom, params, fitd["glm"]["b"], fitd["prevalence"])
    tiles = {e: grid_clusters(geom.x[m], geom.y[m], TILE_M) for e, m in geom.masks.items()}
    _G.update(world=world, tiles=tiles, cond=cond, r_theta=r_theta)


def _main_kit() -> Path | None:
    """The main kit checkout (pillar/repos/liquefaction-model). Worktrees sit deeper than the main checkout, so the
    kit's parents-based data paths miss; this walks up to the pillar instead."""
    for parent in Path(__file__).resolve().parents:
        cand = parent / "repos" / "liquefaction-model"
        if (cand / "results" / F.ROWS_NPZ.name).exists():
            return cand
    return None


def _rebase_paths() -> None:
    """Point the real-data loaders at the main checkout's data paths (read only; nothing is written there)."""
    kit = _main_kit()
    if kit is None:
        return
    from ... import label_sets
    F.ROWS_NPZ = kit / "results" / F.ROWS_NPZ.name
    label_sets.ROOT, label_sets.PILLAR = kit, kit.parents[1]
    label_sets.ROWS_NPZ = F.ROWS_NPZ
    old = label_sets.CGD0300_KML
    kml = label_sets.PILLAR / old.relative_to(Path(label_sets.__file__).resolve().parents[3])
    # The defaults are overwritten in this worker only. Fail loudly if the signatures no longer match (review nit);
    # a kit-level data root setting would replace this (left to the integrator).
    if label_sets.label_sources.__defaults__ != (old,) or label_sets._kml_root.__defaults__ != (old, True):
        raise RuntimeError("label_sets signatures changed: update two_truths._rebase_paths")
    label_sets.CGD0300_KML = kml
    label_sets.label_sources.__defaults__ = (kml,)
    label_sets._kml_root.__defaults__ = (kml, True)


def world_seed(cond: str, i: int) -> int:
    return BASE_SEED * 1000 + COND_ORDER.index(cond) * 100000 + i


def run_world(i: int) -> dict:
    world, cond = _G["world"], _G["cond"]
    seed = world_seed(cond, i)
    rng = np.random.default_rng(seed)
    world._reset()
    fixed = draw_fixed(world, rng)
    evs = draw_events(world, fixed, rng)                 # = World.draw(seed): the interval replicate
    E = len(evs)
    res = np.full((E, 3, 2, 5), np.nan)
    n = np.zeros(E, int)
    npos = np.zeros(E, int)
    ntiles = np.zeros(E, int)
    for k, ev in enumerate(evs):
        t = _G["tiles"][ev["event"]]
        res[k] = interval_event(ev["y"], ev["p"], t, seed * 16 + k * 7)
        n[k], npos[k], ntiles[k] = ev["y"].size, int(ev["y"].sum()), int(np.unique(t).size)
    reps = np.full((_G["r_theta"], E, 3, 2), np.nan)
    for r in range(_G["r_theta"]):
        reset_redrawn(world)
        for k, ev in enumerate(draw_events(world, fixed, rng)):
            reps[r, k] = point_diffs(ev["y"], ev["p"])
    return {"res": res, "reps": reps, "n": n, "n_pos": npos, "n_tiles": ntiles,
            "slot": np.array([ev["slot"] for ev in evs])}


def main_run(fit_path: str, out_dir: str, cond: str, n_worlds: int, r_theta: int = 30, workers: int = 3) -> None:
    t0 = time.time()
    workers = min(workers, 3)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with Pool(workers, initializer=_init, initargs=(fit_path, cond, r_theta)) as pool:
        worlds = []
        for k, w in enumerate(pool.imap(run_world, range(n_worlds), chunksize=1)):
            worlds.append(w)
            if (k + 1) % 25 == 0:
                print(f"{cond}: {k + 1}/{n_worlds} worlds, {time.time() - t0:.0f} s", flush=True)
    reps = np.stack([w["reps"] for w in worlds])               # W, R, E, 3, 2
    arr = {"res": np.stack([w["res"] for w in worlds]),
           "theta_w": np.nanmean(reps, axis=1), "theta_w_sd": np.nanstd(reps, axis=1, ddof=1),
           "theta_w_n": np.isfinite(reps).sum(axis=1),
           "n": np.stack([w["n"] for w in worlds]), "n_pos": np.stack([w["n_pos"] for w in worlds]),
           "n_tiles": np.stack([w["n_tiles"] for w in worlds]), "slot": np.stack([w["slot"] for w in worlds]),
           "r_theta": np.array(r_theta), "seconds": np.array(time.time() - t0)}
    np.savez_compressed(Path(out_dir) / f"{cond}.npz", **arr)
    print(f"{cond}: done {n_worlds} worlds (r_theta {r_theta}) in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Summary: coverage (world-clustered SE) and false SEPARATES under each truth
# ---------------------------------------------------------------------------------------------------------------
def exact_mask(cond: str, E: int) -> np.ndarray:
    """(E, 3, 2) cells where the marginal truth is exactly 0 (entries exchangeable)."""
    ex = np.zeros((E, 3, 2), bool)
    if "NULL" in cond:
        ex[:] = True
    else:
        ex[:, 2, :] = True
    return ex


def truths(cond: str, r: dict) -> dict:
    """theta under each truth, broadcast to (W, E, 3, 2)."""
    tw = r["theta_w"]
    ex = exact_mask(cond, tw.shape[1])
    marg = np.where(ex, 0.0, np.nanmean(tw, axis=0))
    return {"models_fixed": tw, "models_averaged": np.broadcast_to(marg, tw.shape)}


def clustered(x: np.ndarray) -> dict:
    """x: (W, ...) with NaN for left-out cells; mean of per-world means and its world-clustered SE."""
    per_w = np.nanmean(x.reshape(x.shape[0], -1), axis=1)
    per_w = per_w[np.isfinite(per_w)]
    if per_w.size < 2:
        return {"value": None, "se": None, "n_worlds": int(per_w.size)}
    return {"value": round(float(per_w.mean()), 4), "se": round(float(per_w.std(ddof=1) / np.sqrt(per_w.size)), 4),
            "n_worlds": int(per_w.size), "n_cells": int(np.isfinite(x).sum())}


def band(c: dict) -> str:
    if c["value"] is None:
        return "NOT RUN"
    lo, hi = c["value"] - 1.96 * c["se"], c["value"] + 1.96 * c["se"]
    return "IN BAND" if (hi >= LO and lo <= HI) else ("UNDER" if hi < LO else "OVER")


def holm_separates(p: np.ndarray) -> np.ndarray:
    """p (W, E, 3, 2) -> bool SEPARATES per comparison, Holm over the 3 comparisons of one metric and earthquake."""
    W, E = p.shape[:2]
    out = np.zeros(p.shape, bool)
    for w in range(W):
        for e in range(E):
            for mi in range(2):
                pv = p[w, e, :, mi]
                if np.all(np.isfinite(pv)):
                    out[w, e, :, mi] = resample.holm(list(pv))
    return out


def summarise_condition(cond: str, r: dict) -> dict:
    res = r["res"]
    ok = np.isfinite(res[..., LOW]) & np.isfinite(res[..., HIGH])
    sep = holm_separates(res[..., PVAL]) & ok
    est = res[..., DIFF]
    gate = (r["n"] >= 200) & (np.minimum(r["n_pos"], r["n"] - r["n_pos"]) >= 30) & (r["n_tiles"] >= 20)
    th = truths(cond, r)
    ex = np.broadcast_to(exact_mask(cond, res.shape[1])[None], est.shape)
    out = {"n_worlds": int(res.shape[0]), "r_theta": int(r["r_theta"]), "seconds": round(float(r["seconds"]), 1),
           "slots": [EVENTS[int(s)] for s in r["slot"][0]], "metrics": {}}
    # noise of theta_w: mean replicate-to-replicate sd over the interval SE, so a reader can bound its effect
    ratio = r["theta_w_sd"] / np.maximum(res[..., SEB], 1e-12)
    out["theta_w_sd_over_boot_se_median"] = round(float(np.nanmedian(ratio)), 3)
    for mi, m in enumerate(METRICS):
        mo = {}
        for tname, t in th.items():
            cover = np.where(ok[..., mi], ((res[..., LOW] <= t) & (t <= res[..., HIGH]))[..., mi].astype(float), np.nan)
            c_all = clustered(cover)
            c_gate = clustered(np.where(gate[..., None], cover, np.nan))
            per_slot = {f"{out['slots'][e]}#{e}": clustered(cover[:, e]) for e in range(cover.shape[1])}
            sm = sep[..., mi]
            if tname == "models_averaged":
                # false SEPARATES: the word on a comparison whose truth is exactly 0
                fs = np.where(ok[..., mi] & ex[..., mi], sm.astype(float), np.nan)
                fam = np.full(sm.shape[:2], np.nan)
                anyx = ex[..., mi].any(axis=2)
                fam[anyx] = (sm & ex[..., mi]).any(axis=2)[anyx]
                rule = "SEPARATES on a comparison whose models-averaged truth is exactly 0"
            else:
                # false SEPARATES: the word with the opposite sign to this world's fixed-model truth
                wrong = sm & (np.sign(est[..., mi]) != np.sign(t[..., mi]))
                fs = np.where(ok[..., mi] & np.isfinite(t[..., mi]), wrong.astype(float), np.nan)
                fam = wrong.any(axis=2).astype(float)
                rule = "SEPARATES with the opposite sign to this world's models-fixed truth"
            mo[tname] = {"coverage": {**c_all, "band": band(c_all)},
                         "coverage_gate_passing": {**c_gate, "band": band(c_gate)},
                         "coverage_per_slot": {k: {**v, "band": band(v)} for k, v in per_slot.items()},
                         "false_separates_rule": rule,
                         "false_separates_per_comparison": clustered(fs),
                         "false_separates_family": clustered(fam)}
        mo["separates_rate_all"] = clustered(np.where(ok[..., mi], sep[..., mi].astype(float), np.nan))
        out["metrics"][m] = mo
    out["gate_pass_share_by_slot"] = {f"{out['slots'][e]}#{e}": round(float(gate[:, e].mean()), 3)
                                      for e in range(gate.shape[1])}
    out["n_tiles_4km_by_slot"] = {f"{out['slots'][e]}#{e}": int(np.median(r["n_tiles"][:, e]))
                                  for e in range(gate.shape[1])}
    return out


def load(out_dir: Path, cond: str) -> dict | None:
    f = Path(out_dir) / f"{cond}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {k: z[k] for k in z.files}


def main_summarise(out_dir: str, out_path: str) -> None:
    res = {"item": "C2 step 1 (ruling 45, correction 4)", "method": "plain 4 km tile bootstrap, percentile 95%, "
           "k = 1, Holm over 3 comparisons", "target_band": [LO, HI], "n_boot": N_BOOT, "conditions": {}}
    for cond in COND_ORDER:
        r = load(Path(out_dir), cond)
        res["conditions"][cond] = summarise_condition(cond, r) if r is not None else "NOT RUN"
    Path(out_path).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run":
        main_run(a[1], a[2], a[3], int(a[4]), int(a[5]) if len(a) > 5 else 30, int(a[6]) if len(a) > 6 else 3)
    elif a[0] == "summarise":
        main_summarise(a[1], a[2])
    else:
        raise SystemExit(__doc__)

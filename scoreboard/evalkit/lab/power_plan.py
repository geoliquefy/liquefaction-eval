"""C3 power plan for the forward track (ruling 47: the "models fixed" truth). No em dashes.

  <venv python> -m scoreboard.evalkit.lab.power_plan run <fit.json> <out_dir> <condition> <n_worlds> [r_theta] [workers]
  <venv python> -m scoreboard.evalkit.lab.power_plan summarise <out_dir> <out.json> [n_sets] [condition ...]

Question: how many earthquakes must the pooled line (rule book R1, kit pool.py) hold before it has 80% power to give
the word HOLDS for a gap of 0.05 AUC, and for a calibration gap of 0.5 logit, under the models-fixed truth?

Design (forward track: every earthquake is a different place):
- One pooling unit = one earthquake in its own independent lab world (own ground, own three model-error fields, own
  shaking). Within one world the slots Sep, Feb and Jun are three earthquake templates; a pooled set never takes two
  earthquakes from one world, so the units in a set are independent (LAB-CERTIFICATES section 5: same-city
  earthquakes break Hartung-Knapp). Dec is left out: at 4 km it never passes the count gate (17 tiles).
- Earthquake size: all labelled sites of the template ("full"), or a random subset of 500 or 250 of them (the subset is
  part of the fixed ground: these are the CPTs that exist). The count gate (200 sites, 30 per class, 20 tiles) is
  applied per earthquake and size; a failing earthquake does not enter the pooled line (pool.py) and is not counted.
- Per earthquake: the kit's plain 4 km tile bootstrap (two_truths.interval_event: resample.paired, 400 draws) gives d_i
  and s_i; the models-fixed truth theta_i is the mean of r_theta replicate point differences with shaking, outcomes
  and labels redrawn on the same ground, models and sites (two_truths.draw_events).
- Pooled line: plain Hartung-Knapp on (d_i, s_i) as pool.random_effects (vectorised here; equal draw for draw, tested),
  Holm family of one at alpha 0.05 (and 0.05 / 3 beside). Word HOLDS: p < alpha and every leave-one-out interval at
  1 - alpha excludes 0 with the sign of the mean (pool.pooled_lines). Power = P(HOLDS with the sign of the pooled
  models-fixed truth, mean theta_i over the set). Coverage of the pooled interval against that truth is printed too.
- Interval cases: (1) the plain 4 km bootstrap SEs (the kit today); (2) a certified per-earthquake interval from C4,
  modelled as the same SEs times the tile4-inflated k (1.27 AUC, 1.66 log loss; LAB-CERTIFICATES 11.2), and the
  equal-weight line (pool.equal_weight, which uses no SE) beside. HK uses the SEs only as weights, so case 2 moves
  power little.

Nothing per site is stored or written. Seeds: BASE_SEED 20261124 (fresh; C2 used 20261123, the lab 20261119).

Review 1 additions (2026-11-23). The ten registered conditions run exactly as before (seed recipe 1: the subset stream
is seed + 7, which equals the main stream of world i + 7 in the same condition; disclosed, not changed). Three new
conditions carry a true AUC gap of about 0.05 at 2 km and 8 km by scaling E2's model error up ("f2"), so K80 at 0.05
is interpolated between simulated gaps, not extrapolated. They use seed recipe 2 (the subset stream is a SeedSequence
child, disjoint from every main stream) and have no null line (no two entries are exchangeable). The summary also
gives a mixed-size curve: each unit's size drawn from MIX (the per-place gate-share mix of the forward catalogue).
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

from ...scorer import grid_clusters
from . import fit as F
from . import two_truths as TT
from .run import COMPS, N_BOOT, REFIT
from .world import EVENTS, World

BASE_SEED = 20261124
SLOTS = (0, 1, 2)                                  # Sep, Feb, Jun templates (Dec fails the 4 km count gate)
SIZES = (0, 500, 250)                              # 0 = all labelled sites of the template
SIZE_NAMES = ("full", "n500", "n250")
METRICS = ("auc", "log_loss")
K_GRID = tuple(range(3, 21)) + (25, 30, 40, 50, 60, 80, 100)
INFLATE = {"auc": 1.27, "log_loss": 1.66}          # tile4-inflated k (LAB-CERTIFICATES 11.2), case 2
ALPHA = 0.05
DIFF, LOW, HIGH, PVAL, SEB = TT.DIFF, TT.LOW, TT.HIGH, TT.PVAL, TT.SEB

_R2 = {}                                           # model-error range 2 km, large independent noise (first lab)
_R8 = dict(REFIT)                                  # range 8 km, sd_n 0.05 (amendment 1 refit)
CONDITIONS = {
    "P2-A15": {"f1": 0.15, **_R2}, "P2-A30": {"f1": 0.30, **_R2}, "P2-A50": {"f1": 0.50, **_R2},
    "P2-CALP": {"f1": 1.0, "cal_shift": 0.5, **_R2}, "P2-CALM": {"f1": 1.0, "cal_shift": -0.5, **_R2},
    "P8-A15": {"f1": 0.15, **_R8}, "P8-A30": {"f1": 0.30, **_R8}, "P8-A50": {"f1": 0.50, **_R8},
    "P8-CALP": {"f1": 1.0, "cal_shift": 0.5, **_R8}, "P8-CALM": {"f1": 1.0, "cal_shift": -0.5, **_R8},
    # review 1: true AUC gap near 0.05 by scaling E2's model error up (f2); seed recipe 2
    "P2-A15-E2x115": {"f1": 0.15, "f2": 1.15, "recipe": 2, **_R2},
    "P8-A15-E2x130": {"f1": 0.15, "f2": 1.30, "recipe": 2, **_R8},
    "P8-A15-E2x160": {"f1": 0.15, "f2": 1.60, "recipe": 2, **_R8},
}
LAB_ONLY = ("f2", "recipe")                        # power_plan's own keys, not World parameters
# Size mix of gate-passing units, one unit per place (POWER-PLAN-C3 section 5): 2 full-size places, 3 at 500 to 800
# catalogue CPTs (gate share about 0.68), 3 at 200 to 500 (about 0.145).
MIX = (2.0 / 4.475, 2.04 / 4.475, 0.435 / 4.475)
COND_ORDER = tuple(CONDITIONS)
# Which comparison carries which gap: E1 - E2 for the AUC gap (f1 < 1); E2 - E3 for the calibration gap (E3 shifted).
# In the A* conditions E2 - E3 is exchangeable (a null line); in the CAL* conditions E1 - E2 is.
GAP_COMP = {"A": 0, "CAL": 2}
NULL_COMP = {"A": 2, "CAL": 0}


def family(cond: str) -> str:
    return "CAL" if "CAL" in cond else "A"


def world_seed(cond: str, i: int) -> int:
    return BASE_SEED * 1000 + COND_ORDER.index(cond) * 10000 + i


def subset_rng(cond: str, seed: int) -> np.random.Generator:
    """Stream for the subset draw. Recipe 1 (registered conditions): seed + 7, kept so their run reproduces; it
    equals world_seed(cond, i + 7). Recipe 2 (review 1 conditions): a SeedSequence child, disjoint from main streams."""
    if CONDITIONS[cond].get("recipe", 1) == 2:
        return np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    return np.random.default_rng(seed + 7)


def subset(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Positions (into the template's labelled sites) of the sites this earthquake has, sorted."""
    if size == 0 or size >= n:
        return np.arange(n)
    return np.sort(rng.choice(n, size, replace=False))


def gate(n: int, n_pos: int, n_tiles: int) -> bool:
    """The kit's count gate: 200 sites, 30 per class, 20 tiles."""
    return n >= 200 and min(n_pos, n - n_pos) >= 30 and n_tiles >= 20


# ---------------------------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------------------------
_G: dict = {}


def _init(fit_path: str, cond: str, r_theta: int) -> None:
    fitd = json.loads(Path(fit_path).read_text())
    TT._rebase_paths()
    real = F.load_real()
    real.pop("labels", None)                       # positions and masks only; no real label is used
    geom = F.geometry(real, fitd["glm"]["quad"])
    cp = {k: v for k, v in CONDITIONS[cond].items() if k not in LAB_ONLY}
    params = {"sigma_s": fitd["chosen"]["sigma_s"], "rho_s": fitd["chosen"]["rho_s"], **cp, "slots": SLOTS}
    world = World(geom, params, fitd["glm"]["b"], fitd["prevalence"])
    tiles = {e: grid_clusters(geom.x[m], geom.y[m], TT.TILE_M) for e, m in geom.masks.items()}
    _G.update(world=world, tiles=tiles, cond=cond, r_theta=r_theta)


def run_world(i: int) -> dict:
    world, cond = _G["world"], _G["cond"]
    seed = world_seed(cond, i)
    rng = np.random.default_rng(seed)
    world._reset()
    fixed = TT.draw_fixed(world, rng)
    f2 = CONDITIONS[cond].get("f2", 1.0)
    if f2 != 1.0:                                      # review 1: scale E2's model error (fixed with the ground)
        fixed["mod"][1] = fixed["mod"][1] * f2
    evs = TT.draw_events(world, fixed, rng)            # the replicate that carries the interval
    sub_rng = subset_rng(cond, seed)
    subs = [[subset(ev["y"].size, s, sub_rng) for s in SIZES] for ev in evs]
    E, S = len(evs), len(SIZES)
    res = np.full((E, S, 3, 2, 5), np.nan)
    ok = np.zeros((E, S), bool)
    for k, ev in enumerate(evs):
        t_all = _G["tiles"][ev["event"]]
        for j, sb in enumerate(subs[k]):
            y, t = ev["y"][sb], t_all[sb]
            ok[k, j] = gate(y.size, int(y.sum()), int(np.unique(t).size))
            if ok[k, j]:
                res[k, j] = TT.interval_event(y, [p[sb] for p in ev["p"]], t, seed * 64 + k * 8 + j)
    reps = np.full((_G["r_theta"], E, S, 3, 2), np.nan)
    for r in range(_G["r_theta"]):
        TT.reset_redrawn(world)
        for k, ev in enumerate(TT.draw_events(world, fixed, rng)):
            for j, sb in enumerate(subs[k]):
                if ok[k, j]:
                    reps[r, k, j] = TT.point_diffs(ev["y"][sb], [p[sb] for p in ev["p"]])
    return {"res": res, "gate": ok, "theta": np.nanmean(reps, axis=0)}


def main_run(fit_path: str, out_dir: str, cond: str, n_worlds: int, r_theta: int = 20, workers: int = 3) -> None:
    t0 = time.time()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with Pool(min(workers, 3), initializer=_init, initargs=(fit_path, cond, r_theta)) as pool:
        worlds = []
        for k, w in enumerate(pool.imap(run_world, range(n_worlds), chunksize=1)):
            worlds.append(w)
            if (k + 1) % 50 == 0:
                print(f"{cond}: {k + 1}/{n_worlds} worlds, {time.time() - t0:.0f} s", flush=True)
    arr = {"res": np.stack([w["res"] for w in worlds]), "gate": np.stack([w["gate"] for w in worlds]),
           "theta": np.stack([w["theta"] for w in worlds]), "r_theta": np.array(r_theta),
           "seconds": np.array(time.time() - t0)}
    np.savez_compressed(Path(out_dir) / f"{cond}.npz", **arr)
    print(f"{cond}: done {n_worlds} worlds (r_theta {r_theta}) in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Pooled line, vectorised over sets (equal to pool.random_effects and pool.equal_weight; tested)
# ---------------------------------------------------------------------------------------------------------------
def hk(d: np.ndarray, s: np.ndarray, level: float = 0.95, method: str = "hk") -> dict:
    """d, s: (M, K). Plain Hartung-Knapp (DL tau^2) or equal weight, per row."""
    K = d.shape[1]
    df = K - 1
    if method == "equal":
        mean = d.mean(1)
        se = np.sqrt(((d - mean[:, None]) ** 2).sum(1) / (K * (K - 1)))
    else:
        w = 1.0 / s ** 2
        sw = w.sum(1)
        d_fe = (w * d).sum(1) / sw
        q = (w * (d - d_fe[:, None]) ** 2).sum(1)
        c = sw - (w ** 2).sum(1) / sw
        tau2 = np.maximum(0.0, (q - df) / c)
        ws = 1.0 / (s ** 2 + tau2[:, None])
        mean = (ws * d).sum(1) / ws.sum(1)
        se = np.sqrt((ws * (d - mean[:, None]) ** 2).sum(1) / (df * ws.sum(1)))
    tq = stats.t.ppf(0.5 + level / 2, df)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(se > 0, 2.0 * stats.t.sf(np.abs(mean) / se, df), np.where(mean != 0, 0.0, 1.0))
    return {"mean": mean, "se": se, "lo": mean - tq * se, "hi": mean + tq * se, "p": p}


def holds(d: np.ndarray, s: np.ndarray, alpha: float = ALPHA, method: str = "hk") -> dict:
    """R1 word per row for a Holm family of one at alpha: reject, HOLDS (leave-one-out at 1 - alpha), sign."""
    full = hk(d, s, 0.95, method)
    rej = full["p"] < alpha
    sign = np.sign(full["mean"])
    loo_ok = np.ones(d.shape[0], bool)
    K = d.shape[1]
    if K >= 3:
        for i in range(K):
            keep = np.r_[0:i, i + 1:K]
            r = hk(d[:, keep], s[:, keep], 1.0 - alpha, method)
            loo_ok &= np.where(sign > 0, r["lo"] > 0, r["hi"] < 0)
    else:
        loo_ok[:] = False
    return {**full, "reject": rej, "holds": rej & loo_ok, "sign": sign}


# ---------------------------------------------------------------------------------------------------------------
# Summary: power curves, K80, coverage, null rate
# ---------------------------------------------------------------------------------------------------------------
def bank(r: dict, size_j: int, comp: int, mi: int) -> dict:
    """Gate-passing earthquakes of one size: per world the list of usable slots, d, s, theta (W, E)."""
    res = r["res"][:, :, size_j, comp, mi]
    th = r["theta"][:, :, size_j, comp, mi]
    g = r["gate"][:, :, size_j] & np.isfinite(res[..., DIFF]) & (res[..., SEB] > 0) & np.isfinite(th)
    return {"d": res[..., DIFF], "s": res[..., SEB], "theta": th, "ok": g}


def draw_sets(b: dict, K: int, n_sets: int, rng: np.random.Generator) -> tuple | None:
    """n_sets pooled sets of K earthquakes, each from a distinct world, one gate-passing slot drawn per world."""
    worlds = np.flatnonzero(b["ok"].any(1))
    if worlds.size < K:
        return None
    pick_w = np.stack([rng.choice(worlds, K, replace=False) for _ in range(n_sets)])
    okw = b["ok"][pick_w]                                   # (M, K, E)
    u = rng.random(okw.shape) * okw
    slot = u.argmax(-1)
    return tuple(a[pick_w, slot] for a in (b["d"], b["s"], b["theta"]))


def draw_sets_mixed(banks: list[dict], weights, K: int, n_sets: int, rng: np.random.Generator) -> tuple | None:
    """n_sets sets of K earthquakes from K distinct worlds; each unit's size drawn from weights, then one
    gate-passing slot of that size in a world not yet used by the set. None if a set cannot be filled."""
    oks = np.stack([b["ok"].any(1) for b in banks], 1)      # (W, S)
    W = oks.shape[0]
    out = np.empty((3, n_sets, K))
    for m in range(n_sets):
        sz = rng.choice(len(banks), K, p=np.asarray(weights) / np.sum(weights))
        perm = rng.permutation(W)
        used = np.zeros(W, bool)
        col = 0
        for j in range(len(banks)):
            n_j = int((sz == j).sum())
            if n_j == 0:
                continue
            cand = perm[oks[perm, j] & ~used[perm]][:n_j]
            if cand.size < n_j:
                return None
            used[cand] = True
            okw = banks[j]["ok"][cand]
            slot = (rng.random(okw.shape) * okw).argmax(-1)
            for a, key in enumerate(("d", "s", "theta")):
                out[a, m, col:col + n_j] = banks[j][key][cand, slot]
            col += n_j
    return out[0], out[1], out[2]


def power_curve(b, metric: str, n_sets: int, seed: int, case: str, weights=None) -> list[dict]:
    """b: one bank (one size), or a list of banks with weights (mixed sizes)."""
    rng = np.random.default_rng(seed)
    out = []
    for K in K_GRID:
        got = draw_sets(b, K, n_sets, rng) if weights is None else draw_sets_mixed(b, weights, K, n_sets, rng)
        if got is None:
            break
        d, s, th = got
        truth = th.mean(1)
        if case == "inflated":
            s = s * INFLATE[metric]
        meth = "equal" if case == "equal" else "hk"
        row = {"K": K}
        for name, a in (("alpha_0.05", ALPHA), ("alpha_0.0167", ALPHA / 3)):
            w = holds(d, s, a, meth)
            right = w["sign"] == np.sign(truth)
            row[name] = {"holds_right_sign": round(float(np.mean(w["holds"] & right)), 4),
                         "reject_right_sign": round(float(np.mean(w["reject"] & right)), 4),
                         "reject_wrong_sign": round(float(np.mean(w["reject"] & ~right)), 4)}
            if a == ALPHA:
                row["coverage_models_fixed"] = round(float(np.mean((w["lo"] <= truth) & (truth <= w["hi"]))), 4)
        out.append(row)
    return out


def k80(curve: list[dict], key: str = "holds_right_sign", alpha: str = "alpha_0.05") -> int | None:
    for row in curve:
        if row[alpha][key] >= 0.8:
            return row["K"]
    return None


def k80_interp(curve: list[dict], key: str = "holds_right_sign", alpha: str = "alpha_0.05") -> float | None:
    """K where the power curve first reaches 0.8, linear between the grid points around it (None if never)."""
    prev = None
    for row in curve:
        v = row[alpha][key]
        if v >= 0.8:
            if prev is None or prev[1] >= v:
                return float(row["K"])
            k0, v0 = prev
            return round(k0 + (0.8 - v0) * (row["K"] - k0) / (v - v0), 2)
        prev = (row["K"], v)
    return None


def summarise_condition(cond: str, r: dict, n_sets: int) -> dict:
    fam = family(cond)
    out = {"n_worlds": int(r["res"].shape[0]), "r_theta": int(r["r_theta"]), "seconds": round(float(r["seconds"]), 1),
           "params": CONDITIONS[cond], "gap_comparison": "E1 - E2" if fam == "A" else "E2 - E3",
           "null_comparison": "E2 - E3" if fam == "A" else "E1 - E2", "sizes": {}}
    for j, sname in enumerate(SIZE_NAMES):
        so = {"gate_pass_share": round(float(r["gate"][:, :, j].mean()), 4),
              "gate_pass_share_by_slot": {EVENTS[s]: round(float(r["gate"][:, k, j].mean()), 4)
                                          for k, s in enumerate(SLOTS)}, "metrics": {}}
        for mi, m in enumerate(METRICS):
            mo = {}
            roles = (("gap", GAP_COMP[fam]),) if "f2" in CONDITIONS[cond] else \
                (("gap", GAP_COMP[fam]), ("null", NULL_COMP[fam]))
            for role, comp in roles:
                b = bank(r, j, comp, mi)
                th = b["theta"][b["ok"]]
                dd = b["d"][b["ok"]]
                ro = {"n_earthquakes": int(b["ok"].sum()), "theta_mean": round(float(th.mean()), 5),
                      "theta_sd_between_earthquakes": round(float(th.std(ddof=1)), 5),
                      "d_sd": round(float(dd.std(ddof=1)), 5),
                      "s_median": round(float(np.median(b["s"][b["ok"]])), 5),
                      "theta_by_slot": {EVENTS[s]: round(float(np.nanmean(np.where(b["ok"][:, k], b["theta"][:, k],
                                                                                   np.nan))), 5)
                                        for k, s in enumerate(SLOTS)}}
                seed = BASE_SEED + COND_ORDER.index(cond) * 100 + j * 10 + mi * 2 + (role == "null")
                for case in ("plain", "inflated", "equal"):
                    if role == "null" and case != "plain":
                        continue
                    cur = power_curve(b, m, n_sets, seed, case)
                    ro[case] = {"curve": cur, "K80_holds": k80(cur), "K80_holds_interp": k80_interp(cur), "K80_reject": k80(cur, "reject_right_sign"),
                                "K80_holds_alpha_0.0167": k80(cur, alpha="alpha_0.0167")}
                mo[role] = ro
            so["metrics"][m] = mo
        out["sizes"][sname] = so
    # review 1: the gap line with each unit's size drawn from MIX (plain SEs)
    gm = GAP_COMP[fam]
    out["mixed"] = {"weights": [round(w, 4) for w in MIX], "metrics": {}}
    for mi, m in enumerate(METRICS):
        banks = [bank(r, j, gm, mi) for j in range(len(SIZES))]
        th = np.concatenate([bk["theta"][bk["ok"]] for bk in banks])
        cur = power_curve(banks, m, n_sets, BASE_SEED + 5000 + COND_ORDER.index(cond) * 10 + mi, "plain", MIX)
        out["mixed"]["metrics"][m] = {"theta_mean_pooled_banks": round(float(th.mean()), 5),
                                      "theta_mean_weighted": round(float(sum(
                                          w * bk["theta"][bk["ok"]].mean() for w, bk in zip(MIX, banks))), 5),
                                      "curve": cur, "K80_holds": k80(cur), "K80_holds_interp": k80_interp(cur),
                                      "K80_holds_alpha_0.0167": k80(cur, alpha="alpha_0.0167")}
    return out


def load(out_dir: Path, cond: str) -> dict | None:
    f = Path(out_dir) / f"{cond}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {k: z[k] for k in z.files}


def main_summarise(out_dir: str, out_path: str, n_sets: int = 1000, conds: tuple = COND_ORDER) -> None:
    res = {"item": "C3 power plan (ruling 47, models-fixed truth)", "alpha": ALPHA, "n_boot": N_BOOT,
           "base_seed": BASE_SEED, "k_grid": list(K_GRID), "n_sets_per_K": n_sets, "sizes": list(SIZE_NAMES),
           "slots": [EVENTS[s] for s in SLOTS], "inflate_case2": INFLATE, "conditions": {}}
    for cond in conds:
        r = load(Path(out_dir), cond)
        res["conditions"][cond] = summarise_condition(cond, r, n_sets) if r is not None else "NOT RUN"
        print(f"summarised {cond}", flush=True)
    Path(out_path).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        main_run(a[1], a[2], a[3], int(a[4]), int(a[5]) if len(a) > 5 else 20, int(a[6]) if len(a) > 6 else 3)
    elif a and a[0] == "summarise":
        main_summarise(a[1], a[2], int(a[3]) if len(a) > 3 else 1000, tuple(a[4:]) or COND_ORDER)
    else:
        raise SystemExit(__doc__)

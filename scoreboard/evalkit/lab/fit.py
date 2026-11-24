"""Fit the lab simulator to the real Canterbury data (protocol section D1). Aggregates only are written.

Run from the kit root with the model venv:
  <venv python> -m scoreboard.evalkit.lab.fit <out.json>
Reads the 4,542 train/val CPT positions, label set B and the G6 rows file (median PGA, sd ln PGA) in memory.
Nothing per CPT is written: the output holds coefficients, ranges, prevalences and variogram bins only.
No em dashes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import rules
from .world import EVENTS, Geometry, World

KIT = Path(__file__).resolve().parents[3]
PILLAR = KIT.parents[1]
ROWS_NPZ = PILLAR / "repos" / "liquefaction-model" / "results" / "g6-jun2011-rows-2026-10-31.npz"
SIGMA_GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)   # deviation 1: extended past the registered 0.5 to 2.0
RHO_GRID = (300.0, 600.0, 1200.0, 2500.0, 5000.0)
N_SIM = 6
FIT_SEED = 20261119


def load_real() -> dict:
    """Positions, set B labels and masks, ln median PGA and sd ln PGA per earthquake (memory only)."""
    sys.path.insert(0, str(KIT / "scripts"))
    from ...label_sets import build_label_set, load_rows
    rows = load_rows(npz=ROWS_NPZ)
    d = np.load(ROWS_NPZ, allow_pickle=True)
    assert (d["cpt"] == rows.cpt).all()
    S = build_label_set("B", rows)
    x, y = np.asarray(rows.x, float), np.asarray(rows.y, float)
    out = {"x": x, "y": y, "labels": {}, "masks": {}, "ln_pga": {}, "sd_lnpga": {}}
    for e in EVENTS:
        lab, m = S[e]
        out["masks"][e] = np.asarray(m, bool)
        out["labels"][e] = np.asarray(lab, int)
        out["ln_pga"][e] = np.log(np.asarray(d["pga_" + e], float))
        out["sd_lnpga"][e] = float(np.median(np.asarray(d["sdlnpga_" + e], float)))
    out["centre"] = (float(x.mean()), float(y.mean()))
    return out


def pooled_glm(real: dict) -> dict:
    """Per-earthquake intercept and ln PGA slope, shared quadratic NZTM2000 surface (km, centred)."""
    blocks, ys = [], []
    k = len(EVENTS)
    for j, e in enumerate(EVENTS):
        m = real["masks"][e]
        q = rules.trend_design(real["x"][m], real["y"][m], None, real["centre"])[:, 1:]
        X = np.zeros((m.sum(), 2 * k + q.shape[1]))
        X[:, j] = 1.0
        X[:, k + j] = real["ln_pga"][e][m]
        X[:, 2 * k:] = q
        blocks.append(X)
        ys.append(real["labels"][e][m])
    X, yv = np.vstack(blocks), np.concatenate(ys)
    beta = rules.glm_logistic(yv, X)
    quad = beta[2 * k:]
    return {"a": dict(zip(EVENTS, beta[:k].tolist())), "b": dict(zip(EVENTS, beta[k:2 * k].tolist())),
            "quad": quad.tolist()}


def trend_at(real: dict, quad) -> np.ndarray:
    q = rules.trend_design(real["x"], real["y"], None, real["centre"])[:, 1:]
    return q @ np.asarray(quad)


def geometry(real: dict, quad) -> Geometry:
    return Geometry(real["x"], real["y"], real["masks"], real["ln_pga"], real["sd_lnpga"], trend_at(real, quad),
                    real["centre"])


def residual_structure(real: dict, pairs: dict) -> dict:
    """Per earthquake: in-window exponential fit on label minus the registered GLM p, and on label minus its mean."""
    out = {}
    for e in EVENTS:
        m = real["masks"][e]
        yv = real["labels"][e][m].astype(float)
        r = rules.trend_residual(yv, real["x"][m], real["y"][m], real["ln_pga"][e][m], real["centre"])
        vg = pairs[e].variogram(r)
        vg0 = pairs[e].variogram(yv - yv.mean())
        out[e] = {"n": int(m.sum()), "prevalence": float(yv.mean()), "trend_resid": rules.fit_range_in_window(vg),
                  "label_minus_mean": rules.fit_range_in_window(vg0), "gamma_trend_resid": vg["gamma"]}
    return out


ACCEPT_RANGE_FACTOR = 2.0
ACCEPT_PREVALENCE = 0.03
ACCEPT_WORLDS = 20


def acceptance(geom, pairs, real, glm, prevalence, real_struct, sigma_s, rho_s, n_worlds=ACCEPT_WORLDS) -> dict:
    """Protocol D1 acceptance: the median simulated residual range lies within a factor of 2 of the real range for
    every earthquake whose real range is identified; the mean simulated prevalence lies within 0.03 of the real."""
    w = World(geom, {"sigma_s": sigma_s, "rho_s": rho_s}, glm["b"], prevalence)
    ranges, prev = {e: [] for e in EVENTS}, {e: [] for e in EVENTS}
    for k in range(n_worlds):
        for ev in w.draw(FIT_SEED + 500 + k):
            e, idx = ev["event"], ev["idx"]
            r = rules.trend_residual(ev["y"].astype(float), real["x"][idx], real["y"][idx], real["ln_pga"][e][idx],
                                     real["centre"])
            ranges[e].append(rules.fit_range_in_window(pairs[e].variogram(r))["range_m"])
            prev[e].append(float(ev["y"].mean()))
    out, ok = {}, True
    for e in EVENTS:
        real_fit = real_struct[e]["trend_resid"]
        med, pm = float(np.median(ranges[e])), float(np.mean(prev[e]))
        r_ok = (not real_fit["identified"]) or (real_fit["range_m"] / ACCEPT_RANGE_FACTOR <= med
                                                 <= real_fit["range_m"] * ACCEPT_RANGE_FACTOR)
        p_ok = abs(pm - prevalence[e]) <= ACCEPT_PREVALENCE
        ok = ok and r_ok and p_ok
        out[e] = {"sim_median_range_m": med, "real_range_m": real_fit["range_m"],
                  "real_identified": real_fit["identified"], "range_ok": bool(r_ok),
                  "sim_prevalence": pm, "real_prevalence": prevalence[e], "prevalence_ok": bool(p_ok)}
    return {"pass": bool(ok), "events": out, "n_worlds": n_worlds}


def main(out_path: str) -> None:
    real = load_real()
    glm = pooled_glm(real)
    geom = geometry(real, glm["quad"])
    pairs = {e: rules.Pairs(real["x"][real["masks"][e]], real["y"][real["masks"][e]]) for e in EVENTS}
    real_struct = residual_structure(real, pairs)
    prevalence = {e: real_struct[e]["prevalence"] for e in EVENTS}
    real_g = np.array([real_struct[e]["gamma_trend_resid"] for e in EVENTS])
    grid = []
    for sg in SIGMA_GRID:
        for rho in RHO_GRID:
            w = World(geom, {"sigma_s": sg, "rho_s": rho}, glm["b"], prevalence)
            gs, ranges = [], {e: [] for e in EVENTS}
            for k in range(N_SIM):
                world = w.draw(FIT_SEED + k)
                row = []
                for ev in world:
                    e, idx = ev["event"], ev["idx"]
                    r = rules.trend_residual(ev["y"].astype(float), real["x"][idx], real["y"][idx],
                                             real["ln_pga"][e][idx], real["centre"])
                    vg = pairs[e].variogram(r)
                    row.append(vg["gamma"])
                    ranges[e].append(rules.fit_range_in_window(vg)["range_m"])
                gs.append(row)
            sim_g = np.mean(np.array(gs), axis=0)
            loss = float(np.sum((sim_g - real_g) ** 2))
            grid.append({"sigma_s": sg, "rho_s": rho, "loss": loss,
                         "median_range_m": {e: float(np.median(v)) for e, v in ranges.items()}})
            print(f"sigma {sg} rho {rho} loss {loss:.3e}", flush=True)
    # Deviation 3 (logged in the certificates): the least-loss cell is taken only if it passes the acceptance test;
    # otherwise the next cell by loss that passes (at most the 4 best cells are tried).
    tried, chosen = [], None
    for cell in sorted(grid, key=lambda g: g["loss"])[:4]:
        acc = acceptance(geom, pairs, real, glm, prevalence, real_struct, cell["sigma_s"], cell["rho_s"])
        tried.append({"sigma_s": cell["sigma_s"], "rho_s": cell["rho_s"], "loss": cell["loss"], **acc})
        print("acceptance", tried[-1], flush=True)
        if acc["pass"]:
            chosen = cell
            break
    chosen = chosen or min(grid, key=lambda g: g["loss"])
    res = {"protocol": "docs/evals/LAB-PROTOCOL-2026-11-19.md", "label_set": "B", "seed": FIT_SEED,
           "n_sites_total": int(real["x"].size), "glm": glm, "sd_lnpga_median": real["sd_lnpga"],
           "prevalence": prevalence, "real_residual_structure": real_struct, "grid": grid,
           "acceptance_tried": tried,
           "chosen": {"sigma_s": chosen["sigma_s"], "rho_s": chosen["rho_s"]}}
    Path(out_path).write_text(json.dumps(res, indent=1))
    print("chosen", res["chosen"])


if __name__ == "__main__":
    main(sys.argv[1])

"""Measure the spatial range of the real models' errors on the scorecard v1 sites (amendment 1, step A1).

  <venv python> -m scoreboard.evalkit.lab.measure <out.json>

Record track: labels already read; train/val rows only (check_no_test in load_rows); bench v2 test split never read.
Entries: Level 0, LSN + global fragility, LSN + CES fragility, GEM TS22 (v1 inputs), USGS Zhu 2017 when its inputs
load. Quantities per label set (A, B) and earthquake:
  residual      label minus the entry's p
  pair LL       per-site log-loss difference of two entries (clip 1e-6)
  pair AUC      per-site DeLong placement-value difference of two entries
Each quantity is detrended by ordinary least squares on [1, ln median PGA, x, y, x^2, xy, y^2] (km, NZTM2000),
the same columns as the trend-variogram-v2 GLM, then fitted by lab.rules.fit_range_in_window on all labelled sites.
Uncertainty: the same fit on 40 seeded half-samples (5th, 50th, 95th percentile; share identified). A half-sample
spread understates the full uncertainty of one realisation; stated. Aggregates only are written. No em dashes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

from . import engine, fit as F, rules
from .world import EVENTS

HOME_R10 = Path.home() / ".geoliquefy-runs" / "r10" / "inputs"
FRAG = {"LSN + global fragility": (8.314, 1.113), "LSN + CES fragility": (15.910, 1.131)}
PAIRS = (("Level 0", "LSN + global fragility"), ("LSN + global fragility", "GEM TS22"), ("Level 0", "GEM TS22"),
         ("LSN + global fragility", "USGS Zhu 2017"))
N_HALF = 40
SEED = 20261119


def fragility(lsn, xm, beta):
    lsn = np.asarray(lsn, float)
    out = np.zeros_like(lsn)
    pos = lsn > 0
    out[pos] = norm.cdf((np.log(lsn[pos]) - np.log(xm)) / beta)
    return out


def entries(d, e, lsn, gem, lon, lat):
    from ...g5 import p_level0
    out = {"Level 0": p_level0(d["lnk_" + e], d["sdlnpga_" + e])}
    for k, (xm, b) in FRAG.items():
        out[k] = fragility(lsn[f"lsn_{e}"], xm, b)
    out["GEM TS22"] = np.asarray(gem[f"B_{e}_p_gem"], float)
    try:
        from ...ref_usgs_gf import reference_L, to_site_p
        out["USGS Zhu 2017"] = to_site_p(reference_L(e, lon, lat, d["pga_" + e])[0])
    except Exception as exc:  # inputs not on this machine
        print("Zhu not loaded:", exc, flush=True)
    return out


def detrend(v, X):
    b, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ b


def fit_all(v, X, pairs_full, halves):
    r = detrend(v, X)
    full = rules.fit_range_in_window(pairs_full.variogram(r))
    hs = []
    for idx, pr in halves:
        rh = detrend(v[idx], X[idx])
        hs.append(rules.fit_range_in_window(pr.variogram(rh)))
    rng = np.array([h["range_m"] for h in hs])
    return {"range_m": round(full["range_m"], 1), "identified": full["identified"],
            "structured_share": round(full["structured_share"], 3),
            "half_p05_p50_p95_m": [round(float(q), 1) for q in np.percentile(rng, [5, 50, 95])],
            "half_identified_share": round(float(np.mean([h["identified"] for h in hs])), 3)}


def main(out_path: str) -> None:
    sys.path.insert(0, str(F.KIT / "scripts"))
    from ...label_sets import build_label_set, load_rows
    rows = load_rows(npz=F.ROWS_NPZ)
    d = np.load(F.ROWS_NPZ, allow_pickle=True)
    lsn = np.load(HOME_R10 / "lsn_rows.npz", allow_pickle=True)
    gem = np.load(HOME_R10 / "gem_p.npz")
    assert (lsn["cpt"] == rows.cpt).all()
    x, y = np.asarray(rows.x, float), np.asarray(rows.y, float)
    centre = (float(x.mean()), float(y.mean()))
    res = {"note": __doc__.split("\n\n")[1], "sets": {}}
    for sname in ("B", "A"):
        S = build_label_set(sname, rows)
        res["sets"][sname] = {}
        for e in EVENTS:
            lab, m = S[e]
            yv = lab[m].astype(float)
            yb = yv.astype(bool)
            ents = {k: np.clip(v[m], 0.0, 1.0) for k, v in entries(d, e, lsn, gem, rows.lon, rows.lat).items()}
            X = rules.trend_design(x[m], y[m], np.log(d["pga_" + e][m]), centre)
            pf = rules.Pairs(x[m], y[m])
            rng = np.random.default_rng(SEED)
            halves = []
            for _ in range(N_HALF):
                idx = np.sort(rng.choice(m.sum(), m.sum() // 2, replace=False))
                halves.append((idx, rules.Pairs(x[m][idx], y[m][idx])))
            out = {"n": int(m.sum()), "prevalence": round(float(yv.mean()), 3), "label_minus_mean": fit_all(yv - yv.mean(), X, pf, halves)}
            for k, p in ents.items():
                out[f"residual | {k}"] = fit_all(yv - p, X, pf, halves)
            ll = {k: engine.site_log_loss(yv, p) for k, p in ents.items()}
            pv = {k: engine.placement_values(yb, p) for k, p in ents.items()}
            for a, b in PAIRS:
                if a in ents and b in ents:
                    out[f"pair LL | {a} - {b}"] = fit_all(ll[a] - ll[b], X, pf, halves)
                    out[f"pair AUC | {a} - {b}"] = fit_all(pv[a] - pv[b], X, pf, halves)
            res["sets"][sname][e] = out
            print(sname, e, {k: (v["range_m"], v["identified"]) for k, v in out.items() if isinstance(v, dict)}, flush=True)
    Path(out_path).write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------------------------------
# Amendment 1, step A2: refit the lab's model-error range to the measured paired-difference ranges
# ---------------------------------------------------------------------------------------------------------------
RHO_M_GRID = (2000.0, 4000.0, 8000.0, 12000.0, 20000.0)
SD_N_GRID = (0.5, 0.2, 0.05)      # independent model-noise sd, relative to sd_s
REFIT_WORLDS = 8


def summarise(ranges: list[tuple[float, bool]]) -> dict:
    ident = [r for r, ok in ranges if ok]
    return {"n": len(ranges), "share_not_identified": round(1 - len(ident) / max(len(ranges), 1), 3),
            "identified_median_m": round(float(np.median(ident)), 1) if ident else None}


def real_summary(measured: dict, label_set: str = "B") -> dict:
    out = {}
    for kind in ("pair LL", "pair AUC"):
        rr = [(v["range_m"], v["identified"]) for d in measured["sets"][label_set].values()
              for k, v in d.items() if isinstance(v, dict) and k.startswith(kind)]
        out[kind] = summarise(rr)
    return out


def refit(fit_path: str, measured_path: str, out_path: str) -> None:
    """For each (rho_m, sd_n) on the grid (sd_n added after the rho_m-only grid could not reach the real ranges): simulated pair LL and pair AUC ranges (E1 - E2 and E2 - E3, f1 = 1), detrended as
    above. Loss = summed squared difference to the real set B summary (median identified range in km, plus 10 x the
    share not identified). The least-loss rho_m is the refit; the band holds cells with loss <= 2 x best + 1."""
    from .world import World
    fitd = json.loads(Path(fit_path).read_text())
    measured = json.loads(Path(measured_path).read_text())
    real = F.load_real()
    geom = F.geometry(real, fitd["glm"]["quad"])
    target = real_summary(measured)
    pairs = {e: rules.Pairs(real["x"][real["masks"][e]], real["y"][real["masks"][e]]) for e in EVENTS}
    grid = []
    cells = [(rho, sdn) for sdn in SD_N_GRID for rho in RHO_M_GRID]
    for rho, sdn in cells:
        w = World(geom, {"sigma_s": fitd["chosen"]["sigma_s"], "rho_s": fitd["chosen"]["rho_s"], "rho_m": rho,
                         "sd_n": sdn}, fitd["glm"]["b"], fitd["prevalence"])
        got = {"pair LL": [], "pair AUC": []}
        for k in range(REFIT_WORLDS):
            for ev in w.draw(SEED + 900 + k):
                e, idx = ev["event"], ev["idx"]
                yv = ev["y"].astype(float)
                X = rules.trend_design(real["x"][idx], real["y"][idx], np.log(ev["pga_med"]), real["centre"])
                ll = [engine.site_log_loss(yv, p) for p in ev["p"]]
                pv = [engine.placement_values(yv.astype(bool), p) for p in ev["p"]]
                for a, b in ((0, 1), (1, 2)):
                    for kind, q in (("pair LL", ll), ("pair AUC", pv)):
                        f = rules.fit_range_in_window(pairs[e].variogram(detrend(q[a] - q[b], X)))
                        got[kind].append((f["range_m"], f["identified"]))
        sim = {k: summarise(v) for k, v in got.items()}
        loss = 0.0
        for kind in sim:
            s_med = (sim[kind]["identified_median_m"] or 12000.0) / 1000
            t_med = (target[kind]["identified_median_m"] or 12000.0) / 1000
            loss += (s_med - t_med) ** 2 + (10 * (sim[kind]["share_not_identified"]
                                                 - target[kind]["share_not_identified"])) ** 2
        grid.append({"rho_m": rho, "sd_n": sdn, "sim": sim, "loss": round(loss, 3)})
        print(rho, sdn, sim, round(loss, 3), flush=True)
    best = min(grid, key=lambda g: g["loss"])
    band = [g["rho_m"] for g in grid if g["sd_n"] == best["sd_n"] and g["loss"] <= 2 * best["loss"] + 1.0]
    out = {"target_set_B": target, "grid": grid, "rho_m": best["rho_m"], "sd_n": best["sd_n"],
           "rho_m_band": [min(band), max(band)]}
    Path(out_path).write_text(json.dumps(out, indent=1))
    print("refit", out["rho_m"], out["sd_n"], out["rho_m_band"])


if __name__ == "__main__":
    if sys.argv[1] == "refit":
        refit(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main(sys.argv[1])

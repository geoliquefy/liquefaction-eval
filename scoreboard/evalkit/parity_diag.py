"""M3 parity diagnosis helpers (item A2, G1+G5+G6, eval M3). Descriptive only: no word, no new published value.

The two M3 parity checks of 2026-11-19 missed (docs/evals/PARITY-M3-2026-11-19.md):
  round 1, Christchurch 2011 BI14-LPI threshold 15.88 against the published 8.67;
  round 2, CES BI14-LPI "at least minor" fragility median 2.79 against the published 6.261.
This module holds the small, testable pieces the diagnosis uses. The data run is
scripts/parity_diagnosis_2026_11_23.py; the planted cases are tests/test_parity_diagnosis.py. No em dashes.

Pieces
------
porter_fit        the round 2 binned MLE, unchanged (bins k <= LDM / w < k + 1, bin mean LDM, binomial per bin).
cost_threshold    the round 1 rule: Cost(t) = R_FP(t) + R_FN(t), predict yes when LDM >= t, smallest argmin.
cost_minima       the local minima of that cost curve, to show a two-minimum (bimodal) curve.
zero_floor_share  share of positives at LDM below a floor: the mass a lognormal fragility cannot hold.
published_curve   P(at least minor | LDM) from a published (xm, beta).
lateral_spread_holes, case_filter
                  the engrXiv 121 section 2.1.1 case rules that the round 2 population did not apply.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def porter_fit(ldm, y, w: float = 1.0) -> dict:
    """Porter (2016) binned MLE of P(y | LDM) = Phi(ln(LDM / xm) / beta), exactly as the round 2 run script."""
    ldm = np.asarray(ldm, float)
    y = np.asarray(y, bool)
    k = np.floor(ldm / w).astype(int)
    u, inv = np.unique(k, return_inverse=True)
    n = np.bincount(inv).astype(float)
    f = np.bincount(inv, weights=y.astype(float))
    x = np.bincount(inv, weights=ldm) / n
    if np.any((x <= 0) & (f > 0)):
        raise ValueError("a bin with mean LDM 0 holds a positive case: likelihood is 0")
    pos = x > 0

    def nll(th):
        p = np.zeros_like(x)
        p[pos] = norm.cdf((np.log(x[pos]) - th[0]) / np.exp(th[1]))
        p = np.clip(p, 1e-300, 1 - 1e-16)
        return -np.sum(f * np.log(p) + (n - f) * np.log1p(-p))

    r = minimize(nll, x0=np.array([np.log(5.0), 0.0]), method="Nelder-Mead",
                 options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    return {"xm": float(np.exp(r.x[0])), "beta": float(np.exp(r.x[1])), "bins": int(u.size), "ok": bool(r.success)}


def published_curve(ldm, xm: float, beta: float) -> np.ndarray:
    ldm = np.asarray(ldm, float)
    out = np.zeros_like(ldm)
    pos = ldm > 0
    out[pos] = norm.cdf(np.log(ldm[pos] / xm) / beta)
    return out


def _cost_curve(ldm, y):
    ldm = np.asarray(ldm, float)
    y = np.asarray(y, bool)
    t = np.unique(ldm[np.isfinite(ldm)])
    npos, nneg = y.sum(), (~y).sum()
    order = np.sort(ldm[y])
    oneg = np.sort(ldm[~y])
    fn = np.searchsorted(order, t, side="left") / npos          # positives below t are missed
    fp = (nneg - np.searchsorted(oneg, t, side="left")) / nneg   # negatives at or above t are false alarms
    return t, fp + fn


def cost_threshold(ldm, y) -> float:
    """Round 1 rule: the smallest t among the distinct LDM values that minimises R_FP + R_FN (CR = 1)."""
    t, c = _cost_curve(ldm, y)
    return float(t[np.flatnonzero(c == c.min())[0]])


def cost_minima(ldm, y, window: float = 2.0, top: int = 2) -> list[tuple[float, float]]:
    """The `top` lowest local minima of the cost curve, each the lowest point within +/- window of LDM."""
    t, c = _cost_curve(ldm, y)
    picked: list[tuple[float, float]] = []
    for i in np.argsort(c, kind="stable"):
        if all(abs(t[i] - p[0]) > window for p in picked):
            picked.append((float(t[i]), float(c[i])))
        if len(picked) == top:
            break
    return picked


def zero_floor_share(ldm, y, floor: float = 1.0) -> float:
    """Share of positives with LDM below `floor`."""
    ldm = np.asarray(ldm, float)
    y = np.asarray(y, bool)
    return float(np.mean(ldm[y] < floor)) if y.any() else float("nan")


def lateral_spread_holes(hole, grade) -> set:
    """Holes graded lateral spread (4) or severe lateral spread (5) in any earthquake (engrXiv 121 rule 1 drops the
    CPT, so all its earthquakes). A proxy: the paper drops sites where lateral spreading was "the predominant
    manifestation"; we drop a hole on any grade 4 or 5."""
    hole = np.asarray(hole)
    grade = np.asarray(grade, float)
    return set(hole[np.isin(grade, (4.0, 5.0))].tolist())


def case_filter(hole, grade, predrill_m, gwt_m, end_depth_m, short_m: float | None = None) -> dict:
    """Boolean keep masks for the engrXiv 121 case rules: (1) no lateral-spread CPT; (2) pre-drill not deeper than
    the water table; (3) optional proxy for a CPT ended early on gravel: measured end shallower than short_m."""
    ls = lateral_spread_holes(hole, grade)
    k_ls = ~np.isin(np.asarray(hole), list(ls))
    k_pd = ~(np.asarray(predrill_m, float) > np.asarray(gwt_m, float))
    k_sh = np.ones_like(k_ls) if short_m is None else (np.asarray(end_depth_m, float) >= short_m)
    return {"lateral": k_ls, "predrill": k_pd, "short": k_sh, "all": k_ls & k_pd & k_sh}

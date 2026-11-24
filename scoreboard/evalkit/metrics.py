"""Metrics for one card (rule book R9). Pure numpy and scipy; every function takes labels y (0 or 1) and values.

  auc                       Mann-Whitney, ties one half (reused from scoreboard/scorer.py)
  log_loss(y, p, clip)      -mean[y ln p_c + (1 - y) ln(1 - p_c)], p_c = clip(p, c, 1 - c); reported at every CLIPS
  brier(y, p)               mean (p - y)^2; needs no clip
  citl(y, p, clip)          calibration in the large: a in logit P(y) = a + logit(p_c), slope fixed at 1
  recal_slope(y, p, clip)   b in logit P(y) = a + b logit(p_c), the logistic recalibration slope
  *_no01                    the same with exact 0 and 1 removed, so no clip enters (review fix 2)
  isotonic_fit              CORP recalibration: pool adjacent violators on p, equal p pooled first
                            (Dimitriadis, Gneiting and Jordan 2021)
  brier_decomposition       Brier = MCB - DSC + UNC with the CORP fit: MCB = B(p) - B(p_hat), DSC = UNC - B(p_hat),
                            UNC = ybar (1 - ybar); exact by construction
  level_miss(y, p)          logit(ybar) - logit(mean p); printed only for a census label source (R9)
No em dashes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.special import expit
from scipy.special import logit as _logit

from ..scorer import auc

CLIPS = (1e-6, 1e-4, 1e-3, 1e-2)
RELIABILITY_GRID = tuple(round(0.05 * k, 2) for k in range(21))
_NEWTON_MAX = 100
_NEWTON_TOL = 1e-9


def clip_key(clip: float) -> str:
    """'1e-06', '0.0001', ...: the key a clip-dependent statistic is stored under."""
    return f"{clip:g}"


def clipped_logit(p: np.ndarray, clip: float) -> np.ndarray:
    return _logit(np.clip(p, clip, 1.0 - clip))


def log_loss(y: np.ndarray, p: np.ndarray, clip: float) -> float:
    pc = np.clip(p, clip, 1.0 - clip)
    return float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log1p(-pc)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def exact_shares(p: np.ndarray) -> dict[str, float]:
    """Shares of exact 0 and exact 1: the values a clip decides for."""
    return {"share_exact_0": float(np.mean(p == 0.0)), "share_exact_1": float(np.mean(p == 1.0))}


def without_exact(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = (p > 0.0) & (p < 1.0)
    return y[keep], p[keep]


def _loglik(y: np.ndarray, z: np.ndarray) -> float:
    return float(np.sum(y * z - np.logaddexp(0.0, z)))


def _newton_step(x: np.ndarray, y: np.ndarray, w: np.ndarray, slope_free: bool) -> np.ndarray:
    mu = expit(w[0] + w[1] * x)
    v, r = mu * (1.0 - mu), y - mu
    if not slope_free:
        return np.array([r.sum() / v.sum(), 0.0])
    vx = v * x
    h = np.array([[v.sum(), vx.sum()], [vx.sum(), (vx * x).sum()]])
    return np.linalg.solve(h, np.array([r.sum(), (r * x).sum()]))


def _logistic_fit(x: np.ndarray, y: np.ndarray, slope_free: bool) -> np.ndarray | None:
    """Maximum likelihood (a, b) of logit P(y) = a + b x by Newton from (0, 1), halving any step that lowers the
    likelihood. With slope_free False, b stays 1 (x is an offset). None if it does not converge (separation)."""
    w = np.array([0.0, 1.0])
    ll = _loglik(y, w[0] + w[1] * x)
    for _ in range(_NEWTON_MAX):
        try:
            step = _newton_step(x, y, w, slope_free)
        except np.linalg.LinAlgError:
            return None
        t = 1.0
        while True:
            w_new = w + t * step
            ll_new = _loglik(y, w_new[0] + w_new[1] * x)
            if ll_new >= ll - 1e-10 * (1.0 + abs(ll)) or t < 1e-10:
                break
            t *= 0.5
        if t < 1e-10 or not np.all(np.isfinite(w_new)):
            return w if np.max(np.abs(step)) < 1e-6 else None
        w, ll = w_new, ll_new
        if np.max(np.abs(t * step)) < _NEWTON_TOL:
            return w
    return None


def citl(y: np.ndarray, p: np.ndarray, clip: float) -> float:
    """Calibration in the large (Van Calster et al. 2016 level 1): a in logit P(y) = a + logit(p_c), slope fixed 1.

    NaN if p is empty or one class is empty.
    """
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    w = _logistic_fit(clipped_logit(p, clip), y, slope_free=False)
    return float("nan") if w is None else float(w[0])


def recal_slope(y: np.ndarray, p: np.ndarray, clip: float) -> float:
    """Logistic recalibration slope b in logit P(y) = a + b logit(p_c) (Van Calster et al. 2016 level 2; Cox 1958).

    NaN if one class is empty, logit(p) is constant, or the fit does not converge (separation).
    """
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    x = clipped_logit(p, clip)
    if np.ptp(x) == 0.0:
        return float("nan")
    w = _logistic_fit(x, y, slope_free=True)
    return float("nan") if w is None else float(w[1])


def isotonic_fit(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """CORP recalibrated probability per site: PAV (scipy) on the mean label of each distinct p, weighted by count."""
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    _, start, counts = np.unique(ps, return_index=True, return_counts=True)
    means = np.add.reduceat(ys, start) / counts
    fitted = isotonic_regression(means, weights=counts.astype(float), increasing=True).x
    out = np.empty(p.size)
    out[order] = np.repeat(fitted, counts)
    return out


def brier_decomposition(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """CORP decomposition of the Brier score: brier = mcb - dsc + unc (all three are >= 0)."""
    p_hat = isotonic_fit(y, p)
    b, b_hat = brier(y, p), brier(y, p_hat)
    ybar = float(y.mean())
    unc = ybar * (1.0 - ybar)
    return {"brier": b, "mcb": b - b_hat, "dsc": unc - b_hat, "unc": unc}


def reliability_curve(y: np.ndarray, p: np.ndarray, grid: tuple[float, ...] = RELIABILITY_GRID) -> dict:
    """The CORP reliability step function read at fixed forecast values (aggregates only, no per-site points).

    `recalibrated[k]` is the isotonic value at forecast `grid[k]`; None outside the range of the forecasts.
    """
    p_hat = isotonic_fit(y, p)
    order = np.argsort(p, kind="stable")
    ps, fs = p[order], p_hat[order]
    out = []
    for g in grid:
        k = int(np.searchsorted(ps, g, side="right")) - 1
        out.append(None if k < 0 or g > ps[-1] else float(fs[k]))
    _, block_n = np.unique(p_hat, return_counts=True)
    return {"forecast": list(grid), "recalibrated": out, "n_blocks": int(block_n.size),
            "smallest_block_n": int(block_n.min())}


def level_miss(y: np.ndarray, p: np.ndarray) -> float:
    """logit(observed share) - logit(mean p). Positive: the entry under-predicts the level. NaN at a 0 or 1 share."""
    o, q = float(np.mean(y)), float(np.mean(p))
    if not (0.0 < o < 1.0 and 0.0 < q < 1.0):
        return float("nan")
    return float(_logit(o) - _logit(q))


def ranking_stats(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    """Statistics for every entry (R9 first line): AUC only."""
    return {"auc": auc(y.astype(bool), s)}


def probability_stats(y: np.ndarray, p: np.ndarray, clip: float, census: bool) -> dict[str, float]:
    """Statistics for a target probability (R9). Keys with '@c' depend on the clip c; '_no01' drop exact 0 and 1."""
    y0, p0 = without_exact(y, p)
    out = {"auc": auc(y.astype(bool), p), "brier": brier(y, p), "mean_p": float(np.mean(p)),
           "citl": citl(y, p, clip), "citl_no01": citl(y0, p0, clip), "slope_no01": recal_slope(y0, p0, clip)}
    for c in CLIPS:
        out[f"log_loss@{clip_key(c)}"] = log_loss(y, p, c)
        out[f"slope@{clip_key(c)}"] = recal_slope(y, p, c)
    dec = brier_decomposition(y, p)
    out.update(mcb=dec["mcb"], dsc=dec["dsc"], unc=dec["unc"])
    if census:
        out["level_miss"] = level_miss(y, p)
    return out

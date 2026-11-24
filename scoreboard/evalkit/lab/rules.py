"""Tile rules under test in the lab (protocol section M1). No em dashes.

  variogram               the kit's current rule (resample.choose_tile), residual = label - shaking-only p
  trend-variogram-v2      residual = label - p of a registered logistic GLM (ln median PGA + quadratic NZTM2000
                          surface) on the earthquake's own labels; all labelled sites; r inside the lag window only;
                          RANGE NOT IDENTIFIED at the window edge -> registered fallback; >= 20 tiles required
  pairdiff-variogram-v2   the same fit on the per-site paired contribution of one comparison (lab only)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from ...scorer import grid_clusters
from .. import resample

CANDIDATES_M = resample.TILE_CANDIDATES_M
MAX_LAG_M = 12000.0
N_BINS = 24
N_GRID = 120
MIN_TILES = 20
FALLBACK_M = 4000.0
PGA_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, np.inf)


# ---------------------------------------------------------------------------------------------------------------
# Residual sources
# ---------------------------------------------------------------------------------------------------------------

def trend_design(x_m: np.ndarray, y_m: np.ndarray, ln_pga: np.ndarray | None,
                 centre: tuple[float, float]) -> np.ndarray:
    """[1, ln PGA, x, y, x^2, xy, y^2] in km about `centre` (the ln PGA column is left out when None)."""
    xk = (np.asarray(x_m, float) - centre[0]) / 1000.0
    yk = (np.asarray(y_m, float) - centre[1]) / 1000.0
    cols = [np.ones_like(xk)]
    if ln_pga is not None:
        cols.append(np.asarray(ln_pga, float))
    cols += [xk, yk, xk * xk, xk * yk, yk * yk]
    return np.column_stack(cols)


def glm_logistic(y: np.ndarray, X: np.ndarray, *, ridge: float = 1e-6, iters: int = 25) -> np.ndarray:
    """Logistic regression coefficients by Newton steps (a tiny ridge keeps a separated fit finite)."""
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    m = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
    beta[0] = np.log(m / (1 - m))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = np.clip(p * (1 - p), 1e-9, None)
        g = X.T @ (y - p) - ridge * beta
        H = (X * w[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def trend_residual(y: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, ln_pga: np.ndarray,
                   centre: tuple[float, float]) -> np.ndarray:
    """Label minus the registered GLM p (trend-variogram-v2 residual)."""
    X = trend_design(x_m, y_m, ln_pga, centre)
    b = glm_logistic(y, X)
    return np.asarray(y, float) - 1.0 / (1.0 + np.exp(-(X @ b)))


def shaking_rates(earlier_pga: np.ndarray, earlier_y: np.ndarray) -> list[list[float]]:
    """R10 shaking-only bins, as docs/scorecard-v1-2026-11-15/build_v1.py shaking_rates."""
    bins = []
    for lo, hi in zip(PGA_EDGES[:-1], PGA_EDGES[1:]):
        k = (earlier_pga >= lo) & (earlier_pga < hi)
        bins.append([lo, hi, int(k.sum()), int(earlier_y[k].sum())])
    i = len(bins) - 1
    while i > 0:
        if bins[i][2] < 30:
            lo_b = bins[i - 1]
            bins[i - 1] = [lo_b[0], bins[i][1], lo_b[2] + bins[i][2], lo_b[3] + bins[i][3]]
            del bins[i]
        i -= 1
    if len(bins) > 1 and bins[0][2] < 30:
        bins[1] = [bins[0][0], bins[1][1], bins[0][2] + bins[1][2], bins[0][3] + bins[1][3]]
        del bins[0]
    return bins


def shaking_only_p(earlier: list[tuple[np.ndarray, np.ndarray]], pga: np.ndarray) -> np.ndarray | None:
    """Earlier earthquakes' observed rate in PGA bins, read at this earthquake's PGA; None with no earlier one."""
    if not earlier:
        return None
    ep = np.concatenate([e[0] for e in earlier])
    ey = np.concatenate([e[1] for e in earlier]).astype(float)
    bins = shaking_rates(ep, ey)
    out = np.full(pga.size, bins[0][3] / bins[0][2])
    for lo, hi, n, yes in bins:
        out[(pga >= lo) & (pga < hi)] = yes / n
    return out


# ---------------------------------------------------------------------------------------------------------------
# Semivariogram on precomputed pairs
# ---------------------------------------------------------------------------------------------------------------

class Pairs:
    """Site pairs closer than max_lag_m with their bin, for fast semivariograms on a fixed geometry."""

    def __init__(self, x_m: np.ndarray, y_m: np.ndarray, *, max_lag_m: float = MAX_LAG_M, n_bins: int = N_BINS):
        xy = np.column_stack([np.asarray(x_m, float), np.asarray(y_m, float)])
        n = xy.shape[0]
        h = pdist(xy)
        k = np.floor(h * (n_bins / max_lag_m)).astype(np.int64)
        keep = k < n_bins
        i, j = np.triu_indices(n, 1)
        idt = np.int16 if n < 32000 else np.int32
        self.i, self.j = i[keep].astype(idt), j[keep].astype(idt)
        self.k = k[keep].astype(np.int8)
        self.n_pairs = np.bincount(self.k, minlength=n_bins)
        edges = np.linspace(0.0, max_lag_m, n_bins + 1)
        self.lag = 0.5 * (edges[:-1] + edges[1:])
        self.n_bins, self.max_lag_m = n_bins, max_lag_m

    def variogram(self, r: np.ndarray) -> dict:
        d = r[self.i] - r[self.j]
        s = np.bincount(self.k, weights=0.5 * d * d, minlength=self.n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            g = s / self.n_pairs
        return {"lag_m": self.lag.tolist(), "gamma": g.tolist(), "n_pairs": self.n_pairs.tolist()}


def fit_range_in_window(vg: dict, *, max_lag_m: float = MAX_LAG_M, n_grid: int = N_GRID) -> dict:
    """Exponential fit with r on n_grid steps from the first non-empty lag centre to max_lag_m (inside the window).
    Not identified when the best r is the top step and the structure is real. r = 0 when the structured share is
    under resample.STRUCTURE_SHARE."""
    lag, gamma, n = (np.asarray(vg[k], dtype=float) for k in ("lag_m", "gamma", "n_pairs"))
    ok = n > 0
    lag, gamma, n = lag[ok], gamma[ok], n[ok]
    grid = np.linspace(lag[0], max_lag_m, n_grid)
    best = None
    for idx, r in enumerate(grid):
        c0, c1, sse = resample._nonneg_wls(1.0 - np.exp(-3.0 * lag / r), gamma, n)
        if best is None or sse < best[4]:
            best = (idx, float(r), c0, c1, sse)
    idx, r, c0, c1, _ = best
    share = c1 / (c0 + c1) if c0 + c1 > 0 else 0.0
    structured = share >= resample.STRUCTURE_SHARE
    identified = (idx < n_grid - 1) or not structured
    return {"fitted_range_m": r, "range_m": (r if structured else 0.0), "identified": bool(identified),
            "structured_share": float(share), "nugget": c0, "partial_sill": c1}


def tile_counts(x_m: np.ndarray, y_m: np.ndarray, sizes=CANDIDATES_M) -> dict[float, int]:
    return {float(s): int(np.unique(grid_clusters(x_m, y_m, s)).size) for s in sizes}


def choose_v2(fit: dict, counts: dict[float, int], *, candidates=CANDIDATES_M, min_tiles: int = MIN_TILES,
              fallback_m: float = FALLBACK_M) -> dict:
    """Tile side from an in-window fit (protocol M1b): the smallest candidate >= r with >= min_tiles tiles; when the
    range is not identified, or no candidate qualifies, the fallback: 4 km if it has >= min_tiles tiles, else the
    largest candidate with >= min_tiles tiles, else the smallest candidate (the count gate then decides)."""
    ok = [c for c in candidates if counts[float(c)] >= min_tiles]
    if fit["identified"]:
        fits = [c for c in ok if c >= fit["range_m"]]
        if fits:
            return {"tile_m": float(min(fits)), "word": None}
    if counts[float(fallback_m)] >= min_tiles:
        tile = fallback_m
    elif ok:
        tile = max(ok)
    else:
        tile = min(candidates)
    word = "RANGE NOT IDENTIFIED" if not fit["identified"] else "RANGE ABOVE THE LARGEST TILE WITH 20 TILES"
    return {"tile_m": float(tile), "word": word}

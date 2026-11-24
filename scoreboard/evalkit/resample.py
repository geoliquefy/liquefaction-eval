"""Tile bootstrap, paired differences, MDD, Holm, and the tile-size chooser (rule book R11). No em dashes.

Pairing: `bootstrap` draws whole tiles with replacement through scoreboard/scorer.py (`_members`, `_draw`), so two
calls with the same tiles, n_boot and seed see the same draws. A paired difference is then the difference of the two
draw arrays, element by element.

Tile-size chooser (a proposal that needs calibration, see README): fit an exponential semivariogram
gamma(h) = c0 + c1 (1 - exp(-3 h / r)) to binned residual semivariances by weighted least squares over a grid of
practical ranges r; if the partial sill c1 is under STRUCTURE_SHARE of the total sill c0 + c1 the range is 0. The tile
is the smallest candidate side at least the range (Roberts et al. 2017: blocks at least the residual range).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.spatial.distance import pdist

from ..scorer import Z_ALPHA, Z_POWER, _boot_p, _draw, _members, _pct, grid_clusters

TILE_CANDIDATES_M = (500.0, 1000.0, 2000.0, 4000.0, 8000.0)
SENSITIVITY_TILES_M = (1000.0, 2000.0, 4000.0)
STRUCTURE_SHARE = 0.05


def bootstrap(stat_fn: Callable[[np.ndarray], dict[str, float]], tiles: np.ndarray, n_boot: int,
              seed: int) -> dict[str, np.ndarray]:
    """Tile block bootstrap (R11): stat_fn(row indices) on each of n_boot draws. Returns one array per statistic.

    A draw where a statistic is undefined (for example one class empty) gives NaN; intervals use the finite draws.
    """
    members = _members(tiles)
    rng = np.random.default_rng(seed)
    rows = [stat_fn(_draw(members, rng)) for _ in range(n_boot)]
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


def interval(point: float, draws: np.ndarray, level: float = 0.95) -> dict:
    """Percentile interval of the finite draws. `n_draws` counts them."""
    d = draws[np.isfinite(draws)]
    tail = 50.0 * (1.0 - level)
    lo, hi = _pct(d, tail, 100.0 - tail)
    return {"value": float(point), "lo": lo, "hi": hi, "n_draws": int(d.size)}


def paired(point_a: float, point_b: float, draws_a: np.ndarray, draws_b: np.ndarray) -> dict:
    """Paired difference a - b with its 95% and 90% intervals, two-sided bootstrap p, SE and MDD (R11 a).

    MDD = (z_0.975 + z_0.80) SE: the smallest true difference this card would find with 80% power. The 90% interval
    (5th and 95th percentiles of the same draws) decides EQUIVALENT (C2).
    """
    d = draws_a - draws_b
    d = d[np.isfinite(d)]
    lo, hi = _pct(d)
    lo90, hi90 = _pct(d, 5.0, 95.0)
    se = float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
    return {"diff": float(point_a - point_b), "lo": lo, "hi": hi, "lo90": lo90, "hi90": hi90, "p": _boot_p(d),
            "se": se, "mdd": (Z_ALPHA + Z_POWER) * se, "n_draws": int(d.size)}


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down: True where the hypothesis is rejected. Same rule as scorer.apply_holm, on a plain list."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    rejected = [False] * len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] < alpha / (len(pvalues) - rank):
            rejected[i] = True
        else:
            break
    return rejected


def tiles_for(x_m: np.ndarray, y_m: np.ndarray, tile_m: float) -> np.ndarray:
    """Square tiles of side tile_m (scorer.grid_clusters)."""
    return grid_clusters(x_m, y_m, tile_m)


# ---------------------------------------------------------------------------------------------------------------
# Tile-size chooser
# ---------------------------------------------------------------------------------------------------------------

def semivariogram(x_m: np.ndarray, y_m: np.ndarray, resid: np.ndarray, *, max_lag_m: float = 12000.0,
                  n_bins: int = 24, max_points: int = 1500, seed: int = 0) -> dict:
    """Binned empirical semivariogram 0.5 mean (r_i - r_j)^2 by distance; at most max_points sites (seeded draw)."""
    x, y, r = (np.asarray(v, dtype=float) for v in (x_m, y_m, resid))
    if r.size > max_points:
        pick = np.random.default_rng(seed).choice(r.size, max_points, replace=False)
        x, y, r = x[pick], y[pick], r[pick]
    h = pdist(np.column_stack([x, y]))
    g = 0.5 * pdist(r[:, None], "sqeuclidean")
    edges = np.linspace(0.0, max_lag_m, n_bins + 1)
    k = np.digitize(h, edges) - 1
    keep = (k >= 0) & (k < n_bins)
    n_pairs = np.bincount(k[keep], minlength=n_bins)
    sums = np.bincount(k[keep], weights=g[keep], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = sums / n_pairs
    return {"lag_m": (0.5 * (edges[:-1] + edges[1:])).tolist(), "gamma": gamma.tolist(), "n_pairs": n_pairs.tolist()}


def _nonneg_wls(b: np.ndarray, g: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Weighted least squares g ~ c0 + c1 b with c0, c1 >= 0. Returns (c0, c1, weighted SSE)."""
    X = np.column_stack([np.ones_like(b), b])
    sw = np.sqrt(w)
    c0, c1 = np.linalg.lstsq(X * sw[:, None], g * sw, rcond=None)[0]
    if c1 < 0:
        c0, c1 = float(np.average(g, weights=w)), 0.0
    elif c0 < 0:
        c0, c1 = 0.0, float(np.sum(w * b * g) / np.sum(w * b * b))
    return float(c0), float(c1), float(np.sum(w * (g - c0 - c1 * b) ** 2))


def variogram_range(vg: dict, n_grid: int = 120) -> dict:
    """Practical range of an exponential model fitted to a binned semivariogram (module docstring rule)."""
    lag, gamma, n = (np.asarray(vg[k], dtype=float) for k in ("lag_m", "gamma", "n_pairs"))
    ok = n > 0
    lag, gamma, n = lag[ok], gamma[ok], n[ok]
    best = None
    for r in np.linspace(lag[0], 2.0 * lag[-1], n_grid):
        c0, c1, sse = _nonneg_wls(1.0 - np.exp(-3.0 * lag / r), gamma, n)
        if best is None or sse < best[3]:
            best = (float(r), c0, c1, sse)
    r, c0, c1, _ = best
    share = c1 / (c0 + c1) if c0 + c1 > 0 else 0.0
    return {"range_m": r if share >= STRUCTURE_SHARE else 0.0, "fitted_range_m": r, "nugget": c0,
            "partial_sill": c1, "structured_share": share}


def choose_tile(x_m: np.ndarray, y_m: np.ndarray, resid: np.ndarray, *,
                candidates: Sequence[float] = TILE_CANDIDATES_M, seed: int = 0) -> dict:
    """R11 tile size from a residual semivariogram: the smallest candidate side at least the practical range."""
    vg = semivariogram(x_m, y_m, resid, seed=seed)
    fit = variogram_range(vg)
    fits = [c for c in candidates if c >= fit["range_m"]]
    tile = min(fits) if fits else max(candidates)
    rule = (f"smallest of {list(candidates)} m at least the exponential practical range; structure needs a partial "
            f"sill >= {STRUCTURE_SHARE} of the sill")
    return {"tile_m": float(tile), "range_beyond_candidates": not fits, "rule": rule, **fit, "variogram": vg}


def tile_sensitivity(stat_fn: Callable[[np.ndarray], dict[str, float]], x_m: np.ndarray, y_m: np.ndarray,
                     key: str, *, n_boot: int, seed: int, sizes: Sequence[float] = SENSITIVITY_TILES_M) -> dict:
    """The interval of statistic `key` at each tile side in `sizes` (R11: 1, 2 and 4 km print beside)."""
    point = stat_fn(np.arange(np.asarray(x_m).size))[key]
    out = {}
    for s in sizes:
        draws = bootstrap(stat_fn, tiles_for(x_m, y_m, s), n_boot, seed)[key]
        out[f"{s:g}"] = interval(point, draws)
    return out

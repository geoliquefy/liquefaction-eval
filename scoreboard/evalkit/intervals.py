"""Three candidate interval methods (own-work list item C1, eval milestone M2). Candidates only. No em dashes.

Nothing here is certified. The card still uses the tile bootstrap of resample.py: every method below sits behind a
flag (`method`) whose default is the current one ("tile" within an earthquake, "hartung-knapp" for a pooled line).
Item C2 runs these candidates in the lab under both truths; item C4 certifies one and wires it in.

  (a) "spatial-block"   a spatial block bootstrap. The block side comes from a measured range of the paired
                        per-site difference, not from a residual variogram: LAB-CERTIFICATES-2026-11-19.md section
                        11.1, set B, identified medians AUC placement difference 5.5 km, log-loss difference 2.7 km.
                        Side = the range rounded up to 500 m, so AUC uses 5.5 km blocks and log loss 3.0 km blocks.
                        Blocks at least the range of the resampled quantity (Roberts et al. 2017). The measured
                        medians are statistics of a noisy fit (certificates section 12.1); they are inputs, not truths.
  (b) "survey-cluster"  a cluster bootstrap on survey units. The caller supplies one unit id per site (for example
                        the survey team, the walked street block or the label batch). Whole units are drawn with
                        replacement, so a shared label error inside a unit is resampled as one piece.
  (c) "event-resample"  for a pooled line: resample earthquakes with replacement, optionally with a second stage that
                        draws one within-earthquake bootstrap value for each picked earthquake. SMALL-K CAVEAT: with k
                        earthquakes there are only C(2k - 1, k) distinct resamples (10 at k = 3, 35 at k = 4), the
                        one-stage variance of the mean is (k - 1) / k of the usual one, and the Canterbury earthquakes
                        share one city and the same models, so they are not independent units (certificates 11.4).

Pairing: two entries resampled with the same ids, n_boot and seed see the same draws (resample.bootstrap), so a paired
difference is the element-wise difference of two draw arrays, as in resample.paired. Interval forms: "percentile"
(resample.paired, the kit default) and "t" (SE times sqrt(G / (G - 1)), Student t on G - 1 degrees of freedom, G the
number of blocks, units or earthquakes; the lab's tile8-t form). Both are proposals for C2 to test.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats

from ..scorer import Z_ALPHA, Z_POWER, _boot_p, _pct, grid_clusters
from . import pool, resample

WITHIN_METHODS = ("tile", "spatial-block", "survey-cluster")
POOLED_METHODS = ("hartung-knapp", "event-resample")
DEFAULT_WITHIN = "tile"
DEFAULT_POOLED = "hartung-knapp"
FORMS = ("percentile", "t")

# LAB-CERTIFICATES-2026-11-19.md section 11.1, set B, 4 earthquakes, detrended: identified median ranges
MEASURED_PAIRDIFF_RANGE_M = {"auc": 5500.0, "log_loss": 2700.0}
RANGE_SOURCE = "docs/evals/LAB-CERTIFICATES-2026-11-19.md section 11.1 (set B, identified medians)"
BLOCK_ROUND_M = 500.0
MIN_BLOCKS = 20            # the R11 count gate, as for tiles
MIN_UNITS = 20
EXACT_MAX_K = 6            # event resampling enumerates all k^k ordered draws up to this k (46,656 at k = 6)


# ---------------------------------------------------------------------------------------------------------------
# Interval forms on paired draws
# ---------------------------------------------------------------------------------------------------------------

def paired(point_a: float, point_b: float, draws_a: np.ndarray, draws_b: np.ndarray, *, form: str = "percentile",
           n_units: int | None = None, level: float = 0.95) -> dict:
    """Paired difference a - b. form "percentile" is resample.paired unchanged (level 0.95 only). form "t": SE of the
    draws times sqrt(G / (G - 1)), Student t on G - 1 degrees of freedom, G = n_units (needs G >= 2)."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")
    if form == "percentile":
        if level != 0.95:
            raise ValueError("the percentile form prints 95% (and 90%) intervals only")
        return {**resample.paired(point_a, point_b, draws_a, draws_b), "form": "percentile"}
    if n_units is None or n_units < 2:
        raise ValueError("the t form needs n_units >= 2 (blocks, units or earthquakes)")
    d = np.asarray(draws_a, float) - np.asarray(draws_b, float)
    d = d[np.isfinite(d)]
    return {**_t_interval(float(point_a - point_b), d, n_units, level), "form": "t", "n_units": int(n_units)}


def _t_interval(point: float, d: np.ndarray, g: int, level: float, ddof: int = 1) -> dict:
    df = g - 1
    se = float(np.std(d, ddof=ddof) * math.sqrt(g / df)) if d.size > 1 else float("nan")
    tq = float(stats.t.ppf(0.5 + level / 2, df))
    t90 = float(stats.t.ppf(0.95, df))
    if se > 0:
        p = float(2.0 * stats.t.sf(abs(point) / se, df))
    else:
        p = 0.0 if point != 0 else 1.0
    return {"diff": point, "lo": point - tq * se, "hi": point + tq * se, "lo90": point - t90 * se,
            "hi90": point + t90 * se, "p": p, "se": se, "mdd": (Z_ALPHA + Z_POWER) * se, "df": df,
            "level": level, "n_draws": int(d.size)}


# ---------------------------------------------------------------------------------------------------------------
# (a) spatial block bootstrap, block side from a measured paired-difference range
# ---------------------------------------------------------------------------------------------------------------

def block_side(range_m: float, *, round_m: float = BLOCK_ROUND_M) -> float:
    """Block side for a range: the range rounded up to round_m, at least round_m."""
    if not np.isfinite(range_m) or range_m < 0:
        raise ValueError(f"range must be a finite non-negative distance, got {range_m!r}")
    return float(max(round_m, math.ceil(range_m / round_m) * round_m))


def spatial_block_bootstrap(stat_fn: Callable[[np.ndarray], dict[str, float]], x_m: np.ndarray, y_m: np.ndarray, *,
                            n_boot: int, seed: int, metric: str | None = None, range_m: float | None = None,
                            min_blocks: int = MIN_BLOCKS) -> dict:
    """Candidate (a). Square blocks of side block_side(range); the range is range_m, or the measured median for
    `metric` ("auc" or "log_loss"). Each metric gets its own blocks, so call once per metric and read that key.

    Returns {"draws": {stat: array}, "block_m", "range_m", "range_source", "n_blocks", "under_min_blocks"}.
    `under_min_blocks` is the R11 count gate: True means the card would read UNDERPOWERED at this block side.
    """
    if (metric is None) == (range_m is None):
        raise ValueError("give exactly one of metric or range_m")
    if metric is not None:
        if metric not in MEASURED_PAIRDIFF_RANGE_M:
            raise ValueError(f"no measured range for metric {metric!r}; known {sorted(MEASURED_PAIRDIFF_RANGE_M)}")
        range_m, source = MEASURED_PAIRDIFF_RANGE_M[metric], RANGE_SOURCE
    else:
        source = "caller"
    side = block_side(float(range_m))
    blocks = grid_clusters(x_m, y_m, side)
    n_blocks = int(np.unique(blocks).size)
    return {"draws": resample.bootstrap(stat_fn, blocks, n_boot, seed), "block_m": side, "range_m": float(range_m),
            "range_source": source, "n_blocks": n_blocks, "under_min_blocks": n_blocks < min_blocks,
            "method": "spatial-block"}


# ---------------------------------------------------------------------------------------------------------------
# (b) cluster bootstrap on survey units
# ---------------------------------------------------------------------------------------------------------------

def _is_missing(v) -> bool:
    """None, an empty string, or any NA: Python, numpy or pandas NaN, NaT, and pd.NA (whose self-comparison cannot
    be cast to bool, so it raises TypeError)."""
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    try:
        return bool(v != v)
    except TypeError:
        return True


def _id_kind(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "bool"
    if isinstance(v, (int, np.integer)):
        return "int"
    if isinstance(v, (float, np.floating)):
        return "float"
    if isinstance(v, str):
        return "str"
    return type(v).__name__


def unit_codes(unit_ids: Sequence, n_sites: int) -> np.ndarray:
    """Integer codes for one survey-unit id per site. Refuses a length other than n_sites, a missing id (None, an
    empty string, or any NA such as a numpy NaN or pd.NA) and mixed id kinds (7 and "7", or 1 and 1.0, would
    otherwise merge by accident). Ids are coded by value in sorted order (first-seen order if they do not sort)."""
    u = list(unit_ids)
    if len(u) != n_sites:
        raise ValueError(f"one unit id per site: {len(u)} ids for {n_sites} sites")
    if any(_is_missing(v) for v in u):
        raise ValueError("every site needs a survey-unit id; a missing id would silently form its own unit")
    kinds = {_id_kind(v) for v in u}
    if len(kinds) > 1:
        raise ValueError(f"survey-unit ids must all be one kind, got {sorted(kinds)}")
    vals = [v.item() if isinstance(v, np.generic) else v for v in u]
    try:
        order = sorted(set(vals))
    except TypeError:
        order = list(dict.fromkeys(vals))
    code = {v: i for i, v in enumerate(order)}
    return np.array([code[v] for v in vals], dtype=np.int64)


def cluster_bootstrap(stat_fn: Callable[[np.ndarray], dict[str, float]], unit_ids: Sequence, *, n_sites: int,
                      n_boot: int, seed: int, min_units: int = MIN_UNITS) -> dict:
    """Candidate (b). Whole survey units drawn with replacement (resample.bootstrap on unit codes), so the lab's
    vectorised engine.weights(codes, n_boot, seed) gives the same draws. n_sites (the number of sites stat_fn
    indexes) is required: a short id list would otherwise drop the sites past its end from every draw, silently.

    Returns {"draws": {stat: array}, "n_units", "under_min_units", "unit_sizes": {min, median, max}}.
    """
    codes = unit_codes(unit_ids, n_sites)
    sizes = np.bincount(codes)
    return {"draws": resample.bootstrap(stat_fn, codes, n_boot, seed), "n_units": int(sizes.size),
            "under_min_units": int(sizes.size) < min_units,
            "unit_sizes": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max())},
            "method": "survey-cluster"}


# ---------------------------------------------------------------------------------------------------------------
# (c) event-level resampling for a pooled line
# ---------------------------------------------------------------------------------------------------------------

def n_distinct_resamples(k: int) -> int:
    """Distinct multisets of k earthquakes drawn with replacement from k: C(2k - 1, k)."""
    return math.comb(2 * k - 1, k)


def small_k_caveat(k: int) -> str:
    return (f"k = {k} earthquakes: only {n_distinct_resamples(k)} distinct resamples; the one-stage bootstrap "
            f"variance of the mean is {k - 1}/{k} of the usual one; earthquakes that share one city and the same "
            f"models are not independent units. Candidate, not certified.")


def event_bootstrap(diffs: Sequence[float], *, within: Sequence[np.ndarray] | None = None, n_boot: int = 2000,
                    seed: int = 0, form: str = "percentile", level: float = 0.95,
                    min_events: int = pool.MIN_EVENTS) -> dict:
    """Candidate (c). Equal-weight pooled mean of per-earthquake paired differences d_i, with an interval from
    resampling earthquakes.

    One stage (`within` None): earthquakes drawn with replacement. Up to EXACT_MAX_K earthquakes every one of the k^k
    ordered draws is enumerated (exact, no seed); above it, n_boot seeded draws.
    Two stage (`within` given: one array of within-earthquake bootstrap differences per earthquake, for example a
    paired tile or block bootstrap): pick k earthquakes with replacement, then one within draw for each pick. This
    keeps a floor of within-earthquake spread when the k values happen to agree, and double counts it when the
    between-earthquake spread is small, so it leans wide.
    form "percentile": percentile interval and bootstrap p of the pooled draws. form "t": SE of the draws times
    sqrt(k / (k - 1)) (undoes the one-stage shrink), Student t on k - 1 degrees of freedom. With exact enumeration
    the SE is the exact bootstrap SE (ddof 0), so the one-stage t form equals pool.equal_weight (tested).

    New for C2: only the two-stage form. The one-stage t form is pool.equal_weight already, and the one-stage
    percentile form undercovers at k = 3 to 4 (the small-k caveat). ZERO WIDTH: when every pooled draw is equal (for
    example three equal differences, one stage) the interval has zero width and p = 0. That is not a decisive result
    at small k. Such a result carries "zero_width": True and says so in its caveat.
    """
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")
    d = np.asarray(diffs, dtype=float)
    k = int(d.size)
    if k < min_events:
        raise ValueError(f"NOT ALLOWED: {k} earthquake(s), a pooled line needs {min_events} (R1 c)")
    if not np.all(np.isfinite(d)):
        raise ValueError("every per-earthquake difference must be finite")
    rng = np.random.default_rng(seed)
    if within is None:
        if k <= EXACT_MAX_K:
            picks = np.array(list(itertools.product(range(k), repeat=k)), dtype=np.int64)
            scheme = "exact enumeration"
        else:
            picks = rng.integers(0, k, size=(n_boot, k))
            scheme = f"{n_boot} seeded draws"
        draws = d[picks].mean(axis=1)
        exact = k <= EXACT_MAX_K
    else:
        exact = False
        w = [np.asarray(a, dtype=float) for a in within]
        if len(w) != k:
            raise ValueError(f"within needs one draw array per earthquake: {len(w)} for {k}")
        w = [a[np.isfinite(a)] for a in w]
        if any(a.size == 0 for a in w):
            raise ValueError("every earthquake needs at least one finite within draw")
        picks = rng.integers(0, k, size=(n_boot, k))
        cols = rng.random((n_boot, k))
        vals = np.empty((n_boot, k))
        for e in range(k):
            sel = picks == e
            idx = np.minimum((cols[sel] * w[e].size).astype(np.int64), w[e].size - 1)
            vals[sel] = w[e][idx]
        draws = vals.mean(axis=1)
        scheme = f"two stage, {n_boot} seeded draws"
    mean = float(d.mean())
    out = {"mean": mean, "k": k, "method": "event-resample", "scheme": scheme, "form": form, "level": level,
           "n_distinct_resamples": n_distinct_resamples(k), "caveat": small_k_caveat(k), "n_draws": int(draws.size)}
    out["zero_width"] = bool(np.ptp(draws) == 0)
    if out["zero_width"]:
        out["caveat"] += " ZERO WIDTH: every resample gives the same mean, so p = 0 here is not a decisive result."
    if form == "percentile":
        tail = 50.0 * (1.0 - level)
        lo, hi = _pct(draws, tail, 100.0 - tail)
        se = float(np.std(draws, ddof=0 if exact else 1))
        return {**out, "lo": lo, "hi": hi, "p": _boot_p(draws), "se": se}
    t = _t_interval(mean, draws, k, level, ddof=0 if exact else 1)
    return {**out, "lo": t["lo"], "hi": t["hi"], "p": t["p"], "se": t["se"], "df": t["df"]}


# ---------------------------------------------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------------------------------------------

def within_event_draws(stat_fn: Callable[[np.ndarray], dict[str, float]], *, method: str = DEFAULT_WITHIN,
                       n_boot: int, seed: int, tiles: np.ndarray | None = None, x_m: np.ndarray | None = None,
                       y_m: np.ndarray | None = None, metric: str | None = None, range_m: float | None = None,
                       unit_ids: Sequence | None = None, n_sites: int | None = None) -> dict:
    """Within-earthquake draws by flag. "tile" (default) is resample.bootstrap on the caller's tiles, unchanged.
    "survey-cluster" needs unit_ids and n_sites (the number of sites stat_fn indexes)."""
    if method == "tile":
        if tiles is None:
            raise ValueError("method 'tile' needs tiles")
        return {"draws": resample.bootstrap(stat_fn, tiles, n_boot, seed), "n_units": int(np.unique(tiles).size),
                "method": "tile"}
    if method == "spatial-block":
        if x_m is None or y_m is None:
            raise ValueError("method 'spatial-block' needs x_m and y_m")
        r = spatial_block_bootstrap(stat_fn, x_m, y_m, n_boot=n_boot, seed=seed, metric=metric, range_m=range_m)
        return {**r, "n_units": r["n_blocks"]}
    if method == "survey-cluster":
        if unit_ids is None or n_sites is None:
            raise ValueError("method 'survey-cluster' needs unit_ids and n_sites")
        return cluster_bootstrap(stat_fn, unit_ids, n_sites=n_sites, n_boot=n_boot, seed=seed)
    raise ValueError(f"method must be one of {WITHIN_METHODS}, got {method!r}")


def pooled_interval(diffs: Sequence[float], ses: Sequence[float] | None = None, *, method: str = DEFAULT_POOLED,
                    within: Sequence[np.ndarray] | None = None, n_boot: int = 2000, seed: int = 0,
                    form: str = "percentile", level: float = 0.95) -> dict:
    """Pooled-line interval by flag. "hartung-knapp" (default) is pool.random_effects unchanged."""
    if method == "hartung-knapp":
        if ses is None:
            raise ValueError("method 'hartung-knapp' needs per-earthquake SEs")
        return pool.random_effects(diffs, ses, level=level)
    if method == "event-resample":
        return event_bootstrap(diffs, within=within, n_boot=n_boot, seed=seed, form=form, level=level)
    raise ValueError(f"method must be one of {POOLED_METHODS}, got {method!r}")

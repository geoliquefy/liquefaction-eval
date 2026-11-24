"""Placement control (own-work list B7, eval M4). Off by default. No em dashes.

Two parts, both built here so card.py keeps only small hooks:

(a) AUC within CPT-density strata.

    Strata rule (fixed here, registered by B8 before any real run):
      1. For each task site, count the CPTs within RADIUS_M (default 1000 m), leaving out records within SELF_M
         (2 m) of the site, which are the site's own record. Density = count / (pi RADIUS_M^2), in CPTs per km2.
      2. Cut points are the 1/3 and 2/3 quantiles of the density over ALL task sites, labelled or not. No label
         is read. Stratum k = number of cut points strictly below the site's density (np.searchsorted, side left),
         so 0 = low, 1 = middle, 2 = high. Equal cut points (many sites with the same density) merge strata, and
         an empty stratum is dropped and the rest renumbered 0, 1, ...; the rule prints the strata it kept.
      3. Basis names which CPTs are counted. "pre_event": pushed before the first earthquake (cannot descend
         from the outcome; the clean control). "all_index": the index today, which includes CPTs pushed after the
         earthquakes where damage was seen; it descends from the outcome, so it can explain the proxy but it can
         also remove true skill. The card prints the basis and its note. B8 picks the basis in its registration.
         The basis is checked, not claimed (review 1): "pre_event" needs every CPT's push date and the first
         earthquake's date, and `build` keeps only CPTs pushed strictly before that date; an undated CPT is left
         out. The rule prints the counts left out by the date filter and as undated.

    What 3 quantile strata can and cannot remove (review 1). They remove placement that steps WITH the strata.
    Placement that steps inside a stratum (finer steps, or steps at other cut points) or that slopes smoothly with
    density stays in the residual. So an OPEN BIAS can come from placement left inside a stratum, not only from a
    confound or from real skill. Planted worlds pin this: labels that step at 5 quantile levels of density (no cut
    at 1/3 or 2/3) give a proxy AUC of about 0.74 overall and about 0.59 inside 3 strata, and the card prints OPEN
    BIAS; inside 5 strata the same proxy falls to about 0.5. The stratum count (N_STRATA, default 3) is therefore a
    choice that B8 must register before any real run. The strata also control density only: the real proxy also
    uses `pushed_after_event` (evalkit.effort.PROXY_FEATURES), which density strata cannot remove, so part of a
    real OPEN BIAS may come from that feature.

    Pooled stratified AUC = sum_k w_k AUC_k with w_k = n_yes,k n_no,k / sum_j n_yes,j n_no,j (the within-stratum
    pairs; the same rule as evalkit.effort.stratified_auc). A stratum with one class empty has weight 0 and is
    counted as skipped. Intervals: the card's tile bootstrap (same tiles, n_boot and seed as the card), so the
    "within strata minus overall" difference is paired.

    The proxy question (the real proxy AUC 0.72 is NOT recomputed here; B8 runs it under the v3 registration):
    given the investigation-density proxy as an entry (feature_allowlist "proxy", diagnostic only), the block reads
      EXPLAINED BY DENSITY STRATA  the 90% interval of the within-strata proxy AUC lies inside 0.5 +/- MARGIN
      OPEN BIAS                    not EXPLAINED, and the 95% interval of the within-strata proxy AUC excludes 0.5
      CANNOT TELL                  neither
    The order is chosen (review 1): EXPLAINED is checked first, because it is an equivalence test (the residual is
    smaller than the margin the card uses for "no difference"). With thousands of sites a small residual can lie
    inside the margin and still exclude 0.5. The word then stays EXPLAINED and the flag `residual_excludes_0.5` is
    true. The size (within-strata proxy AUC - 0.5, with its 95% interval) is printed under every word.
    The block also prints the share of the proxy's overall lift (AUC - 0.5) that the strata remove. When the
    overall lift is below MARGIN the share is not computed (NaN, with a note), since it would swing without bound.

(b) The effort baseline on the imagery footprint (known limit N4, docs/evals/STEP1-REVIEW-2-2026-11-19.md): the
    old `in_aerial_footprint` is the union of the CGD0200 mapped classes, not the extent of the imagery. If the
    mappers chose the mapped extent by looking at the damage, it leaks. `imagery_footprint_feature` builds
    `in_imagery_footprint` from the imagery extent and refuses a footprint that is the class union or that leaves
    a mapped class outside it. The feature lives on its own allowlist, evalkit.effort.FOOTPRINT_FEATURES
    (manifest name "pre-event-imagery"); the card's effort slot takes it only with `effort_imagery_footprint=True`,
    and the card then prints a note naming the allowlist. The flag without an effort row is refused.
    Limit (review 1): the union refusal is weak. A union buffered by a few metres, or one with more than
    FOOTPRINT_TOL of its area added, passes as the "imagery footprint". The real defence is B8's check that the
    polygon comes from the ECan imagery boundary layer (for example AERIAL_IMAGERY_ChCh_PostEQ_NZAM_Sep2010_Boundary).

`planted_world` is a synthetic world for kit tests and gates only, never a card or a result (correction 1).
"""
from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit

from . import metrics as M
from . import resample
from .spec import Entry, align

RADIUS_M = 1000.0
SELF_M = 2.0
N_STRATA = 3
MARGIN = 0.05            # card.EQUIVALENCE_MARGIN["auc"] (a test keeps the two equal)
BASES = {
    "pre_event": "CPTs pushed before the first earthquake: cannot descend from the outcome (the clean control)",
    "all_index": ("every CPT in the index today, including CPTs pushed after the earthquakes where damage was "
                  "seen: descends from the outcome, so it can explain the proxy but can also remove true skill"),
}
PROXY_ALLOWLIST = "proxy"            # evalkit.effort.PROXY_ALLOWLIST_NAME
PROXY_NAME = "diagnostic: investigation-density proxy"      # evalkit.effort.PROXY_NAME
EFFORT_IMAGERY_ALLOWLIST = "pre-event-imagery"              # evalkit.effort.FOOTPRINT_ALLOWLIST_NAME
WORDS = ("EXPLAINED BY DENSITY STRATA", "OPEN BIAS", "CANNOT TELL")
FOOTPRINT_TOL = 0.01                 # share of area for the two footprint refusals
STRATA_LIMIT_NOTE = ("quantile strata remove only placement that steps with the strata; an OPEN BIAS can come "
                     "from placement left inside a stratum (finer steps or a smooth slope), not only from a confound "
                     "or real skill; the stratum count is a choice B8 must register. The strata control density "
                     "only: the proxy's pushed_after_event feature stays in the residual")
EFFORT_IMAGERY_NOTE = ("B7 effort row: the effort-only baseline uses feature allowlist 'pre-event-imagery' "
                       "(in_imagery_footprint from the imagery extent, known limit N4), not 'pre-event'")


class PlacementRefused(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------------
# (a) Strata
# ---------------------------------------------------------------------------------------------------------------

def cpt_density_per_km2(site_x: np.ndarray, site_y: np.ndarray, cpt_x: np.ndarray, cpt_y: np.ndarray,
                        radius_m: float = RADIUS_M, self_m: float = SELF_M) -> np.ndarray:
    """CPTs within radius_m of each site, without those within self_m (the site's own record), per km2."""
    sx, sy = np.asarray(site_x, float), np.asarray(site_y, float)
    cx, cy = np.asarray(cpt_x, float), np.asarray(cpt_y, float)
    if sx.shape != sy.shape or cx.shape != cy.shape or sx.ndim != 1 or cx.ndim != 1:
        raise PlacementRefused("site and CPT coordinates must be 1-D arrays of equal length in pairs")
    if not (np.isfinite(sx).all() and np.isfinite(sy).all() and np.isfinite(cx).all() and np.isfinite(cy).all()):
        raise PlacementRefused("non-finite coordinates")
    if not radius_m > self_m >= 0.0:
        raise PlacementRefused("need radius_m > self_m >= 0")
    if cx.size == 0:
        return np.zeros(sx.size)
    tree = cKDTree(np.column_stack([cx, cy]))
    pts = np.column_stack([sx, sy])
    n_in = np.array([len(v) for v in tree.query_ball_point(pts, radius_m)], float)
    n_self = np.array([len(v) for v in tree.query_ball_point(pts, self_m)], float) if self_m > 0 else 0.0
    return (n_in - n_self) / (math.pi * radius_m ** 2 / 1e6)


def strata_from_density(density: np.ndarray, n_strata: int = N_STRATA) -> tuple[np.ndarray, list[float]]:
    """Quantile strata of the density over all sites (no labels). Returns stratum per site and the cut points."""
    d = np.asarray(density, float)
    if d.ndim != 1 or d.size == 0 or not np.isfinite(d).all():
        raise PlacementRefused("density must be a non-empty finite 1-D array")
    if n_strata < 2:
        raise PlacementRefused("need at least two strata")
    cuts = np.quantile(d, [k / n_strata for k in range(1, n_strata)])
    cuts = np.unique(cuts)
    raw = np.searchsorted(cuts, d, side="left")
    _, st = np.unique(raw, return_inverse=True)          # empty strata dropped, the rest numbered 0, 1, ...
    return st.astype(int).ravel(), [float(c) for c in cuts]


def pooled_stratified_auc(y: np.ndarray, s: np.ndarray, strata: np.ndarray) -> dict:
    """Pooled within-stratum AUC with weights n_yes,k n_no,k, and the per-stratum aggregates (counts, AUC, weight).

    Equal to evalkit.effort.stratified_auc on the pooled value (a test checks it)."""
    y = np.asarray(y).astype(bool)
    s, strata = np.asarray(s, float), np.asarray(strata)
    per, num, den = [], 0.0, 0.0
    for k in np.unique(strata):
        m = strata == k
        ny, nn = int(y[m].sum()), int((~y[m]).sum())
        a = M.auc(y[m], s[m]) if ny and nn else float("nan")
        w = float(ny) * nn
        per.append({"stratum": int(k), "n": int(m.sum()), "n_yes": ny, "n_no": nn, "auc": a, "pairs": w})
        if w:
            num += w * a
            den += w
    for p in per:
        p["weight"] = p["pairs"] / den if den else float("nan")
    return {"auc": num / den if den else float("nan"), "strata_used": sum(1 for p in per if p["pairs"]),
            "strata_skipped": sum(1 for p in per if not p["pairs"]), "by_stratum": per}


@dataclass(frozen=True, eq=False)
class PlacementStrata:
    """Strata for every site of the task plus the rule summary. Only an object `build` returned passes `kit_built`.

    `proxy` is the investigation-density proxy as an Entry (feature_allowlist "proxy"), or None."""
    site_ids: tuple[str, ...]
    stratum: np.ndarray
    rule: dict
    proxy: Entry | None = None


_BUILT: "weakref.WeakKeyDictionary[PlacementStrata, tuple]" = weakref.WeakKeyDictionary()


def _key(ps: PlacementStrata) -> tuple:
    return (ps.site_ids, ps.stratum.tobytes(), repr(sorted(ps.rule.items())), id(ps.proxy))


def kit_built(ps: object) -> bool:
    return isinstance(ps, PlacementStrata) and _BUILT.get(ps) == _key(ps)


def _cpt_dates(cpt_dates, first_event_date, n: int) -> tuple[np.ndarray, np.datetime64]:
    """Push dates as datetime64[D] (NaT = undated) and the first earthquake's date. Refuses a length mismatch."""
    try:
        d = np.asarray(cpt_dates, dtype="datetime64[D]")
        f = np.datetime64(first_event_date, "D")
    except (TypeError, ValueError) as err:
        raise PlacementRefused(f"CPT push dates and the first earthquake date must be dates: {err}") from err
    if d.ndim != 1 or d.size != n:
        raise PlacementRefused("need one push date per CPT")
    if np.isnat(f):
        raise PlacementRefused("the first earthquake date is missing")
    return d, f


def build(site_ids: Sequence[str], site_x: np.ndarray, site_y: np.ndarray, cpt_x: np.ndarray, cpt_y: np.ndarray, *,
          basis: str, radius_m: float = RADIUS_M, n_strata: int = N_STRATA, proxy: Entry | None = None,
          cpt_dates: Sequence | None = None, first_event_date=None) -> PlacementStrata:
    """Density strata for every task site (labelled or not). Takes no labels. `basis` must be named (BASES).

    basis "pre_event" needs `cpt_dates` (one push date per CPT; None or NaT when unknown) and `first_event_date`.
    Only CPTs pushed strictly before that date are counted; undated CPTs are left out. basis "all_index" counts
    every CPT; dates, when given, only print how many were pushed on or after the first earthquake."""
    if basis not in BASES:
        raise PlacementRefused(f"basis must be one of {sorted(BASES)}; got {basis!r}")
    ids = tuple(site_ids)
    if len(set(ids)) != len(ids) or len(ids) != np.asarray(site_x).size:
        raise PlacementRefused("site ids must be unique and match the coordinates")
    if proxy is not None:
        if proxy.manifest.feature_allowlist != PROXY_ALLOWLIST:
            raise PlacementRefused("the proxy must be the investigation-density proxy (feature_allowlist 'proxy')")
        align(proxy, ids)            # refuses a missing site id
    cx, cy = np.asarray(cpt_x, float), np.asarray(cpt_y, float)
    date_rule: dict = {"n_cpts_given": int(cx.size)}
    if basis == "pre_event" and (cpt_dates is None or first_event_date is None):
        raise PlacementRefused("basis 'pre_event' needs every CPT push date and the first earthquake date; "
                               "without them the clean control is a claim the kit cannot check")
    if cpt_dates is not None or first_event_date is not None:
        if cpt_dates is None or first_event_date is None:
            raise PlacementRefused("give both cpt_dates and first_event_date, or neither")
        d, f = _cpt_dates(cpt_dates, first_event_date, cx.size)
        undated = np.isnat(d)
        before = ~undated & (d < f)
        late = int((~undated & ~before).sum())
        date_rule.update(first_event_date=str(f), n_cpts_undated=int(undated.sum()),
                         n_cpts_on_or_after_first_event=late)
        if basis == "pre_event":
            cx, cy = cx[before], cy[before]
            date_rule.update(n_cpts_left_out_by_date=late, n_cpts_left_out_undated=int(undated.sum()),
                             date_filter="pushed strictly before first_event_date; undated CPTs left out")
    dens = cpt_density_per_km2(site_x, site_y, cx, cy, radius_m)
    st, cuts = strata_from_density(dens, n_strata)
    rule = {"rule": f"quantile strata ({n_strata} asked) of CPTs per km2 within {radius_m:g} m of each site, own "
                    f"record within {SELF_M:g} m left out; cut points over all task sites, no labels read",
            "basis": basis, "basis_note": BASES[basis], "radius_m": float(radius_m), "n_strata_asked": int(n_strata),
            "cut_points_per_km2": cuts, "n_strata_kept": int(st.max()) + 1,
            "sites_per_stratum": [int((st == k).sum()) for k in range(int(st.max()) + 1)],
            "n_cpts_counted": int(cx.size), **date_rule,
            "weights": "w_k = n_yes,k n_no,k / sum_j n_yes,j n_no,j (pairs inside a stratum)",
            "strata_limit": STRATA_LIMIT_NOTE}
    out = PlacementStrata(ids, st, rule, proxy)
    _BUILT[out] = _key(out)
    return out


def bias_word(strat_point: float, strat_draws: np.ndarray, margin: float = MARGIN) -> dict:
    """The proxy word from its within-strata AUC (see the module docstring)."""
    d = np.asarray(strat_draws, float)
    d = d[np.isfinite(d)]
    if not math.isfinite(strat_point) or d.size < 2:
        return {"word": "CANNOT TELL", "why": "no finite within-strata AUC"}
    lo, hi = (float(v) for v in np.percentile(d, [2.5, 97.5]))
    lo90, hi90 = (float(v) for v in np.percentile(d, [5.0, 95.0]))
    size = {"value": strat_point - 0.5, "lo": lo - 0.5, "hi": hi - 0.5}
    excludes = lo > 0.5 or hi < 0.5
    if 0.5 - margin <= lo90 and hi90 <= 0.5 + margin:          # checked first, on purpose (module docstring)
        word = "EXPLAINED BY DENSITY STRATA"
    elif excludes:
        word = "OPEN BIAS"
    else:
        word = "CANNOT TELL"
    return {"word": word, "open_bias_size_auc": size, "margin": margin, "residual_excludes_0.5": bool(excludes),
            "rule": ("checked in this order. EXPLAINED: 90% interval inside 0.5 +/- margin (an equivalence test, so "
                     "it wins even when the 95% interval excludes 0.5; residual_excludes_0.5 then says so). "
                     "OPEN BIAS: 95% interval excludes 0.5. Else CANNOT TELL")}


def lift_share_removed(overall: float, within: float, margin: float = MARGIN) -> float:
    """(overall - within) / (overall - 0.5); NaN when the overall lift is below the margin (it would swing)."""
    lift = overall - 0.5
    return (overall - within) / lift if math.isfinite(lift) and lift >= margin else float("nan")


def card_block(ps: PlacementStrata, rows: Sequence[dict], ids: tuple[str, ...], y: np.ndarray, x: np.ndarray,
               yy: np.ndarray, tile_m: float, n_boot: int, seed: int, scored: bool) -> dict:
    """The card's `placement` block. `rows` are card rows (name, v aligned to `ids`, constant). Aggregates only."""
    st = strata_for(ps, ids)
    block = {"rule": dict(ps.rule), "status": "SCORED" if scored else "UNDERPOWERED: rule only"}
    n_lab = [int((st == k).sum()) for k in range(ps.rule["n_strata_kept"])]
    block["labelled_sites_per_stratum"] = n_lab
    if not scored:
        return block
    yb = y.astype(bool)
    series = {r["name"]: np.asarray(r["v"], float) for r in rows if not r["constant"]}
    proxy_name = None
    if ps.proxy is not None:
        proxy_name = PROXY_NAME
        series[proxy_name] = align(ps.proxy, ids)

    def stat(idx: np.ndarray) -> dict:
        yi = yb[idx]
        out = {}
        for n, v in series.items():
            if yi.all() or not yi.any():
                out[f"all|{n}"] = out[f"strat|{n}"] = float("nan")
                continue
            out[f"all|{n}"] = M.auc(yi, v[idx])
            out[f"strat|{n}"] = pooled_stratified_auc(yi, v[idx], st[idx])["auc"]
        return out

    full = np.arange(y.size)
    point = stat(full)
    tiles = resample.tiles_for(x, yy, tile_m)
    draws = resample.bootstrap(stat, tiles, n_boot, seed)
    per = {}
    for n, v in series.items():
        ps_full = pooled_stratified_auc(yb, v, st)
        per[n] = {"auc_overall": resample.interval(point[f"all|{n}"], draws[f"all|{n}"]),
                  "auc_within_density_strata": resample.interval(point[f"strat|{n}"], draws[f"strat|{n}"]),
                  "within_minus_overall": resample.paired(point[f"strat|{n}"], point[f"all|{n}"],
                                                          draws[f"strat|{n}"], draws[f"all|{n}"]),
                  "by_stratum": [{k: p[k] for k in ("stratum", "n", "n_yes", "n_no", "auc", "weight")}
                                 for p in ps_full["by_stratum"]],
                  "strata_skipped": ps_full["strata_skipped"]}
    block["auc"] = {n: v for n, v in per.items() if n != proxy_name}
    if proxy_name is None:
        block["proxy"] = {"word": "NOT SUPPLIED"}
    else:
        o, s = point[f"all|{proxy_name}"], point[f"strat|{proxy_name}"]
        share = lift_share_removed(o, s)
        block["proxy"] = {"name": proxy_name, **per[proxy_name],
                          **bias_word(s, draws[f"strat|{proxy_name}"]),
                          "share_of_lift_removed": share,
                          "note": "diagnostic only, never a verdict baseline; its overall lift is the placement "
                                  "signal (verification bias). " + STRATA_LIMIT_NOTE}
        if not math.isfinite(share):
            block["proxy"]["share_of_lift_note"] = (f"lift too small: overall AUC - 0.5 is below the margin "
                                                    f"{MARGIN:g}, so the share is not computed")
    return block


def strata_for(ps: PlacementStrata, ids: Sequence[str]) -> np.ndarray:
    """The strata in the order of `ids`, joined by site id only (R7). Refuses a missing id."""
    pos = {s: k for k, s in enumerate(ps.site_ids)}
    missing = sum(1 for s in ids if s not in pos)
    if missing:
        raise PlacementRefused(f"placement strata: no stratum for {missing} labelled site(s)")
    return ps.stratum[np.array([pos[s] for s in ids], dtype=int)]


# ---------------------------------------------------------------------------------------------------------------
# (b) Imagery footprint (N4)
# ---------------------------------------------------------------------------------------------------------------

def imagery_footprint_feature(x: np.ndarray, y: np.ndarray, footprint, mapped_classes: Sequence | None = None
                              ) -> np.ndarray:
    """`in_imagery_footprint` (0 / 1) from the imagery extent, a shapely (Multi)Polygon in the site CRS.

    With `mapped_classes` (the aerial-map class geometries) it refuses (N4):
      - a footprint equal to their union (symmetric difference under FOOTPRINT_TOL of the footprint area);
      - a footprint that leaves more than FOOTPRINT_TOL of the mapped area outside it (no one maps off the imagery).
    """
    import shapely
    if footprint is None or shapely.is_empty(footprint) or not shapely.is_valid(footprint):
        raise PlacementRefused("the imagery footprint must be a valid, non-empty polygon")
    area = float(shapely.area(footprint))
    if area <= 0.0:
        raise PlacementRefused("the imagery footprint has no area")
    if mapped_classes is not None:
        geoms = [g for g in mapped_classes if g is not None and not shapely.is_empty(g)]
        if geoms:
            u = shapely.union_all(geoms)
            ua = float(shapely.area(u))
            if float(shapely.area(shapely.symmetric_difference(footprint, u))) < FOOTPRINT_TOL * area:
                raise PlacementRefused("N4: the footprint is the union of the mapped classes, not the imagery extent")
            if ua > 0 and float(shapely.area(shapely.difference(u, footprint))) > FOOTPRINT_TOL * ua:
                raise PlacementRefused("a mapped class lies outside the footprint: it is not the imagery extent")
    return shapely.contains_xy(footprint, np.asarray(x, float), np.asarray(y, float)).astype(float)


# ---------------------------------------------------------------------------------------------------------------
# Planted world (kit tests and gates only)
# ---------------------------------------------------------------------------------------------------------------

PLANTED_MODES = ("placement_only", "placement_off_strata", "continuous")


def planted_world(seed: int, *, n_sites: int = 3000, n_cpts: int = 6000, extent_m: float = 20000.0,
                  mode: str = "placement_only", stratum_logits: Sequence[float] = (-1.6, -0.2, 1.2),
                  skill: float = 1.0, density_slope: float = 1.1,
                  off_strata_logits: Sequence[float] = (-1.8, -1.0, -0.2, 0.6, 1.4)) -> dict:
    """A synthetic city where CPTs cluster in some districts and the labels follow the clusters.

    In every mode the label comes from placement (density) plus a site susceptibility s that is independent of
    density. No mode has a confound or proxy skill of its own. The modes differ in the SHAPE of the placement:
      "placement_only"        a step per tertile stratum, the same cuts the control uses. The proxy (the density
                              itself) ranks well overall and at chance inside 3 strata. This 0.5 is built in by the
                              alignment: it shows the control works when the steps match the strata, nothing more.
      "placement_off_strata"  a step per quantile level of density (default 5 levels, cuts at 0.2, 0.4, 0.6 and
                              0.8, none at 1/3 or 2/3). The proxy keeps ranking inside 3 strata (OPEN BIAS) and
                              falls to chance inside 5 strata.
      "continuous"            label smooth in log density. The proxy keeps ranking inside any number of strata.
    Returns coordinates, CPT points, the tertile stratum, s, labels and two scores.
    """
    if mode not in PLANTED_MODES:
        raise PlacementRefused(f"mode must be one of {PLANTED_MODES}")
    rng = np.random.default_rng(seed)
    # CPT intensity: a few Gaussian clusters on a sparse background
    centres = rng.uniform(0.15 * extent_m, 0.85 * extent_m, (6, 2))
    k = rng.integers(0, 6, n_cpts)
    clustered = centres[k] + rng.normal(0.0, 1500.0, (n_cpts, 2))
    bg = rng.uniform(0.0, extent_m, (n_cpts // 3, 2))
    cpts = np.clip(np.vstack([clustered, bg]), 0.0, extent_m)
    sx, sy = rng.uniform(0.0, extent_m, n_sites), rng.uniform(0.0, extent_m, n_sites)
    dens = cpt_density_per_km2(sx, sy, cpts[:, 0], cpts[:, 1])
    st, _ = strata_from_density(dens)
    s = rng.normal(0.0, 1.0, n_sites)
    if mode == "placement_only":
        eta = np.asarray(stratum_logits, float)[st] + skill * s
    elif mode == "placement_off_strata":
        lv = np.asarray(off_strata_logits, float)
        q, _ = strata_from_density(dens, lv.size)
        eta = lv[q] + skill * s
    else:
        z = np.log1p(dens)
        eta = density_slope * (z - z.mean()) / z.std() + skill * s
    label = (rng.random(n_sites) < expit(eta)).astype(float)
    return {"site_ids": tuple(f"syn:pl{j:05d}" for j in range(n_sites)), "x_m": sx, "y_m": sy,
            "cpt_x": cpts[:, 0], "cpt_y": cpts[:, 1], "density": dens, "stratum": st, "s": s, "label": label,
            "proxy_score": dens + rng.uniform(0.0, 1e-6, n_sites),
            "skill_score": s + rng.normal(0.0, 0.5, n_sites)}

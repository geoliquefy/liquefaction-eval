"""USGS ground-failure liquefaction reference (Zhu et al. 2017, "general" model) at CPT locations.

Serves GOALS.md G5 (no-partner N4, part A). The reference is printed BESIDE the G5 site word, never in it (ruling 28,
amendment 2): the model was trained on data that include Christchurch 2011, and USGS does not publish it for every
earthquake. On Canterbury it is in-sample for USGS; any Canterbury number from it is a pipeline check, not a fresh
test. Registration: docs/REGISTRATION-G5-USGS-GF-REFERENCE-2026-11-04.md. Reads only USGS files (public domain);
reads no label. No em dashes.

The model, exactly as USGS gfail runs it (GitLab ghsc/esi/groundfailure/groundfailure, commit fa7bd28;
src/gfail/models/zhu_2017.py, src/gfail/logbase.py, defaultconfigfiles/models/zhu_2017_general.ini; saved under
data/usgs-ground-failure-canterbury-2026-11-04/raw/gfail-code-fa7bd28/):

    X  = 8.801 + 0.334 ln(PGV_M) - 1.918 ln(Vs30) + 0.0005408 precip - 0.2054 min(dc, dr) - 0.0333 wtd
    PGV_M = clip(PGV, 0, 150 cm/s) / (1 + 2.71828^(-2 (Mw - 6)))       (Mw from the ShakeMap, one decimal)
    P  = 1 / (1 + exp(-X)); P = NaN where Vs30 > 620 m/s, PGV < 3 cm/s or PGA < 10 %g
    L  = 0.4915 / (1 + 42.4 exp(-9.165 P))^2                         ("proportion of area affected")
    L  = L * (0 < slope <= 5 degrees); rounded to 4 decimals

L is an areal proportion on a cell about 460 m x 330 m (15 arc-seconds), not the probability that one CPT site
liquefies. CONVENTION (stated, not derived): the per-CPT p is L of the cell holding the CPT, as-is, with a masked
cell (NaN or 0) taken as 0, then clipped to [eps, 1 - eps] (eps = 1e-6, the G5 scorer's clip, by default).

Two routes, per earthquake:

- "product": USGS published a ground-failure product (Sep 2010 usp000hk46, Feb 2011 usp000huvq). L is read
  directly from its zhu_2017_general.hdf5 grid; nothing is recomputed.
- "site-term": no product (Jun 2011, Dec 2011, Feb 2016 and any other Canterbury earthquake). The site part of the
  model, X0 = X - 0.334 ln(PGV_M), does not depend on the earthquake. It is recovered per cell from the Feb 2011
  product (Sep 2010 where Feb 2011 has no value) by inverting L and subtracting the shaking term computed from that
  earthquake's own ShakeMap. So X0 carries USGS's own Vs30, precip, dc, dr, wtd and masks, not a re-made copy.
  Then X = X0 + 0.334 ln(PGV_M) for the new earthquake. PGV at a CPT: the earthquake's ShakeMap PGV scaled by the
  ratio of the G5 runner's PGA to the ShakeMap PGA at that CPT ("runner-scaled", primary), or the ShakeMap PGV as-is
  ("shakemap", beside). The PGA mask uses the same PGA as the PGV.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PILLAR = ROOT.parents[1]
GF_RAW = PILLAR / "data" / "usgs-ground-failure-canterbury-2026-11-04" / "raw"
F16_RAW = PILLAR / "data" / "usgs-shakemap-feb2016-2026-11-04" / "raw"

GFAIL_COMMIT = "fa7bd28996bfa04eec23e41b9752c579d179041b"
B0, B_PGV, B_VS30, B_PRECIP, B_DIST, B_WTD = 8.801, 0.334, -1.918, 0.0005408, -0.2054, -0.0333
COV_A, COV_B, COV_C = 0.4915, 42.4, 9.165
MIN_PGV, MIN_PGA_PCTG, VS30_MAX, PGV_MAX = 3.0, 10.0, 620.0, 150.0
EPS = 1e-6
L_FLOOR = COV_A / (1.0 + COV_B) ** 2  # L at P = 0, about 2.61e-4

# ComCat ids. The G5 runner's names: sep2010, feb2011, jun2011, dec2011; the bench's: 2010-Darfield,
# 2011-Christchurch, 2016-Valentines.
EVENTS = {
    "sep2010": {"comcat": "usp000hk46", "product": True},
    "feb2011": {"comcat": "usp000huvq", "product": True},
    "jun2011": {"comcat": "usp000j33b", "product": False},
    "dec2011": {"comcat": "usp000jch9", "product": False},  # the M5.9 at 02:18 UTC (runner "2011 M5.9 Lyttelton")
    "feb2016": {"comcat": "us20005019", "product": False},
}
BENCH_EVENT = {"2010-Darfield": "sep2010", "2011-Christchurch": "feb2011", "2016-Valentines": "feb2016"}
SITE_TERM_FROM = ("feb2011", "sep2010")  # products used to recover X0, in order of preference


# ---------------------------------------------------------------------------------------------------------------
# Model pieces (pure)
# ---------------------------------------------------------------------------------------------------------------

def mag_factor(mw: float) -> float:
    """gfail's term: 1 / (1 + 2.71828^(-2 (MW - 6))), MW the ShakeMap magnitude printed to one decimal."""
    m = float(f"{float(mw):.1f}")
    return float(1.0 / (1.0 + np.power(2.71828, -2.0 * (m - 6.0))))


def shaking_term(pgv_cms, mw) -> np.ndarray:
    pgv = np.clip(np.asarray(pgv_cms, float), 0.0, PGV_MAX)
    with np.errstate(divide="ignore"):
        return B_PGV * np.log(pgv * mag_factor(mw))


def full_logit(pgv_cms, vs30, precip_mm, dc_km, dr_km, wtd_m, mw) -> np.ndarray:
    """X of the full model from its input layers (used by the tests and by any open-layer re-implementation)."""
    return (B0 + shaking_term(pgv_cms, mw) + B_VS30 * np.log(np.asarray(vs30, float))
            + B_PRECIP * np.clip(np.asarray(precip_mm, float), 0.0, 2500.0)
            + B_DIST * np.minimum(np.asarray(dc_km, float), np.asarray(dr_km, float)) + B_WTD * np.asarray(wtd_m, float))


def coverage(P) -> np.ndarray:
    P = np.asarray(P, float)
    return COV_A / (1.0 + COV_B * np.exp(-COV_C * P)) ** 2


def coverage_inverse(L) -> np.ndarray:
    """P from L; NaN where L is not in (L_FLOOR, COV_A)."""
    L = np.asarray(L, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = -np.log((np.sqrt(COV_A / L) - 1.0) / COV_B) / COV_C
    return np.where((L > L_FLOOR) & (L < COV_A), P, np.nan)


def L_from_x(X, pgv_cms, pga_pctg, round4: bool = True) -> np.ndarray:
    """L from the logit X with gfail's PGV and PGA masks (masked -> NaN); rounded to 4 decimals as gfail does."""
    with np.errstate(over="ignore"):
        P = 1.0 / (1.0 + np.exp(-np.asarray(X, float)))
    P = np.where((np.asarray(pgv_cms, float) < MIN_PGV) | (np.asarray(pga_pctg, float) < MIN_PGA_PCTG), np.nan, P)
    L = coverage(P)
    return np.round(L, 4) if round4 else L


def to_site_p(L, eps: float = EPS) -> np.ndarray:
    """The stated convention: L as-is, masked (NaN or 0) -> 0, clipped to [eps, 1 - eps]."""
    L = np.asarray(L, float)
    return np.clip(np.where(np.isfinite(L), L, 0.0), eps, 1.0 - eps)


# ---------------------------------------------------------------------------------------------------------------
# USGS files
# ---------------------------------------------------------------------------------------------------------------

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _one(pattern: str, root: Path) -> Path:
    hits = sorted(root.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one file for {pattern} under {root}, found {len(hits)}")
    return hits[0]


def shakemap_path(event: str) -> Path:
    cc = EVENTS[event]["comcat"]
    root = F16_RAW if event == "feb2016" else GF_RAW
    return _one(f"{cc}/shakemap-preferred/*/download/grid.xml", root)


def product_path(event: str) -> Path:
    cc = EVENTS[event]["comcat"]
    return _one(f"{cc}/ground-failure/*/zhu_2017_general.hdf5", GF_RAW)


@dataclass
class ShakeGrid:
    lon0: float
    lat0: float
    dlon: float
    dlat: float
    nlon: int
    nlat: int
    fields: dict  # name -> (nlat, nlon) array, row 0 = lat0 (south)
    mw: float
    event_id: str
    path: str
    sha256: str

    def at(self, lon, lat, field: str) -> np.ndarray:
        """Bilinear value at points (NaN outside the grid)."""
        z = self.fields[field]
        fx = (np.asarray(lon, float) - self.lon0) / self.dlon
        fy = (np.asarray(lat, float) - self.lat0) / self.dlat
        inside = (fx >= 0) & (fx <= self.nlon - 1) & (fy >= 0) & (fy <= self.nlat - 1)
        i0 = np.clip(np.floor(fx).astype(int), 0, self.nlon - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, self.nlat - 2)
        tx, ty = fx - i0, fy - j0
        v = (z[j0, i0] * (1 - tx) * (1 - ty) + z[j0, i0 + 1] * tx * (1 - ty)
             + z[j0 + 1, i0] * (1 - tx) * ty + z[j0 + 1, i0 + 1] * tx * ty)
        return np.where(inside, v, np.nan)


def read_shakemap(path) -> ShakeGrid:
    txt = Path(path).read_text()
    ev = re.search(r"<event [^>]*>", txt).group(0)
    gs = re.search(r"<grid_specification [^>]*>", txt).group(0)

    def attr(s, k):
        return re.search(rf'\b{k}="([^"]*)"', s).group(1)
    names = [m.group(2) for m in re.finditer(r'<grid_field index="(\d+)" name="([^"]+)"', txt)]
    body = txt[txt.index("<grid_data>") + len("<grid_data>"): txt.index("</grid_data>")]
    a = np.array(body.split(), float).reshape(-1, len(names))
    lon, lat = a[:, names.index("LON")], a[:, names.index("LAT")]
    nlon, nlat = int(attr(gs, "nlon")), int(attr(gs, "nlat"))
    if a.shape[0] != nlon * nlat:
        raise ValueError(f"{path}: {a.shape[0]} rows, expected {nlon} x {nlat}")
    ulon, ulat = np.unique(np.round(lon, 6)), np.unique(np.round(lat, 6))
    if ulon.size != nlon or ulat.size != nlat:
        raise ValueError(f"{path}: grid is not regular")
    dlon, dlat = (ulon[-1] - ulon[0]) / (nlon - 1), (ulat[-1] - ulat[0]) / (nlat - 1)
    ii = np.rint((lon - ulon[0]) / dlon).astype(int)
    jj = np.rint((lat - ulat[0]) / dlat).astype(int)
    fields = {}
    for k, nm in enumerate(names):
        if nm in ("LON", "LAT"):
            continue
        z = np.full((nlat, nlon), np.nan)
        z[jj, ii] = a[:, k]
        fields[nm] = z
    return ShakeGrid(float(ulon[0]), float(ulat[0]), float(dlon), float(dlat), nlon, nlat, fields,
                     float(attr(ev, "magnitude")), attr(ev, "event_id"), str(path), _sha(path))


@dataclass
class ProductGrid:
    x: np.ndarray  # cell-centre longitudes, increasing
    y: np.ndarray  # cell-centre latitudes as stored
    L: np.ndarray  # (len(y), len(x)) proportion of area
    shakemap: str
    path: str
    sha256: str

    def cell(self, lon, lat):
        """(row, col, inside) of the cell holding each point (nearest centre)."""
        dx = (self.x[-1] - self.x[0]) / (self.x.size - 1)
        dy = (self.y[-1] - self.y[0]) / (self.y.size - 1)
        c = np.rint((np.asarray(lon, float) - self.x[0]) / dx).astype(int)
        r = np.rint((np.asarray(lat, float) - self.y[0]) / dy).astype(int)
        inside = (c >= 0) & (c < self.x.size) & (r >= 0) & (r < self.y.size)
        return np.clip(r, 0, self.y.size - 1), np.clip(c, 0, self.x.size - 1), inside

    def at(self, lon, lat) -> np.ndarray:
        r, c, inside = self.cell(lon, lat)
        return np.where(inside, self.L[r, c], np.nan)

    def centre(self, lon, lat):
        r, c, _ = self.cell(lon, lat)
        return self.x[c], self.y[r]


def read_product(path) -> ProductGrid:
    """The hdf5 stores y ascending (south first) but the model rows north first: its rows equal the product's
    north-up GeoTIFF (zhu_2017_general_model.tif) cell for cell (checked in scripts/run_usgs_gf_reference.py). So the
    rows are flipped here to match y. Reading the rows in y order instead puts Christchurch on the mirror latitude
    about the grid centre (the first exploration did that; the two products' site terms then disagree, corr 0.29).

    Since 2026-11-23 the row order comes from the shared reader evalkit.zhu_grid (ROW_ORDER_GEOREF): it takes the
    order from the grid's own metadata (the GeoTIFF twin when present, else the writer named in `history`) and gives
    the same arrays as the old fixed flip on every grid on disk (pillar docs/evals/ERRATUM-ZHU-ROW-ORDER-2026-11-23.md).
    A grid whose metadata does not fix the order is refused."""
    from .evalkit import zhu_grid as ZG
    with h5py.File(path, "r") as f:
        x, y, shape = f["x"][:], f["y"][:], f["model"].shape
        sm = f["metadata/model/description"].attrs.get("shakemap")
    if shape != (y.size, x.size):
        raise ValueError(f"{path}: model grid shape {shape} != (y, x) {(y.size, x.size)}")
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError(f"{path}: expected increasing x and y")
    g = ZG.load(path, ZG.ROW_ORDER_GEOREF)
    return ProductGrid(g.x, g.y, np.asarray(g.model, float), str(sm), str(path), g.sha256)


_CACHE: dict = {}


def shakemap(event: str) -> ShakeGrid:
    k = ("sm", event)
    if k not in _CACHE:
        _CACHE[k] = read_shakemap(shakemap_path(event))
    return _CACHE[k]


def product(event: str) -> ProductGrid:
    k = ("gf", event)
    if k not in _CACHE:
        _CACHE[k] = read_product(product_path(event))
    return _CACHE[k]


# ---------------------------------------------------------------------------------------------------------------
# Site term X0 recovered from a product
# ---------------------------------------------------------------------------------------------------------------

def site_term_from_product(event: str, lon, lat, with_cap: bool = False):
    """X0 at the product cell holding each point: logit(P) - 0.334 ln(PGV_M), with P from inverting L and PGV the
    product's own ShakeMap resampled bilinearly to the cell centre (as gfail resamples it). NaN where the cell is
    masked (L NaN or 0: Vs30, slope, PGV or PGA mask). -inf where L sits within 2e-4 of its floor (P about 0; the
    4-decimal rounding leaves no usable logit).

    Amendment 1 (after run 1 failed gate P2b): a floor cell says only that X0 is very low FOR THAT EARTHQUAKE'S
    SHAKING. So the floor holds only while a new earthquake's shaking term is at most the source's ("cap"); above it
    the cell is unknown and L_site_term masks it. with_cap=True returns (X0, cap), cap = the source shaking term at
    floor cells and NaN elsewhere."""
    G, S = product(event), shakemap(event)
    L = G.at(lon, lat)
    cx, cy = G.centre(lon, lat)
    pgv = S.at(cx, cy, "PGV")
    sh = shaking_term(pgv, S.mw)
    P = coverage_inverse(np.where(L > L_FLOOR + 2e-4, L, np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        X0 = np.log(P / (1.0 - P)) - sh
    floor = np.isfinite(L) & (L > 0) & (L <= L_FLOOR + 2e-4)
    X0 = np.where(floor, -np.inf, X0)
    return (X0, np.where(floor, sh, np.nan)) if with_cap else X0


def site_term(lon, lat, sources=SITE_TERM_FROM, with_cap: bool = False):
    """(X0, source index[, cap]): the first product in `sources` with a finite X0 at each point; else -inf if a
    product puts the cell at its floor (cap = the largest such source shaking term); else NaN (masked)."""
    X0 = np.full(np.asarray(lon).size, np.nan)
    cap = np.full(X0.size, np.nan)
    src = np.full(X0.size, -1)
    for k, e in enumerate(sources):
        v, c = site_term_from_product(e, lon, lat, with_cap=True)
        fill = ~np.isfinite(X0) & np.isfinite(v)
        X0[fill], src[fill], cap[fill] = v[fill], k, np.nan
        flo = ~np.isfinite(X0) & np.isneginf(v) & ~(cap >= c)
        X0[flo], src[flo], cap[flo] = -np.inf, k, c[flo]
    return (X0, src, cap) if with_cap else (X0, src)


def apply_cap(X0, cap, shaking) -> np.ndarray:
    """Amendment 1: a floor cell (X0 = -inf) whose new shaking term exceeds its cap is unknown (NaN, masked)."""
    X0 = np.asarray(X0, float)
    return np.where(np.isneginf(X0) & ~(np.asarray(shaking, float) <= np.asarray(cap, float)), np.nan, X0)


# ---------------------------------------------------------------------------------------------------------------
# The reference at CPTs
# ---------------------------------------------------------------------------------------------------------------

def L_site_term(event: str, lon, lat, pga_g=None, *, pgv: str = "runner-scaled", x0=None, cap=None) -> np.ndarray:
    """L by the site-term route. pga_g: the G5 runner's PGA in g at each point (needed for "runner-scaled").
    x0, cap: optional precomputed X0 and floor cap (the shuffle defect passes permuted copies)."""
    S = shakemap(event)
    pgv_sm, pga_sm = S.at(lon, lat, "PGV"), S.at(lon, lat, "PGA")
    if pgv == "runner-scaled":
        if pga_g is None:
            raise ValueError("runner-scaled PGV needs the runner's PGA (g) at each point")
        pga_pctg = 100.0 * np.asarray(pga_g, float)
        pgv_v = pgv_sm * pga_pctg / pga_sm
    elif pgv == "shakemap":
        pga_pctg, pgv_v = pga_sm, pgv_sm
    else:
        raise ValueError(pgv)
    if x0 is None:
        x0, _, cap = site_term(lon, lat, with_cap=True)
    sh = shaking_term(pgv_v, S.mw)
    X0 = apply_cap(x0, np.full(np.size(x0), np.nan) if cap is None else cap, sh)
    return L_from_x(X0 + sh, pgv_v, pga_pctg)


def reference_L(event: str, lon, lat, pga_g=None, *, route: str = "auto", pgv: str = "runner-scaled") -> tuple:
    """(L, route used). route "auto": the product where USGS published one, else the site-term route."""
    if event not in EVENTS:
        raise KeyError(f"no USGS ground-failure inputs registered for {event!r}; known: {sorted(EVENTS)}")
    if route == "auto":
        route = "product" if EVENTS[event]["product"] else "site-term"
    if route == "product":
        return product(event).at(lon, lat), "product"
    return L_site_term(event, lon, lat, pga_g, pgv=pgv), f"site-term/{pgv}"


def p_usgs(event: str, lon, lat, pga_g=None, *, route: str = "auto", pgv: str = "runner-scaled",
           eps: float = EPS) -> np.ndarray:
    """Per-CPT p of the USGS reference under the stated convention (see the module docstring)."""
    return to_site_p(reference_L(event, lon, lat, pga_g, route=route, pgv=pgv)[0], eps)


def p_usgs_bench(bench_event: str, lon, lat, pga_g, *, eps: float = EPS) -> np.ndarray:
    """Entry point for the N9 sealed read. bench_event: the bench's `event` value ("2010-Darfield",
    "2011-Christchurch" or "2016-Valentines"); lon, lat: the bench's lon/lat columns (WGS84 degrees); pga_g: the
    bench's PGA column in g (used only on the site-term route, i.e. Feb 2016). Returns p per row, route "auto",
    PGV "runner-scaled". Opens only USGS files; reads no label."""
    return p_usgs(BENCH_EVENT[bench_event], lon, lat, pga_g, eps=eps)


def provenance() -> dict:
    out = {"gfail_commit": GFAIL_COMMIT, "model": "zhu_2017_general", "eps": EPS, "events": {}}
    for e, v in EVENTS.items():
        d = {"comcat": v["comcat"], "route": "product" if v["product"] else "site-term"}
        try:
            S = shakemap(e)
            d.update({"shakemap": str(Path(S.path).relative_to(PILLAR)), "shakemap_sha256": S.sha256,
                      "shakemap_mw": S.mw})
            if v["product"]:
                G = product(e)
                d.update({"product": str(Path(G.path).relative_to(PILLAR)), "product_sha256": G.sha256,
                          "product_shakemap": G.shakemap})
        except FileNotFoundError as err:
            d["missing"] = str(err)
        out["events"][e] = d
    return out

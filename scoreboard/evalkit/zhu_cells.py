"""Read a USGS Zhu 2017 grid at case sites without turning empty cells into 0 (own-work list B2, eval M4). No em dashes.

Scorecard v2 deviation 3: the world reader in docs/scorecard-v2-2026-11-19/build_v2.py (`world()`, the line
`use.loc[k, "zhu"] = np.nan_to_num(v, nan=0.0)`) read every empty (NaN) Zhu cell as 0. On Hyogoken-Nambu that set
22 of 26 P1 inputs, so that cell rested mostly on tied zeros. That file is frozen with the sealed card; this module is
the replacement reader for any later card.

The rule here: a site whose nearest grid cell is empty (NaN), or that lies outside the grid extent by more than half a
cell, is OFF-GRID. An off-grid site keeps no value. It is dropped from that model's cell, and the cell prints how many
sites were dropped (`drop_note`). On-grid sites get exactly the nearest-cell value the v2 reader gave.

Usage, in place of the v2 lines (`cell_input` is the hook a card builder calls; it wraps read_cells, drop_off_grid
and drop_note):
    c = zhu_cells.cell_input(f["x"][()], f["y"][()], f["model"][()], lon, lat, label)
    auc(c.label, c.values)          # print c.note beside the cell
`rec` and `note` hold counts only (clause 6.2): no per-site row goes on a card.

Row order (erratum 2026-11-23, pillar docs/evals/ERRATUM-ZHU-ROW-ORDER-2026-11-23.md). These functions take arrays and
trust that row i of `model` sits at gy[i]. The USGS hdf5 stores gy ascending but the rows north first, so
`f["model"][()]` as stored is mirrored. Get the arrays from the shared reader instead:
    g = zhu_cells.load_grid(path, zhu_cells.ROW_ORDER_GEOREF)
    c = zhu_cells.cell_input_ruled(g.x, g.y, g.model, lon, lat, label, pga_g, pgv_cms, vs30_ms)
The default ROW_ORDER_AS_STORED is the registered read of the sealed v2 and v3 cards and is kept only so they
rebuild.

Open ruling (B2 review 1): which masks are OFF-GRID and which are MODEL ZERO. The Canterbury Zhu path
(scoreboard/ref_usgs_gf.py `to_site_p`) still reads a masked cell as 0 by a stated convention; there the masks are the
model's applicability rules (PGA, PGV, Vs30). This reader treats an empty world cell as no data. Scorecard v3 (B8) is
not built until a founder ruling says which rule applies to which mask, applied the same way to both readers. The ask
is pillar docs/evals/asks/ZHU-MASK-RULE-2026-11-23.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .zhu_grid import ROW_ORDER_AS_STORED, ROW_ORDER_GEOREF  # noqa: F401  (re-exported for card builders)
from .zhu_grid import load as load_grid  # noqa: F401

OFF_GRID = "OFF-GRID"


@dataclass(frozen=True, eq=False)
class ZhuRead:
    """Values at the sites (NaN where off-grid), the on-grid mask, and why each off-grid site is off."""
    values: np.ndarray
    on_grid: np.ndarray
    outside_extent: np.ndarray
    empty_cell: np.ndarray

    @property
    def n_off_grid(self) -> int:
        return int((~self.on_grid).sum())


def _edges(axis: np.ndarray) -> tuple[float, float]:
    a = np.asarray(axis, float)
    if a.ndim != 1 or a.size < 2 or not np.all(np.isfinite(a)):
        raise ValueError("a grid axis must be a finite 1-D array of at least two cell centres")
    half = 0.5 * float(np.median(np.abs(np.diff(a))))
    return float(a.min()) - half, float(a.max()) + half


def read_cells(gx: np.ndarray, gy: np.ndarray, model: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> ZhuRead:
    """Nearest-cell Zhu value at each (lon, lat), with off-grid sites marked and given NaN, never 0.

    gx: cell-centre longitudes (columns); gy: cell-centre latitudes (rows); model: the grid, shape (len(gy), len(gx)).
    """
    gx, gy = np.asarray(gx, float), np.asarray(gy, float)
    model = np.asarray(model, float)
    lon, lat = np.atleast_1d(np.asarray(lon, float)), np.atleast_1d(np.asarray(lat, float))
    if model.shape != (gy.size, gx.size):
        raise ValueError(f"grid shape {model.shape} does not match the axes ({gy.size}, {gx.size})")
    if lon.shape != lat.shape or lon.ndim != 1:
        raise ValueError("lon and lat must be 1-D arrays of the same length")
    if not (np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))):
        raise ValueError("every site needs a finite lon and lat")
    x0, x1 = _edges(gx)
    y0, y1 = _edges(gy)
    outside = (lon < x0) | (lon > x1) | (lat < y0) | (lat > y1)
    ix = np.abs(gx[None, :] - lon[:, None]).argmin(axis=1)
    iy = np.abs(gy[None, :] - lat[:, None]).argmin(axis=1)
    v = model[iy, ix].astype(float)
    empty = ~np.isfinite(v) & ~outside
    on = ~outside & ~empty
    return ZhuRead(values=np.where(on, v, np.nan), on_grid=on, outside_extent=outside, empty_cell=empty)


def drop_off_grid(read: ZhuRead, label: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """(keep mask, counts record). The keep mask drops every off-grid site from this model's cell.

    With `label` (1 yes, 0 no), the record also says how many yes and no sites were dropped.
    """
    keep = read.on_grid.copy()
    rec = {"n_sites": int(keep.size), "n_kept": int(keep.sum()), "n_off_grid": read.n_off_grid,
           "n_empty_cell": int(read.empty_cell.sum()), "n_outside_extent": int(read.outside_extent.sum())}
    if label is not None:
        y = np.asarray(label)
        if y.shape != keep.shape:
            raise ValueError("label needs one value per site")
        off = ~keep
        rec["n_yes_dropped"] = int((y[off] == 1).sum())
        rec["n_no_dropped"] = int((y[off] == 0).sum())
    return keep, rec


def drop_note(rec: dict) -> str:
    """The text the cell prints beside its score: how many sites were off the grid and dropped."""
    if rec["n_off_grid"] == 0:
        return f"0 of {rec['n_sites']} sites off the model grid"
    s = (f"{rec['n_off_grid']} of {rec['n_sites']} sites off the model grid, dropped from this cell "
         f"({rec['n_empty_cell']} on empty cells, {rec['n_outside_extent']} outside the grid)")
    if "n_yes_dropped" in rec:
        s += f": {rec['n_yes_dropped']} yes, {rec['n_no_dropped']} no"
    return s


@dataclass(frozen=True, eq=False)
class CellInput:
    """What one Zhu cell scores on after the drop: kept labels and values, the counts record and the printed note."""
    label: np.ndarray
    values: np.ndarray
    rec: dict
    note: str


def cell_input(gx: np.ndarray, gy: np.ndarray, model: np.ndarray, lon: np.ndarray, lat: np.ndarray,
               label: np.ndarray) -> CellInput:
    """The card hook for B2 part 2: read the grid, drop off-grid sites, return the cell input and its note.

    `label` and `values` are the per-site arrays the metric needs. They stay in memory and never go on a card.
    The card carries `note` (and `rec` if wanted), which hold counts only.
    """
    r = read_cells(gx, gy, model, lon, lat)
    keep, rec = drop_off_grid(r, label)
    return CellInput(label=np.asarray(label)[keep], values=r.values[keep], rec=rec, note=drop_note(rec))


# ---------------------------------------------------------------------------------------------------------------
# Ruling 46 (founder 2026-11-23), registered by scorecard v3 (B8, docs/evals/SCORECARD-V3-REGISTRATION-2026-11-23.md
# section 4.3): split empty cells by reason, one test in both readers (Canterbury and world).
# ---------------------------------------------------------------------------------------------------------------
LIMITS = {"pga_g_below": 0.10, "pgv_cms_below": 3.0, "vs30_ms_above": 620.0}    # the model's own published limits
LIMITS_SOURCE = ("Zhu et al. 2017 limits as the USGS ground-failure code applies them: PGA < 0.10 g, PGV < 3 cm/s, "
                 "Vs30 > 620 m/s give no hazard (pillar docs/evals/asks/ZHU-MASK-RULE-2026-11-23.md)")
MODEL_ZERO = "MODEL ZERO"
NOT_CHECKABLE = "NOT CHECKABLE"
RULING_46 = ("ruling 46: an empty Zhu cell is 0 only where the site fails one of the model's own limits (PGA, PGV or "
             "Vs30) from our inputs; any other empty cell is OFF-GRID and dropped; where the reason cannot be "
             "checked, the cell is dropped; a site outside the grid extent has no cell and is OFF-GRID")


def cell_input_ruled(gx: np.ndarray, gy: np.ndarray, model: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                     label: np.ndarray, pga_g: np.ndarray, pgv_cms: np.ndarray, vs30_ms: np.ndarray) -> CellInput:
    """Ruling 46 reader, the same in both readers. `pga_g`, `pgv_cms` and `vs30_ms` are our inputs at the sites (nan
    where we have none). An empty cell whose site fails any limit we can check is 0 (MODEL ZERO). An empty cell whose
    site passes every limit, all three checked, is OFF-GRID and dropped. An empty cell with no failing limit and at
    least one input missing is NOT CHECKABLE and dropped. Counts only go in `rec` and `note` (clause 6.2)."""
    r = read_cells(gx, gy, model, lon, lat)
    n = r.values.size
    ins = [np.atleast_1d(np.asarray(a, float)) for a in (pga_g, pgv_cms, vs30_ms)]
    if any(a.shape != (n,) for a in ins):
        raise ValueError("pga_g, pgv_cms and vs30_ms need one value per site (nan where missing)")
    pga, pgv, vs = ins
    y = np.asarray(label)
    if y.shape != (n,):
        raise ValueError("label needs one value per site")
    fail = {"pga": np.isfinite(pga) & (pga < LIMITS["pga_g_below"]),
            "pgv": np.isfinite(pgv) & (pgv < LIMITS["pgv_cms_below"]),
            "vs30": np.isfinite(vs) & (vs > LIMITS["vs30_ms_above"])}
    any_fail = fail["pga"] | fail["pgv"] | fail["vs30"]
    all_known = np.isfinite(pga) & np.isfinite(pgv) & np.isfinite(vs)
    zero = r.empty_cell & any_fail
    off = (r.empty_cell & ~any_fail & all_known) | r.outside_extent
    unk = r.empty_cell & ~any_fail & ~all_known
    keep = r.on_grid | zero
    values = np.where(zero, 0.0, r.values)
    rec = {"rule": RULING_46, "limits_source": LIMITS_SOURCE, "n_sites": int(n), "n_kept": int(keep.sum()),
           "n_on_grid_value": int(r.on_grid.sum()), "n_model_zero": int(zero.sum()),
           "n_model_zero_by_limit": {k: int((r.empty_cell & v).sum()) for k, v in fail.items()},
           "n_off_grid": int(off.sum()), "n_outside_extent": int(r.outside_extent.sum()),
           "n_not_checkable": int(unk.sum()),
           "n_yes_dropped": int((y[~keep] == 1).sum()), "n_no_dropped": int((y[~keep] == 0).sum())}
    note = (f"ruling 46: {rec['n_kept']} of {n} sites kept ({rec['n_model_zero']} set to 0 by a model limit); "
            f"{rec['n_off_grid']} OFF-GRID and {rec['n_not_checkable']} NOT CHECKABLE dropped: "
            f"{rec['n_yes_dropped']} yes, {rec['n_no_dropped']} no")
    return CellInput(label=y[keep], values=values[keep], rec=rec, note=note)


ON_GRID_VALUE = "ON-GRID VALUE"
REASONS = (ON_GRID_VALUE, MODEL_ZERO, OFF_GRID, NOT_CHECKABLE)


def split_by_reason(gx: np.ndarray, gy: np.ndarray, model: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                    pga_g: np.ndarray, pgv_cms: np.ndarray, vs30_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ruling 46 split of `cell_input_ruled`, label free, for a frozen prediction file (D8 Zhu refreeze).

    Returns (values, reason) per site, in site order. reason is one of REASONS. values: the cell value where ON-GRID
    VALUE, 0.0 where MODEL ZERO, NaN where OFF-GRID or NOT CHECKABLE (both dropped at scoring). The classes are the
    same masks as `cell_input_ruled` (a test checks this); nothing here reads or takes a label."""
    r = read_cells(gx, gy, model, lon, lat)
    n = r.values.size
    ins = [np.atleast_1d(np.asarray(a, float)) for a in (pga_g, pgv_cms, vs30_ms)]
    if any(a.shape != (n,) for a in ins):
        raise ValueError("pga_g, pgv_cms and vs30_ms need one value per site (nan where missing)")
    pga, pgv, vs = ins
    any_fail = ((np.isfinite(pga) & (pga < LIMITS["pga_g_below"]))
                | (np.isfinite(pgv) & (pgv < LIMITS["pgv_cms_below"]))
                | (np.isfinite(vs) & (vs > LIMITS["vs30_ms_above"])))
    all_known = np.isfinite(pga) & np.isfinite(pgv) & np.isfinite(vs)
    zero = r.empty_cell & any_fail
    off = (r.empty_cell & ~any_fail & all_known) | r.outside_extent
    reason = np.full(n, NOT_CHECKABLE, dtype=object)
    reason[r.on_grid] = ON_GRID_VALUE
    reason[zero] = MODEL_ZERO
    reason[off] = OFF_GRID
    return np.where(zero, 0.0, r.values), reason

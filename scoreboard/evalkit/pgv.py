"""PGV baselines from ShakeMap grids (own-work list B5, eval M4). Optional card rows, off by default. No em dashes.

Two rows, both built from the ShakeMap PGV alone (no soil, no site memory, no PGA):

  baseline: pgv score     score = PGV (cm/s) read from the task's ShakeMap grid at each site. Output kind "score",
                          AUC only (R3). No fit, so it exists on any task whose grid carries PGV.
  baseline: pgv only      p = expit(a + b ln PGV), a and b fitted by maximum likelihood on the pooled labelled rows
                          of EARLIER earthquakes only (never the scored one; refused by event id). Same Newton fit
                          and the same guards as the B4 shaking-only fit (evalkit.shaking: MIN_CLASS yes and no,
                          no separation, convergence, slope above 0).

No fill (the B5 "done when"). PGV is never filled from PGA, from a neighbour, from a GMM or from a default. When a
site has no PGV the row is not built, and the card prints why in the `pgv` block:
  - NO SHAKEMAP GRID           the task names no grid for this earthquake;
  - PGV NOT IN GRID            the grid has no PGV field;
  - PGV MISSING AT k OF n SITES  k listed sites fall outside the grid or on an empty node (bilinear needs all four
                               corners);
  - for the probability row, PGV MISSING IN EARLIER EARTHQUAKE <id> when an earlier earthquake lacks PGV at any of
    its labelled rows: the fit is NOT AVAILABLE (the score row may still be built).
A NOT AVAILABLE object carries no entry; the card prints the block and a note and adds no row.

Grid reading: ShakeMap grid.xml (or a .zip holding one), the USGS layout: <event magnitude event_id>, one
<grid_field> per column, PGV in cm/s, rows of a regular lon/lat grid. Sampling is bilinear on the regular grid
(NaN outside or next to an empty node), the convention of scoreboard/ref_usgs_gf.py ShakeGrid.at. B8 (scorecard v3
registration section 5) adds `sampling="nearest"` to `score_baseline`: the nearest grid node, NaN outside or on an
empty node, recorded in the summary; the default stays bilinear and its output is unchanged. The grid's
sha256, event id, ShakeMap version and PGV units print in the summary.

Task id and grid id: a task names its earthquake (for example "feb2011"); a ShakeMap grid carries its ComCat id
(for example "usp000huvq"). The builders take an explicit `shakemap_id` when the two differ, check the grid against
it, and print both ids in the summary and the row statement. With no `shakemap_id` the grid id must equal the task id.

Grids on disk (pillar data/, checked 2026-11-23, header read only): 49 grid.xml and grid.xml.zip files carry PGV
(3 of them Kaikoura, refused), among them the five Canterbury atlas v1 grids
(data/usgs-ground-failure-canterbury-2026-11-04: Sep 2010, Feb 2011, Jun 2011, two Dec 2011) and the eleven
non-US world-set grids (data/comcat-shakemap-nonus-2026-11-07). The scorecard v2 PGA is not a USGS ShakeMap,
so a v3 card that wants both must name both sources (B8).
Other grid names (shakemap_grid.xml, v21_grid.xml in the discovery workdir) were not counted.
Kaikoura 2016 (us1000778i) stays out (ruling 45 correction 6): read_grid refuses it (EXCLUDED_EVENTS).

Limits (printed in the summary):
- PGV with no magnitude factor. A magnitude-scaled form (as in Zhu et al. 2017) is not built; within one earthquake
  M is one number, so the ranking is the same; it only changes how earlier earthquakes pool in the fit.
- The atlas ShakeMaps are retrospective versions, not the maps issued at the time (R21 names the version).

Provenance: `score_baseline`, `probability_baseline` and `not_available` register what they return (a weak
registry keyed by the object). The card takes only a registered object whose entry and summary are unchanged.

Pure numpy; reads a file only in `read_grid`.
"""
from __future__ import annotations

import hashlib
import json
import re
import weakref
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import shaking as SH
from .spec import Entry, EntryManifest, entry_sha256

NAME_SCORE = "baseline: pgv score"
NAME_PROB = "baseline: pgv only"
FIT_MARKER = "B5 PGV fit:"
STATUS_BUILT = "BUILT"
STATUS_NA = "NOT AVAILABLE"
NO_FILL = "no fill: PGV is never filled from PGA, a neighbour, a GMM or a default"
EXCLUDED_EVENTS = {"us1000778i": "Kaikoura 2016 stays out (ruling 45 correction 6)"}
LIMITS = ("PGV with no magnitude factor; a magnitude-scaled form (Zhu et al. 2017) is not built; within one "
          "earthquake the ranking is the same",
          "atlas ShakeMaps are retrospective versions, not the maps issued at the time (R21 names the version)",
          NO_FILL)


class PgvRefused(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------------
# Grid reading and sampling
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Grid:
    """One ShakeMap grid: regular lon/lat axes, the fields as (nlat, nlon) arrays (row 0 = south), and provenance."""
    event_id: str
    magnitude: float
    version: str
    lon0: float
    lat0: float
    dlon: float
    dlat: float
    nlon: int
    nlat: int
    fields: dict = field(repr=False)
    units: dict = field(repr=False)
    sha256: str = ""

    def has(self, name: str) -> bool:
        return name in self.fields

    def provenance(self) -> dict:
        return {"event_id": self.event_id, "shakemap_version": self.version, "magnitude": self.magnitude,
                "sha256": self.sha256, "fields": sorted(self.fields), "pgv_units": self.units.get("PGV")}


def _attr(tag: str, key: str, what: str) -> str:
    m = re.search(rf'\b{key}="([^"]*)"', tag)
    if m is None:
        raise PgvRefused(f"{what}: no {key} attribute")
    return m.group(1)


def parse_grid(text: str, sha256: str = "", what: str = "grid") -> Grid:
    """Parse ShakeMap grid.xml text. Refuses a grid that is not regular or has the wrong row count."""
    root = re.search(r"<shakemap_grid [^>]*>", text)
    ev = re.search(r"<event [^>]*>", text)
    gs = re.search(r"<grid_specification [^>]*>", text)
    if root is None or ev is None or gs is None or "<grid_data>" not in text:
        raise PgvRefused(f"{what}: not a ShakeMap grid.xml (shakemap_grid, event, grid_specification, grid_data)")
    event_id = _attr(ev.group(0), "event_id", what)
    if event_id in EXCLUDED_EVENTS:
        raise PgvRefused(f"{what}: {event_id}: {EXCLUDED_EVENTS[event_id]}")
    fields = re.findall(r'<grid_field index="(\d+)" name="([^"]+)" units="([^"]*)"', text)
    fields.sort(key=lambda t: int(t[0]))
    names = [f[1] for f in fields]
    if "LON" not in names or "LAT" not in names:
        raise PgvRefused(f"{what}: the grid needs LON and LAT fields")
    body = text[text.index("<grid_data>") + len("<grid_data>"): text.index("</grid_data>")]
    try:
        a = np.array(body.split(), float)
    except ValueError as exc:
        raise PgvRefused(f"{what}: grid_data must be numeric ({exc})") from None
    if a.size % len(names):
        raise PgvRefused(f"{what}: grid_data does not split into rows of {len(names)} values")
    a = a.reshape(-1, len(names))
    nlon, nlat = int(_attr(gs.group(0), "nlon", what)), int(_attr(gs.group(0), "nlat", what))
    if nlon < 2 or nlat < 2 or a.shape[0] != nlon * nlat:
        raise PgvRefused(f"{what}: {a.shape[0]} rows, expected {nlon} x {nlat} (at least 2 x 2)")
    lon, lat = a[:, names.index("LON")], a[:, names.index("LAT")]
    ulon, ulat = np.unique(np.round(lon, 6)), np.unique(np.round(lat, 6))
    if ulon.size != nlon or ulat.size != nlat:
        raise PgvRefused(f"{what}: the grid is not regular")
    dlon, dlat = (ulon[-1] - ulon[0]) / (nlon - 1), (ulat[-1] - ulat[0]) / (nlat - 1)
    ii = np.rint((lon - ulon[0]) / dlon).astype(int)
    jj = np.rint((lat - ulat[0]) / dlat).astype(int)
    out, units = {}, {}
    for k, (_, nm, un) in enumerate(fields):
        if nm in ("LON", "LAT"):
            continue
        z = np.full((nlat, nlon), np.nan)
        z[jj, ii] = a[:, k]
        out[nm], units[nm] = z, un
    version = re.search(r'\bshakemap_version="([^"]*)"', root.group(0))
    return Grid(event_id, float(_attr(ev.group(0), "magnitude", what)), version.group(1) if version else "",
                float(ulon[0]), float(ulat[0]), float(dlon), float(dlat), nlon, nlat, out, units, sha256)


def read_grid(path: str | Path) -> Grid:
    """Read grid.xml, or the one grid .xml inside a .zip. The sha256 is of the file as stored."""
    p = Path(path)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            xml = [n for n in z.namelist() if n.endswith("grid.xml")]
            if len(xml) != 1:
                raise PgvRefused(f"{p}: the zip must hold exactly one grid.xml, holds {len(xml)}")
            text = z.read(xml[0]).decode("utf-8")
    else:
        text = raw.decode("utf-8")
    return parse_grid(text, sha, str(p))


def sample(grid: Grid, name: str, lon, lat) -> np.ndarray:
    """Bilinear value of a field at points; NaN outside the grid or next to an empty node. Never fills."""
    if not grid.has(name):
        raise PgvRefused(f"grid {grid.event_id}: no {name} field")
    z = grid.fields[name]
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    fx, fy = (lon - grid.lon0) / grid.dlon, (lat - grid.lat0) / grid.dlat
    inside = (fx >= 0) & (fx <= grid.nlon - 1) & (fy >= 0) & (fy <= grid.nlat - 1) & np.isfinite(fx) & np.isfinite(fy)
    fx, fy = np.where(inside, fx, 0.0), np.where(inside, fy, 0.0)
    i0 = np.clip(np.floor(fx).astype(int), 0, grid.nlon - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, grid.nlat - 2)
    tx, ty = fx - i0, fy - j0
    v = (z[j0, i0] * (1 - tx) * (1 - ty) + z[j0, i0 + 1] * tx * (1 - ty)
         + z[j0 + 1, i0] * (1 - tx) * ty + z[j0 + 1, i0 + 1] * tx * ty)
    return np.where(inside, v, np.nan)


SAMPLINGS = {"bilinear": "bilinear on the regular ShakeMap grid; NaN outside or next to an empty node",
             "nearest": "nearest grid node (nearest cell); NaN outside the grid by more than half a cell or on an "
                        "empty node"}


def sample_nearest(grid: Grid, name: str, lon, lat) -> np.ndarray:
    """Value of a field at the nearest grid node; NaN outside the grid (beyond half a cell) or on an empty node.
    Never fills (B8: scorecard v3 registration section 5 names nearest cell for the B5 PGV score)."""
    if not grid.has(name):
        raise PgvRefused(f"grid {grid.event_id}: no {name} field")
    z = grid.fields[name]
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    fx, fy = (lon - grid.lon0) / grid.dlon, (lat - grid.lat0) / grid.dlat
    ok = np.isfinite(fx) & np.isfinite(fy)
    i = np.rint(np.where(ok, fx, -1.0)).astype(int)
    j = np.rint(np.where(ok, fy, -1.0)).astype(int)
    inside = ok & (i >= 0) & (i <= grid.nlon - 1) & (j >= 0) & (j <= grid.nlat - 1)
    v = z[np.clip(j, 0, grid.nlat - 1), np.clip(i, 0, grid.nlon - 1)]
    return np.where(inside, v, np.nan)


def _check_sampling(sampling: str) -> str:
    if sampling not in SAMPLINGS:
        raise PgvRefused(f"sampling must be one of {sorted(SAMPLINGS)}, got {sampling!r}")
    return sampling


def pgv_at(grid: Grid | None, lon, lat, sampling: str = "bilinear") -> tuple[np.ndarray, str | None]:
    """PGV (cm/s) at the points and the reason it is missing (None when every point has a PGV). Never fills.
    `sampling` "bilinear" (default) or "nearest" (B8)."""
    _check_sampling(sampling)
    n = np.asarray(lon, float).size
    if grid is None:
        return np.full(n, np.nan), "NO SHAKEMAP GRID"
    if not grid.has("PGV"):
        return np.full(n, np.nan), "PGV NOT IN GRID"
    if grid.units.get("PGV") not in ("cm/s", "cms"):
        raise PgvRefused(f"grid {grid.event_id}: PGV units {grid.units.get('PGV')!r}, expected cm/s")
    v = sample(grid, "PGV", lon, lat) if sampling == "bilinear" else sample_nearest(grid, "PGV", lon, lat)
    bad = ~np.isfinite(v) | (v <= 0)
    return v, (f"PGV MISSING AT {int(bad.sum())} OF {n} SITES" if bad.any() else None)


# ---------------------------------------------------------------------------------------------------------------
# The baseline objects
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class PgvBaseline:
    """A PGV row (entry) or a NOT AVAILABLE record (entry None), with the aggregate summary the card prints."""
    entry: Entry | None
    summary: dict


@dataclass(frozen=True)
class Earlier:
    """One earlier earthquake's labelled rows for the fit: id, PGV (cm/s, NaN where missing) and 0/1 labels."""
    event_id: str
    pgv: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "pgv", np.asarray(self.pgv, float))
        object.__setattr__(self, "y", np.asarray(self.y, float))
        if self.pgv.ndim != 1 or self.pgv.shape != self.y.shape or self.pgv.size == 0:
            raise PgvRefused(f"earlier {self.event_id!r}: needs one PGV and one label per row")
        if not np.all(np.isin(self.y, (0.0, 1.0))):
            raise PgvRefused(f"earlier {self.event_id!r}: labels must be 0 or 1")


_BUILT: "weakref.WeakKeyDictionary[PgvBaseline, tuple]" = weakref.WeakKeyDictionary()


def _key(summary: dict) -> str:
    return json.dumps(summary, sort_keys=True, default=str)


def _register(out: PgvBaseline) -> PgvBaseline:
    _BUILT[out] = (out.entry, entry_sha256(out.entry) if out.entry is not None else None, _key(out.summary))
    return out


def kit_built(obj: object) -> bool:
    """True only for an object this module returned, with the same entry object, entry hash and summary."""
    if not isinstance(obj, PgvBaseline):
        return False
    rec = _BUILT.get(obj)
    if rec is None or rec[0] is not obj.entry or rec[2] != _key(obj.summary):
        return False
    return rec[1] == (entry_sha256(obj.entry) if obj.entry is not None else None)


def _grid_id(event_id: str, shakemap_id: str | None) -> str:
    """The ComCat id the grid must carry: `shakemap_id` when given, else the task's event id."""
    if shakemap_id is None:
        return event_id
    if not isinstance(shakemap_id, str) or not shakemap_id.strip():
        raise PgvRefused("shakemap_id must be a non-empty string (the grid's ComCat id)")
    if shakemap_id in EXCLUDED_EVENTS:
        raise PgvRefused(f"{shakemap_id}: {EXCLUDED_EVENTS[shakemap_id]}")
    return shakemap_id


def _check_grid(event_id: str, grid: Grid | None, shakemap_id: str | None) -> str:
    want = _grid_id(event_id, shakemap_id)
    if grid is not None and grid.event_id != want:
        hint = "" if shakemap_id is not None else "; pass shakemap_id when the task id and the ComCat id differ"
        raise PgvRefused(f"the grid is for {grid.event_id!r}, not the scored earthquake {event_id!r} "
                         f"(expected grid id {want!r}){hint}")
    return want


def _base_summary(event_id: str, grid: Grid | None, version: str, shakemap_id: str | None = None) -> dict:
    return {"scored_event": event_id, "shakemap_id": _grid_id(event_id, shakemap_id), "version": version,
            "grid": grid.provenance() if grid is not None else None,
            "sampling": "bilinear on the regular ShakeMap grid; NaN outside or next to an empty node",
            "no_fill": NO_FILL, "limits": list(LIMITS)}


def not_available(event_id: str, reason: str, grid: Grid | None = None, version: str = "score",
                  n_sites: int | None = None, shakemap_id: str | None = None) -> PgvBaseline:
    """The NOT AVAILABLE record: no entry, the reason, and (for missing sites) the count."""
    s = _base_summary(event_id, grid, version, shakemap_id)
    s.update(status=f"{STATUS_NA}: {reason}; {NO_FILL}", reason=reason, n_sites=n_sites)
    return _register(PgvBaseline(None, s))


def _manifest(name: str, kind: str, statement: str, event_id: str) -> EntryManifest:
    return EntryManifest(model=name, output_kind=kind, statement=statement, exposure={event_id: "unseen"},
                         house=False)


def score_baseline(event_id: str, site_ids: Sequence[str], lon, lat, grid: Grid | None,
                   shakemap_id: str | None = None, sampling: str = "bilinear") -> PgvBaseline:
    """The ranking row from the grid at every listed site, or NOT AVAILABLE with the reason (no fill).

    `event_id` is the task's id; `shakemap_id` is the grid's ComCat id when it differs (both print).
    `sampling`: "bilinear" (default, unchanged) or "nearest" (nearest grid node; B8 scorecard v3).
    """
    site_ids = tuple(site_ids)
    _check_grid(event_id, grid, shakemap_id)
    v, why = pgv_at(grid, lon, lat, sampling)
    if v.size != len(site_ids):
        raise PgvRefused("one longitude and one latitude per site id")
    if why is not None:
        out = not_available(event_id, why, grid, "score", len(site_ids), shakemap_id)
        if sampling == "bilinear":
            return out
        s = dict(out.summary)
        s["sampling"] = SAMPLINGS[sampling]
        return _register(PgvBaseline(None, s))
    how = "bilinear" if sampling == "bilinear" else "nearest grid node"
    stmt = (f"scored earthquake {event_id}; PGV (cm/s) from ShakeMap {grid.event_id} version {grid.version or '?'} "
            f"(sha256 {grid.sha256[:12]}), {how} at each site; ranking only, no fit; {NO_FILL}")
    s = _base_summary(event_id, grid, "score", shakemap_id)
    s.update(status=STATUS_BUILT, n_sites=len(site_ids), ties=SH.tie_count(v))
    if sampling != "bilinear":
        s["sampling"] = SAMPLINGS[sampling]
    return _register(PgvBaseline(Entry(_manifest(NAME_SCORE, "score", stmt, event_id), site_ids, v), s))


def fit(earlier: Sequence[Earlier], scored_event_id: str, shakemap_id: str | None = None) -> dict:
    """a, b of p = expit(a + b ln PGV) on the pooled earlier rows, with the B4 guards. Raises PgvRefused.

    The scored earthquake is refused in the fit set under its task id and under its ShakeMap id.
    """
    if not earlier:
        raise PgvRefused("the PGV probability needs at least one earlier earthquake")
    ids = [e.event_id for e in earlier]
    for sid in {scored_event_id, _grid_id(scored_event_id, shakemap_id)}:
        if sid in ids:
            raise PgvRefused(f"the scored earthquake {sid!r} is in the fit set; fit on earlier ones only")
    if len(set(ids)) != len(ids):
        raise PgvRefused("each earlier earthquake may enter the fit once")
    for e in earlier:
        if not (np.all(np.isfinite(e.pgv)) and np.all(e.pgv > 0)):
            raise PgvRefused(f"PGV MISSING IN EARLIER EARTHQUAKE {e.event_id}")
    x = np.concatenate([np.log(e.pgv) for e in earlier])
    y = np.concatenate([e.y for e in earlier])
    n_yes, n_no = int(y.sum()), int(y.size - y.sum())
    if n_yes < SH.MIN_CLASS or n_no < SH.MIN_CLASS:
        raise PgvRefused(f"the earlier rows have {n_yes} yes and {n_no} no; the fit needs at least {SH.MIN_CLASS} "
                         "of each")
    if SH._separated(x, y):
        raise PgvRefused("the earlier rows are separated by PGV alone; the slope has no finite estimate")
    a, b, converged, n_iter = SH._logistic(x, y, SH.MAX_ITER)
    if not converged:
        raise PgvRefused(f"the PGV fit did not converge in {SH.MAX_ITER} Newton steps")
    if not b > 0:
        raise PgvRefused(f"the PGV fit has slope b = {b:.3g}, not above 0")
    return {"form": "p = expit(a + b ln PGV)", "a": a, "b": b, "fitted_on": ids, "n_rows": int(y.size),
            "n_yes": n_yes, "scored_event_id": scored_event_id, "converged": True, "n_iter": n_iter,
            "separated": False, "slope_positive": True, "min_class": SH.MIN_CLASS}


def probability_baseline(event_id: str, site_ids: Sequence[str], lon, lat, grid: Grid | None,
                         earlier: Sequence[Earlier], shakemap_id: str | None = None) -> PgvBaseline:
    """The probability row (fit on earlier earthquakes only), or NOT AVAILABLE with the reason (no fill).

    A missing PGV, at a scored site or in an earlier earthquake, gives NOT AVAILABLE. A fit guard that fails
    (too few yes or no, separation, no convergence, slope not above 0) raises PgvRefused: that is a refusal, not
    a missing input. `shakemap_id` is the grid's ComCat id when it differs from the task id (both print).
    """
    site_ids = tuple(site_ids)
    _check_grid(event_id, grid, shakemap_id)
    v, why = pgv_at(grid, lon, lat)
    if v.size != len(site_ids):
        raise PgvRefused("one longitude and one latitude per site id")
    if why is not None:
        return not_available(event_id, why, grid, "probability", len(site_ids), shakemap_id)
    try:
        f = fit(earlier, event_id, shakemap_id)
    except PgvRefused as exc:
        if str(exc).startswith("PGV MISSING IN EARLIER"):
            return not_available(event_id, str(exc), grid, "probability", len(site_ids), shakemap_id)
        raise
    p = SH._expit(f["a"] + f["b"] * np.log(v))
    stmt = (f"{FIT_MARKER} p = expit(a + b ln PGV), a = {f['a']:.4f}, b = {f['b']:.4f}, scored earthquake "
            f"{event_id}, PGV (cm/s) from ShakeMap "
            f"{grid.event_id} version {grid.version or '?'} (sha256 {grid.sha256[:12]}); fitted on the earlier "
            f"earthquakes {', '.join(f['fitted_on'])} only, never on this one; n_rows {f['n_rows']}, "
            f"n_yes {f['n_yes']}; {NO_FILL}")
    s = _base_summary(event_id, grid, "probability", shakemap_id)
    s.update(status=STATUS_BUILT, n_sites=len(site_ids), fit=f, n_earlier=len(earlier))
    return _register(PgvBaseline(Entry(_manifest(NAME_PROB, "target_probability", stmt, event_id), site_ids, p), s))


# ---------------------------------------------------------------------------------------------------------------
# Card hooks (evalkit.card calls these; it raises CardRefused on a message)
# ---------------------------------------------------------------------------------------------------------------

def card_problem(obj: object, event_id: str, earlier_label_sets: int) -> str | None:
    """Why the card must refuse this object, or None."""
    if not isinstance(obj, PgvBaseline):
        return "the PGV baseline must be built by evalkit.pgv (a PgvBaseline)"
    if not kit_built(obj):
        return ("the PGV baseline was not returned by evalkit.pgv, or its entry or summary changed after the build "
                "(provenance registry)")
    if obj.summary.get("scored_event") != event_id:
        return f"the PGV baseline was built for another earthquake, not {event_id!r}"
    grid = obj.summary.get("grid")
    if grid is not None and grid.get("event_id") != obj.summary.get("shakemap_id"):
        return "the PGV baseline's grid id differs from its recorded shakemap_id"
    if obj.entry is None:
        return None
    m = obj.entry.manifest
    if (m.model, m.output_kind) not in ((NAME_SCORE, "score"), (NAME_PROB, "target_probability")):
        return f"the PGV baseline must be {NAME_SCORE!r} (score) or {NAME_PROB!r} (target_probability)"
    if m.exposure != {event_id: "unseen"}:
        return "the PGV baseline reads the scored earthquake's ShakeMap only: exposure unseen"
    if m.output_kind == "target_probability":
        if earlier_label_sets == 0:
            return "R10: a PGV probability baseline was supplied but the task names no earlier earthquake"
        if obj.summary.get("n_earlier", 0) > earlier_label_sets:
            return (f"the PGV probability baseline uses {obj.summary.get('n_earlier')} earlier earthquakes; the task "
                    f"names {earlier_label_sets}")
    return None


def card_note(obj: PgvBaseline) -> str:
    if obj.entry is None:
        return f"B5 PGV baseline ({obj.summary['version']}): {obj.summary['status']}"
    return f"B5 PGV baseline ({obj.summary['version']}): built from ShakeMap PGV; {NO_FILL}"

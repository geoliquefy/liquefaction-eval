"""Sequences (rule R25, own-work list item D6, eval M1): attribution window, shock list, task kind, the
"after an earlier shock" mark, and the per-shock combine (the old two_shock logic). Pure python plus numpy; reads
only the catalogue files a caller names. No em dashes.

Rule R25 (docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md, pillar root; DRAFT, not adopted) in five steps:

  survey dates --> attribution_window --> Window (start, end]
  ComCat rows ---> shock_list(window, sites) --> ShockList: census shocks whose radius covers a task site,
                   |                             time order, plus UNRESOLVED rows (no Mw-family magnitude yet)
  (fallback) ----> sequence_shock_list --> re-anchor the 30-day window on the first shock until it is stable
  GeoNet rows ---> cross_check --> each ComCat shock's GeoNet id; GeoNet-only M >= 5 rows listed as notes
  ShockList -----> plan_tasks(attributes_per_shock) --> SINGLE / PER SHOCK (one task per shock) / SEQUENCE
  entry time ----> after_earlier_shock --> the mark "after an earlier shock"
  per-shock runs -> combine_runs --> a site's value is its largest over the runs that include it

Definitions used (rule book D6 and R25):
- Census magnitude: the largest Mw-family magnitude ComCat lists for the event (mw, mww, mwc, mwb, mwr; not mwp),
  preferred or not. A ComCat summary CSV lists only the preferred magnitude. When that is Mw-family it is a LOWER
  BOUND on the census magnitude (a detail record may list a larger one), and the shock says so in
  `magnitude_source`. When it is not Mw-family (mb, ml, ...) the summary cannot give a census magnitude: a row with
  a preferred magnitude of 5.0 or more is UNRESOLVED and `plan_tasks` refuses until the caller resolves it from the
  detail record (`census_from_detail`) through `census_magnitudes`.
- Completeness limit: a row whose preferred magnitude is below 5.0 is dropped by default, Mw-family or not, even
  though its census magnitude (the largest Mw-family value) could be 5.0 or more. The on-disk source
  (data/usgs-comcat-m5-2026-11-04) is a M >= 5.0 query, so such an event is not in it at all. The shock list is
  therefore complete only for events whose ComCat preferred magnitude is 5.0 or more. `cross_check` against GeoNet
  is a partial net. With a broader catalogue, `unresolved_from` (for example 4.5) marks rows with a preferred
  magnitude in [unresolved_from, 5.0) and no detail or caller magnitude as UNRESOLVED too.
- Radius: R = min(200, 30 x 10^(0.5 (Mw - 6))) km, hypocentral, with Mw the census magnitude.
- Window: from the end of this source's previous survey of the region (or 30 days before the origin, if none) to
  the end of this survey. The kit reads it as (start, end]: a shock exactly at the start is out, one exactly at
  the end is in. The start/end edge reading is the kit's, not v1's (v1 is silent).
- Fallback anchor (kit reading; v1 is silent): with no previous survey the window hangs on the origin, but for a
  sequence the origin is the first shock's, which can come before the origin the caller passed. That is circular.
  The kit takes the fixed point: `sequence_shock_list` re-anchors the window on the first shock and rebuilds until
  the first shock is not before the anchor. `plan_tasks` refuses a fallback window whose first shock comes before
  its anchor. The fixed point can walk back through a long run of M 5 shocks; `max_iter` bounds it.
- Task kind: one shock is SINGLE. More, and the source attributes each yes to one shock (spec S3.2), each shock is
  its own task (PER SHOCK). Otherwise SEQUENCE: the target is "manifestation seen after any shock in the window",
  the origin is the first shock's.
- Mark: "after an earlier shock" when the entry time, or the time of an observation among the entry's inputs, is
  after the origin of a window shock that precedes the task's last scored shock.

The old two_shock logic (scripts/run_g6b_p3v2.py: FORE and p_of) took, for Jun and Dec 2011, the main shock and one
named foreshock, and gave each site p = max over the two shocks. It is copied here with a parity test; the script
stays as the registered source and is unchanged. `combine_runs` is that max in general form (any number of shocks,
a site can be outside a run). `p_max_over_shocks` is a copy of p_of (tests check it against the script's own
source). The script's gm_o variant (max over shocks with a shared eta shift) has no counterpart here.
LEGACY_TWO_SHOCK keeps the old map for reference only: R25 builds the shock list from the catalogue instead, and
for Dec 2011 the catalogue gives more than two shocks in reach of the city.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

MW_FAMILY = ("mw", "mww", "mwc", "mwb", "mwr")  # D6: "mw, mww, mwc, mwb, mwr; not mwp"
CENSUS_MIN = 5.0
FALLBACK_DAYS = 30
R_CAP_KM = 200.0
EARTH_RADIUS_KM = 6371.0088
SINGLE, PER_SHOCK, SEQUENCE = "SINGLE", "PER SHOCK", "SEQUENCE"
SEQUENCE_TARGET = "manifestation seen after any shock in the window"
MARK = "after an earlier shock"
FROM_DETAIL = "detail (largest Mw-family)"
FROM_SUMMARY = "summary preferred Mw-family (lower bound of the D6 read)"
FROM_CALLER = "caller (census_magnitudes)"

# The old map from scripts/run_g6b_p3v2.py (FORE), kept for reference only. Magnitudes are the GeoNet values the
# old scripts used (ComCat's preferred for the Jun foreshock is mwr 5.2).
LEGACY_TWO_SHOCK = {"jun2011": ("jun2011_fore", 5.3), "dec2011": ("dec2011_fore", 5.8)}


class SequenceError(ValueError):
    """R25 cannot be applied as asked (bad window, unresolved magnitudes, no shock)."""


def _utc(t) -> datetime:
    if isinstance(t, datetime):
        return t.astimezone(timezone.utc) if t.tzinfo else t.replace(tzinfo=timezone.utc)
    s = str(t).strip().replace("Z", "+00:00")
    d = datetime.fromisoformat(s)
    return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Event:
    """One catalogue row."""
    catalog: str          # "comcat" or "geonet"
    event_id: str
    time: datetime        # UTC
    lat: float
    lon: float
    depth_km: float
    mag: float
    mag_type: str
    event_type: str = "earthquake"

    @property
    def mw_family(self) -> bool:
        return self.mag_type.strip().lower() in MW_FAMILY


# ------------------------------------------------------------------------------------------------ readers

def read_comcat_csv(paths: Iterable[str | Path]) -> list[Event]:
    """ComCat FDSN CSV (columns time, latitude, longitude, depth, mag, magType, ..., id, ..., type, ...)."""
    out, seen = [], set()
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("mag"):
                    continue
                if r["id"] in seen:
                    raise SequenceError(f"duplicate ComCat id {r['id']}")
                seen.add(r["id"])
                out.append(Event("comcat", r["id"], _utc(r["time"]), float(r["latitude"]), float(r["longitude"]),
                                 float(r["depth"] or 0.0), float(r["mag"]), r["magType"] or "",
                                 r.get("type") or "earthquake"))
    return sorted(out, key=lambda e: (e.time, e.event_id))


def read_geonet_text(paths: Iterable[str | Path]) -> list[Event]:
    """GeoNet FDSN text (pipe separated; header line starts with '#')."""
    out, seen = [], set()
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            c = [x.strip() for x in line.split("|")]
            if len(c) < 14 or not c[10]:
                continue
            if c[0] in seen:
                raise SequenceError(f"duplicate GeoNet id {c[0]}")
            seen.add(c[0])
            out.append(Event("geonet", c[0], _utc(c[1]), float(c[2]), float(c[3]), float(c[4] or 0.0),
                             float(c[10]), c[9], c[13] or "earthquake"))
    return sorted(out, key=lambda e: (e.time, e.event_id))


def census_from_detail(detail: Mapping) -> float | None:
    """D6 read from a ComCat detail GeoJSON: the largest Mw-family magnitude over every product, preferred or not."""
    props = detail.get("properties", detail)
    vals = []
    if str(props.get("magType", "")).lower() in MW_FAMILY and props.get("mag") is not None:
        vals.append(float(props["mag"]))
    for prods in (props.get("products") or {}).values():
        for pr in prods:
            q = pr.get("properties", {})
            for tk, mk in (("magnitude-type", "magnitude"), ("derived-magnitude-type", "derived-magnitude")):
                if str(q.get(tk, "")).lower() in MW_FAMILY and q.get(mk) not in (None, ""):
                    vals.append(float(q[mk]))
    return max(vals) if vals else None


# ------------------------------------------------------------------------------------------------ geometry

def radius_km(mw: float) -> float:
    """D6: R = min(200, 30 x 10^(0.5 (Mw - 6))) km, hypocentral."""
    return float(min(R_CAP_KM, 30.0 * 10.0 ** (0.5 * (mw - 6.0))))


def hypocentral_km(ev: Event, lat, lon) -> np.ndarray:
    """Great-circle epicentral distance (haversine) combined with depth."""
    la1, lo1 = math.radians(ev.lat), math.radians(ev.lon)
    la2, lo2 = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    a = np.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    epi = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.sqrt(epi ** 2 + ev.depth_km ** 2)


# ------------------------------------------------------------------------------------------------ window

@dataclass(frozen=True)
class Window:
    start: datetime   # exclusive
    end: datetime     # inclusive
    rule: str
    anchor: datetime | None = None   # the origin the fallback start hangs on (None: previous survey end)

    def contains(self, t: datetime) -> bool:
        return self.start < _utc(t) <= self.end


def attribution_window(survey_end, origin, previous_survey_end=None) -> Window:
    """R25 window: (previous survey end, or origin minus 30 days) to this survey's end."""
    end, org = _utc(survey_end), _utc(origin)
    if end < org:
        raise SequenceError("the survey ends before the origin")
    if previous_survey_end is None:
        start, rule = org - timedelta(days=FALLBACK_DAYS), f"origin minus {FALLBACK_DAYS} days (no previous survey)"
        return Window(start, end, rule, org)
    start = _utc(previous_survey_end)
    if start >= org:
        raise SequenceError("the previous survey ends at or after the origin")
    return Window(start, end, "end of the previous survey of the region")


# ------------------------------------------------------------------------------------------------ shock list

@dataclass(frozen=True)
class Shock:
    event: Event
    census_mw: float
    magnitude_source: str
    radius_km: float
    n_sites_covered: int
    covered: tuple = field(repr=False, compare=False, default=())   # bool per task site (in memory only)


@dataclass(frozen=True)
class ShockList:
    window: Window
    shocks: tuple            # Shock, sorted by (time, event id)
    unresolved: tuple        # Event: no census magnitude yet (see shock_list), reach unknown
    n_sites: int

    @property
    def complete(self) -> bool:
        return not self.unresolved


def shock_list(comcat: Sequence[Event], window: Window, site_lat, site_lon,
               census_magnitudes: Mapping[str, float | None] | None = None,
               detail_magnitudes: Mapping[str, float | None] | None = None,
               unresolved_from: float | None = None) -> ShockList:
    """Every ComCat earthquake in the window with census magnitude >= 5.0 whose radius covers a task site.

    The shocks come out sorted by (time, event id), whatever the order of `comcat`.
    detail_magnitudes: event id -> census_from_detail(...) for the events whose detail record is on disk.
    census_magnitudes: event id -> a census magnitude the caller sets (None: the caller states it has no Mw-family
    magnitude, so the event is not a census shock). Either one resolves an UNRESOLVED row.
    UNRESOLVED: a non-Mw preferred magnitude of 5.0 or more. With unresolved_from set, also any summary-only row
    (Mw-family or not) with a preferred magnitude in [unresolved_from, 5.0).
    Limit: complete only for events whose ComCat preferred magnitude is 5.0 or more (module docstring).
    """
    lat, lon = np.asarray(site_lat, float), np.asarray(site_lon, float)
    if lat.shape != lon.shape or lat.ndim != 1 or lat.size == 0:
        raise SequenceError("site latitudes and longitudes must be two equal, non-empty 1-d arrays")
    census_magnitudes, detail_magnitudes = dict(census_magnitudes or {}), dict(detail_magnitudes or {})
    shocks, unresolved = [], []
    for ev in comcat:
        if ev.catalog != "comcat":
            raise SequenceError("the shock list is built from ComCat rows only (R25)")
        if ev.event_type.lower() != "earthquake" or not window.contains(ev.time):
            continue
        if ev.event_id in census_magnitudes:
            mw, src = census_magnitudes[ev.event_id], FROM_CALLER
        elif detail_magnitudes.get(ev.event_id) is not None:
            mw, src = float(detail_magnitudes[ev.event_id]), FROM_DETAIL
        elif unresolved_from is not None and unresolved_from <= ev.mag < CENSUS_MIN:
            unresolved.append(ev)
            continue
        elif ev.mw_family:
            mw, src = ev.mag, FROM_SUMMARY
        else:
            if ev.mag >= CENSUS_MIN:
                unresolved.append(ev)
            continue
        if mw is None or mw < CENSUS_MIN:
            continue
        r = radius_km(mw)
        cov = hypocentral_km(ev, lat, lon) <= r
        if cov.any():
            shocks.append(Shock(ev, float(mw), src, r, int(cov.sum()), tuple(bool(x) for x in cov)))
    order = lambda e: (e.time, e.event_id)  # noqa: E731
    return ShockList(window, tuple(sorted(shocks, key=lambda s: order(s.event))),
                     tuple(sorted(unresolved, key=order)), int(lat.size))


def sequence_shock_list(comcat: Sequence[Event], survey_end, origin, site_lat, site_lon,
                        previous_survey_end=None, max_iter: int = 50, **kw) -> ShockList:
    """R25 window plus shock list, with the fallback anchor taken to its fixed point (kit reading, module docstring).

    With a previous survey end the window does not depend on the origin and one pass is final. With the 30-day
    fallback the window is re-anchored on the first shock until the first shock is not before the anchor. Each pass
    moves the anchor strictly earlier, so it stops; more than max_iter passes is refused.
    """
    org = _utc(origin)
    for _ in range(max_iter):
        sl = shock_list(comcat, attribution_window(survey_end, org, previous_survey_end), site_lat, site_lon, **kw)
        if previous_survey_end is not None or not sl.shocks or sl.shocks[0].event.time >= org:
            return sl
        org = sl.shocks[0].event.time
    raise SequenceError(f"the fallback window did not settle in {max_iter} passes (a long run of M 5 shocks)")


# ------------------------------------------------------------------------------------------------ GeoNet check

@dataclass(frozen=True)
class CrossCheck:
    matched: tuple       # (comcat id, geonet id, geonet mag, geonet mag type, dt seconds, distance km)
    comcat_only: tuple   # comcat ids with no GeoNet row within the tolerances
    geonet_only: tuple   # GeoNet events M >= 5.0 in the window, within reach of a site, matched to no ComCat row


def cross_check(sl: ShockList, comcat: Sequence[Event], geonet: Sequence[Event], site_lat, site_lon,
                max_dt_s: float = 30.0, max_km: float = 50.0) -> CrossCheck:
    """Match each census shock to GeoNet; list GeoNet M >= 5.0 rows in reach that ComCat does not match.

    GeoNet is a second catalogue, not the rule's source: a GeoNet-only row is a note for the task file (a shock the
    ComCat list may be missing), never a shock by itself.
    """
    def nearest(ev, pool):
        best = None
        for g in pool:
            dt = abs((g.time - ev.time).total_seconds())
            if dt > max_dt_s:
                continue
            d = float(hypocentral_km(Event("x", "", ev.time, g.lat, g.lon, 0.0, 0.0, ""), [ev.lat], [ev.lon])[0])
            if d <= max_km and (best is None or (dt, d) < best[0]):
                best = ((dt, d), g)
        return best

    matched, comcat_only = [], []
    for s in sl.shocks:
        b = nearest(s.event, geonet)
        if b is None:
            comcat_only.append(s.event.event_id)
        else:
            (dt, d), g = b
            matched.append((s.event.event_id, g.event_id, g.mag, g.mag_type, round(dt, 3), round(d, 3)))
    lat, lon = np.asarray(site_lat, float), np.asarray(site_lon, float)
    geonet_only = []
    for g in geonet:
        if g.event_type.lower() != "earthquake" or g.mag < CENSUS_MIN or not sl.window.contains(g.time):
            continue
        if not (hypocentral_km(g, lat, lon) <= radius_km(g.mag)).any():
            continue
        if nearest(g, [e for e in comcat if e.catalog == "comcat"]) is None:
            geonet_only.append(g.event_id)
    return CrossCheck(tuple(matched), tuple(comcat_only), tuple(geonet_only))


# ------------------------------------------------------------------------------------------------ task plan

@dataclass(frozen=True)
class TaskPlan:
    kind: str                 # SINGLE, PER SHOCK or SEQUENCE
    origin: datetime          # the shock's origin; for SEQUENCE the first shock's
    shock_ids: tuple          # the shocks this task scores (time order)
    window_shock_ids: tuple   # every census shock in the window
    window_shock_times: tuple
    target: str

    def summary(self) -> dict:
        """Aggregates only (no site, no coordinate): what a card or task file prints."""
        return {"kind": self.kind, "origin_utc": self.origin.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "shocks": list(self.shock_ids), "window_shocks": list(self.window_shock_ids),
                "target": self.target, "card_word": SEQUENCE if self.kind == SEQUENCE else None}


def plan_tasks(sl: ShockList, attributes_per_shock: bool = False, task_target: str = "task target") -> list[TaskPlan]:
    """R25 task kind. Refuses an incomplete list (UNRESOLVED rows), a window with no census shock, and a fallback
    window whose first shock comes before its anchor (rebuild it with sequence_shock_list)."""
    if not sl.complete:
        raise SequenceError("UNRESOLVED: ComCat rows with preferred magnitude >= 5.0 and no Mw-family magnitude: "
                            + ", ".join(e.event_id for e in sl.unresolved)
                            + "; read the detail records (census_from_detail) and pass census_magnitudes")
    if not sl.shocks:
        raise SequenceError("no census shock (Mw >= 5.0 within reach of a task site) in the window")
    shocks = sorted(sl.shocks, key=lambda s: (s.event.time, s.event.event_id))
    ids = tuple(s.event.event_id for s in shocks)
    times = tuple(s.event.time for s in shocks)
    if sl.window.anchor is not None and times[0] < sl.window.anchor:
        raise SequenceError("the first shock comes before the fallback window's anchor (the origin is the first "
                            "shock's, R25): rebuild the window with sequence_shock_list")
    if len(sl.shocks) == 1:
        return [TaskPlan(SINGLE, times[0], ids, ids, times, task_target)]
    if attributes_per_shock:
        return [TaskPlan(PER_SHOCK, t, (i,), ids, times, task_target) for i, t in zip(ids, times)]
    return [TaskPlan(SEQUENCE, times[0], ids, ids, times, SEQUENCE_TARGET)]


def after_earlier_shock(plan: TaskPlan, entry_time, input_observation_times: Iterable = ()) -> bool:
    """R25 mark: the entry, or an observation among its inputs, comes after a window shock that precedes the task's
    last scored shock. A single-shock task, or the first shock of a PER SHOCK set, has no earlier shock."""
    last = max(plan.window_shock_times[plan.window_shock_ids.index(i)] for i in plan.shock_ids)
    earlier = [t for t in plan.window_shock_times if t < last]
    if not earlier:
        return False
    first = min(earlier)
    times = [_utc(entry_time)] + [_utc(t) for t in input_observation_times]
    return any(t > first for t in times)


# ------------------------------------------------------------------------------------------------ combine

def combine_runs(runs: Mapping[str, np.ndarray], include: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    """A sequence task's value per site: the largest over the runs that include the site; NaN if none does.

    runs: shock id -> values per task site (NaN where the run has no value). include: shock id -> bool per site
    (default: every finite value counts). This is the two_shock "p = max" of scripts/run_g6b_p3v2.py in general form.
    """
    if not runs:
        raise SequenceError("no run to combine")
    keys = list(runs)
    V = np.vstack([np.asarray(runs[k], float) for k in keys])
    if include is not None:
        if set(include) - set(keys):
            raise SequenceError("include names a shock with no run")
        M = np.vstack([np.asarray(include.get(k, np.isfinite(V[i])), bool) for i, k in enumerate(keys)])
        if M.shape != V.shape:
            raise SequenceError("include masks must match the runs")
        V = np.where(M, V, np.nan)
    ok = np.isfinite(V)
    out = np.where(ok, V, -np.inf).max(axis=0)
    out[~ok.any(axis=0)] = np.nan
    return out


def p_max_over_shocks(lnks: Sequence[np.ndarray], se, mu: float, sig_r: float) -> np.ndarray:
    """A copy of scripts/run_g6b_p3v2.py p_of (parity tested; the script stays the registered source):
    p = max over shocks of Phi((ln k - mu) / sqrt(sig_r^2 + se^2))."""
    from scipy.special import ndtr
    s = np.sqrt(sig_r ** 2 + np.asarray(se, float) ** 2)
    return np.max([ndtr((np.asarray(lk, float) - mu) / s) for lk in lnks], axis=0)

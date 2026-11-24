"""Standard baseline "baseline: site memory": did this site liquefy last time? (own-work list B3, eval M4).

No em dashes.

For a later earthquake, each site of the scored task gets the label of the same site, or of the nearest labelled site
within 20 m, in the most recent earlier earthquake. The 20 m is NOT settled for this use: see "Radius" below.
The scored earthquake's labels never enter: the builder takes the scored site list as ids and coordinates only
(`Sites`), and the loader drops the scored labels on read.

The rule, step by step (all fixed, no search):

1. Memory earthquake: the latest earlier earthquake (by origin date, strictly before the scored one).
2. Match: the same site id if it is labelled there; else the nearest labelled site within RADIUS_M (distance <=
   20 m, ties broken by the smaller site id); else no memory.
3. Probability version (needs two earlier earthquakes): apply steps 1 and 2 one step back, with the memory
   earthquake as the "scored" one and the earthquake before it as its memory. Its labelled sites give three observed
   rates: yes share among sites whose memory was yes, whose memory was no, and that had no memory. Each rate is the
   Jeffreys estimate (k + 0.5) / (n + 1), so no rate is exactly 0 or 1. A cell with fewer than MIN_CELL sites takes
   the Jeffreys estimate over all labelled sites of the memory earthquake instead (stated on the card). The scored
   sites get the rate of their cell. Output kind target_probability.
4. Ranking-only version (one earlier earthquake): memory yes scores 1, memory no scores 0, no memory scores the
   memory earthquake's yes share (between the two). Output kind score: the card gives it AUC only (R3).
5. Label source: every earlier earthquake carries its label source. By default (memory level "label source") build
   refuses a mix and records the source in the summary, and the card refuses a source other than the task's.
   Memory level "label set" (scorecard v3.1, B8 review 2 finding R2-3; opt in with `label_set=` and
   `scored_label_source=`): each earlier earthquake keeps its REAL source string, every string (and the task's)
   must belong to the named label set, a mix of sources is allowed, and the summary records every earlier source,
   the scored source and whether all are the same maker. The card then checks the scored source against the task.
   When the sources differ, the label persistence note is replaced by CROSS_SOURCE_NOTE.
6. Sites with no memory are counted on the card (`n_no_memory`) and handled by the rule in step 3 or 4. They are
   never dropped: every task site gets a value, so the baseline is scored on the same sites as every entry.

Limits (printed on the card):
- The probability version carries one earthquake pair's conditional rates to the next pair. Shaking differs between
  earthquakes, so its level can be off; the ranking part (yes above no) is the robust part.
- Exposure is SEEN PLACE: the baseline reads other earthquakes' labels at the same places. That is the point of the
  baseline, not a leak; the scored earthquake's labels are never read.
- Label persistence (correction B3, founder check 2026-11-23). The card requires the memory earthquakes to share
  the task's label source, so the baseline always compares a label with a label from the same maker (the same map
  method, the same streets walked). On the same label source it therefore also measures label persistence: a
  label error at a place can repeat at that place in the next earthquake. The error is large. Aerial specificity
  is 0.47 [0.30, 0.63] in Feb 2011 (pillar docs/evals/LABEL-ERROR-RATES-2026-11-19.md), so about half of the
  aerial sites that did not liquefy carry a yes label. A high site-memory score is not, on its own, evidence of
  ground behaviour. The card prints this (PERSISTENCE_NOTE) in the row statement, the limits and the notes.
- Radius (correction B3). Pillar ruling 16 set the same-site distance of 20 m for the twin test and the display.
  It did not set a radius for this baseline. Using 20 m here is a new choice, made without a registration; it is
  not settled, and the v3 registration (item B8) must fix the radius before any real run. A label source with
  coarser locations (a mapped polygon, a street segment) may need another radius. The radius is an argument and
  prints on the card with this status (RADIUS_NOTE).

Provenance: `build` registers every object it returns (a weak registry keyed by the object, holding the entry,
its hash and the summary). The card slot takes only a registered object whose entry and summary are unchanged.
A hand-made SiteMemory with the right name and allowlist is refused.

Pure numpy and scipy.
"""
from __future__ import annotations

import csv
import json
import datetime as dt
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .spec import Entry, EntryManifest, Labels, SpecError, _check_ids, entry_sha256

NAME = "baseline: site memory"
ALLOWLIST_NAME = "earlier-labels"     # recorded in the manifest; the card's site-memory slot requires it
EXPOSURE = "seen_place"
RADIUS_M = 20.0                       # NEW CHOICE, not settled: ruling 16 set 20 m for the twin test and the
                                      # display only; the v3 registration (B8) fixes the radius for this use
MIN_CELL = 20
CELLS = ("memory_yes", "memory_no", "no_memory")
HISTORY_HEADER = ["event_id", "site_id", "label", "x_m", "y_m"]
SAME_ID, NEAREST, NONE = 0, 1, 2
PERSISTENCE_NOTE = ("label persistence: on the same label source, site memory also measures label persistence "
                    "(a label error at a place can repeat there in the next earthquake), not only ground behaviour; "
                    "aerial specificity is 0.47 [0.30, 0.63] in Feb 2011 (docs/evals/LABEL-ERROR-RATES-2026-11-19.md)")
RADIUS_NOTE = ("radius status: ruling 16 set 20 m for the twin test and the display only; its use here is a new "
               "choice, not settled, to be fixed in the v3 registration (B8); coarser label locations may need "
               "another radius")
CROSS_SOURCE_NOTE = ("label sources differ: the memory labels come from another label maker than the scored labels "
                     "(memory level 'label set'), so this is not a same-maker comparison and does not measure label "
                     "persistence on one source; it reads across label makers of the same label set")
MEMORY_LEVEL_SOURCE = "label source"
MEMORY_LEVEL_SET = "label set"
LIMITS = ("the probability version carries one earthquake pair's conditional rates to the next pair; shaking "
          "differs, so its level can be off; the ranking part is the robust part",
          "exposure SEEN PLACE: other earthquakes' labels at the same places; the scored labels are never read",
          RADIUS_NOTE,
          PERSISTENCE_NOTE)


class SiteMemoryRefused(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class Sites:
    """A task's site list without labels: ids and projected coordinates in metres."""
    site_ids: tuple[str, ...]
    x_m: np.ndarray = field(repr=False)
    y_m: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_ids", tuple(self.site_ids))
        for k in ("x_m", "y_m"):
            object.__setattr__(self, k, np.asarray(getattr(self, k), dtype=float))
        n = len(self.site_ids)
        if n == 0 or self.x_m.shape != (n,) or self.y_m.shape != (n,):
            raise SpecError("sites: x_m and y_m need one value per site id")
        _check_ids(self.site_ids, "sites")
        if not (np.all(np.isfinite(self.x_m)) and np.all(np.isfinite(self.y_m))):
            raise SpecError("sites: coordinates must be finite")


def sites_of(labels: Labels) -> Sites:
    """The site list of a label set with the labels removed."""
    return Sites(labels.site_ids, labels.x_m, labels.y_m)


@dataclass(frozen=True, eq=False)
class Earlier:
    """One earlier earthquake: id, origin date (ISO yyyy-mm-dd), labels and their label source."""
    event_id: str
    date: str
    labels: Labels
    label_source: str


def _date(s: str, what: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except (TypeError, ValueError):
        raise SiteMemoryRefused(f"{what}: date must be ISO yyyy-mm-dd, got {s!r}") from None


def recall(sites: Sites, earlier: Labels, radius_m: float = RADIUS_M) -> tuple[np.ndarray, np.ndarray]:
    """Step 2: each site's label in `earlier` (1, 0 or nan) and how it was found (SAME_ID, NEAREST or NONE)."""
    if not radius_m > 0:
        raise SiteMemoryRefused("radius_m must be positive")
    lab = earlier.labelled
    e_ids = np.array([s for s, k in zip(earlier.site_ids, lab) if k], dtype=object)
    e_lab = earlier.label[lab]
    n = len(sites.site_ids)
    mem, how = np.full(n, np.nan), np.full(n, NONE, dtype=int)
    if e_ids.size == 0:
        return mem, how
    pos = {s: k for k, s in enumerate(e_ids)}
    tree = cKDTree(np.column_stack([earlier.x_m[lab], earlier.y_m[lab]]))
    pts = np.column_stack([sites.x_m, sites.y_m])
    for i, s in enumerate(sites.site_ids):
        if s in pos:
            mem[i], how[i] = e_lab[pos[s]], SAME_ID
            continue
        near = tree.query_ball_point(pts[i], radius_m)
        if not near:
            continue
        d = np.hypot(*(tree.data[near] - pts[i]).T)
        best = min(range(len(near)), key=lambda k: (d[k], e_ids[near[k]]))
        mem[i], how[i] = e_lab[near[best]], NEAREST
    return mem, how


def _cell_of(mem: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(mem), 2, np.where(mem == 1, 0, 1))


def _jeffreys(k: int, n: int) -> float:
    return (k + 0.5) / (n + 1.0)


def _rates(mem_back: np.ndarray, y_back: np.ndarray, min_cell: int) -> dict:
    cell = _cell_of(mem_back)
    n_all, k_all = int(y_back.size), int(y_back.sum())
    out = {}
    for c, name in enumerate(CELLS):
        m = cell == c
        n, k = int(m.sum()), int(y_back[m].sum())
        if n >= min_cell:
            out[name] = {"n": n, "k_yes": k, "p": _jeffreys(k, n), "source": "Jeffreys (k + 0.5) / (n + 1)"}
        else:
            out[name] = {"n": n, "k_yes": k, "p": _jeffreys(k_all, n_all),
                         "source": f"fewer than {min_cell} sites: Jeffreys (K + 0.5) / (N + 1) over all "
                                   f"{n_all} labelled sites of the memory earthquake"}
    return out


@dataclass(frozen=True, eq=False)
class SiteMemory:
    """The baseline as a kit Entry plus the aggregate summary the card prints.

    Anyone can construct one, but only an object returned by `build` passes `kit_built` (and so the card slot).
    """
    entry: Entry
    summary: dict


_BUILT: "weakref.WeakKeyDictionary[SiteMemory, tuple[Entry, str, str]]" = weakref.WeakKeyDictionary()


def resolve_sources(earlier: Sequence["Earlier"], label_set: str | None, scored_label_source: str | None,
                    refused: type = ValueError) -> dict:
    """The label-source fields of a site-memory or geographic-null summary.

    label_set None (default, memory level "label source"): one non-empty source for every earlier earthquake;
    returns {"label_source": that source}, exactly the fields of the sealed v3 card.
    label_set given (memory level "label set"): every earlier source and the scored source must be non-empty and
    start with "<label_set>: "; a mix is allowed. Returns the real strings and whether they are all the same maker.
    """
    sources = [e.label_source for e in earlier]
    if label_set is None:
        if scored_label_source is not None:
            raise refused("scored_label_source is only taken with label_set (memory level 'label set')")
        uniq = set(sources)
        src = next(iter(uniq))
        if len(uniq) != 1 or not isinstance(src, str) or not src.strip():
            raise refused(f"earlier earthquakes need one non-empty label source, got {sorted(map(repr, uniq))}")
        return {"label_source": src}
    if not isinstance(label_set, str) or not label_set.strip():
        raise refused("label_set must be a non-empty string")
    pre = f"{label_set}: "
    for s_ in [*sources, scored_label_source]:
        if not isinstance(s_, str) or not s_.startswith(pre) or not s_[len(pre):].strip():
            raise refused(f"memory level 'label set': every label source must read '{pre}<source>', got {s_!r}")
    order = []
    for s_ in sources:
        if s_ not in order:
            order.append(s_)
    same = all(s_ == scored_label_source for s_ in sources)
    return {"memory_level": MEMORY_LEVEL_SET, "label_set": label_set, "scored_label_source": scored_label_source,
            "earlier_label_sources": {e.event_id: e.label_source for e in earlier},
            "label_source": " + ".join(order), "same_label_source_as_scored": bool(same)}


def source_problem(summary: Mapping, task_label_source: str, what: str) -> str | None:
    """The card's label-source check for a site-memory or geographic-null summary (both memory levels)."""
    if summary.get("memory_level") == MEMORY_LEVEL_SET:
        if summary.get("scored_label_source") != task_label_source:
            return (f"the {what} was built for scored label source {summary.get('scored_label_source')!r}, the task "
                    f"is scored on {task_label_source!r}")
        return None
    if summary.get("label_source") != task_label_source:
        return (f"the {what} reads label source {summary.get('label_source')!r}, the task is scored on "
                f"{task_label_source!r}")
    return None


def kit_built(sm: object) -> bool:
    """True only for an object `build` returned, with the same Entry object, entry hash and summary."""
    if not isinstance(sm, SiteMemory):
        return False
    rec = _BUILT.get(sm)
    return (rec is not None and rec[0] is sm.entry and rec[1] == entry_sha256(sm.entry)
            and rec[2] == _summary_key(sm.summary))


def _summary_key(summary: dict) -> str:
    return json.dumps(summary, sort_keys=True, default=str)


def build(scored_event: str, scored_date: str, sites: Sites, earlier: Sequence[Earlier],
          radius_m: float = RADIUS_M, min_cell: int = MIN_CELL, label_set: str | None = None,
          scored_label_source: str | None = None) -> SiteMemory:
    """The site-memory baseline for every site of the scored task. Takes no scored labels (see `Sites`).

    label_set and scored_label_source (both or neither) select memory level "label set" (see step 5)."""
    if isinstance(sites, Labels) or not isinstance(sites, Sites):
        raise SiteMemoryRefused("pass the scored site list as Sites (sites_of(labels)); the scored labels never enter")
    t = _date(scored_date, scored_event)
    if not earlier:
        raise SiteMemoryRefused("site memory needs at least one earlier earthquake")
    src_fields = resolve_sources(earlier, label_set, scored_label_source, SiteMemoryRefused)
    label_source = src_fields["label_source"]
    ids = [e.event_id for e in earlier]
    if len(set(ids)) != len(ids) or scored_event in ids:
        raise SiteMemoryRefused("earlier earthquakes need distinct ids, none equal to the scored earthquake")
    dates = [_date(e.date, e.event_id) for e in earlier]
    if any(d >= t for d in dates):
        raise SiteMemoryRefused("every earlier earthquake must come strictly before the scored one")
    if len(set(dates)) != len(dates):
        raise SiteMemoryRefused("two earlier earthquakes share a date; their order is not defined")
    order = sorted(range(len(earlier)), key=lambda k: dates[k])
    prev = earlier[order[-1]]
    if not prev.labels.labelled.any():
        raise SiteMemoryRefused(f"{prev.event_id}: the memory earthquake has no labelled site")
    overall = float(prev.labels.label[prev.labels.labelled].mean())
    mem, how = recall(sites, prev.labels, radius_m)
    cell = _cell_of(mem)
    counts = {"n_sites": int(mem.size), "n_same_id": int((how == SAME_ID).sum()),
              "n_nearest_within_radius": int((how == NEAREST).sum()), "n_no_memory": int((how == NONE).sum()),
              "n_memory_yes": int((cell == 0).sum()), "n_memory_no": int((cell == 1).sum())}
    summary = {"rule": f"label of the same site, or the nearest labelled site within {radius_m:g} m, in the most "
                       f"recent earlier earthquake", "radius_m": float(radius_m), "scored_event": scored_event,
               "memory_event": prev.event_id, "memory_yes_share": overall, "label_source": label_source,
               "n_earlier": len(earlier), "counts": counts, "limits": list(LIMITS),
               "label_persistence": PERSISTENCE_NOTE, "radius_status": RADIUS_NOTE}
    persistence = PERSISTENCE_NOTE
    if label_set is not None:
        summary.update(src_fields)
        summary["memory_label_source"] = prev.label_source
        if not src_fields["same_label_source_as_scored"]:
            persistence = CROSS_SOURCE_NOTE
            summary["label_persistence"] = CROSS_SOURCE_NOTE
            summary["limits"] = [x for x in LIMITS if x != PERSISTENCE_NOTE] + [CROSS_SOURCE_NOTE]
    if len(earlier) >= 2:
        back = earlier[order[-2]]
        lab = prev.labels.labelled
        back_sites = Sites(tuple(s for s, k in zip(prev.labels.site_ids, lab) if k),
                           prev.labels.x_m[lab], prev.labels.y_m[lab])
        mem_back, _ = recall(back_sites, back.labels, radius_m)
        rates = _rates(mem_back, prev.labels.label[lab], min_cell)
        p = np.array([rates[CELLS[c]]["p"] for c in cell])
        kind = "target_probability"
        summary.update(version="probability", calibration_event=back.event_id, rates=rates, min_cell=int(min_cell),
                       no_memory_rule="the observed yes share of the memory earthquake's own sites that had no "
                                      "memory in the earthquake before it")
    else:
        p = np.where(cell == 0, 1.0, np.where(cell == 1, 0.0, overall))
        kind = "score"
        summary.update(version="ranking only", calibration_event=None,
                       no_memory_rule="score = the memory earthquake's yes share, between memory no (0) and yes (1)")
    statement = (f"{summary['rule']} ({prev.event_id}); {summary['version']}; no memory: {summary['no_memory_rule']}"
                 f"; {persistence}; radius {radius_m:g} m, a new choice to be fixed in the v3 registration")
    m = EntryManifest(model=NAME, output_kind=kind, statement=statement, exposure={scored_event: EXPOSURE},
                      house=False, feature_allowlist=ALLOWLIST_NAME)
    out = SiteMemory(Entry(m, sites.site_ids, p), summary)
    _BUILT[out] = (out.entry, entry_sha256(out.entry), _summary_key(out.summary))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Loader (wave 3 feeds the Canterbury history through this; tested on a synthetic fixture only)
# ---------------------------------------------------------------------------------------------------------------

def load_history(csv_path: str | Path, dates: Mapping[str, str], scored_event: str,
                 label_source: str) -> tuple[Sites, list[Earlier]]:
    """Read a long table `event_id,site_id,label,x_m,y_m` (label 1, 0 or blank) for one label source.

    Returns the scored earthquake's site list (labels dropped on read: the label cell of a scored row is never
    parsed) and every earthquake dated strictly before it as `Earlier`, tagged with `label_source` (the file holds
    one label source). Rows of later earthquakes are skipped after the id checks. Every event id in the file needs
    a date in `dates`. Duplicate (event, site) rows stop the run (R7), in any earthquake: the check reads ids only.
    """
    if scored_event not in dates:
        raise SiteMemoryRefused(f"no date for the scored earthquake {scored_event!r}")
    t = _date(dates[scored_event], scored_event)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != HISTORY_HEADER:
        raise SpecError(f"{csv_path}: header must be exactly {','.join(HISTORY_HEADER)}")
    by_event: dict[str, list[list[str]]] = {}
    seen: set[tuple[str, str]] = set()
    for r in rows[1:]:
        if len(r) != len(HISTORY_HEADER):
            raise SpecError(f"{csv_path}: every row needs {len(HISTORY_HEADER)} cells")
        if r[0] not in dates:
            raise SiteMemoryRefused(f"{csv_path}: no date for earthquake {r[0]!r}")
        if (r[0], r[1]) in seen:
            raise SpecError(f"{csv_path}: duplicate (event, site) row in earthquake {r[0]!r} (R7)")
        seen.add((r[0], r[1]))
        by_event.setdefault(r[0], []).append(r)
    if scored_event not in by_event:
        raise SiteMemoryRefused(f"{csv_path}: no rows for the scored earthquake {scored_event!r}")

    def xy(rs):
        try:
            return np.array([float(r[3]) for r in rs]), np.array([float(r[4]) for r in rs])
        except ValueError as exc:
            raise SpecError(f"{csv_path}: coordinates must be numeric ({exc})") from None

    scored_rows = by_event.pop(scored_event)          # the scored labels go no further than this line
    sites = Sites(tuple(r[1] for r in scored_rows), *xy(scored_rows))
    earlier = []
    for ev, rs in sorted(by_event.items(), key=lambda kv: _date(dates[kv[0]], kv[0])):
        if _date(dates[ev], ev) >= t:
            continue
        bad = [r[2] for r in rs if r[2] not in ("1", "0", "")]
        if bad:
            raise SpecError(f"{csv_path}: label must be 1, 0 or blank, got {bad[0]!r}")
        lab = np.array([float(r[2]) if r[2] else np.nan for r in rs])
        earlier.append(Earlier(ev, dates[ev], Labels(tuple(r[1] for r in rs), lab, *xy(rs)), label_source))
    return sites, earlier

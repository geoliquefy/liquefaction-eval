"""Geographic null baseline (own-work list B5, eval M4). Optional card row, off by default. No em dashes.

The design's own rule, quoted from pillar docs/evals/EVAL-SYSTEM-DESIGN-2026-11-19.md, R2.2 item 2:

    "Geographic null: distance to known liquefaction only; report calibrated AUC = AUC - null AUC + 0.5 (Hijmans
    2012) [A]. Test "no" points matched in distance to training "yes" points."

The rule book draft (RULEBOOK-V1-DRAFT-2026-11-15.md) does not name this baseline; R10 lists the coin, the
earlier-earthquake rate and the shaking-only baseline. This row sits beside them as an optional baseline.

What the kit builds, step by step (all fixed, no search):

1. Known liquefaction: every labelled yes site of the EARLIER earthquakes of the task's label source (the R10
   meaning of "earlier earthquake"), pooled. "Earlier" is checked: `build` takes the scored date and refuses any
   earlier earthquake dated on or after it (as site_memory.build); the dates print in the summary. The scored
   earthquake's labels never enter: the builder takes the scored site list as ids and coordinates only
   (site_memory.Sites).
2. Score at each scored site: minus the distance (m) to the nearest known-liquefaction site. Output kind "score",
   AUC only (R3). It ranks as Hijmans' geographic-distance model (inverse distance to the nearest training
   presence), with no division by zero at distance 0.
3. Calibrated AUC, on the card, for each entry: cAUC = AUC - null AUC + 0.5. Its 95% interval is 0.5 plus the
   card's paired AUC difference interval (entry minus null, tile bootstrap). Baselines get no calibrated AUC
   (the card tests no pair of two baselines, so there is no paired interval).
4. Matched-distance test set (`matched_sites`): pairwise distance sampling after Hijmans 2012. For each scored yes
   site, in site-id order, take the unused scored no site whose distance to the nearest known-liquefaction site is
   closest, if the gap is at most TR times the yes site's distance. Unpaired yes sites drop. It returns the site ids
   of the matched set. It reads the scored labels, so it defines a scoring domain: a card on it is a separate task
   with its own site-list hash (R2). It is NOT wired into the card; the v3 registration (B8) decides if it is used.

Interpretation choices NOT settled (the v3 registration, B8, must fix them before any real run):
- "known liquefaction" read as earlier earthquakes' yes labels (an event-held-out eval), not a within-event
  spatial training fold. A within-event fold would read the scored labels; that version is not built.
- The calibrated AUC form follows the design's quote: cAUC = AUC - null AUC + 0.5. Hijmans 2012 (Ecology 93:679)
  may instead use AUC + 0.5 - max(0.5, null AUC), which gives no reward when the null ranks worse than chance.
  Not checked against the paper; B8 checks it and registers which form is used.
- TR = 0.33 is the pairing tolerance; it follows the default of dismo's pwdSample as we recall it, not checked
  against the package source; the number is a choice to fix in B8.

Exposure: SEEN PLACE (other earthquakes' labels near the scored sites), as the site-memory baseline.
Provenance: `build` registers what it returns; the card takes only a registered, unchanged object.
Pure numpy and scipy.
"""
from __future__ import annotations

import json
import weakref
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from . import site_memory as _SM
from .site_memory import Earlier, SiteMemoryRefused, Sites, _date
from .spec import Entry, EntryManifest, Labels, entry_sha256

NAME = "baseline: geographic null"
ALLOWLIST_NAME = "earlier-labels"
EXPOSURE = "seen_place"
TR = 0.33
RULE_QUOTE = ('"Geographic null: distance to known liquefaction only; report calibrated AUC = AUC - null AUC + 0.5 '
              '(Hijmans 2012) [A]. Test "no" points matched in distance to training "yes" points." '
              "(docs/evals/EVAL-SYSTEM-DESIGN-2026-11-19.md, R2.2 item 2)")
INTERPRETATION_NOTE = ("interpretation not settled: known liquefaction read as earlier earthquakes' yes labels of "
                       "the task's label source; the matched-distance set and its tolerance TR = 0.33 are built but "
                       "not wired; the v3 registration (B8) fixes both")
LIMITS = ("exposure SEEN PLACE: earlier earthquakes' yes labels near the scored sites; the scored labels are never "
          "read", "the null rewards models that track where liquefaction was found before, including where people "
          "looked (R2.2); calibrated AUC removes that part from the AUC", INTERPRETATION_NOTE)


class GeoNullRefused(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class GeoNull:
    entry: Entry
    summary: dict


_BUILT: "weakref.WeakKeyDictionary[GeoNull, tuple]" = weakref.WeakKeyDictionary()


def _key(summary: dict) -> str:
    return json.dumps(summary, sort_keys=True, default=str)


def kit_built(obj: object) -> bool:
    if not isinstance(obj, GeoNull):
        return False
    rec = _BUILT.get(obj)
    return (rec is not None and rec[0] is obj.entry and rec[1] == entry_sha256(obj.entry)
            and rec[2] == _key(obj.summary))


def known_yes(earlier: Sequence[Earlier]) -> np.ndarray:
    """Step 1: the (n, 2) coordinates of every labelled yes site of the earlier earthquakes, pooled."""
    pts = [np.column_stack([e.labels.x_m, e.labels.y_m])[e.labels.labelled & (np.nan_to_num(e.labels.label) == 1)]
           for e in earlier]
    return np.vstack(pts) if pts else np.empty((0, 2))


def distance_to_known(x_m, y_m, known: np.ndarray) -> np.ndarray:
    """Distance (m) from each point to the nearest known-liquefaction site."""
    if known.shape[0] == 0:
        raise GeoNullRefused("no known liquefaction: the earlier earthquakes have no labelled yes site")
    d, _ = cKDTree(known).query(np.column_stack([np.asarray(x_m, float), np.asarray(y_m, float)]))
    return np.asarray(d, float)


def _iso(s: str, what: str):
    try:
        return _date(s, what)
    except SiteMemoryRefused as exc:
        raise GeoNullRefused(str(exc)) from None


def build(scored_event: str, scored_date: str, sites: Sites, earlier: Sequence[Earlier],
          label_set: str | None = None, scored_label_source: str | None = None) -> GeoNull:
    """The geographic null for every scored site. Takes no scored labels (see site_memory.Sites).

    label_set and scored_label_source (both or neither) select memory level "label set": the earlier earthquakes keep
    their real label source strings, a mix within the label set is allowed, and the summary records them
    (site_memory.resolve_sources). The default keeps one label source, as the sealed v3 card.

    Every earlier earthquake must be dated strictly before `scored_date` (ISO yyyy-mm-dd): a later earthquake's yes
    labels would leak the future into "known liquefaction".
    """
    if isinstance(sites, Labels) or not isinstance(sites, Sites):
        raise GeoNullRefused("pass the scored site list as Sites (site_memory.sites_of(labels)); the scored labels "
                             "never enter")
    t = _iso(scored_date, scored_event)
    if not earlier:
        raise GeoNullRefused("the geographic null needs at least one earlier earthquake")
    ids = [e.event_id for e in earlier]
    if len(set(ids)) != len(ids) or scored_event in ids:
        raise GeoNullRefused("earlier earthquakes need distinct ids, none equal to the scored earthquake")
    dates = {e.event_id: _iso(e.date, e.event_id) for e in earlier}
    if any(d >= t for d in dates.values()):
        raise GeoNullRefused("every earlier earthquake must come strictly before the scored one")
    src_fields = _SM.resolve_sources(earlier, label_set, scored_label_source, GeoNullRefused)
    src = src_fields["label_source"]
    known = known_yes(earlier)
    d = distance_to_known(sites.x_m, sites.y_m, known)
    q = np.percentile(d, [10, 50, 90])
    summary = {"rule": "score = minus the distance (m) to the nearest labelled yes site of the earlier earthquakes",
               "rule_quote": RULE_QUOTE, "scored_event": scored_event, "scored_date": t.isoformat(),
               "earlier_events": sorted(ids), "earlier_dates": {k: dates[k].isoformat() for k in sorted(dates)},
               "n_earlier": len(earlier), "label_source": src, "n_known_yes": int(known.shape[0]),
               "n_sites": int(d.size), "distance_m_p10_p50_p90": [float(v) for v in q],
               "calibrated_auc_rule": "cAUC = AUC - null AUC + 0.5; 95% interval = 0.5 + paired AUC difference "
                                      "interval (entry minus null)",
               "matched_set": f"built (matched_sites, TR {TR:g}), not wired into the card", "limits": list(LIMITS),
               "interpretation_status": INTERPRETATION_NOTE}
    if label_set is not None:
        summary.update(src_fields)
    stmt = (f"minus the distance to the nearest labelled yes site of {', '.join(sorted(ids))} ({src}); ranking only; "
            f"Hijmans 2012 geographic null per the eval design R2.2 item 2; {INTERPRETATION_NOTE}")
    m = EntryManifest(model=NAME, output_kind="score", statement=stmt, exposure={scored_event: EXPOSURE},
                      house=False, feature_allowlist=ALLOWLIST_NAME)
    out = GeoNull(Entry(m, sites.site_ids, -d), summary)
    _BUILT[out] = (out.entry, entry_sha256(out.entry), _key(out.summary))
    return out


def matched_sites(labels: Labels, earlier: Sequence[Earlier], tr: float = TR) -> dict:
    """Step 4: the matched-distance test set (site ids and counts). Reads the scored labels: a domain choice.

    Each scored yes site, in site-id order, takes the unused scored no site with the closest distance to known
    liquefaction (ties: smaller site id), if |d_no - d_yes| <= tr * d_yes. Unpaired yes sites drop.
    """
    if not tr > 0:
        raise GeoNullRefused("tr must be positive")
    d = distance_to_known(labels.x_m, labels.y_m, known_yes(earlier))
    lab = labels.labelled
    ids = np.array(labels.site_ids, dtype=object)
    yes = sorted(np.flatnonzero(lab & (np.nan_to_num(labels.label) == 1)), key=lambda k: ids[k])
    no = sorted(np.flatnonzero(lab & (np.nan_to_num(labels.label) == 0)), key=lambda k: ids[k])
    no_d = np.array([d[k] for k in no])
    used = np.zeros(len(no), bool)
    kept_yes, kept_no = [], []
    for k in yes:
        if used.all():
            break
        gap = np.where(used, np.inf, np.abs(no_d - d[k]))
        j = int(np.argmin(gap))                         # first minimum: no list is in site-id order
        if gap[j] <= tr * d[k]:
            used[j] = True
            kept_yes.append(ids[k])
            kept_no.append(ids[no[j]])
    return {"site_ids": tuple(sorted(kept_yes + kept_no)), "n_yes_in": len(yes), "n_no_in": len(no),
            "n_pairs": len(kept_yes), "tr": float(tr)}


# ---------------------------------------------------------------------------------------------------------------
# Card hooks
# ---------------------------------------------------------------------------------------------------------------

def card_problem(obj: object, event_id: str, label_source: str, earlier_label_sets: int) -> str | None:
    if not isinstance(obj, GeoNull):
        return "the geographic null must be built by evalkit.geo_null.build (a GeoNull)"
    if not kit_built(obj):
        return ("the geographic null was not returned by evalkit.geo_null.build, or its entry or summary changed "
                "after the build (provenance registry)")
    m = obj.entry.manifest
    if m.model != NAME or m.output_kind != "score" or m.feature_allowlist != ALLOWLIST_NAME:
        return f"the geographic null must be {NAME!r}, output kind score, feature_allowlist {ALLOWLIST_NAME!r}"
    if obj.summary.get("scored_event") != event_id or m.exposure != {event_id: EXPOSURE}:
        return f"the geographic null was built for another earthquake, not {event_id!r}"
    if earlier_label_sets == 0:
        return "R10: a geographic null was supplied but the task names no earlier earthquake"
    why = _SM.source_problem(obj.summary, label_source, "geographic null")
    if why:
        return why
    if obj.summary.get("n_earlier", 0) > earlier_label_sets:
        return (f"the geographic null uses {obj.summary.get('n_earlier')} earlier earthquakes; the task names "
                f"{earlier_label_sets}")
    return None


def calibrated_auc(card: dict) -> dict:
    """Step 3 on a scored card: cAUC for each entry row, with 0.5 plus the paired AUC difference interval.
    A row B2 marks NOT RUN BY AUTHOR SETTINGS has no score; it gets that mark, not a cAUC (B8 step 2 fix)."""
    null_auc = card["metrics"][NAME]["auc"]["value"]
    out = {}
    for e in card["entries"]:
        if e["role"] != "entry":
            continue
        name = e["name"]
        if not isinstance(card["metrics"].get(name), dict):     # B2: a NOT RUN BY AUTHOR SETTINGS row has no score
            out[name] = card["metrics"].get(name, "NOT SCORED")
            continue
        row = {"auc": card["metrics"][name]["auc"]["value"], "null_auc": null_auc,
               "value": card["metrics"][name]["auc"]["value"] - null_auc + 0.5}
        for c in card["comparisons"]:
            if c["family"] == "ranking" and {c["a"], c["b"]} == {name, NAME}:
                sign = 1.0 if c["a"] == name else -1.0
                lo, hi = sorted((0.5 + sign * c["lo"], 0.5 + sign * c["hi"]))
                row.update(lo=lo, hi=hi, word_vs_null=c["word"])
        out[name] = row
    return out

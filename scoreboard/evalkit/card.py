"""The card for one task: a plain dict of aggregates, refused unless every rule below holds. No em dashes.

Rule book rules enforced here (docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md):
  R2   the task hash must equal the registered hash when one is given; the label site list must match the task's hash
  R3   only probability-eligible entries (spec.probability_eligible) get probability metrics; others get AUC only
  R5   only a forward-track card can name a best entry; a record-track card reads RETROSPECTIVE and names none
  R6   every entry declares its exposure for the task earthquake; the mark prints; only UNSEEN can be named best
  R7   join by site id only (spec.align); the card prints the label caveats and the label rule hash
  R8   the card prints the label hash and the label error (or NOT MEASURED) and, with it, the AUC ceiling
  R9   AUC for all; log loss, Brier, calibration intercept and slope (each also without exact 0 and 1), every clip,
       exact 0 and 1 shares, CORP reliability and Brier decomposition for probabilities; level miss on a census only
  R10  coin always, earlier-earthquake rate and shaking-only when supplied, scored as entries; when the task says an
       earlier earthquake exists, a missing one refuses the card ("scorer fault: baseline not built"); with none,
       both print NOT AVAILABLE. B4: the shaking-only p is the continuous evalkit.shaking fit (earlier earthquakes
       only); the continuous rows use the CPT-free MSF form MSF-M-ONLY-1.3 (B&I 2014 eq. 2.19, page 14,
       MSFmax fixed at 1.3; evalkit.shaking.MSF_SOURCE), named in each row statement. `shaking_score` (PGA / MSF(M), ranking only,
       no fit) may sit beside on any task, and
       `shaking_binned` (the old binned row) may sit beside the shaking-only p for one release. With the old row
       beside, the shaking-only slot must hold the continuous fit (name, FIT_MARKER statement) and differ from the
       old row; without it, the slot still takes any target probability (old callers, the CLI CSV path, the demo),
       and full enforcement starts with scorecard v3 (B8).
       Optional "baseline: site memory" (evalkit.site_memory, B3): only an object
       site_memory.build returned is taken (provenance registry), only for the task earthquake and label source,
       never with more earlier earthquakes than the task names (probability version: two or more); its rule,
       counts and limits print under `site_memory`, and the notes say that on the same label source it also
       measures label persistence and that its radius is a new choice (site_memory.PERSISTENCE_NOTE, RADIUS_NOTE)
       B5 (eval M4), both optional and off by default: `pgv` (evalkit.pgv: PGV score or PGV-only fit from the
       ShakeMap grid; a NOT AVAILABLE object adds no row and the `pgv` block says why, no fill) and `geo_null`
       (evalkit.geo_null: distance to earlier known liquefaction; the `geographic_null` block adds calibrated
       AUC = AUC - null AUC + 0.5 per entry). Only kit-built objects are taken (provenance registry)
  R11  every metric has a tile-bootstrap interval; under the count gate the card reads UNDERPOWERED with counts only;
       the tile size comes from a residual semivariogram before scoring (tile_rule "variogram"); two Holm families
       (ranking: paired AUC; probability: paired log loss), no pair of two baselines; words SEPARATES BETTER or
       WORSE, EQUIVALENT, CANNOT TELL (C2); MDD per comparison; words at 1, 2 and 4 km and at every clip print
       beside, and a word that moves reads TILE-SENSITIVE or CLIP-SENSITIVE
  R12  the scorer gates (gates.run_gates) must pass before any card is built
  R15  an entry stating another division is refused
  R16  house entries are marked and listed first, comparisons a house entry loses print first, a house entry is never
       named best (no outside rerun path yet), and every card says who scored it
  R17  `scan_aggregates` refuses any per-site array, long list, site-id-shaped string or known site id
  B2   option `author_confirmed` (off by default): an outside-model row its authors have not confirmed prints
       NOT RUN BY AUTHOR SETTINGS and is not scored; `outside.check_card` refuses a card that prints its score
  B1   (eval M4) a house other_probability may declare itself a probability of the target (spec `declared_target`);
       `losses_first=True` puts the LOSS block (evalkit.losses) first; `require_house_proper_score=True` refuses a
       house other_probability with no declaration and no conversion; all off by default, so older cards keep
       their bytes
  B6   (eval M4) option `breakdowns` (off by default): tables by PGA bin, Ic class, crust thickness and severity
       class (evalkit.breakdowns), n per cell, a floor below which a cell prints "too few", no winner named
  2b   every card hash is `spec.card_sha256` (canonical JSON, floats at 12 significant digits), never the file bytes;
       the effort-only slot needs feature_allowlist "pre-event" in the entry manifest (review 2 N3)
  B7   (eval M4) option `placement` (evalkit.placement.build, off when None): AUC within CPT-density strata for every
       row and the proxy word (EXPLAINED BY DENSITY STRATA / OPEN BIAS / CANNOT TELL) under `placement`;
       `effort_imagery_footprint=True` makes the effort slot take only feature_allowlist "pre-event-imagery"
       (known limit N4) and adds a note naming it (review 1; the flag without an effort row is refused); both off
       by default, so older cards keep their bytes
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..label_ceiling import ceiling, pi_moments
from . import breakdowns as BD
from . import gates, losses, metrics, outside, resample
from . import interval_policy as IP
from . import placement as PL
from .shaking import FIT_MARKER
from . import geo_null as GN
from . import pgv as PGV
from . import site_memory as SM
from .spec import (Entry, EntryManifest, Labels, Task, align, card_sha256, entry_sha256, labels_sha256,
                   probability_eligible, site_list_sha256)

KIT_VERSION = "0.1.0"
RULEBOOK = "docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md (v1.0 DRAFT, not adopted)"
BASELINE_PREFIX = "baseline:"
EFFORT_ALLOWLIST = "pre-event"          # evalkit.effort.ALLOWLIST_NAME; the only allowlist the effort slot takes
ALPHA = 0.05
EQUIVALENCE_MARGIN = {"auc": 0.05, "log_loss": 0.05}      # R27: AUC 0.05, log loss 0.05 nats
FAMILIES = ("ranking", "probability")
MAX_LIST_LEN = 32
MAX_PAIR_RECORDS = 512
SITE_ID_SHAPE = re.compile(r"\b(nzgd|rer|usgs):\S", re.IGNORECASE)
EXPOSURE_MARKS = {"unseen": "UNSEEN", "seen_inputs": "SEEN INPUTS", "fitted_on": "FITTED ON THIS EARTHQUAKE",
                  "unknown": "NOT CHECKED", "seen_place": "SEEN PLACE (OTHER EARTHQUAKES)"}
SCORED_BY = "scored by the organiser, who also enters models"
CLAIMS = ("Scores rate models on a named label source, not on the ground.",
          "A record-track score is not a test of skill.",
          "One earthquake or one city is not the world.",
          "UNDERPOWERED is not a tie.",
          "No score is design advice.")


class CardRefused(RuntimeError):
    """A rule the card enforces is broken; no card is written. The message names the rule."""


# ---------------------------------------------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------------------------------------------

def tile_choice(task: Task, labels: Labels, shaking_only: Entry | None = None) -> dict:
    """R11 c: the main tile size from a residual semivariogram, fixed before any entry is scored.

    Residual = label minus the shaking-only p; without it, minus the earlier-earthquake rate; with neither, minus 0.5.
    Only labels and baselines enter, never an entry. The variogram sites are drawn with the task seed. Returns the
    choice with the sha256 of the binned semivariogram (canonical JSON) and no per-site value.
    """
    lab = labels.labelled
    ids = tuple(s for s, k in zip(labels.site_ids, lab) if k)
    y, x, yy = labels.label[lab], labels.x_m[lab], labels.y_m[lab]
    if shaking_only is not None:
        base, source = align(shaking_only, ids), "label minus the shaking-only p"
    elif task.earlier_base_rate is not None:
        base, source = np.full(y.size, task.earlier_base_rate), "label minus the earlier-earthquake rate"
    else:
        base, source = np.full(y.size, 0.5), "label minus 0.5 (no earlier earthquake)"
    c = resample.choose_tile(x, yy, y - base, seed=task.seed)
    vg = c.pop("variogram")
    c["variogram_sha256"] = hashlib.sha256(json.dumps(vg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    c["residual"] = source
    if c["range_beyond_candidates"]:
        c["mark"] = "range exceeds the largest tile"
    return c


def build_card(task: Task, entries: Sequence[Entry], labels: Labels, *, shaking_only: Entry | None = None,
               registered_task_sha256: str | None = None, effort_only: Entry | None = None,
               author_confirmed: dict | None = None, shaking_score: Entry | None = None,
               shaking_binned: Entry | None = None,
               site_memory: "SM.SiteMemory | None" = None, losses_first: bool = False,
               require_house_proper_score: bool = False, pgv: "PGV.PgvBaseline | None" = None,
               geo_null: "GN.GeoNull | None" = None, breakdowns: "BD.Breakdowns | None" = None,
               placement: "PL.PlacementStrata | None" = None, effort_imagery_footprint: bool = False,
               card_version: str = "v2") -> dict:
    """Score every entry and baseline on one task and return the card as a JSON-ready dict of aggregates.

    `author_confirmed` (B2, off when None): maps outside-model row names to whether their authors confirmed the
    settings; a row missing from it counts as not confirmed and prints NOT RUN BY AUTHOR SETTINGS (evalkit.outside).
    B4: `shaking_score` is evalkit.shaking.score_entry (ranking only); `shaking_binned` is
    evalkit.shaking.binned_entry, the old row kept beside for one release. Both are optional.
    `losses_first` (B1): put the house LOSS block (evalkit.losses.loss_block) first. Off by default.
    `require_house_proper_score` (B1): refuse the card when a house other_probability entry has no declared target
    and no conversion, so every house probability row carries a proper score. Off by default.
    `breakdowns` (B6): an evalkit.breakdowns.Breakdowns; adds the `breakdowns` section. Off by default.
    `card_version` (C4): "v2" (default) builds the card exactly as before; "v3" prints the C4 per-earthquake outcome
    (evalkit.interval_policy) as the only interval, needs fixed 4 km tiles, and withholds every per-earthquake word
    on the card (comparisons, the geographic null's word_vs_null, the placement proxy word). B8: "v3" also hashes
    refit rows (effort slot, B4 and B5 fits) with spec hash recipe "v2" and records `hash_recipe` on the card; "v2"
    keeps recipe "v1" and every byte.
    """
    IP.check_version(card_version)
    IP.check_task(task, card_version)
    if require_house_proper_score:
        missing = losses.house_rows_without_proper_score(e.manifest for e in entries)
        if missing:
            raise CardRefused(f"B1: house other_probability rows with no declared target and no conversion carry no "
                              f"proper score: {missing}")
    gate_result = gates.run_gates()
    if not gate_result["pass"]:
        raise CardRefused("R12: the scorer gates failed; no card is built")
    _check_inputs(task, entries, labels, shaking_only, registered_task_sha256)
    _check_shaking_extras(task, shaking_only, shaking_score, shaking_binned)
    _check_effort(effort_only, PL.EFFORT_IMAGERY_ALLOWLIST if effort_imagery_footprint else EFFORT_ALLOWLIST)
    _check_site_memory(site_memory, task)
    _check_b5(pgv, geo_null, task)
    if placement is not None and not PL.kit_built(placement):
        raise CardRefused("B7: the placement strata must come from evalkit.placement.build, unchanged since")
    if effort_imagery_footprint and effort_only is None:
        raise CardRefused("B7: effort_imagery_footprint=True needs an effort_only row; alone it does nothing")
    tiles_block = {"rule": task.tile_rule, "tile_m": task.tile_m}
    if task.tile_rule == "variogram":
        choice = tile_choice(task, labels, shaking_only)
        if choice["tile_m"] != task.tile_m:
            raise CardRefused(f"R11 c: the task tile {task.tile_m:g} m differs from the variogram choice "
                              f"{choice['tile_m']:g} m")
        tiles_block.update(choice)
    lab = labels.labelled
    ids = tuple(s for s, k in zip(labels.site_ids, lab) if k)
    y, x, yy = labels.label[lab], labels.x_m[lab], labels.y_m[lab]
    recipe = hash_recipe(card_version)
    rows = _rows(task, entries, shaking_only, ids, recipe)
    for extra in (shaking_score, shaking_binned):
        if extra is not None:
            m = _shaking_extra_manifest(extra, task.event_id)
            rows.append(_row(m.model, "baseline", m, align(extra, ids), entry_sha256(extra)))
    if effort_only is not None:
        m = _effort_manifest(effort_only.manifest.statement, task.event_id)
        rows.append(_row(m.model, "baseline", m, align(effort_only, ids), entry_sha256(effort_only, recipe, True)))
    if site_memory is not None:
        e = site_memory.entry
        rows.append(_row(e.manifest.model, "baseline", e.manifest, align(e, ids), entry_sha256(e)))
    for b5 in (pgv, geo_null):
        if b5 is not None and b5.entry is not None:
            e = b5.entry
            rows.append(_row(e.manifest.model, "baseline", e.manifest, align(e, ids),
                             entry_sha256(e, recipe, is_refit(e))))
    all_rows, not_run = rows, []
    if author_confirmed is not None:
        rows, not_run = outside.split_rows(rows, author_confirmed)
    counts = _counts(task, y, x, yy)
    card = {"kit": f"evalkit {KIT_VERSION}", "rulebook": RULEBOOK, "task": task.to_dict(),
            "task_sha256": task.sha256(), "track": "FORWARD" if task.track == "forward" else "RETROSPECTIVE",
            "scored_by": SCORED_BY, "claims": list(CLAIMS),
            "gates": {"pass": True, "seed": gate_result["seed"],
                      **{k: v["pass"] for k, v in gate_result["gates"].items()}},
            "label": _label_block(task, labels, y), "count_gate": counts, "tiles": tiles_block,
            "entries": [outside.entry_marks(_entry_meta(r, task), r) for r in all_rows],
            "notes": _notes(task)}
    if site_memory is not None:
        card["site_memory"] = site_memory.summary
        card["notes"] += [f"B3 site memory, {site_memory.summary.get('label_persistence', SM.PERSISTENCE_NOTE)}",
                          f"B3 site memory, {SM.RADIUS_NOTE}"]
    if pgv is not None:
        card["pgv"] = pgv.summary
        card["notes"].append(PGV.card_note(pgv))
    if geo_null is not None:
        card["geographic_null"] = dict(geo_null.summary)
        card["notes"].append(f"B5 geographic null, {GN.INTERPRETATION_NOTE}")
    if effort_imagery_footprint:
        card["notes"].append(PL.EFFORT_IMAGERY_NOTE)
    if counts["passes"]:
        card["status"] = "SCORED"
        card.update(_score(task, rows, y, x, yy))
    else:
        card["status"] = "UNDERPOWERED"
        why = "UNDERPOWERED (R11): counts only; this is not a tie"
        card.update(comparisons=[], best={"ranking": {"name": None, "reasons": [why]},
                                          "probability": {"name": None, "reasons": [why]}})
    if breakdowns is not None:
        card["breakdowns"] = BD.block(breakdowns, task, rows, ids, y, x, yy, scored=counts["passes"])
    if placement is not None:
        card["placement"] = PL.card_block(placement, rows, ids, y, x, yy, task.tile_m, task.n_boot, task.seed,
                                          scored=counts["passes"])
    if author_confirmed is not None:
        outside.mark_card(card, not_run)
    if geo_null is not None and card["status"] == "SCORED":
        card["geographic_null"]["calibrated_auc"] = GN.calibrated_auc(card)
    card = _json_safe(card)
    if losses_first:
        card = losses.put_first(card)
    if card_version != "v2":
        card = IP.apply_per_earthquake(card, card_version)
        card["hash_recipe"] = {"card": recipe, "refit_entries": recipe,
                               "rule": "spec.card_sha256(recipe='v2'); refit rows via spec.entry_sha256(recipe='v2', "
                                       "refit=True) (B8, ruling 48)"}
    author_gate(card)
    scan_aggregates(card, labels.site_ids)
    return card


def author_gate(card: dict) -> None:
    """B2 gate: refuse a card that prints a score for an outside-model row its authors have not confirmed."""
    try:
        outside.check_card(card)
    except outside.AuthorGateError as e:
        raise CardRefused(str(e)) from None


DESCRIPTIVE = "DESCRIPTIVE"
_WORD_KEYS = ("word", "a_is", "favours", "words_by_tile_m", "words_by_clip", "flags", "p", "mdd")


def descriptive(card: dict, note: str) -> dict:
    """The card with every within-earthquake word withheld (scorecard v2 registration K2).

    Every comparison keeps its difference, 95% and 90% intervals, SE and draw count; its word, the BETTER / WORSE
    direction, the favoured entry, the words at other tile sizes and clips, the sensitivity flags, the bootstrap p and
    the MDD are removed, and `word` reads DESCRIPTIVE. Comparisons are re-sorted by (family, a, b), so the order no
    longer shows which comparisons a house entry loses. The best-entry fields read DESCRIPTIVE. Every metric value
    and interval is kept. `note` (the interval caveat) is stored on the card. The input card is not changed.
    """
    if not isinstance(note, str) or not note.strip():
        raise ValueError("descriptive needs a non-empty interval note")
    out = json.loads(json.dumps(card))
    comps = []
    for c in out.get("comparisons", []):
        c = {k: v for k, v in c.items() if k not in _WORD_KEYS}
        c["word"] = DESCRIPTIVE
        comps.append(c)
    comps.sort(key=lambda c: (c.get("family", ""), c.get("a", ""), c.get("b", "")))
    out["comparisons"] = comps
    out.pop("holm_family_sizes", None)
    out["best"] = {f: {"name": None, "reasons": [DESCRIPTIVE]} for f in FAMILIES}
    out["words"] = DESCRIPTIVE
    out["interval_note"] = note
    author_gate(out)
    scan_aggregates(out)
    return out


def write_card(card: dict, path: str | Path, site_ids: Sequence[str] = ()) -> None:
    """Scan, then write the card as JSON. Refuses (R17) rather than write per-site content, and (B2) rather than
    write a score for an unconfirmed outside-model row."""
    author_gate(card)
    scan_aggregates(card, site_ids)
    Path(path).write_text(json.dumps(card, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def scan_aggregates(obj: object, site_ids: Sequence[str] = (), path: str = "card") -> None:
    """R17 output scan. Refuses an array, a list longer than MAX_LIST_LEN, a site-id-shaped string, or a known id.

    One exception: a list under a key named `comparisons` may hold up to MAX_PAIR_RECORDS items if every item is a
    pair record (a dict naming `a` and `b`). Pairs grow with the square of the rows (9 rows give 45), while a per-site
    list would be scalars; every item is still scanned.
    """
    known = site_ids if isinstance(site_ids, (set, frozenset)) else frozenset(site_ids)
    if isinstance(obj, dict):
        for k, v in obj.items():
            scan_aggregates(k, known, f"{path} key")
            scan_aggregates(v, known, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        pair_list = (path.endswith(".comparisons") and len(obj) <= MAX_PAIR_RECORDS
                     and all(isinstance(v, dict) and "a" in v and "b" in v for v in obj))
        if len(obj) > MAX_LIST_LEN and not pair_list:
            raise CardRefused(f"R17: {path} is a list of {len(obj)} items; a card holds aggregates only")
        for k, v in enumerate(obj):
            scan_aggregates(v, known, f"{path}[{k}]")
    elif isinstance(obj, str):
        if SITE_ID_SHAPE.search(obj) or obj in known:
            raise CardRefused(f"R17: {path} holds a site id ({obj[:40]!r})")
    elif obj is not None and not isinstance(obj, (bool, int, float)):
        raise CardRefused(f"R17: {path} is a {type(obj).__name__}; a card holds plain JSON aggregates only")


# ---------------------------------------------------------------------------------------------------------------
# Inputs, rows, counts
# ---------------------------------------------------------------------------------------------------------------

def _check_inputs(task: Task, entries: Sequence[Entry], labels: Labels, shaking_only: Entry | None,
                  registered: str | None) -> None:
    if registered is not None and registered != task.sha256():
        raise CardRefused("R2: the task hash differs from the registered hash")
    if site_list_sha256(labels.site_ids) != task.site_list_sha256:
        raise CardRefused("R2: the label file's site list does not match the task's site list hash")
    names = [e.manifest.model for e in entries]
    if len(set(names)) != len(names):
        raise CardRefused("each entry needs its own model name")
    for e in entries:
        m = e.manifest
        if m.model.lower().startswith(BASELINE_PREFIX):
            raise CardRefused(f"{m.model!r}: names starting {BASELINE_PREFIX!r} are kept for baselines (R10)")
        if task.event_id not in m.exposure:
            raise CardRefused(f"R6: {m.model!r} declares no exposure for {task.event_id!r}")
        if m.division is not None and m.division != task.division:
            raise CardRefused(f"R15: {m.model!r} is a {m.division} entry on a {task.division} card")
    if shaking_only is not None and shaking_only.manifest.output_kind != "target_probability":
        raise CardRefused("R10: the shaking-only baseline must be a probability of the task target")
    if task.earlier_label_sets > 0 and (task.earlier_base_rate is None or shaking_only is None):
        missing = [n for n, v in (("earlier-earthquake rate", task.earlier_base_rate), ("shaking only", shaking_only))
                   if v is None]
        raise CardRefused(f"R10: scorer fault: baseline not built ({', '.join(missing)}); an earlier earthquake exists")
    if task.earlier_label_sets == 0 and (task.earlier_base_rate is not None or shaking_only is not None):
        raise CardRefused("R10: an earlier-earthquake baseline was supplied but the task names no earlier earthquake")


SHAKING_ONLY_NAME = f"{BASELINE_PREFIX} shaking only"              # evalkit.shaking.NAME_PROB
SHAKING_SCORE_NAME = f"{BASELINE_PREFIX} shaking score"            # evalkit.shaking.NAME_SCORE
SHAKING_BINNED_NAME = f"{BASELINE_PREFIX} shaking only (binned, old)"   # evalkit.shaking.NAME_BINNED


def _check_shaking_extras(task: Task, shaking_only: Entry | None, score: Entry | None,
                          binned: Entry | None) -> None:
    """B4: the continuous shaking score (ranking only) and the old binned row kept beside for one release."""
    if score is not None:
        if score.manifest.model != SHAKING_SCORE_NAME:
            raise CardRefused(f"the shaking score must be named {SHAKING_SCORE_NAME!r} (evalkit.shaking.score_entry)")
        if score.manifest.output_kind != "score" or score.manifest.conversion_id is not None:
            raise CardRefused("the shaking score is PGA / MSF(M): output kind score, no conversion (ranking only)")
    if binned is not None:
        if binned.manifest.model != SHAKING_BINNED_NAME:
            raise CardRefused(f"the old binned row must be named {SHAKING_BINNED_NAME!r} "
                              "(evalkit.shaking.binned_entry)")
        if binned.manifest.output_kind != "target_probability":
            raise CardRefused("R10: the old binned row must be a probability of the task target")
        if shaking_only is None or task.earlier_label_sets == 0:
            raise CardRefused("the old binned row sits beside the shaking-only p and needs an earlier earthquake")
        if (shaking_only.manifest.model != SHAKING_ONLY_NAME
                or not shaking_only.manifest.statement.startswith(FIT_MARKER)):
            raise CardRefused("with the old binned row beside, the shaking-only slot must hold the continuous fit "
                              "(evalkit.shaking.probability_entry), not another row under its name")
        if _same_values(shaking_only, binned):
            raise CardRefused("the old binned row and the shaking-only p carry the same values; the shaking-only "
                              "slot must hold the continuous fit")


def _same_values(a: Entry, b: Entry) -> bool:
    """True when two entries give the same value at every site (same site set, order free)."""
    if set(a.site_ids) != set(b.site_ids):
        return False
    pa = dict(zip(a.site_ids, a.p))
    return bool(all(pa[s] == v for s, v in zip(b.site_ids, b.p)))


def _shaking_extra_manifest(e: Entry, event_id: str) -> EntryManifest:
    return EntryManifest(model=e.manifest.model, output_kind=e.manifest.output_kind, statement=e.manifest.statement,
                         exposure={event_id: "unseen"}, house=False)


def _check_effort(effort_only: Entry | None, allowlist: str = EFFORT_ALLOWLIST) -> None:
    """Step 1c: the effort-only baseline is the kit's own kind, built by evalkit.effort (allowlist, LOEO).
    B7: with effort_imagery_footprint the slot takes only the imagery-footprint allowlist (N4)."""
    if effort_only is None:
        return
    if effort_only.manifest.model != f"{BASELINE_PREFIX} effort only":
        raise CardRefused("the effort-only baseline must be named 'baseline: effort only' (evalkit.effort.entry)")
    if effort_only.manifest.output_kind != "other_probability":
        raise CardRefused("the effort-only baseline is a probability of being labelled, not of the target: "
                          "output kind other_probability (ranking only)")
    if effort_only.manifest.feature_allowlist != allowlist:
        raise CardRefused(f"the effort-only baseline must declare feature_allowlist {allowlist!r} in its "
                          f"manifest (review 2 N3); it declares {effort_only.manifest.feature_allowlist!r}")


def _check_site_memory(sm: "SM.SiteMemory | None", task: Task) -> None:
    """B3: the site-memory baseline is the kit's own kind, built by evalkit.site_memory.build (no scored labels)."""
    if sm is None:
        return
    if not isinstance(sm, SM.SiteMemory):
        raise CardRefused("the site-memory baseline must be built by evalkit.site_memory.build (a SiteMemory)")
    if not SM.kit_built(sm):
        raise CardRefused("the site-memory baseline was not returned by evalkit.site_memory.build, or its entry or "
                          "summary changed after the build (provenance registry)")
    m = sm.entry.manifest
    if m.model != SM.NAME or m.feature_allowlist != SM.ALLOWLIST_NAME:
        raise CardRefused(f"the site-memory baseline must be named {SM.NAME!r} with feature_allowlist "
                          f"{SM.ALLOWLIST_NAME!r}")
    if m.output_kind not in ("target_probability", "score"):
        raise CardRefused("the site-memory baseline is a target probability or a ranking-only score")
    if sm.summary.get("scored_event") != task.event_id or m.exposure != {task.event_id: SM.EXPOSURE}:
        raise CardRefused(f"the site-memory baseline was built for another earthquake, not {task.event_id!r}")
    if task.earlier_label_sets == 0:
        raise CardRefused("R10: a site-memory baseline was supplied but the task names no earlier earthquake")
    why = SM.source_problem(sm.summary, task.label_source, "site-memory baseline")
    if why:
        raise CardRefused(why)
    if sm.summary.get("n_earlier", 0) > task.earlier_label_sets:
        raise CardRefused(f"the site-memory baseline uses {sm.summary.get('n_earlier')} earlier earthquakes; the task "
                          f"names {task.earlier_label_sets}")
    if m.output_kind == "target_probability" and task.earlier_label_sets < 2:
        raise CardRefused("the probability version of site memory needs two earlier earthquakes; the task names one")


def _check_b5(pgv, geo_null, task: Task) -> None:
    """B5: the PGV and geographic-null baselines are the kit's own kinds (evalkit.pgv, evalkit.geo_null)."""
    if pgv is not None:
        why = PGV.card_problem(pgv, task.event_id, task.earlier_label_sets)
        if why:
            raise CardRefused(why)
    if geo_null is not None:
        why = GN.card_problem(geo_null, task.event_id, task.label_source, task.earlier_label_sets)
        if why:
            raise CardRefused(why)


def _effort_manifest(statement: str, event_id: str) -> EntryManifest:
    return EntryManifest(model=f"{BASELINE_PREFIX} effort only", output_kind="other_probability",
                         statement=statement, exposure={event_id: "seen_place"}, house=False)


def _baseline_manifest(name: str, statement: str, event_id: str) -> EntryManifest:
    return EntryManifest(model=f"{BASELINE_PREFIX} {name}", output_kind="target_probability", statement=statement,
                         exposure={event_id: "unseen"}, house=False)


def _row(name: str, role: str, m: EntryManifest, v: np.ndarray, sha: str | None) -> dict:
    return {"name": name, "role": role, "manifest": m, "v": v, "eligible": probability_eligible(m),
            "constant": bool(np.ptp(v) == 0.0), "sha": sha}


HASH_RECIPE_BY_VERSION = {"v2": "v1", "v3": "v2"}     # B8: card version -> spec hash recipe (ruling 48)


def hash_recipe(card_version: str) -> str:
    """The spec hash recipe a card of this version uses for its refit rows and its card hash (B8, ruling 48)."""
    return HASH_RECIPE_BY_VERSION[IP.check_version(card_version)]


def is_refit(e: Entry) -> bool:
    """A row the card fits on each build (B4 continuous shaking fit, B5 PGV fit); the effort slot is refit too."""
    s = e.manifest.statement
    return s.startswith(FIT_MARKER) or s.startswith(PGV.FIT_MARKER)


def _rows(task: Task, entries: Sequence[Entry], shaking_only: Entry | None, ids: tuple[str, ...],
          recipe: str = "v1") -> list[dict]:
    """House entries first, then outside entries, then baselines (R16 b, R10)."""
    ordered = sorted(entries, key=lambda e: not e.manifest.house)
    rows = [_row(e.manifest.model, "entry", e.manifest, align(e, ids), entry_sha256(e)) for e in ordered]
    n, ev = len(ids), task.event_id
    rows.append(_row(f"{BASELINE_PREFIX} coin", "baseline", _baseline_manifest("coin", "0.5 at every site", ev),
                     np.full(n, 0.5), None))
    if task.earlier_base_rate is not None:
        m = _baseline_manifest("earlier-earthquake rate", "yes share of earlier earthquakes at every site", ev)
        rows.append(_row(m.model, "baseline", m, np.full(n, task.earlier_base_rate), None))
    if shaking_only is not None:
        m = _baseline_manifest("shaking only", shaking_only.manifest.statement, ev)
        rows.append(_row(m.model, "baseline", m, align(shaking_only, ids),
                         entry_sha256(shaking_only, recipe, is_refit(shaking_only))))
    return rows


def _counts(task: Task, y: np.ndarray, x: np.ndarray, yy: np.ndarray) -> dict:
    n_yes = int(y.sum())
    c = {"n_labelled": int(y.size), "n_yes": n_yes, "n_no": int(y.size - n_yes),
         "n_tiles": int(np.unique(resample.tiles_for(x, yy, task.tile_m)).size) if y.size else 0,
         "min_sites": task.min_sites, "min_per_class": task.min_per_class, "min_tiles": task.min_tiles}
    c["passes"] = (c["n_labelled"] >= task.min_sites and min(n_yes, c["n_no"]) >= task.min_per_class
                   and c["n_tiles"] >= task.min_tiles)
    return c


def _label_block(task: Task, labels: Labels, y: np.ndarray) -> dict:
    out = {"source": task.label_source, "target": task.target, "rule_sha256": task.label_rule_sha256,
           "labels_sha256": labels_sha256(labels), "n_sites_listed": len(labels.site_ids),
           "caveats": task.label_caveats, "census": task.census}
    if task.label_error is None:
        out["error"] = "NOT MEASURED (R8)"
        out["auc_ceiling"] = "NOT MEASURED (R8)"
        return out
    se, sp = task.label_error["sensitivity"], task.label_error["specificity"]
    out["error"] = {"sensitivity": se, "specificity": sp}
    pi = float(pi_moments(float(y.mean()), se, sp)) if y.size else float("nan")
    out["auc_ceiling"] = float(ceiling(pi, se, sp)["C"]) if 0.0 < pi < 1.0 else "NOT IDENTIFIED at this yes share"
    return out


def _entry_meta(r: dict, task: Task) -> dict:
    m = r["manifest"]
    return {"name": r["name"], "role": r["role"], "house": m.house, "mark": "HOUSE" if m.house else "",
            "output_kind": m.output_kind, "statement": m.statement,
            "exposure": EXPOSURE_MARKS[m.exposure[task.event_id]], "conversion_id": m.conversion_id,
            "probability_metrics": r["eligible"], "entry_sha256": r["sha"], **losses.entry_meta_extra(m)}


def _notes(task: Task) -> list[str]:
    notes = []
    if task.n_boot < 2000:
        notes.append(f"NOT FOR RELEASE: n_boot {task.n_boot} is below the 2,000 draws R11 requires")
    if task.earlier_label_sets == 0:
        notes.append("R10: no earlier earthquake of this label source: earlier-earthquake rate and shaking-only "
                     "baseline NOT AVAILABLE")
    return notes


# ---------------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------------

def _stat_fn(r: dict, y: np.ndarray, task: Task):
    v = r["v"]
    if r["eligible"]:
        return lambda idx: metrics.probability_stats(y[idx], v[idx], task.clip, task.census)
    return lambda idx: metrics.ranking_stats(y[idx], v[idx])


def _slim_fn(r: dict, y: np.ndarray, ll_key: str, clip: float):
    """AUC and the registered log loss only: what the tile-sensitivity words need."""
    v = r["v"]
    if r["eligible"]:
        return lambda idx: {"auc": metrics.auc(y[idx].astype(bool), v[idx]),
                            ll_key: metrics.log_loss(y[idx], v[idx], clip)}
    return lambda idx: metrics.ranking_stats(y[idx], v[idx])


def _pairs(rows: list[dict]) -> list[tuple[str, str, str]]:
    """(family, a, b), R11 d. Ranking: paired AUC for every pair not both constant. Probability: paired log loss for
    every pair of probability-eligible rows. Neither family tests a pair of two baselines."""
    out = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["role"] == "baseline" and b["role"] == "baseline":
                continue
            if not (a["constant"] and b["constant"]):
                out.append(("ranking", a["name"], b["name"]))
    elig = [r for r in rows if r["eligible"]]
    for i, a in enumerate(elig):
        for b in elig[i + 1:]:
            if a["role"] == "baseline" and b["role"] == "baseline":
                continue
            out.append(("probability", a["name"], b["name"]))
    return out


def _word_table(pairs, points: dict, draws: dict, ll_key: str) -> list[dict]:
    """Paired differences, Holm within each family (R11 d), and the C2 word for each.

    SEPARATES: Holm rejects; `a_is` says BETTER (higher AUC or lower log loss for a) or WORSE. EQUIVALENT: not
    SEPARATES and the 90% paired interval lies strictly inside plus or minus the R27 margin. CANNOT TELL: neither.
    """
    res = []
    for fam, a, b in pairs:
        key = "auc" if fam == "ranking" else ll_key
        res.append(resample.paired(points[a][key], points[b][key], draws[a][key], draws[b][key]))
    rejected = [False] * len(pairs)
    sizes = {}
    for fam in FAMILIES:
        idx = [k for k, pr in enumerate(pairs) if pr[0] == fam]
        sizes[fam] = len(idx)
        for k, rej in zip(idx, resample.holm([res[k]["p"] for k in idx], ALPHA)):
            rejected[k] = rej
    for (fam, a, b), r, rej in zip(pairs, res, rejected):
        metric = "auc" if fam == "ranking" else "log_loss"
        a_wins = (r["diff"] > 0) == (fam == "ranking")
        margin = EQUIVALENCE_MARGIN[metric]
        if rej:
            word = "SEPARATES"
        elif -margin < r["lo90"] and r["hi90"] < margin:
            word = "EQUIVALENT"
        else:
            word = "CANNOT TELL"
        r.update(family=fam, family_size=sizes[fam], metric=metric, a=a, b=b, word=word,
                 a_is=("BETTER" if a_wins else "WORSE") if rej else None,
                 favours=(a if a_wins else b) if rej else None)
    return res


def _score(task: Task, rows: list[dict], y: np.ndarray, x: np.ndarray, yy: np.ndarray) -> dict:
    tiles = resample.tiles_for(x, yy, task.tile_m)
    ll_key = f"log_loss@{metrics.clip_key(task.clip)}"
    full = np.arange(y.size)
    points, draws, tile_draws = {}, {}, {s: {} for s in resample.SENSITIVITY_TILES_M}
    for r in rows:
        fn = _stat_fn(r, y, task)
        points[r["name"]] = fn(full)
        draws[r["name"]] = resample.bootstrap(fn, tiles, task.n_boot, task.seed)
        slim = _slim_fn(r, y, ll_key, task.clip)
        for s in tile_draws:
            tile_draws[s][r["name"]] = resample.bootstrap(slim, resample.tiles_for(x, yy, s), task.n_boot, task.seed)
    pairs = _pairs(rows)
    table = _word_table(pairs, points, draws, ll_key)
    by_clip = {metrics.clip_key(c): _word_table(pairs, points, draws, f"log_loss@{metrics.clip_key(c)}")
               for c in metrics.CLIPS}
    by_tile = {f"{s:g}": _word_table(pairs, points, tile_draws[s], ll_key) for s in tile_draws}
    tile_counts = {f"{s:g}": int(np.unique(resample.tiles_for(x, yy, s)).size) for s in tile_draws}
    for k, c in enumerate(table):
        c["words_by_tile_m"] = {s: t[k]["word"] for s, t in by_tile.items()}
        c["flags"] = []
        if any((t[k]["word"], t[k]["favours"]) != (c["word"], c["favours"]) for s, t in by_tile.items()
               if tile_counts[s] >= task.min_tiles):
            c["flags"].append("TILE-SENSITIVE")
        if c["family"] == "probability":
            c["words_by_clip"] = {s: t[k]["word"] for s, t in by_clip.items()}
            if any((t[k]["word"], t[k]["favours"]) != (c["word"], c["favours"]) for t in by_clip.values()):
                c["flags"].append("CLIP-SENSITIVE")
    entries_out = {r["name"]: _metrics_block(r, points[r["name"]], draws[r["name"]], tile_draws, task, y)
                   for r in rows}
    obs = resample.bootstrap(lambda idx: {"o": float(y[idx].mean())}, tiles, task.n_boot, task.seed)["o"]
    vals = {r["name"]: r["v"] for r in rows if not r["constant"]}
    dups = gates.duplicate_pairs(vals)
    house = {r["name"] for r in rows if r["manifest"].house}
    table.sort(key=lambda c: not (c["favours"] is not None and ({c["a"], c["b"]} - {c["favours"]}) <= house))
    dup_sets = [{d["a"], d["b"]} for d in dups]
    fam_sizes = {f: sum(1 for c in table if c["family"] == f) for f in FAMILIES}
    sens = {s: (n if n >= task.min_tiles else f"{n} tiles: too few tiles at this size") for s, n in tile_counts.items()}
    return {"observed_share": resample.interval(float(y.mean()), obs), "metrics": entries_out, "comparisons": table,
            "holm_family_sizes": fam_sizes, "sensitivity_tile_counts": sens,
            "best": {"ranking": _best("ranking", rows, table, points, ll_key, task, dup_sets),
                     "probability": _best("probability", rows, table, points, ll_key, task, dup_sets)},
            "duplicates": {"rule": f"Spearman above {gates.DUPLICATE_RHO} marks one model", "pairs": dups,
                           "distinct_models": gates.distinct_models(list(vals), dups)}}


def _metrics_block(r: dict, point: dict, draws: dict, tile_draws: dict, task: Task, y: np.ndarray) -> dict:
    def iv(k: str) -> dict:
        return resample.interval(point[k], draws[k])

    name, v = r["name"], r["v"]
    m = {"auc": iv("auc"),
         "auc_by_tile_m": {f"{s:g}": resample.interval(point["auc"], d[name]["auc"]) for s, d in tile_draws.items()}}
    if not r["eligible"]:
        m["probability_metrics"] = (f"NOT APPLICABLE (R3): output kind {r['manifest'].output_kind} with no frozen "
                                    f"conversion; ranking only")
        return m
    c = metrics.clip_key(task.clip)
    m.update(
        log_loss=iv(f"log_loss@{c}"), brier=iv("brier"), mean_p=iv("mean_p"),
        calibration={"clip": task.clip, "intercept": iv("citl"), "slope": iv(f"slope@{c}"),
                     "intercept_no01": iv("citl_no01"), "slope_no01": iv("slope_no01")},
        clip_sensitivity={**metrics.exact_shares(v),
                          "by_clip": {metrics.clip_key(k): {"log_loss": iv(f"log_loss@{metrics.clip_key(k)}"),
                                                            "slope": iv(f"slope@{metrics.clip_key(k)}")}
                                      for k in metrics.CLIPS}},
        brier_decomposition={k: iv(k) for k in ("mcb", "dsc", "unc")},
        reliability=metrics.reliability_curve(y, v),
        level_miss=iv("level_miss") if task.census else "NOT SHOWN (R9): the label source is not a census")
    return m


def _best(family: str, rows: list[dict], table: list[dict], points: dict, ll_key: str, task: Task,
          dup_sets: list[set]) -> dict:
    """Name a best entry only on a forward card, and only if it separates from every other entry in the family.

    In the ranking family a flagged duplicate of the candidate is skipped: the two are one model, and their AUC
    difference is zero by construction.
    """
    if task.track != "forward":
        return {"name": None, "reasons": ["RETROSPECTIVE: a record-track card names no better forecaster (R5)"]}
    members = [r for r in rows if family == "ranking" or r["eligible"]]
    if len(members) < 2:
        return {"name": None, "reasons": [f"fewer than two entries in the {family} family"]}
    if family == "ranking":
        cand = max(members, key=lambda r: points[r["name"]]["auc"])
    else:
        cand = min(members, key=lambda r: points[r["name"]][ll_key])
    name, reasons = cand["name"], []
    for c in table:
        if c["family"] != family or name not in (c["a"], c["b"]):
            continue
        other = c["b"] if c["a"] == name else c["a"]
        if family == "ranking" and {name, other} in dup_sets:
            continue
        if c["favours"] != name:
            reasons.append(f"not separated from {other} after Holm")
        elif c["flags"]:
            reasons.append(f"{' and '.join(c['flags'])} against {other}")
    if cand["role"] != "entry":
        reasons.append(f"role {cand['role']}: only an entry can be named best; a baseline never is (R5, R10)")
    else:
        exposure = cand["manifest"].exposure[task.event_id]
        if exposure != "unseen":
            reasons.append(f"exposure {EXPOSURE_MARKS[exposure]}: no primary word (R6)")
        if cand["manifest"].house:
            reasons.append("house entry: not named until an outside tester reruns the score (R16 c)")
    return {"candidate": name, "name": None if reasons else name, "reasons": reasons}


def _json_safe(obj: object) -> object:
    """Plain JSON scalars: numpy scalars to Python, NaN and inf to None, tuples to lists. Arrays stay for the scan."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj

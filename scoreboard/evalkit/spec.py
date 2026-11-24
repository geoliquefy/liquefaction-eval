"""Task, entry and label specifications for the eval kit. Strict validation; JSON and CSV load and save; hashes.

Rule book: docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md (pillar root). Rules served here:
  R2   a task names the earthquake, site list hash, label source, target, count gate, tile size, clip, seed, track
  R3   each entry declares what its number is (`output_kind`); `probability_eligible` is the only gate to log loss
  R4   an entry's canonical hash (values at 1e-9, reusing scoreboard/g5_amend.canon_cell)
  R6   each entry declares its exposure per earthquake
  R7   labels are yes, no or not labelled; join by site id only; duplicate ids stop the run
  R15  an entry may state its division; the card refuses one that differs from the task's. A closed task may name
       its package hash (`Task.package_sha256`, D5 kit step); it is omitted from the task JSON when unset, so every
       older task keeps its hash and the sealed v2 card is unchanged
  step 2b  `feature_allowlist` names the feature allowlist an entry was built from (review 2 N3); omitted from the
       manifest JSON when unset, so every older entry keeps its hash
  B1 (eval M4)  `declared_target`: a house entry may declare its other_probability a probability of the task target
       (evalkit.losses); then it gets probability metrics. Omitted from the manifest JSON when unset, so every older
       entry keeps its hash and the sealed v2 card is unchanged
  step 2b  `card_sha256`: the hash of a card over canonical JSON with floats at CARD_HASH_SIG significant digits, so
       last-bit platform differences (container vs Mac, 1.7e-16) do not change it
  B8 (eval M4, ruling 48)  hash recipe "v2" for scorecard v3: `card_sha256(card, recipe="v2")` rounds floats to
       9 significant digits, and values within 1e-6 of zero on an absolute 1e-12 grid; `entry_sha256(entry,
       recipe="v2", refit=True)` hashes a row the card fits on each build (effort, B4 and B5 fits) with predictions at
       6 decimals. The default recipe stays "v1", so every sealed card and entry keeps its hash
No em dashes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from ..g5_amend import canon_cell
from .metrics import CLIPS

TRACKS = ("forward", "record")
DIVISIONS = ("closed", "open")
OUTPUT_KINDS = ("target_probability", "other_probability", "score")
EXPOSURES = ("unseen", "seen_inputs", "fitted_on", "unknown", "seen_place")
TILE_RULES = ("fixed", "variogram")
CARD_HASH_SIG = 12
CARD_HASH_ZERO = 1e-12          # |v| below this hashes as 0.0 (review 1 of step 2, MINOR 5)
HASH_RECIPES = ("v1", "v2")     # B8: "v1" is the sealed v2 card's recipe (default); "v2" is ruling 48's fix for v3
HASH_V2_SIG = 9                 # recipe v2: significant digits (ruling 48 tolerance: 1e-9 relative)
HASH_V2_NEAR_ZERO = 1e-6        # recipe v2: values with |v| below this are rounded on an absolute scale ...
HASH_V2_ABS_DECIMALS = 12       # ... to the nearest 1e-12 (ruling 48: 1e-12 absolute within 1e-6 of zero)
REFIT_P_DECIMALS = 6            # recipe v2: predictions of a refit row hash at 6 decimals (fit noise is ~1e-12)
ENTRY_HEADER = ["site_id", "p"]
LABEL_HEADER = ["site_id", "label", "x_m", "y_m"]
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SITE_ID = re.compile(r"[^,\"\s]+")


class SpecError(ValueError):
    """A task, manifest, entry or label file breaks the spec. The message names the field."""


# ---------------------------------------------------------------------------------------------------------------
# Field checks
# ---------------------------------------------------------------------------------------------------------------

def _keys(d: object, required: tuple[str, ...], optional: tuple[str, ...], what: str) -> dict:
    if not isinstance(d, dict):
        raise SpecError(f"{what}: expected a JSON object, got {type(d).__name__}")
    missing = [k for k in required if k not in d]
    unknown = sorted(k for k in d if k not in required and k not in optional)
    if missing:
        raise SpecError(f"{what}: missing field(s) {missing}")
    if unknown:
        raise SpecError(f"{what}: unknown field(s) {unknown}")
    return d


def _text(v: object, name: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise SpecError(f"{name}: must be a non-empty string, got {v!r}")
    return v


def _choice(v: object, name: str, options: tuple[str, ...]) -> str:
    if v not in options:
        raise SpecError(f"{name}: must be one of {list(options)}, got {v!r}")
    return v


def _sha(v: object, name: str) -> str:
    if not isinstance(v, str) or not _HEX64.fullmatch(v):
        raise SpecError(f"{name}: must be a lowercase 64-character sha256 hex string, got {v!r}")
    return v


def _int(v: object, name: str, lo: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise SpecError(f"{name}: must be an integer >= {lo}, got {v!r}")
    return v


def _real(v: object, name: str, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= float(v) <= hi):
        raise SpecError(f"{name}: must be a number in [{lo}, {hi}], got {v!r}")
    return float(v)


def _flag(v: object, name: str) -> bool:
    if not isinstance(v, bool):
        raise SpecError(f"{name}: must be true or false, got {v!r}")
    return v


def canonical_json(d: dict) -> str:
    """Sorted keys, no spaces, ASCII: the text every spec hash is taken over."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canon_value(v: object, sig: int) -> object:
    """Floats with |v| < CARD_HASH_ZERO set to 0.0, then rounded to `sig` significant digits (negative zero to
    zero); containers walked; other scalars kept."""
    if isinstance(v, dict):
        return {str(k): _canon_value(x, sig) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon_value(x, sig) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            raise SpecError("a card hash needs finite values (the card stores NaN and inf as null)")
        if abs(f) < CARD_HASH_ZERO:          # noise near zero (-2.78e-17 on the demo card) is not a difference
            return 0.0
        r = float(f"{f:.{sig}g}")
        return 0.0 if r == 0.0 else r
    return v


def _canon_value_v2(v: object) -> object:
    """Hash recipe v2 (B8, ruling 48): |v| < 1e-6 rounds to the nearest 1e-12 (absolute); every other float rounds
    to 9 significant digits; negative zero to zero; containers walked; other scalars kept."""
    if isinstance(v, dict):
        return {str(k): _canon_value_v2(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon_value_v2(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            raise SpecError("a card hash needs finite values (the card stores NaN and inf as null)")
        r = round(f, HASH_V2_ABS_DECIMALS) if abs(f) < HASH_V2_NEAR_ZERO else float(f"{f:.{HASH_V2_SIG}g}")
        return 0.0 if r == 0.0 else r
    return v


def check_recipe(recipe: str) -> str:
    if recipe not in HASH_RECIPES:
        raise SpecError(f"hash recipe must be one of {list(HASH_RECIPES)}, got {recipe!r}")
    return recipe


def canonical_card_json(card: dict, sig: int = CARD_HASH_SIG, recipe: str = "v1") -> str:
    """The text a card hash is taken over: sorted keys, no spaces, ASCII. Recipe v1 (default): every float at `sig`
    significant digits, |v| < 1e-12 as 0. Recipe v2 (B8): see `_canon_value_v2`; `sig` is not used."""
    if check_recipe(recipe) == "v2":
        return canonical_json(_canon_value_v2(card))
    return canonical_json(_canon_value(card, sig))


def card_sha256(card: dict, sig: int = CARD_HASH_SIG, recipe: str = "v1") -> str:
    """Platform-stable card hash (step 2b): cards that differ only below `sig` significant digits hash the same.

    A value that sits on a rounding boundary at digit `sig` can still split two near-equal cards; the 1.7e-16 gaps
    seen between the container and the Mac are about five orders of magnitude below the 12th digit of a value
    near 1 (1e-11), so such a split needs a value within that gap of a rounding boundary (odds near 1e-5 per value).
    Recipe "v2" (B8, scorecard v3) rounds at 9 significant digits and on an absolute 1e-12 grid near zero; the same
    boundary limit holds there, at lower odds (ruling 48's measured gaps are at most 7.5e-11 relative).
    """
    return _sha256(canonical_card_json(card, sig, recipe))


def load_card_sha256(path: str | Path, sig: int = CARD_HASH_SIG, recipe: str = "v1") -> str:
    """card_sha256 of a card JSON file; the file is read, never rewritten."""
    return card_sha256(json.loads(Path(path).read_text(encoding="utf-8")), sig, recipe)


# ---------------------------------------------------------------------------------------------------------------
# Task (R2)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """One earthquake, one label source, one hashed site list, one target (R2).

    tile_m and clip are the registered values that give the words (R9, R11); the card prints 1, 2 and 4 km tiles and
    every clip in CLIPS beside. The count gate defaults to R11's 200 sites, 30 per class, 20 tiles. `census` says the
    label source grades every site in an area; only then is the level miss printed (R9). `label_error` holds measured
    sensitivity and specificity (R8); None prints NOT MEASURED.

    tile_rule (R11 c): "fixed" takes tile_m as given (synthetic tests); "variogram" means tile_m was chosen by
    card.tile_choice from a residual semivariogram before scoring, and the card refuses a task whose tile_m differs
    from the recomputed choice. earlier_label_sets (R10): how many earlier earthquakes of this label source exist;
    when above 0 the earlier-earthquake rate and the shaking-only baseline are required, and a card without either is
    refused ("scorer fault: baseline not built").

    package_sha256 (R2, R15; D5 kit step): on the closed division, the package hash the task names. The intake refuses
    a closed rule book form entry whose package_sha256 differs from it. Optional, and left out of the task JSON (and
    so of the task hash) when unset; a task without it gets "package: NOT CHECKED" at intake.
    """
    event_id: str
    target: str
    label_source: str
    label_rule_sha256: str
    site_list_sha256: str
    track: str
    division: str
    tile_m: float
    clip: float
    seed: int
    label_caveats: str
    n_boot: int = 2000
    min_sites: int = 200
    min_per_class: int = 30
    min_tiles: int = 20
    census: bool = False
    earlier_base_rate: float | None = None
    label_error: dict | None = None
    tile_rule: str = "fixed"
    earlier_label_sets: int = 0
    package_sha256: str | None = None

    def __post_init__(self) -> None:
        Task.from_dict(asdict(self), _check_only=True)

    @classmethod
    def from_dict(cls, d: object, _check_only: bool = False) -> "Task":
        names = tuple(f.name for f in fields(cls))
        required = names[:11]
        d = _keys(d, required, names[11:], "task")
        for k in ("event_id", "target", "label_source", "label_caveats"):
            _text(d[k], f"task.{k}")
        _sha(d["label_rule_sha256"], "task.label_rule_sha256")
        _sha(d["site_list_sha256"], "task.site_list_sha256")
        _choice(d["track"], "task.track", TRACKS)
        _choice(d["division"], "task.division", DIVISIONS)
        _real(d["tile_m"], "task.tile_m", 100.0, 100000.0)
        if d["clip"] not in CLIPS:
            raise SpecError(f"task.clip: must be one of {list(CLIPS)} (R9), got {d['clip']!r}")
        _int(d["seed"], "task.seed", 0)
        _int(d.get("n_boot", 2000), "task.n_boot", 50)
        for k in ("min_sites", "min_per_class", "min_tiles"):
            _int(d.get(k, 1), f"task.{k}", 1)
        _flag(d.get("census", False), "task.census")
        if d.get("earlier_base_rate") is not None:
            _real(d["earlier_base_rate"], "task.earlier_base_rate", 1e-6, 1 - 1e-6)
        if d.get("label_error") is not None:
            le = _keys(d["label_error"], ("sensitivity", "specificity"), (), "task.label_error")
            for k in ("sensitivity", "specificity"):
                _real(le[k], f"task.label_error.{k}", 0.5, 1.0)
        _choice(d.get("tile_rule", "fixed"), "task.tile_rule", TILE_RULES)
        _int(d.get("earlier_label_sets", 0), "task.earlier_label_sets", 0)
        if d.get("package_sha256") is not None:
            _sha(d["package_sha256"], "task.package_sha256")
            if d["division"] != "closed":
                raise SpecError("task.package_sha256: only a closed division task names a package hash (R2, R15)")
        if _check_only:
            return None
        d = dict(d, tile_m=float(d["tile_m"]), clip=float(d["clip"]))
        if d.get("earlier_base_rate") is not None:
            d["earlier_base_rate"] = float(d["earlier_base_rate"])
        return cls(**d)

    def to_dict(self) -> dict:
        """Plain dict; `package_sha256` appears only when set, so every older task keeps its hash (D5 kit step)."""
        d = asdict(self)
        if d["package_sha256"] is None:
            del d["package_sha256"]
        return d

    def sha256(self) -> str:
        """Hash of the canonical task JSON; the scorer refuses a task whose hash differs from the registration (R2)."""
        return _sha256(canonical_json(self.to_dict()))


# ---------------------------------------------------------------------------------------------------------------
# Entry manifest (R3, R4, R6, R15)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EntryManifest:
    """What an entry's number is, and what it may have seen.

    output_kind (R3): `target_probability` is a probability of the task target; `other_probability` is a probability
    of another event (triggering, areal extent, index exceedance); `score` is any ranking number. `conversion_id` is
    the sha256 of a transform to the task target frozen before the join (R3 `transform_sha256`); with it, the entry's
    numbers are the converted target probabilities. exposure (R6) maps each earthquake id to one of EXPOSURES.
    `house` marks an entry we built, tuned, ran or converted (R16).
    `declared_target` (B1, eval M4): True only on a house `other_probability` entry with no conversion. The house
    declares that it stands behind its number as a probability of the task target, so the entry takes log loss, Brier
    and calibration instead of ranking only. It is a declaration, not a conversion: the numbers are not changed, and
    the card marks the row (evalkit.losses.DECLARED_MARK). Unset (None, or False, which is stored as None) keeps the
    current behaviour.
    """
    model: str
    output_kind: str
    statement: str
    exposure: dict
    house: bool
    conversion_id: str | None = None
    division: str | None = None
    feature_allowlist: str | None = None
    declared_target: bool | None = None

    def __post_init__(self) -> None:
        if self.declared_target is False:
            object.__setattr__(self, "declared_target", None)
        EntryManifest.from_dict(asdict(self), _check_only=True)

    @classmethod
    def from_dict(cls, d: object, _check_only: bool = False) -> "EntryManifest":
        d = _keys(d, ("model", "output_kind", "statement", "exposure", "house"),
                  ("conversion_id", "division", "feature_allowlist", "declared_target"), "manifest")
        _text(d["model"], "manifest.model")
        _choice(d["output_kind"], "manifest.output_kind", OUTPUT_KINDS)
        _text(d["statement"], "manifest.statement")
        if not isinstance(d["exposure"], dict) or not d["exposure"]:
            raise SpecError("manifest.exposure: must map at least one earthquake id to an exposure (R6)")
        for ev, x in d["exposure"].items():
            _text(ev, "manifest.exposure key")
            _choice(x, f"manifest.exposure[{ev!r}]", EXPOSURES)
        _flag(d["house"], "manifest.house")
        if d.get("conversion_id") is not None:
            _sha(d["conversion_id"], "manifest.conversion_id")
            if d["output_kind"] == "target_probability":
                raise SpecError("manifest.conversion_id: only an other_probability or score entry takes a conversion")
        if d.get("division") is not None:
            _choice(d["division"], "manifest.division", DIVISIONS)
        if d.get("feature_allowlist") is not None:
            _text(d["feature_allowlist"], "manifest.feature_allowlist")
        if d.get("declared_target") is not None and _flag(d["declared_target"], "manifest.declared_target"):
            if not d["house"] or d["output_kind"] != "other_probability" or d.get("conversion_id") is not None:
                raise SpecError("manifest.declared_target: only a house other_probability entry with no conversion "
                                "may declare its number a probability of the task target (B1)")
        return None if _check_only else cls(**d)

    def to_dict(self) -> dict:
        """Plain dict; `feature_allowlist` and `declared_target` appear only when set, so older entries keep their
        canonical hash."""
        d = asdict(self)
        for k in ("feature_allowlist", "declared_target"):
            if d[k] is None:
                del d[k]
        return d


def probability_eligible(m: EntryManifest) -> bool:
    """R3: probability metrics apply to a target probability, to another kind with a frozen conversion, or (B1) to a
    house other_probability the house declares a probability of the task target."""
    return m.output_kind == "target_probability" or m.conversion_id is not None or m.declared_target is True


# ---------------------------------------------------------------------------------------------------------------
# Entries (R4)
# ---------------------------------------------------------------------------------------------------------------

def _check_ids(ids: tuple[str, ...], what: str) -> None:
    bad = [s for s in ids if not isinstance(s, str) or not _SITE_ID.fullmatch(s)]
    if bad:
        raise SpecError(f"{what}: site ids must be non-empty with no comma, quote or space; first bad {bad[0]!r}")
    if len(set(ids)) != len(ids):
        raise SpecError(f"{what}: duplicate site ids (R7: duplicate ids stop the run)")


@dataclass(frozen=True, eq=False)
class Entry:
    """One number per site plus a manifest (R4). Probability kinds must lie in [0, 1]; every value must be finite."""
    manifest: EntryManifest
    site_ids: tuple[str, ...]
    p: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_ids", tuple(self.site_ids))
        object.__setattr__(self, "p", np.asarray(self.p, dtype=float))
        what = f"entry {self.manifest.model!r}"
        if self.p.ndim != 1 or self.p.size != len(self.site_ids) or self.p.size == 0:
            raise SpecError(f"{what}: needs one value per site id")
        _check_ids(self.site_ids, what)
        if not np.all(np.isfinite(self.p)):
            raise SpecError(f"{what}: every value must be finite")
        if self.manifest.output_kind != "score" and (self.p.min() < 0 or self.p.max() > 1):
            raise SpecError(f"{what}: a probability must lie in [0, 1]")


def predictions_text(site_ids: tuple[str, ...], p: np.ndarray) -> str:
    """Canonical `site_id,p` text: values via g5_amend.canon_cell ('.9f', unsigned zero), rows in file order.

    Equal to g5_amend.canonical_text(path, str_cols={"site_id"}) for the same CSV, so the G5 predictions hash and this
    one agree (tested).
    """
    rows = ["site_id,p"] + [f"{s},{canon_cell(repr(float(v)))}" for s, v in zip(site_ids, p)]
    return "\n".join(rows) + "\n"


def refit_predictions_text(site_ids: tuple[str, ...], p: np.ndarray) -> str:
    """Hash recipe v2 text of a refit row: `site_id,p` with p at REFIT_P_DECIMALS decimals, unsigned zero."""
    def cell(v: float) -> str:
        s = f"{float(v):.{REFIT_P_DECIMALS}f}"
        return s[1:] if s.startswith("-") and float(s) == 0.0 else s
    return "\n".join(["site_id,p"] + [f"{s},{cell(v)}" for s, v in zip(site_ids, p)]) + "\n"


def entry_sha256(entry: Entry, recipe: str = "v1", refit: bool = False) -> str:
    """R4 canonical hash of an entry: sha256 of the canonical manifest JSON, a newline, then the predictions text.

    g5_amend.canonical_sha256 hashes the predictions alone; the manifest is bound in here so that a changed output
    kind, exposure or conversion is a new entry, as R4 requires of a corrected model.
    B8 (ruling 48): with recipe "v2" and `refit=True` (a row the card fits on each build), the predictions hash at
    6 decimals, so a last-digit platform difference in the fit (about 1e-12) does not change the hash. Every other
    entry, and every entry under the default recipe "v1", hashes exactly as before.
    """
    if check_recipe(recipe) == "v2" and refit:
        text = refit_predictions_text(entry.site_ids, entry.p)
    else:
        text = predictions_text(entry.site_ids, entry.p)
    return _sha256(canonical_json(entry.manifest.to_dict()) + "\n" + text)


def load_predictions(csv_path: str | Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Site ids and values from a `site_id,p` CSV."""
    ids, cols = _read_csv(csv_path, ENTRY_HEADER)
    try:
        return ids, np.array([float(v) for v in cols["p"]])
    except ValueError as exc:
        raise SpecError(f"{csv_path}: column p must be numeric ({exc})") from None


def load_entry(manifest_path: str | Path, csv_path: str | Path) -> Entry:
    m = EntryManifest.from_dict(json.loads(Path(manifest_path).read_text(encoding="utf-8")))
    return Entry(m, *load_predictions(csv_path))


def save_entry(entry: Entry, manifest_path: str | Path, csv_path: str | Path) -> None:
    Path(manifest_path).write_text(json.dumps(entry.manifest.to_dict(), indent=1, sort_keys=True) + "\n",
                                   encoding="utf-8")
    rows = [ENTRY_HEADER] + [[s, repr(float(v))] for s, v in zip(entry.site_ids, entry.p)]
    _write_csv(csv_path, rows)


def align(entry: Entry, site_ids: tuple[str, ...]) -> np.ndarray:
    """The entry's values in the order of `site_ids`, joined by id only (R7). Refuses a missing id."""
    pos = {s: k for k, s in enumerate(entry.site_ids)}
    missing = [s for s in site_ids if s not in pos]
    if missing:
        raise SpecError(f"entry {entry.manifest.model!r}: no value for {len(missing)} labelled site(s)")
    return entry.p[np.array([pos[s] for s in site_ids], dtype=int)]


# ---------------------------------------------------------------------------------------------------------------
# Labels (R7)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Labels:
    """The task's site list with labels 1 (yes), 0 (no) or nan (not labelled), and projected coordinates in metres."""
    site_ids: tuple[str, ...]
    label: np.ndarray = field(repr=False)
    x_m: np.ndarray = field(repr=False)
    y_m: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_ids", tuple(self.site_ids))
        for k in ("label", "x_m", "y_m"):
            object.__setattr__(self, k, np.asarray(getattr(self, k), dtype=float))
        n = len(self.site_ids)
        if n == 0 or any(getattr(self, k).shape != (n,) for k in ("label", "x_m", "y_m")):
            raise SpecError("labels: label, x_m and y_m need one value per site id")
        _check_ids(self.site_ids, "labels")
        lab = self.label[np.isfinite(self.label)]
        if not np.all((lab == 0) | (lab == 1)):
            raise SpecError("labels: label must be 1, 0 or blank (not labelled)")
        if not (np.all(np.isfinite(self.x_m)) and np.all(np.isfinite(self.y_m))):
            raise SpecError("labels: coordinates must be finite")

    @property
    def labelled(self) -> np.ndarray:
        return np.isfinite(self.label)


def _label_cell(v: float) -> str:
    return "" if math.isnan(v) else str(int(v))


def labels_sha256(labels: Labels) -> str:
    """sha256 of the canonical label text (label as 1, 0 or blank; coordinates via canon_cell). R8 label hash."""
    rows = [",".join(LABEL_HEADER)] + [
        f"{s},{_label_cell(v)},{canon_cell(repr(float(x)))},{canon_cell(repr(float(y)))}"
        for s, v, x, y in zip(labels.site_ids, labels.label, labels.x_m, labels.y_m)]
    return _sha256("\n".join(rows) + "\n")


SITE_LIST_RECIPE = ("sha256 of the UTF-8 text of the SORTED site ids, each followed by a newline (one id per line, "
                    "and the text ends with a newline). Row order in the entry CSV does not change this hash; it does "
                    "change the entry hash (D3 hashes the rows in file order)")


def site_list_sha256(site_ids) -> str:
    """The task's site list hash (R2), as SITE_LIST_RECIPE says: sha256 of "".join(s + "\\n" for s in sorted(ids))."""
    return _sha256("".join(f"{s}\n" for s in sorted(site_ids)))


def load_labels(csv_path: str | Path) -> Labels:
    ids, cols = _read_csv(csv_path, LABEL_HEADER)
    bad = [v for v in cols["label"] if v not in ("1", "0", "")]
    if bad:
        raise SpecError(f"{csv_path}: label must be 1, 0 or blank, got {bad[0]!r}")
    lab = np.array([float(v) if v else np.nan for v in cols["label"]])
    try:
        x, y = (np.array([float(v) for v in cols[k]]) for k in ("x_m", "y_m"))
    except ValueError as exc:
        raise SpecError(f"{csv_path}: coordinates must be numeric ({exc})") from None
    return Labels(ids, lab, x, y)


def save_labels(labels: Labels, csv_path: str | Path) -> None:
    rows = [LABEL_HEADER] + [[s, _label_cell(v), repr(float(x)), repr(float(y))]
                             for s, v, x, y in zip(labels.site_ids, labels.label, labels.x_m, labels.y_m)]
    _write_csv(csv_path, rows)


# ---------------------------------------------------------------------------------------------------------------
# JSON and CSV
# ---------------------------------------------------------------------------------------------------------------

def load_task(path: str | Path) -> Task:
    return Task.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_task(task: Task, path: str | Path) -> None:
    Path(path).write_text(json.dumps(task.to_dict(), indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: str | Path, header: list[str]) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != header:
        raise SpecError(f"{path}: header must be exactly {','.join(header)}")
    if any(len(r) != len(header) for r in rows[1:]):
        raise SpecError(f"{path}: every row needs {len(header)} cells")
    cols = {h: [r[k] for r in rows[1:]] for k, h in enumerate(header)}
    return tuple(cols["site_id"]), cols


def _write_csv(path: str | Path, rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)

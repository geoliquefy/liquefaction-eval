"""Entry intake in both manifest forms (R4, B3; work item D5, eval milestone M5). No em dashes.

D5 found that an outsider who follows rule book B3 is refused by the kit, and that the receipt hash differs from the
published recipe. This module reads both manifest forms and says which one it read:

  manifest.json --+-- has "rule_book_version" --> RULE BOOK FORM (B3)
                  |     every B3 field required; only conversion_sha256 and package_sha256 may be null;
                  |     extra fields allowed and hashed; values in [0, 1] for every output kind (R4);
                  |     entry hash = the B3 recipe on the manifest EXACTLY AS SUBMITTED;
                  |     entrant = the manifest's entrant_id (inside the hash)
                  |
                  +-- otherwise --------------------> KIT FORM (spec.EntryManifest)
                        fields: model, output_kind, statement, exposure, house (required);
                        conversion_id, division, feature_allowlist, declared_target (optional);
                        no other field;
                        entry hash = spec.entry_sha256 on the manifest AFTER the kit fills its defaults:
                        conversion_id and division are written as null when absent;
                        feature_allowlist and declared_target are left out when unset;
                        entrant = the --entrant flag (NOT inside the hash)

Both forms: entry hash = sha256( canonical manifest JSON + "\\n" + predictions text ), with canonical JSON and the
predictions text as in rule book D3. `Intake.manifest_text` is the exact canonical manifest JSON that was hashed, so
an entrant can check the receipt with the published recipe in either form.

A rule book form manifest maps onto the kit's checks as: conversion_sha256 -> conversion_id, the other kit fields by
the same name. The card builder still hashes the kit form (`spec.entry_sha256`); binding a rule book form entry's
receipt hash on a card is left for the v3 card (B8).

`check_against_task` checks an entry against a task file (R2): the site list hash, the exposure key for the task
earthquake, the division and (rule book form) task_sha256. The register intake runs it when given --task.

Task hash (R2; review 1 of D5). It is NOT the hash of the task file's bytes, and NOT the canonical JSON of the file as
written. It is `spec.Task.sha256()`: sha256 of the canonical JSON of the task AFTER the kit reads it, which
  1. fills every optional field the file leaves out with its default (n_boot, min_sites, min_per_class, min_tiles,
     census, earlier_base_rate, label_error, tile_rule, earlier_label_sets: 9 fields, 20 in all), and
  2. writes tile_m, clip and (when set) earlier_base_rate as floats (2000 -> 2000.0).
`task_hash` returns that hash and the exact text hashed; `register hash --task TASK.json` prints both. The scorer
checks the same hash against a registration, so it is not changed here.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .spec import (DIVISIONS, Entry, EntryManifest, OUTPUT_KINDS, SpecError, Task, canonical_json, entry_sha256,
                   load_predictions, load_task, predictions_text, site_list_sha256)

RULEBOOK_FORM = "rule book B3"
KIT_FORM = "kit"
FORM_KEY = "rule_book_version"
# Rule book v1.0 B3, "Write manifest.json with every field below"; order as printed there.
RULEBOOK_FIELDS = ("rule_book_version", "entrant_id", "task_sha256", "model", "model_version", "primary",
                   "output_kind", "conversion_sha256", "division", "package_sha256", "inputs", "exposure",
                   "training_cases_sha256", "choices_after_outcomes", "text_read", "statement", "house")
RULEBOOK_NULLABLE = ("conversion_sha256", "package_sha256")
KIT_REQUIRED = ("model", "output_kind", "statement", "exposure", "house")
KIT_OPTIONAL = ("conversion_id", "division", "feature_allowlist", "declared_target")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ENTRANT = re.compile(r"[A-Za-z0-9._:-]{1,64}")


@dataclass(frozen=True)
class Intake:
    """One entry as read at intake."""
    form: str                     # RULEBOOK_FORM or KIT_FORM
    entry: Entry                  # the kit's view, for the schema checks and for scoring
    entry_sha256: str             # the hash the receipt carries
    manifest_text: str            # the canonical manifest JSON that was hashed
    entrant: str | None           # rule book form: the manifest's entrant_id; kit form: None
    primary: bool | None          # rule book form only
    task_sha256: str | None       # rule book form only


def manifest_form(d: object) -> str:
    """RULEBOOK_FORM if the manifest names a rule book version, else KIT_FORM."""
    return RULEBOOK_FORM if isinstance(d, dict) and FORM_KEY in d else KIT_FORM


def _str(d: dict, k: str) -> None:
    if not isinstance(d[k], str) or not d[k].strip():
        raise SpecError(f"manifest.{k}: must be a non-empty string (B3), got {d[k]!r}")


def _hex(d: dict, k: str) -> None:
    if not isinstance(d[k], str) or not _HEX64.fullmatch(d[k]):
        raise SpecError(f"manifest.{k}: must be a lowercase 64-character sha256 hex string (B3)")


def rulebook_to_kit(d: object) -> EntryManifest:
    """Check a rule book form manifest (B3) and return the kit manifest it maps to. SpecError on any breach."""
    if not isinstance(d, dict):
        raise SpecError(f"manifest: expected a JSON object, got {type(d).__name__}")
    missing = [k for k in RULEBOOK_FIELDS if k not in d]
    if missing:
        raise SpecError(f"manifest (rule book form, B3): missing field(s) {missing}")
    nulls = [k for k in RULEBOOK_FIELDS if d[k] is None and k not in RULEBOOK_NULLABLE]
    if nulls:
        raise SpecError(f"manifest (rule book form, B3): only {list(RULEBOOK_NULLABLE)} may be null; null: {nulls}")
    for k in ("rule_book_version", "model", "model_version", "choices_after_outcomes", "text_read", "statement"):
        _str(d, k)
    if not isinstance(d["entrant_id"], str) or not _ENTRANT.fullmatch(d["entrant_id"]):
        raise SpecError("manifest.entrant_id: must be 1 to 64 characters of letters, digits and ._:-")
    for k in ("task_sha256", "training_cases_sha256"):
        _hex(d, k)
    for k in ("primary", "house"):
        if not isinstance(d[k], bool):
            raise SpecError(f"manifest.{k}: must be true or false, got {d[k]!r}")
    if d["output_kind"] not in OUTPUT_KINDS:
        raise SpecError(f"manifest.output_kind: must be one of {list(OUTPUT_KINDS)}, got {d['output_kind']!r}")
    if d["division"] not in DIVISIONS:
        raise SpecError(f"manifest.division: must be one of {list(DIVISIONS)}, got {d['division']!r}")
    for k in RULEBOOK_NULLABLE:
        if d[k] is not None:
            _hex(d, k)
    if d["division"] == "open" and d["package_sha256"] is not None:
        raise SpecError("manifest.package_sha256: must be null on the open division (B3)")
    if d["division"] == "closed" and d["package_sha256"] is None:
        raise SpecError("manifest.package_sha256: the closed division names its package hash (B3)")
    if d["output_kind"] == "target_probability" and d["conversion_sha256"] is not None:
        raise SpecError("manifest.conversion_sha256: must be null on a target_probability entry; only an "
                        "other_probability or score entry takes a conversion (kit rule; rule book v1 B3 does not "
                        "state it yet)")
    if not isinstance(d["inputs"], list) or not all(isinstance(x, str) and x.strip() for x in d["inputs"]):
        raise SpecError("manifest.inputs: must be a list of non-empty strings (B3)")
    return EntryManifest.from_dict({"model": d["model"], "output_kind": d["output_kind"], "statement": d["statement"],
                                    "exposure": d["exposure"], "house": d["house"],
                                    "conversion_id": d["conversion_sha256"], "division": d["division"]})


def read_entry(manifest_path: str | Path, csv_path: str | Path) -> Intake:
    """Read an entry in either manifest form, check it, and hash it as its form says (see the module text)."""
    try:
        raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(f"{manifest_path}: not JSON ({exc})") from None
    ids, p = load_predictions(csv_path)
    if manifest_form(raw) == KIT_FORM:
        if isinstance(raw, dict):
            unknown = sorted(k for k in raw if k not in KIT_REQUIRED + KIT_OPTIONAL)
            if unknown:
                raise SpecError(f"manifest: unknown field(s) {unknown}. The kit form takes {list(KIT_REQUIRED)} and "
                                f"optionally {list(KIT_OPTIONAL)}; a rule book form manifest (B3) must carry every "
                                f"B3 field, starting with '{FORM_KEY}'")
        e = Entry(EntryManifest.from_dict(raw), ids, p)
        return Intake(KIT_FORM, e, entry_sha256(e), canonical_json(e.manifest.to_dict()), None, None, None)
    m = rulebook_to_kit(raw)
    e = Entry(m, ids, p)
    if p.min() < 0 or p.max() > 1:
        raise SpecError(f"{Path(csv_path).name}: every value must lie in [0, 1] (R4; a score model rescales)")
    try:
        mj = canonical_json(raw)
    except ValueError as exc:
        raise SpecError(f"{manifest_path}: not canonical JSON ({exc})") from None
    sha = hashlib.sha256((mj + "\n" + predictions_text(ids, p)).encode("utf-8")).hexdigest()
    return Intake(RULEBOOK_FORM, e, sha, mj, raw["entrant_id"], raw["primary"], raw["task_sha256"])


def check_against_task(intake: Intake, task: Task) -> dict:
    """Check an entry against a task file (R2, R6, R15, B3 step 1). Returns {check: "PASS" | "FAIL: why"}."""
    m = intake.entry.manifest
    out = {}
    got = site_list_sha256(intake.entry.site_ids)
    out["site_list"] = "PASS" if got == task.site_list_sha256 else \
        "FAIL: the entry's site ids do not hash to the task's site list (one row per task site, no other row)"
    out["exposure_key"] = "PASS" if task.event_id in m.exposure else \
        f"FAIL: exposure names no exposure for the task earthquake {task.event_id!r}"
    out["division"] = "PASS" if m.division in (None, task.division) else \
        f"FAIL: the entry states division {m.division!r}; the task is {task.division!r}"
    if intake.form == RULEBOOK_FORM:
        want = task.sha256()
        out["task_sha256"] = "PASS" if intake.task_sha256 == want else \
            (f"FAIL: manifest task_sha256 is {intake.task_sha256}; the task hash is {want} (the task hash is "
             f"Task.sha256, the canonical JSON after the kit fills its defaults, not the file's bytes; print it "
             f"with `register hash --task TASK.json`)")
    return out


def task_hash(path: str | Path) -> tuple[str, str]:
    """(task hash, exact text hashed) for a task file, as `check_against_task` and the scorer take it (R2)."""
    t = load_task(path)
    return t.sha256(), canonical_json(t.to_dict())     # Task.sha256 = sha256 of this exact UTF-8 text


def task_failures(checks: dict) -> list[str]:
    return [f"{k}: {v}" for k, v in checks.items() if v != "PASS"]


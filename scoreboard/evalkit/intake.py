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
earthquake, the division, (rule book form) task_sha256, (closed division) the package hash, and (closed division)
the package files. The register intake runs it when given --task. Each check reads PASS, FAIL (the intake refuses) or NOT CHECKED (the intake goes on and
the submit output says so beside the receipt).

Site list hash (R2; D5 kit step, dry run 2 point 7): spec.SITE_LIST_RECIPE, the sha256 of the SORTED site ids, each
followed by a newline. Row order does not change it. A FAIL prints the hash the entry's ids give, the hash the task
needs and the recipe; `register hash ENTRY.json` prints the entry's site list hash.

Package hash (R15 check; D5 kit step, dry run 2 point 10). A closed rule book form entry names package_sha256. When
the task file names one too (spec.Task.package_sha256), a different hash is refused. When the task file names none,
the check reads NOT CHECKED: the kit has nothing to check the entry's hash against, and it does not invent a rule.
A kit form manifest has no package field, so it cannot name a package hash. A closed kit form entry is therefore
refused (D5 intake, wave 11; ruling 57 c, rule book amendment 2, makes the package table required): against a
closed task the package check reads FAIL, and the register intake refuses a kit form manifest that states
division "closed" even without --task. The kit form does not gain a package field: that would move every kit form
entry hash.

Package files (R15, B3 step 2; D5 intake, wave 11). On a closed task that names package_sha256, the intake rebuilds
the package hash from the submitted package folder (`package.rebuild_sha256`: every file but hidden ones, in the
kit table form) and refuses a hash that differs. With no package folder the check reads FAIL: the intake runs the
package check on every closed entry it checks against such a task. On a closed task that names no package hash it
reads NOT CHECKED (nothing to rebuild against), and says that a folder given was not used. On an open task a
package folder is refused.

  closed task + package_sha256
     |
     +-- --package DIR only ----------------> rebuild DIR (hidden files out) == task hash?   PASS / FAIL
     +-- --package DIR --package-table T ---> sha256(T) == task hash?  then package.check_files(T, DIR):
                                              no FAIL or MISSING row, no non-hidden file outside T?  PASS / FAIL

The table path (D5 intake review 1 F1) serves a table with a hidden input row, which rule book amendment 2 section
4.3 point 5 allows and no rebuild can give. Both paths refuse an extra non-hidden file, though `package check` reads
PASS for one and lists it under not_in_table (F2). With no task file, `entry_package_files_check` checks a folder
against a closed rule book form entry's own package_sha256 (F4); without --package that reads NOT CHECKED.

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

from . import package as _package
from .spec import (DIVISIONS, Entry, EntryManifest, OUTPUT_KINDS, SITE_LIST_RECIPE, SpecError, Task, canonical_json,
                   entry_sha256, load_predictions, load_task, predictions_text, site_list_sha256)

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
PASS = "PASS"
FAIL = "FAIL"
NOT_CHECKED = "NOT CHECKED"
CLOSED_KIT_FORM = ("a closed division entry names its package hash (R15; B3; ruling 57 c, rule book amendment 2: "
                   "the package table lists every input file an entry reads), and a kit form manifest has no "
                   "package hash field. A closed kit form entry is refused. Submit a rule book form manifest (B3) "
                   "that names rule_book_version and package_sha256")
PACKAGE_REBUILD = ("Rebuild: the kit table of every file in the folder but hidden ones (header file_id,sha256; rows "
                   "sorted; lowercase hex; LF), as `python -m scoreboard.evalkit.package write DIR --out TABLE.csv` "
                   "writes it; the package hash is the sha256 of that table. Keep the table file outside DIR")
PACKAGE_REFUSES = ("The intake refuses a FAIL or MISSING row, and also any file that `package check` lists under "
                   "not_in_table, hidden files apart (`package check` itself reads PASS for such a file)")
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
    package_sha256: str | None = None   # rule book form only (null on the open division)


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
    return Intake(RULEBOOK_FORM, e, sha, mj, raw["entrant_id"], raw["primary"], raw["task_sha256"],
                  raw["package_sha256"])


def site_list_failure(site_ids, want: str) -> str | None:
    """None if the ids hash to `want`, else the FAIL text: both hashes, the row count and the recipe (point 7)."""
    got = site_list_sha256(site_ids)
    if got == want:
        return None
    return (f"{FAIL}: the entry's {len(site_ids)} site ids hash to {got}; the task's site list hash is {want}. "
            f"The entry needs one row per task site and no other row. Site list hash: {SITE_LIST_RECIPE}")


def package_check(intake: Intake, task: Task) -> str:
    """The R15 package hash check at intake: PASS, FAIL (refused) or NOT CHECKED (said beside the receipt)."""
    if task.division != "closed":
        return PASS                                   # an open task has no package; B3 makes the entry's hash null
    if intake.form != RULEBOOK_FORM:
        return (f"{FAIL}: {CLOSED_KIT_FORM}")
    if task.package_sha256 is None:
        return (f"{NOT_CHECKED}: the task file names no package_sha256, so the entry's package hash "
                f"{intake.package_sha256} was not checked against the package (R15)")
    if intake.package_sha256 == task.package_sha256:
        return PASS
    return (f"{FAIL}: manifest package_sha256 is {intake.package_sha256}; the task's package hash is "
            f"{task.package_sha256} (R15: a closed record entry with another package hash is refused)")


def _not_used(package_dir, package_table) -> str:
    given = [f"--package {package_dir}" if package_dir is not None else "",
             f"--package-table {package_table}" if package_table is not None else ""]
    given = [g for g in given if g]
    return f"; the {' and '.join(given)} given {'was' if len(given) == 1 else 'were'} not used" if given else ""


def _list(names: list[str], cap: int = 20) -> str:
    more = f" and {len(names) - cap} more" if len(names) > cap else ""
    return ", ".join(names[:cap]) + more


def files_against(want: str, package_dir: str | Path | None, package_table: str | Path | None = None, *,
                  whose: str = "the task's") -> str:
    """Check a package folder against package hash `want` (R15, B3 step 2). PASS or FAIL (refused).

    Without `package_table`: rebuild the hash from the folder (`package.rebuild_sha256`: every file but hidden ones,
    in the kit table form) and compare. With `package_table` (D5 intake review 1 F1): the table's hash must equal
    `want`; then every row must read PASS in `package.check_files` (no FAIL or MISSING row), and no file that check
    lists under not_in_table may be there, hidden files apart. This is the path for a table with a hidden input row,
    which rule book amendment 2 section 4.3 point 5 allows and which no rebuild can give."""
    if package_dir is None:
        return (f"{FAIL}: {whose} package hash is {want}, and the intake checks it against the package the entry "
                f"read; no package folder was given. Give --package DIR, the folder of input files as fetched "
                f"(R15; B3 step 2). {PACKAGE_REBUILD}")
    if not Path(package_dir).is_dir():
        return f"{FAIL}: the package folder {package_dir} is not a folder"
    if package_table is not None:
        try:
            _rows, tsha = _package.read_table(package_table)
        except (OSError, SpecError) as exc:
            return f"{FAIL}: the package table {package_table} could not be read: {exc}"
        if tsha != want:
            return (f"{FAIL}: the package table {package_table} hashes to {tsha}; {whose} package hash is {want}. "
                    f"Give the task's own package table (R15: the package hash is the sha256 of that table)")
        try:
            res = _package.check_files(package_table, package_dir)
        except (OSError, SpecError) as exc:
            return f"{FAIL}: the package folder {package_dir} could not be checked: {exc}"
        bad = [f"{fid} {s}" for fid, s in sorted(res["files"].items()) if s != PASS]
        extra = [fid for fid in res["not_in_table"] if not _package._hidden(fid)]
        if not bad and not extra:
            return PASS
        parts = ([f"rows that do not read PASS: {_list(bad)}"] if bad else []) + \
                ([f"files the table does not name: {_list(extra)}"] if extra else [])
        return (f"{FAIL}: the package folder {package_dir} does not fit the package table {package_table} "
                f"(package hash {want}): {'; '.join(parts)}. {PACKAGE_REFUSES}.")
    try:
        got, n = _package.rebuild_sha256(package_dir)
    except (OSError, SpecError) as exc:
        return f"{FAIL}: the package folder {package_dir} could not be rebuilt: {exc}; {whose} package hash is {want}"
    if got == want:
        return PASS
    return (f"{FAIL}: the package folder {package_dir} ({n} files) rebuilds to package hash {got}; {whose} "
            f"package hash is {want}. A file differs, is missing, or is extra (R15: the entrant checks each file "
            f"against the table). To see which, run: python -m scoreboard.evalkit.package check TABLE.csv "
            f"{package_dir} with the task's package table. {PACKAGE_REFUSES}. A table with a hidden input row "
            f"(rule book amendment 2, section 4.3 point 5) never rebuilds: give --package-table TABLE.csv too, and "
            f"the intake checks each file against that table. {PACKAGE_REBUILD}")


def package_files_check(task: Task, package_dir: str | Path | None, package_table: str | Path | None = None) -> str:
    """The R15 package files check at intake (B3 step 2; D5 intake, wave 11): check the submitted package folder
    against the task's package hash (`files_against`). PASS, FAIL (refused) or NOT CHECKED."""
    if task.division != "closed":
        if package_dir is None and package_table is None:
            return PASS
        return (f"{FAIL}: a package folder or table was given, but the task is {task.division!r}; only a closed "
                f"division task has a package (R15). Leave out --package and --package-table")
    if task.package_sha256 is None:
        return (f"{NOT_CHECKED}: the task file names no package_sha256, so the package files were not checked "
                f"against it (R15){_not_used(package_dir, package_table)}")
    return files_against(task.package_sha256, package_dir, package_table)


def entry_package_files_check(intake: Intake, package_dir: str | Path | None,
                              package_table: str | Path | None = None) -> str:
    """The package files check with no task file (D5 intake review 1 F4). A closed rule book form entry names its
    own package_sha256: with --package the intake checks the folder against that hash. That shows the folder is the
    package the entry names; it does not show the package is the task's (only --task does). Without --package it
    reads NOT CHECKED. An entry that names no package hash refuses a folder."""
    own = intake.package_sha256 if intake.form == RULEBOOK_FORM else None
    if own is None:
        if package_dir is None and package_table is None:
            return PASS
        return (f"{FAIL}: a package folder or table was given, but the entry names no package_sha256 (only a "
                f"closed rule book form entry does; R15, B3). Leave out --package and --package-table")
    if package_dir is None and package_table is None:
        return (f"{NOT_CHECKED}: no --task and no --package given, so the package files were not checked. A closed "
                f"entry without --task is registered with package NOT CHECKED; give --package DIR to check the "
                f"folder against the entry's own package_sha256 (R15; B3 step 2)")
    return files_against(own, package_dir, package_table, whose="the entry's own")


def check_against_task(intake: Intake, task: Task, package_dir: str | Path | None = None,
                       package_table: str | Path | None = None) -> dict:
    """Check an entry against a task file (R2, R6, R15, B3 steps 1 and 2). Returns {check: "PASS" | "FAIL: why" |
    "NOT CHECKED: why"}. Only a FAIL refuses (`task_failures`). `package_dir`: the submitted package folder, needed
    on a closed task that names a package hash; `package_table`: the task's package table, optional
    (`package_files_check`)."""
    m = intake.entry.manifest
    out = {}
    out["site_list"] = site_list_failure(intake.entry.site_ids, task.site_list_sha256) or PASS
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
    out["package"] = package_check(intake, task)
    out["package_files"] = package_files_check(task, package_dir, package_table)
    return out


def task_hash(path: str | Path) -> tuple[str, str]:
    """(task hash, exact text hashed) for a task file, as `check_against_task` and the scorer take it (R2)."""
    t = load_task(path)
    return t.sha256(), canonical_json(t.to_dict())     # Task.sha256 = sha256 of this exact UTF-8 text


def task_failures(checks: dict) -> list[str]:
    """The checks that refuse an entry: every FAIL. A NOT CHECKED is not a failure; the submit output prints it."""
    return [f"{k}: {v}" for k, v in checks.items() if v != PASS and not v.startswith(NOT_CHECKED)]


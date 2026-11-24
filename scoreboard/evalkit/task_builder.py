"""Public task builder (rules R2 and R15; D5 kit review 1 N4, eval M5; D5 follow-ups, wave 10). No em dashes.

Rule book: docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md (pillar root), with amendment 1
(docs/evals/RULEBOOK-V1-AMENDMENT-1-2026-11-24.md, ruling 52c). R2: "on the closed division, the package hash (R15)"
is part of every task file; "the kit's public task builder makes every task file from committed items only".
Amended R2: a task names clip, tile_m and seed, each equal to the value pinned for its card version. R15: "the
package hash is the sha256 of that table"; "a closed record entry with another package hash is refused".

  caller fields --+
  task_pins.json -+--> pinned clip, tile_m, seed, tile_rule (a caller value that differs: REFUSED)
  package table --+--> closed: package_sha256 = package.table_sha256(TABLE) (a caller value that differs: REFUSED;
                  |            no table: REFUSED); open: no table, no package hash
                  v
            spec.Task.from_dict (every field check) --> task file (new path only; read back; then linked in place)

Committed items this builder reads. (1) `task_pins.json` beside this module: amendment 1, section 4, per card
version (v2, v3, v3.1, later). The caller may leave clip, tile_m, seed and tile_rule out; the builder fills them from
the pins. A caller value that differs is refused, nothing written. (2) On the closed division, the package table
(`evalkit.package`, kit form TABLE_RECIPE): the builder computes package_sha256 from it. A closed task without a
table is refused, and so is a caller package_sha256 that differs from the table's hash. So a closed rule book form
(B3) entry against a task this builder writes never reads "package: NOT CHECKED" at intake, and the task's package
hash is the sha256 of a table someone can check. A closed kit form entry still reads NOT CHECKED: the kit form
manifest has no package field (intake.py).

The builder never replaces a file: it refuses an --out path that exists (a task file is a public item). It writes a
temporary file beside --out, reads it back, and refuses (and deletes the temporary file) when the task hash or
package hash differs from what it built. Only then does it link the file into place; the link fails if --out
appeared in the meantime (task builder review 1 N2).

What it does not do (R2, still open). It does not derive the other fields from committed items (site universe and
footprint, label source standing items, census read, ShakeMap rule, rule-book version): the kit holds none of them
yet, so the caller still gives event_id, target, label source, label rule hash, site list hash, track, division and
caveats. spec.Task itself still reads a closed task without `package_sha256`, and any clip, tile_m and seed in
range: every sealed card and every older task file rehashes through it, so that reader does not change.

Command line (run from the repo root without -I):
  python -m scoreboard.evalkit.task_builder FIELDS.json --out TASK.json [--package-table TABLE.csv]
                                            [--card-version later]
Prints the task hash and the package hash as JSON. Exit codes: 0 written; 2 refused, nothing written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping

from . import package as _package
from .spec import SpecError, Task, load_task, save_task

PINS_FILE = Path(__file__).with_name("task_pins.json")
DEFAULT_CARD_VERSION = "later"
CLOSED_NEEDS_PACKAGE = ("task builder: a closed division task names its package hash (package_sha256), and the "
                        "builder takes it only from the package table; give --package-table TABLE.csv (R2: \"on the "
                        "closed division, the package hash\"; R15: the package hash is the sha256 of that table)")


def load_pins(card_version: str = DEFAULT_CARD_VERSION) -> dict:
    """The pinned task fields of one card version, from the committed task_pins.json (rule book v1 amendment 1)."""
    doc = json.loads(PINS_FILE.read_text(encoding="utf-8"))
    versions = doc["card_versions"]
    if card_version not in versions:
        raise SpecError(f"task builder: card version {card_version!r} has no pinned values; known: "
                        f"{sorted(versions)} (rule book v1 amendment 1, section 4)")
    pins = versions[card_version]
    if sorted(pins) != sorted(doc["pinned_fields"]):
        raise SpecError(f"task builder: {PINS_FILE.name} card version {card_version!r} does not pin exactly "
                        f"{doc['pinned_fields']}")
    return dict(pins)


def _same(given, pin) -> bool:
    if isinstance(given, bool) or isinstance(pin, bool):
        return False
    if isinstance(pin, str):
        return given == pin
    if isinstance(pin, int):
        return isinstance(given, int) and given == pin
    return isinstance(given, (int, float)) and float(given) == float(pin)


def build_task(fields: Mapping, *, package_table: str | Path | None = None,
               card_version: str = DEFAULT_CARD_VERSION) -> Task:
    """A checked Task from the caller's fields, the committed pins and (closed division) the package table.

    Refused: a pinned field (clip, tile_m, seed, tile_rule) that differs from task_pins.json for `card_version`;
    a closed task without a package table; a caller package_sha256 that differs from the table's hash; a package
    table on an open task (R2 as amended, R15)."""
    if not isinstance(fields, Mapping):
        raise SpecError(f"task builder: the task fields must be a JSON object, got {type(fields).__name__}")
    pins = load_pins(card_version)
    for k, pin in pins.items():
        if k in fields and not _same(fields[k], pin):
            raise SpecError(f"task builder: {k} is pinned at {pin!r} for card version {card_version!r} (rule book "
                            f"v1 amendment 1, section 4; a task cannot choose it); the caller gave {fields[k]!r}")
    d = dict(fields, **pins)
    if d.get("division") == "closed":
        if package_table is None:
            raise SpecError(CLOSED_NEEDS_PACKAGE)
        pkg = _package.table_sha256(package_table)
        given = d.get("package_sha256")
        if given is not None and given != pkg:
            raise SpecError(f"task builder: the caller gave package_sha256 {given}; the package table "
                            f"{package_table} hashes to {pkg} (R15). The builder takes it from the table")
        d["package_sha256"] = pkg
    elif package_table is not None:
        raise SpecError("task builder: only a closed division task has a package table (R15); this task is "
                        f"{d.get('division')!r}")
    return Task.from_dict(d)


def write_task(fields: Mapping, path: str | Path, *, package_table: str | Path | None = None,
               card_version: str = DEFAULT_CARD_VERSION) -> dict:
    """Build the task, write it to a new file at `path` (an existing path is refused), read it back.

    Returns {task_sha256, package_sha256, division, card_version, path}."""
    task = build_task(fields, package_table=package_table, card_version=card_version)
    path = Path(path)
    if os.path.lexists(path):
        raise SpecError(f"task builder: {path} exists; the builder never replaces a task file (a public item). "
                        "Give a new --out path")
    tmp = path.with_name(f".{path.name}.task_builder-{os.getpid()}.tmp")
    try:
        save_task(task, tmp)
        back = load_task(tmp)
        if back.sha256() != task.sha256() or back.package_sha256 != task.package_sha256:
            raise SpecError(f"task builder: {path} read back with task hash {back.sha256()} and package hash "
                            f"{back.package_sha256}; built {task.sha256()} and {task.package_sha256}")
        try:
            os.link(tmp, path)                     # never replaces: fails if --out appeared after the check
        except FileExistsError:
            raise SpecError(f"task builder: {path} exists; the builder never replaces a task file (a public "
                            "item). Give a new --out path") from None
    finally:
        tmp.unlink(missing_ok=True)
    return {"task_sha256": task.sha256(), "package_sha256": task.package_sha256, "division": task.division,
            "card_version": card_version, "path": str(path)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scoreboard.evalkit.task_builder",
                                 description="Write a task file from its fields, the pinned clip, tile_m, seed "
                                             "and tile_rule (task_pins.json, rule book v1 amendment 1) and, on the "
                                             "closed division, the package table. A closed task always names "
                                             "package_sha256, taken from the table (R2, R15).")
    ap.add_argument("fields", help="JSON object of spec.Task fields (README, 'Task file'); clip, tile_m, seed and "
                                   "tile_rule may be left out (they come from the pins)")
    ap.add_argument("--out", required=True, help="task file to write")
    ap.add_argument("--package-table", help="closed division: the package table (python -m "
                                            "scoreboard.evalkit.package write DIR --out TABLE.csv)")
    ap.add_argument("--card-version", default=DEFAULT_CARD_VERSION,
                    help="card version whose pins apply (task_pins.json; default: later)")
    a = ap.parse_args(argv)
    try:
        fields = json.loads(Path(a.fields).read_text(encoding="utf-8"))
        res = write_task(fields, a.out, package_table=a.package_table, card_version=a.card_version)
    except (OSError, ValueError, SpecError) as e:
        print(json.dumps({"status": "REFUSED", "reason": str(e)}, indent=1))
        return 2
    print(json.dumps(dict(res, status="WRITTEN"), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

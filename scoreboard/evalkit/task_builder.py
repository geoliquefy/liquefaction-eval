"""Public task builder, first slice (rules R2 and R15; D5 kit review 1 N4, eval M5). No em dashes.

Rule book: docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md (pillar root). R2: "on the closed division, the package hash
(R15)" is part of every task file. R15: "a closed record entry with another package hash is refused".

  task fields (JSON object) --> spec.Task.from_dict (every field check) --> closed? package_sha256 named? --> task file
                                                                              |
                                                                              no --> REFUSED, nothing written

What this slice does. It writes a task file from given fields, and it always writes `package_sha256` on a closed
task: a closed task without it is refused, so no task file this builder writes reads "package: NOT CHECKED" at
intake. An open task names no package hash (spec.Task refuses one). The builder never replaces a file: it refuses an
--out path that exists (a task file is a public item). It writes a temporary file beside --out, reads it back, and
refuses (and deletes the temporary file) when the task hash or package hash differs from what it built. Only then
does it link the file into place; the link fails if --out appeared in the meantime (task builder review 1 N2).

What it does not do (R2, still open). It does not make the task from committed items only (site universe, standing
items, census read, ShakeMap rule, rule-book version); the caller gives every field. It does not compute the package
hash from a package table (R15 defines the hash as the sha256 of that table; the kit has no table format yet).

spec.Task itself still reads a closed task without `package_sha256`: every sealed card and every older task file
rehashes through it, so that reader does not change. Only this builder requires the field.

Command line (run from the repo root without -I):
  python -m scoreboard.evalkit.task_builder FIELDS.json --out TASK.json
Prints the task hash and the package hash as JSON. Exit codes: 0 written; 2 refused, nothing written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping

from .spec import SpecError, Task, load_task, save_task

CLOSED_NEEDS_PACKAGE = ("task builder: a closed division task names its package hash; give package_sha256, 64 "
                        "lowercase hex characters (R2: \"on the closed division, the package hash\"; R15)")


def build_task(fields: Mapping) -> Task:
    """A checked Task from the given fields. A closed task without package_sha256 is refused (R2, R15)."""
    if not isinstance(fields, Mapping):
        raise SpecError(f"task builder: the task fields must be a JSON object, got {type(fields).__name__}")
    task = Task.from_dict(dict(fields))
    if task.division == "closed" and task.package_sha256 is None:
        raise SpecError(CLOSED_NEEDS_PACKAGE)
    return task


def write_task(fields: Mapping, path: str | Path) -> dict:
    """Build the task, write it to a new file at `path` (an existing path is refused), read it back.

    Returns {task_sha256, package_sha256, division, path}."""
    task = build_task(fields)
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
            "path": str(path)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scoreboard.evalkit.task_builder",
                                 description="Write a task file from its fields. A closed task always names "
                                             "package_sha256; a closed task without it is refused (R2, R15).")
    ap.add_argument("fields", help="JSON object of spec.Task fields (README, 'Task file')")
    ap.add_argument("--out", required=True, help="task file to write")
    a = ap.parse_args(argv)
    try:
        fields = json.loads(Path(a.fields).read_text(encoding="utf-8"))
        res = write_task(fields, a.out)
    except (OSError, ValueError, SpecError) as e:
        print(json.dumps({"status": "REFUSED", "reason": str(e)}, indent=1))
        return 2
    print(json.dumps(dict(res, status="WRITTEN"), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

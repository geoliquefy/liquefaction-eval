"""Closed division package table (rule R15; D5 follow-ups, wave 10, eval M5). No em dashes.

Rule book: docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md (pillar root). R15: "The closed package is the site id list
plus a table of the sha256 of every input file; the package hash is the sha256 of that table. An entrant who fetches
files itself (as NZGD terms require) checks each against the table." B3 step 2: "get each input file by its id and
check it against the package table".

  package folder --> write_table --> TABLE.csv (header file_id,sha256; one row per file; sorted; LF)
                                        |
                                        +--> table_sha256 = sha256 of the table file's bytes = the package hash (R15)
                                        +--> check_files(TABLE.csv, folder): PASS, or FAIL with each file that differs

What the rule book fixes: one sha256 per input file, and the package hash is the sha256 of the table. What it does
not fix: the table's byte form. This kit form (TABLE_RECIPE) is the smallest one that makes "the sha256 of that
table" one number for one set of files: every reader refuses a table that is not in it. The site id list is in the
task already (site_list_sha256, spec.SITE_LIST_RECIPE), so it is not a row here. No rule moves (correction 3).

Command line (run from the repo root without -I):
  python -m scoreboard.evalkit.package write DIR --out TABLE.csv   write the table of every file in DIR but hidden
                                                                   ones (never replaces)
  python -m scoreboard.evalkit.package hash TABLE.csv              print the package hash
  python -m scoreboard.evalkit.package check TABLE.csv DIR         check every file in DIR against the table
Exit codes: 0 written, printed or PASS; 1 FAIL (a file differs or is missing); 2 refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Mapping

from .spec import SpecError

TABLE_HEADER = "file_id,sha256"
TABLE_RECIPE = ("UTF-8 text; first line exactly 'file_id,sha256'; then one line per input file, 'FILE_ID,SHA256', "
                "sorted by FILE_ID (Python str order), no repeated FILE_ID; SHA256 is 64 lowercase hex characters of "
                "the file's bytes; FILE_ID is the file's path inside the package folder with '/' separators (no "
                "leading '/', no '.' or '..' part, no comma, backslash or control character); every line ends "
                "with one LF, the last one too. The package hash is the sha256 of these bytes (R15). 'package "
                "write' leaves out hidden files (any path part that starts with '.', such as .DS_Store), so a "
                "fetcher who lacks one never reads MISSING; a hidden input file goes in a table only by hand")
_HEX = re.compile(r"[0-9a-f]{64}")


def _check_id(fid: str) -> str:
    if (not isinstance(fid, str) or not fid or fid.startswith("/") or any(c in fid for c in ",\\")
            or any(ord(c) < 32 or ord(c) == 127 for c in fid)
            or any(p in ("", ".", "..") for p in fid.split("/"))):
        raise SpecError(f"package table: bad file_id {fid!r}. {TABLE_RECIPE}")
    return fid


def table_text(rows: Mapping[str, str]) -> str:
    """The canonical table text for {file_id: sha256}. Refuses a bad id or hash."""
    for fid, h in rows.items():
        _check_id(fid)
        if not isinstance(h, str) or not _HEX.fullmatch(h):
            raise SpecError(f"package table: {fid!r} has bad sha256 {h!r}. {TABLE_RECIPE}")
    if not rows:
        raise SpecError("package table: a closed package names at least one input file (R15)")
    return TABLE_HEADER + "\n" + "".join(f"{fid},{rows[fid]}\n" for fid in sorted(rows))


def table_sha256_of_rows(rows: Mapping[str, str]) -> str:
    """The package hash (R15) of {file_id: sha256}: sha256 of the canonical table text."""
    return hashlib.sha256(table_text(rows).encode("utf-8")).hexdigest()


def read_table(path: str | Path) -> tuple[dict, str]:
    """({file_id: sha256}, package hash) of a table file. Refuses a table not in the kit form (TABLE_RECIPE)."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise SpecError(f"package table {path}: not UTF-8. {TABLE_RECIPE}") from None
    lines = text.split("\n")
    if not text.endswith("\n") or not lines or lines[0] != TABLE_HEADER:
        raise SpecError(f"package table {path}: not in the kit form. {TABLE_RECIPE}")
    rows: dict[str, str] = {}
    for ln in lines[1:-1]:
        parts = ln.split(",")
        if len(parts) != 2 or parts[0] in rows:
            raise SpecError(f"package table {path}: bad or repeated line {ln!r}. {TABLE_RECIPE}")
        rows[parts[0]] = parts[1]
    if table_text(rows) != text:
        raise SpecError(f"package table {path}: not in the kit form (row order, spacing or line ends). "
                        f"{TABLE_RECIPE}")
    return rows, hashlib.sha256(raw).hexdigest()


def table_sha256(path: str | Path) -> str:
    """The package hash (R15): sha256 of the table file's bytes, after the kit form check."""
    return read_table(path)[1]


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hidden(fid: str) -> bool:
    return any(part.startswith(".") for part in fid.split("/"))


def _files(folder: Path, *, hidden: bool = True) -> dict:
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file() or p.is_symlink():
            fid = p.relative_to(folder).as_posix()
            if hidden or not _hidden(fid):
                out[_check_id(fid)] = p
    return out


def write_table(folder: str | Path, out: str | Path) -> dict:
    """Write the table of every file in `folder` to a new file `out` (an existing path is refused).

    Hidden files (any path part that starts with '.', such as .DS_Store) are left out (D5 follow-ups review 1):
    a stray Finder file must not become a row that every fetcher then reads as MISSING. Returns {package_sha256, n_files, path}."""
    folder, out = Path(folder), Path(out)
    if not folder.is_dir():
        raise SpecError(f"package: {folder} is not a folder")
    files = {fid: p for fid, p in _files(folder, hidden=False).items() if p.resolve() != out.resolve()}
    text = table_text({fid: _sha_file(p) for fid, p in files.items()})
    try:
        with open(out, "x", encoding="utf-8", newline="") as f:   # "x": never replaces a table
            f.write(text)
    except FileExistsError:
        raise SpecError(f"package: {out} exists; a package table is never replaced. Give a new --out path") from None
    return {"package_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "n_files": len(files),
            "path": str(out)}


def check_files(table: str | Path, folder: str | Path) -> dict:
    """Check each file of the table against the files in `folder` (R15, B3 step 2).

    status PASS when every table file is there with its sha256; FAIL otherwise. `files` maps each file_id to PASS,
    FAIL or MISSING. Files in the folder that the table does not name are listed in `not_in_table` (they are not
    package inputs, so they do not change the status)."""
    rows, pkg = read_table(table)
    folder = Path(folder)
    have = _files(folder) if folder.is_dir() else {}
    tpath = Path(table).resolve()
    files = {fid: ("MISSING" if fid not in have else "PASS" if _sha_file(have[fid]) == h else "FAIL")
             for fid, h in rows.items()}
    extra = sorted(fid for fid, p in have.items() if fid not in rows and p.resolve() != tpath)
    status = "PASS" if all(v == "PASS" for v in files.values()) else "FAIL"
    return {"status": status, "package_sha256": pkg, "files": files, "not_in_table": extra}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scoreboard.evalkit.package",
                                 description="Closed division package table (R15). " + TABLE_RECIPE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="write the table of every file in DIR but hidden ones (never replaces --out)")
    w.add_argument("dir")
    w.add_argument("--out", required=True)
    h = sub.add_parser("hash", help="print the package hash of a table")
    h.add_argument("table")
    c = sub.add_parser("check", help="check every file in DIR against the table")
    c.add_argument("table")
    c.add_argument("dir")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "write":
            res = dict(write_table(a.dir, a.out), status="WRITTEN")
        elif a.cmd == "hash":
            res = {"package_sha256": table_sha256(a.table), "status": "OK"}
        else:
            res = check_files(a.table, a.dir)
    except (OSError, ValueError, SpecError) as e:
        print(json.dumps({"status": "REFUSED", "reason": str(e)}, indent=1))
        return 2
    print(json.dumps(res, indent=1, sort_keys=True))
    return 1 if res["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())

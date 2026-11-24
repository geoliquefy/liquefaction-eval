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

Rule book v1 amendment 2 (pillar docs/evals/RULEBOOK-V1-AMENDMENT-2-2026-11-25.md, section 4) names this byte form;
the bytes did not change. Ruling 58 (c) (B4 layout, wave 12) adds two practice lines to the FILE_ID check and the
file walk, so every reader and writer agrees (amendment 2 review 1, F1, F5 and F6):
  - FILE_ID is printable ASCII (codes 33 to 126) with no space, no '"' and no "'" (F1, F5). A file named otherwise
    is refused by write, rebuild and check, and a table that holds such an id is refused by every reader. No table
    the kit wrote for a task held one: the only kit tables were the forward runner's (ids shakemap_grid.xml and
    cpt_list.csv before B4), so every older kit table stays readable.
  - A symbolic link to a file is hashed as its target's bytes, under the link's own path, in write, rebuild and
    check alike (F6); `hash` reads the table file the same way. A link to a folder, or a link whose target is
    missing, is refused: no reader could hash it. The walk never enters a linked folder.

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
                "the file's bytes (a symbolic link to a file is hashed as its target); FILE_ID is the file's path "
                "inside the package folder with '/' separators (on the forward track the folder is B4 /inputs/), no "
                "leading '/', no '.' or '..' part, printable ASCII only, no comma, backslash, space or quote "
                "(ruling 58c); every line ends "
                "with one LF, the last one too. The package hash is the sha256 of these bytes (R15). 'package "
                "write' leaves out hidden files (any path part that starts with '.', such as .DS_Store), so a "
                "fetcher who lacks one never reads MISSING; a hidden input file goes in a table only by hand")
_HEX = re.compile(r"[0-9a-f]{64}")
_ID_BANNED = ",\\\"'"           # comma, backslash, double quote, single quote. A space (code 32) is refused by
                                # the range check 33..126 in _check_id (B4-LAYOUT review 1, N1)


def _check_id(fid: str) -> str:
    if (not isinstance(fid, str) or not fid or fid.startswith("/")
            or any(not 33 <= ord(c) <= 126 or c in _ID_BANNED for c in fid)
            or any(p in ("", ".", "..") for p in fid.split("/"))):
        why = ("not printable ASCII, or holds a space or a quote (ruling 58c; amendment 2 review F1, F5)"
               if isinstance(fid, str) and any(not 33 <= ord(c) <= 126 or c in " \"'" for c in fid) else "bad form")
        raise SpecError(f"package table: bad file_id {fid!r}: {why}. {TABLE_RECIPE}")
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


def _walk(folder: Path):
    """(file_id, path) of every entry under `folder` that the walk does not enter: files, links and other non-folders.
    A symbolic link to a folder is given as an entry and never entered. No test is run here."""
    for root, dirs, names in os.walk(folder, followlinks=False):
        r = Path(root)
        for n in list(dirs):
            if (r / n).is_symlink():              # os.walk lists a link to a folder as a folder; never enter it
                dirs.remove(n)
                names.append(n)
        for n in names:
            p = r / n
            yield p.relative_to(folder).as_posix(), p


def _row_path(fid: str, p: Path) -> Path:
    """The tests a row's path passes (ruling 58c, review F6): a symbolic link must point at a file (it is hashed as
    its target); any other non-file is refused; the id is printable ASCII with no space or quote."""
    if p.is_symlink() and not p.is_file():
        raise SpecError(f"package: {fid} is a symbolic link to a folder or to nothing; a link is hashed as "
                        f"its target file, so it must point at a file (ruling 58c)")
    if not p.is_file():
        raise SpecError(f"package: {fid} is not a regular file")
    _check_id(fid)
    return p


def _files(folder: Path, *, hidden: bool = True) -> dict:
    """{file_id: path} of every file under `folder`. One walk (`_walk`) and one set of tests (`_row_path`) for write,
    rebuild and check (ruling 58c, review F6). With hidden=False a hidden path is left out before any test, so a
    hidden broken link or a hidden name with a space never refuses a write or a rebuild (B4-LAYOUT review 1, F1)."""
    out = {}
    for fid, p in _walk(folder):
        if not hidden and _hidden(fid):
            continue
        out[fid] = _row_path(fid, p)
    return dict(sorted(out.items()))


def hidden_ids(folder: str | Path) -> list:
    """The hidden paths under `folder` (any path part that starts with '.'), untested, sorted. The forward runner
    uses it to refuse any added file in its own /inputs/ (B4-LAYOUT review 1, N4)."""
    folder = Path(folder)
    return sorted(fid for fid, _ in _walk(folder) if _hidden(fid)) if folder.is_dir() else []


def folder_rows(folder: str | Path) -> dict:
    """{file_id: sha256} of the table `package write` would write for `folder` (every file but hidden ones)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise SpecError(f"package: {folder} is not a folder")
    return {fid: _sha_file(p) for fid, p in _files(folder, hidden=False).items()}


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


def rebuild_sha256(folder: str | Path) -> tuple[str, int]:
    """(package hash, number of files) of the table that `package write` would write for `folder`: every file but
    hidden ones, in the kit form (TABLE_RECIPE). Nothing is written. The intake uses it to rebuild a submitted
    package's hash (D5 intake, wave 11). A table file kept inside the folder becomes a row, so keep it outside."""
    folder = Path(folder)
    if not folder.is_dir():
        raise SpecError(f"package: {folder} is not a folder")
    rows = folder_rows(folder)
    return table_sha256_of_rows(rows), len(rows)


def check_files(table: str | Path, folder: str | Path) -> dict:
    """Check each file of the table against the files in `folder` (R15, B3 step 2).

    status PASS when every table file is there with its sha256; FAIL otherwise. `files` maps each file_id to PASS,
    FAIL or MISSING. Files in the folder that the table does not name are listed in `not_in_table` (they are not
    package inputs, so they do not change the status).

    The walk and its tests are the ones write and rebuild use, with the hidden rule of the table (B4-LAYOUT review
    1, F1): a hidden path is tested only when the table names it (a hidden input row, put in by hand). A hidden path
    the table does not name is listed in `not_in_table` and never tested, so a hidden editor lock that is a broken
    link, or a hidden folder whose name holds a space, never refuses the check (write and rebuild skip it too). A
    non-hidden path is tested whether the table names it or not, as write and rebuild test it: a non-hidden extra
    file with a bad name (not printable ASCII, a space or a quote), a link to a folder or a broken link makes the
    check REFUSED (exit 2), not FAIL (B4-LAYOUT review 1, N2)."""
    rows, pkg = read_table(table)
    folder = Path(folder)
    tpath = Path(table).resolve()
    have, extra = {}, []
    if folder.is_dir():
        for fid, p in _walk(folder):
            if fid not in rows and _hidden(fid):
                if p.resolve() != tpath:
                    extra.append(fid)             # hidden, not a row: listed, never tested
                continue
            have[fid] = _row_path(fid, p)
    files = {fid: ("MISSING" if fid not in have else "PASS" if _sha_file(have[fid]) == h else "FAIL")
             for fid, h in rows.items()}
    extra = sorted(extra + [fid for fid, p in have.items() if fid not in rows and p.resolve() != tpath])
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

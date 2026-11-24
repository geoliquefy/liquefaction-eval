G1+G5+G6 (eval M5): rule book v1 amendment 2, ruling 57 (c). Names the package table byte form in R15.

# Liquefaction model eval: rule book v1, amendment 2 (R15: the package table byte form)

This file amends rule R15 of `docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md` (revision 4). That file is not edited.
Amendment 1 (`docs/evals/RULEBOOK-V1-AMENDMENT-1-2026-11-24.md`) is not edited. This is the second rule book
amendment. It is committed alone, as one file. It is frozen once committed. The kit does not change: it stays at main
`c052221` (tag `evalkit-v0.10.0`). No card, card hash or task hash moves. No other rule moves (correction 3). No em
dashes.

```
 RULE BOOK R15 (draft, 2026-11-15)                 AMENDMENT 2 (this file)
 "a table of the sha256 of every input file;  -->  the table has one byte form (section 4);
  the package hash is the sha256 of that table"     it lists every input file an entry reads,
                                                    the ShakeMap grid included (section 5)

 input files --sha256 each--> rows "file_id,sha256" --sort, header, LF--> TABLE bytes --sha256--> package hash
                                                                              |
                                                     check: each file vs its row (PASS / FAIL / MISSING)

 kit package.py at c052221                          unchanged: it already writes and reads this form
 sealed cards v2, v3, v3.1                          unchanged: no card hash or task hash moves
```

## 1. The ruling

Founder ruling 57 (c), `GOALS.md`, 2026-11-25, quoted:

> (c) Rule book amendment 2, committed alone, names the package table byte form the kit uses (header
> file_id,sha256; rows sorted; lowercase hex; LF line ends) and says the table lists every input file an entry
> reads, the ShakeMap grid included.

## 2. Why

1. R15 says "the package hash is the sha256 of that table". It does not fix the table's bytes. Two tables can list
   the same files and hashes and still differ in bytes: row order, letter case, line ends, a final newline. Each
   such table gives another package hash. So R15 alone does not give one package hash for one set of files.
2. The kit fixed one form in wave 10 (D5 follow-ups, `scoreboard/evalkit/package.py`, constant `TABLE_RECIPE`).
   Every kit reader refuses a table in another form. The kit README notes that the rule book does not fix it.
3. The wave 10 record (`docs/EVAL-WAVE10-2026-11-24.md`, section 7, item 6, question Q-PKG) asked two things. Should
   the rule book name the kit form? Which files does a closed forward package hold, since B4 lists
   `shakemap/grid.xml` apart from `package/`? Ruling 57 (c) answers both.

## 3. Amended R15

### 3.1 Text kept

R15, **Rule**, second sentence. The draft reads, and it stays:

> The closed package is the site id list plus a table of the sha256 of every input file; the package hash is the
> sha256 of that table.

### 3.2 Text added after it

> The table lists every input file an entry reads, the ShakeMap grid included, one row per file. It is in the byte
> form of rule book v1 amendment 2, section 4: the header line `file_id,sha256`, then one row `FILE_ID,SHA256` per
> file, rows sorted by `FILE_ID`, each SHA256 64 lowercase hex characters, every line ending in one LF, the last
> line too. The package hash is the sha256 of exactly these bytes. A table in any other form is not a package table.

### 3.3 Check sentence added to R15 **Check**

> A package table not in the amendment 2 byte form is refused, and so is a package hash that is not the sha256 of
> such a table; an input file that is missing, or whose sha256 differs from its row, reads FAIL.

### 3.4 Effect on B3 step 2 and B4 step 2

B3 and B4 are not edited. B3 step 2 says "get each input file by its id and check it against the package table".
Under amended R15 the id is the `FILE_ID` of section 4.3, and the check is the sha256 of the file's bytes against
its row. B4 step 2 lists `event.json`, `shakemap/grid.xml`, `sites.csv` and `package/`. Under amended R15 each file
that the entry reads from that list is a row of the table (section 5).

## 4. The byte form

This is the form the kit writes and reads today (`package.py` at `c052221`: `TABLE_HEADER` line 38,
`TABLE_RECIPE` lines 39 to 45, `_check_id` line 49, `table_text` line 57, `read_table` line 73, `write_table`
line 122). Each point below is what that code does.

### 4.1 Encoding and lines

1. The table is UTF-8 text with no byte order mark.
2. The first line is exactly `file_id,sha256`.
3. Then comes one line per input file: `FILE_ID,SHA256`. There is no space, no quote and no other column.
4. Every line ends with one LF (byte `0x0A`). The header line ends with LF. The last row ends with LF too, so the
   file ends with exactly one LF. There is no CR anywhere and no empty line.
5. A table has at least one row. A table with only the header is refused.

### 4.2 Rows

1. Rows are sorted by `FILE_ID` in ascending order. The order is Python `str` order: Unicode code point order. For
   UTF-8 text this is the same as plain byte order of the encoded ids. Example: `cpt/a.csv` sorts before
   `cpt_list.csv`, because `/` is `0x2F` and `_` is `0x5F`. Case counts: `Z` (`0x5A`) sorts before `a` (`0x61`).
2. A `FILE_ID` appears once. A repeated id is refused.
3. `SHA256` is the sha256 of the file's bytes as stored, written as 64 lowercase hex characters (`0-9`, `a-f`). A
   gzip file is hashed as its gzip bytes, not as its unzipped content.

### 4.3 How FILE_ID is formed

1. `FILE_ID` is the file's path inside the folder the table covers (the package folder), with `/` between parts.
2. It has no leading `/`. No part is empty, `.` or `..`. So it has no `//` and no trailing `/`.
3. It holds no comma, no backslash and no control character (code below 32, or 127).
4. It is case sensitive and is written exactly as the path is spelled.
5. `package write` leaves out hidden files (any path part that starts with `.`, such as `.DS_Store`). A hidden input
   file goes in a table only by hand. The form allows it.

### 4.4 The package hash

The package hash is the sha256 of the table bytes of 4.1 to 4.3, written as 64 lowercase hex characters. It is the
`package_sha256` that R2 puts in a closed task and that B3 puts in a closed record manifest.

## 5. Which files the table lists

1. The table lists every input file an entry reads, the ShakeMap grid included. So one package hash covers every
   input of a closed entry.
2. Record track: every file of the published package.
3. Forward track: every file the entry reads from its input folder (B4 `/inputs/`). That is the ShakeMap grid
   (`shakemap/grid.xml`), and every other B4 input the entry reads (`event.json`, `sites.csv`, the `package/`
   files). The `FILE_ID` is the path inside that folder, for example `shakemap/grid.xml`.
4. The site id list held only in the task file is not a row. Its hash is the task's `site_list_sha256` (R2). A site
   list that is a file the entry reads (B4 `sites.csv`; the kit's `cpt_list.csv`) is a row.
5. R21 still has the organiser commit the hash of each input it publishes. This amendment does not change that.

## 6. Worked example (made-up files)

Two made-up input files in one package folder. Each is shown as the exact `printf` that makes it. You can recompute
every hash below with standard tools.

| FILE_ID | bytes (printf form) | size | sha256 |
|---|---|---|---|
| `cpt_list.csv` | `site_id,lon,lat\nex:0001,172.60,-43.50\n` | 38 | `4bc526d2c850cad30a6f4255fb457e1d349baa86ca0e9d6c6544b4060c0dac1a` |
| `shakemap_grid.xml` | `<shakemap_grid example="1"/>\n` | 29 | `45ec07521080f99c204ab0afa13484e4f67a9f0514fcc52ff32c072625a34c43` |

The table bytes (176 bytes; each line ends in LF, the last one too):

```
file_id,sha256
cpt_list.csv,4bc526d2c850cad30a6f4255fb457e1d349baa86ca0e9d6c6544b4060c0dac1a
shakemap_grid.xml,45ec07521080f99c204ab0afa13484e4f67a9f0514fcc52ff32c072625a34c43
```

The package hash:

```
1c858d5ec6fa1c733a7d4b6fda6530f69ef286831ebd77f57885f1fa87efeb10
```

Recompute it in a shell:

```
mkdir pkg
printf 'site_id,lon,lat\nex:0001,172.60,-43.50\n' > pkg/cpt_list.csv
printf '<shakemap_grid example="1"/>\n' > pkg/shakemap_grid.xml
shasum -a 256 pkg/cpt_list.csv pkg/shakemap_grid.xml
printf 'file_id,sha256\ncpt_list.csv,%s\nshakemap_grid.xml,%s\n' \
  "$(shasum -a 256 < pkg/cpt_list.csv | cut -c1-64)" \
  "$(shasum -a 256 < pkg/shakemap_grid.xml | cut -c1-64)" | shasum -a 256
```

Or in Python 3 (standard library):

```
import hashlib
files = {"cpt_list.csv": b"site_id,lon,lat\nex:0001,172.60,-43.50\n",
         "shakemap_grid.xml": b'<shakemap_grid example="1"/>\n'}
rows = {fid: hashlib.sha256(b).hexdigest() for fid, b in files.items()}
text = "file_id,sha256\n" + "".join(f"{fid},{rows[fid]}\n" for fid in sorted(rows))
print(hashlib.sha256(text.encode("utf-8")).hexdigest())
```

Kit check (2026-11-25, kit `c052221`, run from the repo root): `python -m scoreboard.evalkit.package write pkg --out
TABLE.csv` wrote these 176 bytes and printed the package hash above. `package hash TABLE.csv` printed the same hash.
`package check TABLE.csv pkg` read PASS for both files. The same table with the last LF removed, with CRLF line
ends, with uppercase hex, or with the two rows swapped gives another sha256, and `package hash` refuses each.

## 7. What the kit does today, and what it does not

| What | Kit at `c052221` |
|---|---|
| Writes the form (`package write DIR --out TABLE.csv`; never replaces a table) | `package.py` `write_table`, line 122 |
| Refuses a table not in the form; prints the package hash (`package hash`) | `package.py` `read_table`, line 73; `table_sha256`, line 95 |
| Checks each file against its row: PASS, FAIL or MISSING (`package check`) | `package.py` `check_files`, line 141 |
| Forward track: the closed task's package hash is the sha256 of a table of the frozen ShakeMap grid and `cpt_list.csv`, written as `scores/<type>/package_table.csv` | `forward.py` line 866 (`pkg_rows`), grid file name at line 357 (`shakemap_grid.xml`, or `.gz`) |
| Intake runs `package check` on an entrant's files | no (`scoreboard/evalkit/README.md` lines 111 to 112) |

1. The kit already meets section 4. No kit change is needed for the byte form.
2. The kit's forward table holds the two frozen files its own entries read: the ShakeMap grid and `cpt_list.csv`.
   Its ids are `shakemap_grid.xml` and `cpt_list.csv`, the names in the kit's frozen input folder. B4 names the
   grid `shakemap/grid.xml` and lists `event.json` and `sites.csv`. The kit has no `event.json` file; it keeps the
   event in its input manifest. Matching the kit's forward input folder to the B4 layout is a kit change. It needs
   a separate go. It does not change the byte form.
3. The intake does not run `package check`. An entrant or runner runs it (B3 step 2). Adding it to the intake is a
   kit change and needs a separate go.

## 8. What this amendment does not change

1. The draft `docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md` is not edited. Amendment 1 is not edited. No other rule
   text is edited.
2. The kit does not change. It stays at main `c052221`, tag `evalkit-v0.10.0`.
3. No card is rebuilt. No card hash and no task hash moves. The sealed v2 card hash stays
   `cc9f7b744d6bd6b3b4d5ce0956f04e3a17539ef9014de0277223033c08c8847d`; v3.1 stays
   `2ca66984e8262e9ab79e9c32fc7f1b604053f515bf3af3ad310815f7007b96f3`. Every task block of the sealed v2, v3 and
   v3.1 cards is open division (14, 12 and 12 blocks), so none names a package hash.
4. R21 is not changed: the organiser still commits each published input hash.
5. Versions (R14). The rule book is a draft and is not adopted. So no rule book version number moves now. The
   amended R15 is part of the text that an adoption fixes.

"""Register and receipt for the eval kit (rule R20; work item D3, eval milestone M6). No em dashes.

Rule book: docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md, R20 (pillar root).

  entry files ---> schema check (spec.EntryManifest, spec.Entry) ---> entry hash (spec.entry_sha256, R4)
       |
       v
  register (JSONL, append only)          one line per item, written as canonical JSON:
    line 1: {seq 1, intake_utc, item_type, item_sha256, prev_sha256 = GENESIS, entrant}
    line 2: {seq 2, ..., prev_sha256 = sha256(line 1 text)}
    ...
       |
       v
  receipt: entry hash, line hash, previous-line hash, intake time, line number (seq), entrant id

What a check finds:
  - a changed line: its sha256 no longer equals the next line's prev_sha256 (and no longer equals a receipt's
    line_sha256);
  - a removed line: a gap in seq and a broken prev_sha256 link; a receipt whose line is gone prints REGISTER FAULT;
  - a decrease in intake time along the chain: REGISTER FAULT (R20).
A receipt binds every line at or before its own: changing any earlier line changes the chain up to the receipt's
line. Lines removed after the newest receipt are found only by a stamped head (OpenTimestamps), not by a receipt.

OpenTimestamps (R20: the head is stamped): `stamp_head` writes the head line hash to a small file beside the
register and runs `ots stamp` on it. It runs only behind the explicit `--stamp` flag. Stamping sends a hash to public
calendar servers, which is a publication, so it needs a founder go; tests mock the runner and never stamp.

Not built here (left for later items): the organiser's signature on a receipt (R20 "signed with the organiser's
key"; the key and its custody are D7 and E2), the public mirror (R20, R27), and block-time reading from upgraded
proofs. Three more R20 and R4 gaps stay open:
  - a refused intake (an entry that fails the schema) leaves no line, so R20's "one that cannot be scored prints
    REFUSED" has no register record yet;
  - a receipt is printed but is not itself appended as a register line ('receipt' is a valid item_type that nothing
    writes yet);
  - the hash-first path (R4, R20: send the hash now, the files later) exists only in the Python API (`append_item`
    with a bare hash), not on the command line.

Clock recovery: the intake refuses a clock earlier than the head, so one line written with a wrong far-future clock
blocks every later intake. The register is append only and no line is ever edited. The recovery is an erratum
(R14 errata log) that names the bad line, and a new register whose first item records the old register's head hash.

Line hash (D5): line_sha256 = sha256 of the UTF-8 bytes of the line's canonical JSON text, without its trailing
newline. Canonical JSON = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).
The first line's prev_sha256 is 64 zeros (GENESIS).

Entry hash (D5): intake.read_entry. A rule book form manifest (B3) is hashed exactly as submitted, by the B3 recipe;
a kit form manifest is hashed after the kit fills conversion_id and division as null when absent. `hash` prints the
manifest text that was hashed, the predictions file read and the entry's site list hash (spec.SITE_LIST_RECIPE).

Closed entries (D5 intake, wave 11; R15, B3 step 2; ruling 57 c). A kit form manifest that states division
"closed" is refused: it has no package hash field. With --task on a closed task that names a package hash, the
intake needs --package DIR: it rebuilds the package hash from the files in DIR (hidden files left out, the kit
table form) and refuses a hash that differs from the task's. With --package-table TABLE.csv (the task's table) it
checks that the table's hash is the task's, then checks each file against the table: this serves a table with a
hidden input row (rule book amendment 2, section 4.3 point 5). Without --task, --package checks the folder against
a closed rule book form entry's own package_sha256; a closed entry without --task and without --package is
registered with package NOT CHECKED. See intake.package_files_check and intake.entry_package_files_check.

Predictions file (D5 kit step, dry run 2 point 9): `--csv PATH` if given; else NAME.csv beside NAME.json; else, for a
manifest named manifest.json (the rule book B3 names), entry.csv beside it. A refusal names every file it looked for.

Entry files in hand (D5 kit step, dry run 2 point 11): `verify --entry` on files that hash differently from the
receipt, with the register chain and the receipt's line intact, prints ENTRY MISMATCH, not REGISTER FAULT. R20 keeps
REGISTER FAULT for faults of the register; it sets no word for an entrant's own file mismatch. ENTRY MISMATCH is a
status of this command, not a card mark. A register fault and a mismatch together print REGISTER FAULT with both
reasons.

Command line (run from the repo root without -I: isolated mode drops the current directory from the import path):
  python -m scoreboard.evalkit.register submit  --register R.jsonl ENTRY.json [--csv ENTRY.csv]
        [--entrant ID] [--task TASK.json] [--package DIR] [--receipt-out receipt.json] [--stamp]
  python -m scoreboard.evalkit.register hash    ENTRY.json [--csv ENTRY.csv]
  python -m scoreboard.evalkit.register verify  --register R.jsonl --receipt receipt.json
        [--entry ENTRY.json [--csv ENTRY.csv]]
  python -m scoreboard.evalkit.register check   --register R.jsonl
Exit codes: 0 done or verified; 1 REGISTER FAULT or ENTRY MISMATCH, or (submit) the line is registered and its
receipt printed but the stamp or the receipt file failed; 2 refused, nothing registered (bad input or the spec is
broken).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import intake as _intake
from .spec import SITE_LIST_RECIPE, SpecError, canonical_json, load_task, site_list_sha256

REGISTER_VERSION = 1
RECEIPT_VERSION = 1
GENESIS = "0" * 64                         # prev_sha256 of the first line
ITEM_TYPES = ("entry", "task", "registration", "card", "receipt")
LINE_KEYS = ("v", "seq", "intake_utc", "item_type", "item_sha256", "prev_sha256", "entrant")
RECEIPT_KEYS = ("v", "item_type", "entry_sha256", "line_sha256", "prev_sha256", "seq", "intake_utc", "entrant")
VERIFIED = "VERIFIED"
REGISTER_FAULT = "REGISTER FAULT"
ENTRY_MISMATCH = "ENTRY MISMATCH"         # files in hand differ from the receipt; the register is intact (D5 point 11)
B3_MANIFEST_NAME = "manifest.json"        # rule book B3 file names
B3_CSV_NAME = "entry.csv"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
_ENTRANT = re.compile(r"[A-Za-z0-9._:-]{1,64}")


class RegisterError(ValueError):
    """The intake refuses: a bad item, a broken register, or a clock that would go backwards."""


# ---------------------------------------------------------------------------------------------------------------
# Lines and hashes
# ---------------------------------------------------------------------------------------------------------------

def utc_text(t: datetime) -> str:
    """UTC time as fixed-width text with microseconds and a Z, so text order is time order."""
    if t.tzinfo is None:
        raise RegisterError("intake time must carry a time zone")
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def line_text(line: dict) -> str:
    """The exact text of one register line (no newline): canonical JSON, sorted keys, no spaces, ASCII."""
    return canonical_json(line)


def line_sha256(text: str) -> str:
    """sha256 of one register line's text (without its newline)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_line_fields(d: object) -> str | None:
    """None if `d` is a well-formed line, else what is wrong with it."""
    if not isinstance(d, dict) or sorted(d) != sorted(LINE_KEYS):
        return "fields differ from the register line spec"
    if d["v"] != REGISTER_VERSION:
        return f"version {d['v']!r} is not {REGISTER_VERSION}"
    if isinstance(d["seq"], bool) or not isinstance(d["seq"], int) or d["seq"] < 1:
        return "seq must be an integer >= 1"
    if not isinstance(d["intake_utc"], str) or not _UTC.fullmatch(d["intake_utc"]):
        return "intake_utc must read YYYY-MM-DDTHH:MM:SS.ffffffZ"
    if d["item_type"] not in ITEM_TYPES:
        return f"item_type must be one of {list(ITEM_TYPES)}"
    for k in ("item_sha256", "prev_sha256"):
        if not isinstance(d[k], str) or not _HEX64.fullmatch(d[k]):
            return f"{k} must be a lowercase 64-character sha256 hex string"
    if d["entrant"] is not None and (not isinstance(d["entrant"], str) or not _ENTRANT.fullmatch(d["entrant"])):
        return "entrant must be null or 1 to 64 characters of letters, digits and ._:-"
    return None


# ---------------------------------------------------------------------------------------------------------------
# Checking the whole register
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class RegisterCheck:
    """The result of reading a register end to end. `status` is VERIFIED or REGISTER FAULT."""
    n_lines: int
    head_sha256: str
    head_seq: int
    head_intake_utc: str | None
    faults: list[dict] = field(default_factory=list)
    lines: list[dict] = field(default_factory=list, repr=False)
    hashes: list[str] = field(default_factory=list, repr=False)

    @property
    def status(self) -> str:
        return VERIFIED if not self.faults else REGISTER_FAULT

    def summary(self) -> dict:
        return {"status": self.status, "n_lines": self.n_lines, "head_seq": self.head_seq,
                "head_sha256": self.head_sha256, "head_intake_utc": self.head_intake_utc, "faults": self.faults}


def check_register(path: str | Path) -> RegisterCheck:
    """Read every line and check the chain: each line well formed and canonical, seq 1, 2, 3 with no gap, each
    prev_sha256 equal to the sha256 of the line before (GENESIS for the first), intake times never decreasing.

    A missing or empty register is an empty chain (head GENESIS, seq 0)."""
    p = Path(path)
    raw = p.read_bytes() if p.exists() else b""
    faults: list[dict] = []
    lines: list[dict] = []
    hashes: list[str] = []
    if raw and not raw.endswith(b"\n"):
        faults.append({"line": raw.count(b"\n") + 1, "fault": "last line has no newline (a cut or partial write)"})
    texts = raw.decode("utf-8", errors="replace").split("\n")
    if texts and texts[-1] == "":
        texts = texts[:-1]
    prev, prev_time, want_seq = GENESIS, None, 1
    for k, text in enumerate(texts, start=1):
        h = line_sha256(text)
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            faults.append({"line": k, "fault": "not JSON"})
            lines.append({})
            hashes.append(h)
            prev, want_seq = h, want_seq + 1
            continue
        bad = _check_line_fields(d)
        if bad:
            faults.append({"line": k, "fault": bad})
        elif line_text(d) != text:
            faults.append({"line": k, "fault": "not canonical JSON (the line was rewritten)"})
        if isinstance(d, dict):
            if d.get("seq") != want_seq:
                faults.append({"line": k, "fault": f"seq {d.get('seq')!r} where {want_seq} was due (a line added "
                                                   "or removed)"})
            if d.get("prev_sha256") != prev:
                faults.append({"line": k, "fault": "prev_sha256 does not match the line before (a line changed, "
                                                   "added or removed)"})
            t = d.get("intake_utc")
            if isinstance(t, str) and prev_time is not None and t < prev_time:
                faults.append({"line": k, "fault": "intake time decreases (R20)"})
            if isinstance(t, str):
                prev_time = t
        lines.append(d if isinstance(d, dict) else {})
        hashes.append(h)
        got = d.get("seq") if isinstance(d, dict) else None
        ok_seq = isinstance(got, int) and not isinstance(got, bool)
        prev, want_seq = h, (got + 1 if ok_seq else want_seq + 1)   # one fault per gap, not one per later line
    head_time = lines[-1].get("intake_utc") if lines else None
    return RegisterCheck(n_lines=len(texts), head_sha256=prev, head_seq=len(texts), head_intake_utc=head_time,
                         faults=faults, lines=lines, hashes=hashes)


# ---------------------------------------------------------------------------------------------------------------
# Appending (intake)
# ---------------------------------------------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def append_item(register: str | Path, item_type: str, item_sha256: str, *, entrant: str | None = None,
                now: Callable[[], datetime] = _now) -> dict:
    """Append one line for an item hash and return its receipt.

    The register is locked for the whole read, check and write, so two intakes cannot take one seq. The intake
    refuses (RegisterError) a register whose chain does not check, and a clock earlier than the head's intake time,
    since writing either would create a REGISTER FAULT. The file is opened for append only; no line is ever
    rewritten."""
    if item_type not in ITEM_TYPES:
        raise RegisterError(f"item_type must be one of {list(ITEM_TYPES)}, got {item_type!r}")
    if not isinstance(item_sha256, str) or not _HEX64.fullmatch(item_sha256):
        raise RegisterError("item_sha256 must be a lowercase 64-character sha256 hex string")
    if entrant is not None and not _ENTRANT.fullmatch(entrant):
        raise RegisterError("entrant must be 1 to 64 characters of letters, digits and ._:-")
    p = Path(register)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            chk = check_register(p)
            if chk.faults:
                raise RegisterError(f"{REGISTER_FAULT}: the register does not check, so nothing was appended "
                                    f"(first fault: line {chk.faults[0]['line']}, {chk.faults[0]['fault']})")
            t = utc_text(now())
            if chk.head_intake_utc is not None and t < chk.head_intake_utc:
                raise RegisterError(f"the clock reads {t}, earlier than the head's intake time "
                                    f"{chk.head_intake_utc}; nothing was appended (R20)")
            line = {"v": REGISTER_VERSION, "seq": chk.head_seq + 1, "intake_utc": t, "item_type": item_type,
                    "item_sha256": item_sha256, "prev_sha256": chk.head_sha256, "entrant": entrant}
            text = line_text(line)
            f.write(text + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return {"v": RECEIPT_VERSION, "item_type": item_type, "entry_sha256": item_sha256,
            "line_sha256": line_sha256(text), "prev_sha256": line["prev_sha256"], "seq": line["seq"],
            "intake_utc": t, "entrant": entrant}


def _csv_candidates(manifest_path: str | Path) -> list[Path]:
    """Where the kit looks for the predictions file when --csv is not given, in order."""
    mp = Path(manifest_path)
    out = [mp.with_suffix(".csv")]                                  # the kit's convention: NAME.csv beside NAME.json
    if mp.name == B3_MANIFEST_NAME:
        out.append(mp.with_name(B3_CSV_NAME))                       # rule book B3: manifest.json beside entry.csv
    return out


def csv_for(manifest_path: str | Path, csv_path: str | Path | None = None) -> Path:
    """The predictions CSV of an entry: `csv_path` if given; else NAME.csv beside NAME.json (the kit's convention);
    else, for a manifest named manifest.json, entry.csv beside it (the rule book B3 names). When none exists, the
    first candidate is returned and `read_entry_files` refuses with every name it tried."""
    if csv_path:
        return Path(csv_path)
    cands = _csv_candidates(manifest_path)
    return next((c for c in cands if c.exists()), cands[0])


def read_entry_files(manifest_path: str | Path, csv_path: str | Path | None = None) -> "_intake.Intake":
    """Read an entry in either manifest form (intake.read_entry: rule book B3 form or kit form), check its schema
    (fields, one finite value per site, probabilities in [0, 1], no duplicate ids) and hash it. SpecError on failure."""
    cp = csv_for(manifest_path, csv_path)
    if not Path(manifest_path).exists():
        raise SpecError(f"{manifest_path}: no such entry manifest")
    if not cp.exists():
        if csv_path:
            raise SpecError(f"--csv {csv_path}: no such predictions file (header site_id,p; one row per task site)")
        tried = " or ".join(c.name for c in _csv_candidates(manifest_path))
        raise SpecError(f"{Path(manifest_path).name}: no predictions file found. Without --csv the kit reads NAME.csv "
                        f"beside NAME.json, and entry.csv beside a manifest.json (rule book B3 names); here it looked "
                        f"for {tried} in {Path(manifest_path).resolve().parent}. Name the predictions file "
                        f"{tried} (header site_id,p), or pass --csv PATH")
    return _intake.read_entry(manifest_path, cp)


def entry_file_sha256(manifest_path: str | Path, csv_path: str | Path | None = None) -> str:
    """The entry hash a receipt carries (intake.read_entry; kit form: spec.entry_sha256, as before D5)."""
    return read_entry_files(manifest_path, csv_path).entry_sha256


def _entrant_for(it: "_intake.Intake", entrant: str | None) -> str | None:
    """Rule book form: the line's entrant is the manifest's entrant_id; a different --entrant is refused (D5)."""
    if it.form != _intake.RULEBOOK_FORM:
        return entrant
    if entrant is not None and entrant != it.entrant:
        raise RegisterError(f"--entrant {entrant!r} differs from the manifest's entrant_id {it.entrant!r}")
    return it.entrant


def submit_entry(register: str | Path, manifest_path: str | Path, csv_path: str | Path | None = None, *,
                 entrant: str | None = None, task: str | Path | None = None, package: str | Path | None = None,
                 package_table: str | Path | None = None, checks_out: dict | None = None,
                 now: Callable[[], datetime] = _now) -> dict:
    """Intake: check the entry's schema (and, with `task`, the entry against that task file and, on a closed task
    that names a package hash, the submitted `package` folder against it, through `package_table` when given),
    hash it as its manifest form says, append it to the register, return the receipt. A kit form manifest that
    states division "closed" is refused (it cannot name a package hash; D5 intake, wave 11). Without `task`,
    `package` is checked against a closed rule book form entry's own package_sha256 (D5 intake review 1 F4).
    `checks_out`, when given, is filled with the checks the intake ran, so the caller prints them without hashing
    the package a second time (review 1 F5)."""
    it = read_entry_files(manifest_path, csv_path)
    if it.form != _intake.RULEBOOK_FORM and it.entry.manifest.division == "closed":
        raise RegisterError(_intake.CLOSED_KIT_FORM)
    if task is not None:
        checks = _intake.check_against_task(it, load_task(task), package, package_table)
        bad = _intake.task_failures(checks)
        if bad:
            raise RegisterError("the entry does not fit the task file: " + "; ".join(bad))
    else:
        checks = {"package": f"{_intake.NOT_CHECKED}: no --task given (R15)"}
        pf = _intake.entry_package_files_check(it, package, package_table)
        if pf != _intake.PASS or package is not None or package_table is not None:
            checks["package_files"] = pf                # left out only when there is nothing to check
        bad = _intake.task_failures(checks)
        if bad:
            raise RegisterError("the entry's package does not fit: " + "; ".join(bad))
    rec = append_item(register, "entry", it.entry_sha256, entrant=_entrant_for(it, entrant), now=now)
    if checks_out is not None:
        checks_out.update(checks)
    return rec


# ---------------------------------------------------------------------------------------------------------------
# Verifying a receipt
# ---------------------------------------------------------------------------------------------------------------

def check_receipt_fields(r: object) -> dict:
    if not isinstance(r, dict) or sorted(r) != sorted(RECEIPT_KEYS):
        raise RegisterError(f"receipt: fields must be exactly {sorted(RECEIPT_KEYS)}")
    if r["v"] != RECEIPT_VERSION:
        raise RegisterError(f"receipt: version {r['v']!r} is not {RECEIPT_VERSION}")
    for k in ("entry_sha256", "line_sha256", "prev_sha256"):
        if not isinstance(r[k], str) or not _HEX64.fullmatch(r[k]):
            raise RegisterError(f"receipt.{k}: must be a lowercase 64-character sha256 hex string")
    if isinstance(r["seq"], bool) or not isinstance(r["seq"], int) or r["seq"] < 1:
        raise RegisterError("receipt.seq: must be an integer >= 1")
    return r


def verify_receipt(register: str | Path, receipt: dict, *, entry_sha: str | None = None) -> dict:
    """Check a receipt against the register. VERIFIED needs all of: the register chain checks from line 1 to the
    head; the line at the receipt's seq exists; that line's sha256, prev_sha256, item hash, item type, intake time
    and entrant equal the receipt's; and, when `entry_sha` is given (the hash of the entry files in hand), it equals
    the receipt's entry hash. When only that last check fails, the status is ENTRY MISMATCH: the register and the
    receipt's line are intact, and the files in hand are not the files received (D5 point 11). Anything else is
    REGISTER FAULT with the reasons listed (R20)."""
    r = check_receipt_fields(receipt)
    chk = check_register(register)
    reasons = [f"register line {x['line']}: {x['fault']}" for x in chk.faults]
    seq = r["seq"]
    at = [k for k, d in enumerate(chk.lines) if d.get("seq") == seq]
    if not at:
        reasons.append(f"the receipt's line {seq} is missing (removed or renumbered; the register has "
                       f"{chk.n_lines} lines)")
    else:
        # No separate position check: a line that sits away from position seq means a seq gap or repeat, which
        # check_register already reports as a chain fault in `reasons`.
        d, h = chk.lines[at[0]], chk.hashes[at[0]]
        if h != r["line_sha256"]:
            reasons.append(f"line {seq} changed: its sha256 differs from the receipt's line_sha256")
        for rk, lk in (("prev_sha256", "prev_sha256"), ("entry_sha256", "item_sha256"),
                       ("item_type", "item_type"), ("intake_utc", "intake_utc"), ("entrant", "entrant"),
                       ("seq", "seq")):
            if d.get(lk) != r[rk]:
                reasons.append(f"line {seq}: {lk} differs from the receipt's {rk}")
    mismatch = entry_sha is not None and entry_sha != r["entry_sha256"]
    if mismatch:
        reasons.append(f"the entry files in hand hash to {entry_sha}, not the receipt's entry_sha256 "
                       f"{r['entry_sha256']} (the files changed after intake, or these are other files)")
    status = REGISTER_FAULT if len(reasons) > int(mismatch) else (ENTRY_MISMATCH if mismatch else VERIFIED)
    return {"status": status, "seq": seq, "entry_sha256": r["entry_sha256"],
            "register_lines": chk.n_lines, "head_sha256": chk.head_sha256, "reasons": reasons}


# ---------------------------------------------------------------------------------------------------------------
# OpenTimestamps (behind --stamp only; never run in tests)
# ---------------------------------------------------------------------------------------------------------------

def head_file(register: str | Path, chk: RegisterCheck) -> Path:
    """Where the head record for a stamp is written: REGISTER.heads/SEQ-HEADSHA.txt."""
    p = Path(register)
    return p.parent / f"{p.name}.heads" / f"{chk.head_seq:08d}-{chk.head_sha256}.txt"


def stamp_head(register: str | Path, *, ots_bin: str = "ots",
               runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict:
    """Stamp the register head with OpenTimestamps: write the head line's seq, sha256 and intake time to a head file
    and run `ots stamp` on it, which leaves HEADFILE.ots beside it. A PUBLICATION: the client sends the file's hash to
    public calendar servers. Call it only behind the explicit --stamp flag on a founder go. Tests pass a fake
    `runner`. Refuses a register that does not check, and an empty one."""
    chk = check_register(register)
    if chk.faults:
        raise RegisterError(f"{REGISTER_FAULT}: the register does not check, so the head was not stamped")
    if chk.n_lines == 0:
        raise RegisterError("the register is empty; there is no head to stamp")
    hf = head_file(register, chk)
    hf.parent.mkdir(parents=True, exist_ok=True)
    hf.write_text(f"register head\nseq {chk.head_seq}\nline_sha256 {chk.head_sha256}\n"
                  f"intake_utc {chk.head_intake_utc}\n", encoding="utf-8")
    cmd = [ots_bin, "stamp", str(hf)]
    try:
        res = runner(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise RegisterError(f"OpenTimestamps client {ots_bin!r} not found; the head file is written, not stamped") \
            from None
    ok = res.returncode == 0
    return {"stamped": ok, "head_seq": chk.head_seq, "head_sha256": chk.head_sha256, "head_file": str(hf),
            "proof": str(hf) + ".ots" if ok else None, "command": cmd,
            "stderr": (res.stderr or "").strip()[-500:] if not ok else ""}


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------

def _finish_submit(a: argparse.Namespace, rec: dict, checks: dict | None = None) -> int:
    """After the line is in the register: always print the receipt. A failed receipt file write or stamp does not
    undo the line, so it is reported beside the receipt (exit 1), never as REFUSED (exit 2). `checks` are the checks
    `submit_entry` ran before the append; they are printed as run, never run again (review 1 F5)."""
    checks = checks or {}
    out: dict = {"receipt": rec}
    try:
        it = read_entry_files(a.entry, a.csv)
        if it.entry_sha256 == rec["entry_sha256"]:
            out["intake"] = {"form": it.form, "hashed_manifest": it.manifest_text,
                             "predictions_file": str(csv_for(a.entry, a.csv)),
                             "task": "checked" if a.task else "not checked (no --task given)",
                             "receipt_signed": False, "register_public": False}
            if a.task:
                out["intake"]["task_checks"] = checks
            out["intake"]["package"] = checks.get("package", f"{_intake.NOT_CHECKED}: no --task given (R15)")
            if "package_files" in checks:
                out["intake"]["package_files"] = checks["package_files"]
        else:                                    # the files changed between the append and the re-read
            out["intake"] = {"error": "the line is registered, but the entry files changed after the append: they "
                                      f"now hash to {it.entry_sha256}, not the receipt's {rec['entry_sha256']}; "
                                      "hashed_manifest is not printed"}
    except (SpecError, OSError) as exc:          # the files changed after the append; the receipt still stands
        out["intake"] = {"error": f"the line is registered, but the entry could not be re-read: {exc}"}
    code = 0
    if a.receipt_out:
        try:
            Path(a.receipt_out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as exc:
            out["receipt_out_error"] = f"the line is registered, but the receipt file was not written: {exc}"
            code = 1
    if a.stamp:
        try:
            out["stamp"] = stamp_head(a.register)
        except (RegisterError, OSError) as exc:
            out["stamp"] = {"stamped": False, "error": f"the line is registered, but the head was not stamped: {exc}"}
        if not out["stamp"]["stamped"]:
            code = 1
    print(json.dumps(out, indent=1, sort_keys=True))
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scoreboard.evalkit.register",
                                 description="Register and receipt (R20): submit, verify, check.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sb = sub.add_parser("submit", help="check an entry's schema, hash it, append it, print the receipt")
    sb.add_argument("--register", required=True)
    sb.add_argument("entry", help="the entry manifest NAME.json")
    sb.add_argument("--csv", help="the predictions file (default NAME.csv beside NAME.json; for a manifest.json, "
                                  "entry.csv beside it, the rule book B3 names)")
    sb.add_argument("--entrant", help="entrant id for a kit form manifest (not inside the entry hash); a rule book "
                                      "form manifest carries entrant_id itself")
    sb.add_argument("--task", help="the task file (R2): refuse an entry whose site list, exposure key, division, "
                                   "task_sha256 or (closed) package_sha256 does not fit it. Site list hash: sha256 "
                                   "of the SORTED site ids, each followed by a newline; row order does not matter "
                                   "for it (it does for the entry hash). `hash ENTRY.json` prints it")
    sb.add_argument("--package", help="closed division: the package folder the entry read (the input files as "
                                      "fetched; keep the package table outside it). With --task, the intake "
                                      "rebuilds the package hash from it and refuses a hash that differs from the "
                                      "task's (R15, B3 step 2). Required on a closed task that names a package "
                                      "hash. Without --task, it is checked against the entry's own package_sha256")
    sb.add_argument("--package-table", help="closed division, with --package: the task's package table. The intake "
                                            "checks that its hash is the package hash, then checks each file in "
                                            "--package against it: no FAIL or MISSING row, no non-hidden file "
                                            "outside the table. Use it when the table has a hidden input row "
                                            "(rule book amendment 2, section 4.3 point 5)")
    sb.add_argument("--receipt-out")
    sb.add_argument("--stamp", action="store_true",
                    help="after the append, stamp the head with OpenTimestamps (a publication: founder go only)")
    vf = sub.add_parser("verify", help="check a receipt against the register: VERIFIED, REGISTER FAULT, or (with "
                                       "--entry) ENTRY MISMATCH when only the files in hand differ")
    vf.add_argument("--register", required=True)
    vf.add_argument("--receipt", required=True)
    vf.add_argument("--entry", help="also check that these entry files hash to the receipt's entry hash "
                                    "(ENTRY MISMATCH if not)")
    vf.add_argument("--csv", help="the predictions file of --entry (default NAME.csv); refused without --entry")
    ck = sub.add_parser("check", help="check the whole register chain")
    ck.add_argument("--register", required=True)
    hs = sub.add_parser("hash", help="print an entry's hash, its manifest form, the exact manifest text hashed, the "
                                     "predictions file read and the entry's site list hash; with --task, also a task "
                                     "file's hash and the exact task text hashed")
    hs.add_argument("entry", nargs="?")
    hs.add_argument("--csv")
    hs.add_argument("--task", help="a task file: print the task hash an entry's task_sha256 must carry (R2)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "submit":
            checks: dict = {}
            rec = submit_entry(a.register, a.entry, a.csv, entrant=a.entrant, task=a.task, package=a.package,
                               package_table=a.package_table, checks_out=checks)
            return _finish_submit(a, rec, checks)
        if a.cmd == "hash":
            if a.entry is None and a.task is None:
                raise RegisterError("hash needs an entry manifest, --task TASK.json, or both")
            if a.csv and not a.entry:
                raise RegisterError("--csv needs an entry manifest")
            out = {}
            if a.entry is not None:
                it = read_entry_files(a.entry, a.csv)
                out.update({"form": it.form, "entry_sha256": it.entry_sha256, "hashed_manifest": it.manifest_text,
                            "entrant": it.entrant, "predictions_file": str(csv_for(a.entry, a.csv)),
                            "site_list_sha256": site_list_sha256(it.entry.site_ids),
                            "site_list_recipe": SITE_LIST_RECIPE})
            if a.task is not None:
                out["task_sha256"], out["hashed_task"] = _intake.task_hash(a.task)
            print(json.dumps(out, indent=1, sort_keys=True))
            return 0
        if a.cmd == "verify":
            if a.csv and not a.entry:
                raise RegisterError("--csv needs --entry (the CSV is checked only with its entry manifest)")
            rec = json.loads(Path(a.receipt).read_text(encoding="utf-8"))
            sha = entry_file_sha256(a.entry, a.csv) if a.entry else None
            res = verify_receipt(a.register, rec, entry_sha=sha)
        else:
            res = check_register(a.register).summary()
        print(json.dumps(res, indent=1, sort_keys=True))
        return 0 if res["status"] == VERIFIED else 1
    except (RegisterError, SpecError, json.JSONDecodeError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

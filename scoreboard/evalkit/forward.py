"""Forward runner (rule R21; eval M6, item D4): freeze, hash, predict, then score when labels arrive.

G1, G5, G6. Item D4 of `docs/evals/OWN-WORK-LIST-2026-11-23.html` (pillar root), founder ruling 45. No em dashes.

```
 watcher flag (G5 state.json)          register (D3 shape, JSONL)
        |                                      |
        v                                      v
 freeze_inputs(type A or B) --> input manifest + sha256 --> log "inputs_frozen"
        |                         '--> publish_input_hash(run=False): anchor text and command only
        v
 predict(type) --> every registered entry on the frozen inputs --> predictions sha256 --> log "predictions_written"
        |
        v   (labels arrive; the label file is opened only here, only after "predictions_written")
 score(type, labels) --> log "labels_opened" --> card (evalkit.card.build_card) per input type, never pooled
```

Input types (as in LEAP, the Liquefaction Experiments and Analysis Projects): type A runs on the inputs frozen at the
flag (the first ShakeMap version and the CPT list known then); type B runs on the final inputs (the last ShakeMap
version). Each type has its own manifest, its own predictions and its own card. The two are never pooled, and a
card names its type.

Blindness. `predict` takes no label argument. `score` is the only function here that opens a label file. It first
checks the run log for this type's "predictions_written" line, checks the log chain and re-hashes every prediction
file; any failure refuses the score before the label file is opened. `predict` refuses a type whose labels were
already opened. `tests/test_evalkit_forward.py` enforces the order with a Python audit hook on file opens.

The run log (`run_log.jsonl`) is append-only and hash-chained: each line carries the sha256 of the line before it.

Watcher flags (read only; `repos/event-scoreboard/scripts/g5_watch.py`, step 9 of 2026-11-05): the watcher writes
`~/.geoliquefy-runs/g5/watch/state.json` with `events: {id: record}`. A record is flagged when `status` is one of
CANDIDATE, ALERT, ALERT UNKNOWN (the M >= 6 census flags) or `class_b.flag` is "CLASS B". `flagged_events` reads
that shape and nothing else; this module never writes the watcher's files.

Register (D3 shape). D3 (register and receipt, R20) was not merged when D4 was built, so this module reads a
register JSONL of the shape below, one line per accepted entry:

    {"seq": 1, "submitted_utc": "2026-11-23T00:00:00Z",
     "entry": {"name": "...", "manifest": {<evalkit.spec.EntryManifest fields, exposure may be {"*": "unseen"}>},
               "runner": {"kind": "builtin", "name": "shaking_score" | "pga_logistic", "params": {...}}
                      or {"kind": "python", "callable": "package.module:function", "params": {...}}},
     "entry_sha256": "<sha256 of canonical_json(entry)>",
     "prev_sha256": "<line_sha256 of the line before, or 64 zeros>",
     "line_sha256": "<sha256 of canonical_json(this line without line_sha256)>"}

`entry_sha256`, `prev_sha256` and `line_sha256` are checked when present. A `python` runner is called as
`fn(inputs: FrozenInputs, **params)` and must return one value per site in `inputs.site_ids` order. It must carry
`code_sha256`, the sha256 of the bytes of the module file that holds the callable (`callable_sha256`); `predict`
re-hashes that file before it imports it and refuses a mismatch, so the code cannot change after registration
unseen. An exposure key "*" is replaced by the flagged earthquake's id at run time.

D3 layout. A register written by D3 itself (evalkit.register, lines {v, seq, intake_utc, item_type, item_sha256,
prev_sha256, entrant}) holds hashes only. `register_entries` reads it with a folder of recipe files: a forward
recipe counts when an "entry" line's item_sha256 equals its `recipe_sha256` (canonical JSON of name, manifest and
runner). That is not the hash D3 `submit_entry` writes (the R4 hash of a manifest plus a predictions CSV), so a
recipe goes on a D3 register through `append_item(register, "entry", recipe_sha256(recipe))`. Entry lines that
match no recipe (ordinary prediction entries) are skipped. A recipe file whose hash is on no line is refused. D3
has no item type for a forward recipe yet; one would make the match explicit (left for D3 and D4 to agree).

D3 and D4 joint test (tests/test_evalkit_d34_joint.py, 2026-11-23) fixed three seams, all on this side:
  1. a D3 register is read only after D3's own `check_register` passes (before, a back-dated line with good links,
     a REGISTER FAULT to D3, was counted as on time here);
  2. `submit_recipe` is the recipe intake: `check_recipe` (the checks `predict` makes that need no frozen input),
     then D3 `append_item`, then the D3 receipt (before, a broken recipe got a VERIFIED receipt and later made
     `predict` refuse every entry of the run);
  3. the input manifest freezes the register prefix (`n_bytes` and its sha256) and `predict` reads only that
     prefix, so a later intake on the live, append-only D3 register does not block a run; any change inside the
     prefix is still refused.
Review 1 of that item (2026-11-23) closed the rest of seam 2 and two trace gaps:
  - `submit_recipe` refuses a recipe hash already registered and (with `recipes_dir`) a name already taken; and
    `predict` never refuses a run for either: it runs the first line of a hash and the first recipe of a name, and
    lists the rest in the predictions record under "register_skipped";
  - a recipe registered after the freeze is listed in "late" with its seq, intake time and the reason
    "registered after the freeze";
  - entry lines on the frozen prefix that match no recipe file are counted in "register_skipped" (count and seqs),
    so a registered recipe whose file is missing from `recipes_dir` leaves a trace.

R21 parts enforced here: the register prefix sha256 is frozen in each input manifest, and `predict` refuses another
register; an entry counts only if its intake time is at least 24 h before the origin (R5, R21), and a LATE entry
is not run and is listed with its intake time; a python entry's code hash is checked; every entry runs twice, and
an entry whose two runs differ by more than 1e-9 is RUN FAILED (not saved, not scored); types A and B must run the
same counted entries. B4 sandbox (ruling 59a, wave 13): every run of every entry is a new process that sees only a
read-only copy of /inputs/ and writes only /outputs/ (evalkit.sandbox; README "B4 sandbox"); an entry that reads,
lists or writes outside, opens a socket or starts a process is SANDBOX REFUSED, listed and not scored. R21 parts NOT
enforced: OCI images, digests and environment hashes; CPU and memory limits outside container mode; the 48 h output limit and the
label join only after it; the ShakeMap choice (R21 takes the first version updated at least 24 h after the origin;
type A takes the version it is given, and the manifest only records whether that version meets the 24 h rule);
the census read at origin plus 7 days; standing items, site universes, primaries and retirement; the runner and
tester rules. A done D4 does not mean R21 is enforced.

B4 input layout (ruling 58c, wave 12). `freeze_inputs` writes each type's input folder `inputs/<type>/` in the
rule book B4 layout, the folder an entry sees as `/inputs/`:

    inputs/<type>/event.json            comcat_id, origin_utc, lon, lat, depth_km, mw, mw_type (from the event the
                                        input manifest freezes; canonical JSON plus one LF)
    inputs/<type>/shakemap/grid.xml     the ShakeMap grid given (a gzip file is stored unzipped, so the name is true;
                                        the manifest keeps the given file's sha256 as source_sha256)
    inputs/<type>/sites.csv             site_id,lon,lat: the run list (sites inside the grid), LF
    inputs/<type>/package/cpt_list.csv  the kit's CPT list (site_id,lat,lon,x_m,y_m) for the kit's own entries

The input manifest is written apart, as `manifests/<type>.json`, so `/inputs/` holds only entry inputs. It freezes
the sha256 of every file in the folder (`inputs.files`) and the package hash of that table (`inputs.package_sha256`,
amendment 2 byte form). FILE_ID is the path inside `/inputs/` (amendment 2 section 5.3; review F2), so the package
table of a closed forward task lists all four files: one table per task (ruling 58b). `load_frozen` refuses a
folder whose files differ from the frozen table in any way (a changed, missing or added file).
Old layout (schema evalkit-forward-inputs/1: `shakemap_grid.xml`, `cpt_list.csv` and `manifest.json` inside
`inputs/<type>/`) is REFUSED with a clear message, not read: freeze again into a new run folder. Only D4 replay runs
with planted labels ever used it; no closed forward task exists. Any package table such a run wrote stays readable by
`python -m scoreboard.evalkit.package hash` and `check` (the byte form and its ids did not change).

Publishing the input hash is wired, not run: `publish_input_hash` writes the anchor text and returns the
OpenTimestamps command; it runs the command only with `run=True` and a non-empty `go`, which nothing in the kit or
its tests passes (ruling 45: PUBLISH items wait for a founder go).
"""
from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import package as _package
from . import register as _register
from . import sandbox as _sandbox
from . import task_builder as _task_builder
from . import shaking
from .spec import (Entry, EntryManifest, SpecError, Task, canonical_json, entry_sha256, load_entry,
                   load_labels, save_entry, site_list_sha256)

SCHEMA_INPUTS = "evalkit-forward-inputs/2"          # B4 layout (ruling 58c)
SCHEMA_INPUTS_OLD = "evalkit-forward-inputs/1"      # kit layout before ruling 58c: refused, not read
SCHEMA_PREDICTIONS = "evalkit-forward-predictions/1"
INPUT_TYPES = {
    "A": "frozen inputs: the first ShakeMap version and the CPT list known at the flag",
    "B": "final inputs: the last ShakeMap version and the CPT list known at scoring",
}
WATCH_FLAG_STATUSES = ("CANDIDATE", "ALERT", "ALERT UNKNOWN")   # g5_watch.FLAG_STATUSES
WATCH_CLASS_B_FLAG = "CLASS B"                                   # g5_class_b.FLAG
ZERO = "0" * 64
CPT_HEADER = ["site_id", "lat", "lon", "x_m", "y_m"]
SITES_HEADER = ["site_id", "lon", "lat"]                       # B4 step 2: sites.csv
B4_EVENT, B4_GRID, B4_SITES, B4_CPT = "event.json", "shakemap/grid.xml", "sites.csv", "package/cpt_list.csv"
B4_FILES = (B4_EVENT, B4_CPT, B4_GRID, B4_SITES)               # sorted: the rows of the forward package table
# B4 event.json field: the kit event field it comes from (event_of)
EVENT_JSON_FIELDS = {"comcat_id": "id", "origin_utc": "time_utc", "lon": "lon", "lat": "lat",
                     "depth_km": "depth_km", "mw": "mag", "mw_type": "mag_type"}
LOG = "run_log.jsonl"


class RunRefused(RuntimeError):
    """The forward runner refused a step (order, hash or blindness)."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(p: Path) -> str:
    return _sha_bytes(Path(p).read_bytes())


def _sha_json(d: dict) -> str:
    return _sha_bytes(canonical_json(d).encode("utf-8"))


def _type(t: str) -> str:
    if t not in INPUT_TYPES:
        raise RunRefused(f"input type must be one of {sorted(INPUT_TYPES)}, got {t!r}")
    return t


# ---------------------------------------------------------------------------------------------------------------
# Watcher flags (read only)
# ---------------------------------------------------------------------------------------------------------------

def is_flagged(rec: dict) -> bool:
    """True when a G5 watcher record carries a census flag or the class (b) flag."""
    return rec.get("status") in WATCH_FLAG_STATUSES or (rec.get("class_b") or {}).get("flag") == WATCH_CLASS_B_FLAG


def flagged_events(state_path: str | Path) -> list[dict]:
    """Flagged records from a watcher state.json, oldest first. Reads the file; writes nothing."""
    S = json.loads(Path(state_path).read_text(encoding="utf-8"))
    ev = [r for r in (S.get("events") or {}).values() if is_flagged(r)]
    return sorted(ev, key=lambda r: r.get("time_utc", ""))


def event_of(rec: dict) -> dict:
    """The event fields the runner freezes: id, time, magnitude, hypocentre, the watcher status."""
    for k in ("id", "time_utc", "mag", "lat", "lon"):
        if rec.get(k) is None:
            raise RunRefused(f"watcher record: {k!r} is missing")
    return {"id": str(rec["id"]), "time_utc": str(rec["time_utc"]), "mag": float(rec["mag"]),
            "mag_type": rec.get("mag_type"), "lat": float(rec["lat"]), "lon": float(rec["lon"]),
            "depth_km": float(rec.get("depth_km") or 0.0), "watch_status": rec.get("status"),
            "watch_class_b": (rec.get("class_b") or {}).get("flag")}


def event_json_bytes(ev: dict) -> bytes:
    """The B4 `event.json` bytes for a frozen kit event (`event_of`): canonical JSON (sorted keys, no spaces, ASCII)
    of the seven B4 fields, then one LF. `mw` and `mw_type` are the watcher's magnitude and its type as given (the
    census read freezes them; a magnitude of another type keeps its own mw_type)."""
    d = {k: ev.get(src) for k, src in EVENT_JSON_FIELDS.items()}
    return (canonical_json(d) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------------------------------------------
# ShakeMap grid
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    """A regular ShakeMap grid of PGA in g, rows from lat_min up, columns from lon_min east."""
    lon_min: float
    lat_min: float
    dlon: float
    dlat: float
    pga_g: np.ndarray = field(repr=False)
    header: dict = field(default_factory=dict)

    def sample(self, lat, lon) -> np.ndarray:
        """Bilinear PGA (g) at each point; NaN outside the grid."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        ny, nx = self.pga_g.shape
        fx, fy = (lon - self.lon_min) / self.dlon, (lat - self.lat_min) / self.dlat
        out = np.full(lat.shape, np.nan)
        ok = (fx >= 0) & (fy >= 0) & (fx <= nx - 1) & (fy <= ny - 1)
        x0 = np.clip(np.floor(fx[ok]).astype(int), 0, nx - 2)
        y0 = np.clip(np.floor(fy[ok]).astype(int), 0, ny - 2)
        tx, ty = fx[ok] - x0, fy[ok] - y0
        g = self.pga_g
        out[ok] = ((1 - tx) * (1 - ty) * g[y0, x0] + tx * (1 - ty) * g[y0, x0 + 1]
                   + (1 - tx) * ty * g[y0 + 1, x0] + tx * ty * g[y0 + 1, x0 + 1])
        return out


def _xml_bytes(path: Path) -> bytes:
    b = Path(path).read_bytes()
    return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b


def read_grid(path: str | Path) -> Grid:
    """Read a ShakeMap grid.xml (plain or gzip). PGA in %g becomes g. Header: event id, version, timestamps."""
    root = ET.fromstring(_xml_bytes(Path(path)))
    ns = {"s": root.tag[1:].split("}")[0]} if root.tag.startswith("{") else {}
    q = (lambda t: f"s:{t}") if ns else (lambda t: t)
    spec = root.find(q("grid_specification"), ns)
    ev = root.find(q("event"), ns)
    fields_ = {f.get("name"): (int(f.get("index")) - 1, f.get("units")) for f in root.findall(q("grid_field"), ns)}
    if "PGA" not in fields_ or "LON" not in fields_ or "LAT" not in fields_:
        raise SpecError(f"{path}: grid needs LON, LAT and PGA fields")
    nlon, nlat = int(spec.get("nlon")), int(spec.get("nlat"))
    data = np.array(root.find(q("grid_data"), ns).text.split(), dtype=float).reshape(-1, len(fields_))
    if data.shape[0] != nlon * nlat:
        raise SpecError(f"{path}: {data.shape[0]} grid rows, expected {nlon * nlat}")
    lon_min, lon_max = float(spec.get("lon_min")), float(spec.get("lon_max"))
    lat_min, lat_max = float(spec.get("lat_min")), float(spec.get("lat_max"))
    dlon, dlat = (lon_max - lon_min) / (nlon - 1), (lat_max - lat_min) / (nlat - 1)
    i_pga, units = fields_["PGA"]
    pga = data[:, i_pga] / (100.0 if units in ("%g", "pctg") else 1.0)
    ix = np.rint((data[:, fields_["LON"][0]] - lon_min) / dlon).astype(int)
    iy = np.rint((data[:, fields_["LAT"][0]] - lat_min) / dlat).astype(int)
    g = np.full((nlat, nlon), np.nan)
    g[iy, ix] = pga
    if not np.all(np.isfinite(g)):
        raise SpecError(f"{path}: grid has holes")
    header = {"event_id": root.get("event_id"), "shakemap_version": root.get("shakemap_version"),
              "process_timestamp": root.get("process_timestamp"), "originator": root.get("shakemap_originator"),
              "map_status": root.get("map_status"),
              "magnitude": float(ev.get("magnitude")) if ev is not None and ev.get("magnitude") else None}
    return Grid(lon_min, lat_min, dlon, dlat, g, header)


# ---------------------------------------------------------------------------------------------------------------
# CPT list
# ---------------------------------------------------------------------------------------------------------------

def write_cpt_list(path: str | Path, site_ids, lat, lon, x_m, y_m) -> None:
    rows = [",".join(CPT_HEADER)] + [f"{s},{float(a)!r},{float(b)!r},{float(c)!r},{float(d)!r}"
                                     for s, a, b, c, d in zip(site_ids, lat, lon, x_m, y_m)]
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_sites(path: str | Path, site_ids, lon, lat) -> None:
    """B4 `sites.csv`: header site_id,lon,lat, one row per site in run order, LF line ends, the last one too."""
    rows = [",".join(SITES_HEADER)] + [f"{s},{float(a)!r},{float(b)!r}" for s, a, b in zip(site_ids, lon, lat)]
    with open(path, "x", encoding="utf-8", newline="") as f:
        f.write("\n".join(rows) + "\n")


def read_sites(path: str | Path) -> tuple[str, ...]:
    """Site ids of a B4 `sites.csv`, in run order. Refuses another header or a repeated id."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split(",") != SITES_HEADER:
        raise SpecError(f"{path}: sites.csv header must be {','.join(SITES_HEADER)} (B4 step 2)")
    ids = tuple(ln.split(",")[0] for ln in lines[1:] if ln.strip())
    if len(set(ids)) != len(ids):
        raise SpecError(f"{path}: duplicate site ids in sites.csv")
    return ids


def read_cpt_list(path: str | Path) -> dict:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split(",") != CPT_HEADER:
        raise SpecError(f"{path}: CPT list header must be {','.join(CPT_HEADER)}")
    rows = [ln.split(",") for ln in lines[1:] if ln.strip()]
    ids = tuple(r[0] for r in rows)
    if len(set(ids)) != len(ids):
        raise SpecError(f"{path}: duplicate site ids in the CPT list")
    arr = np.array([[float(v) for v in r[1:]] for r in rows], dtype=float).reshape(-1, 4)
    return {"site_ids": ids, "lat": arr[:, 0], "lon": arr[:, 1], "x_m": arr[:, 2], "y_m": arr[:, 3]}


# ---------------------------------------------------------------------------------------------------------------
# Run log (append-only, hash-chained)
# ---------------------------------------------------------------------------------------------------------------

def read_log(run_dir: str | Path) -> list[dict]:
    p = Path(run_dir) / LOG
    if not p.exists():
        return []
    out, prev = [], ZERO
    for k, ln in enumerate(p.read_text(encoding="utf-8").splitlines()):
        d = json.loads(ln)
        body = {a: b for a, b in d.items() if a != "line_sha256"}
        if d.get("seq") != k + 1 or d.get("prev_sha256") != prev or _sha_json(body) != d.get("line_sha256"):
            raise RunRefused(f"run log line {k + 1}: broken chain (seq, prev_sha256 or line_sha256)")
        out.append(d)
        prev = d["line_sha256"]
    return out


def _append(run_dir: Path, step: str, input_type: str | None, sha256: str, **extra) -> dict:
    log = read_log(run_dir)
    body = {"seq": len(log) + 1, "utc": _now(), "step": step, "input_type": input_type, "sha256": sha256,
            **extra, "prev_sha256": log[-1]["line_sha256"] if log else ZERO}
    line = {**body, "line_sha256": _sha_json(body)}
    with open(Path(run_dir) / LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")
    return line


def _steps(log: list[dict], step: str, input_type: str) -> list[dict]:
    return [d for d in log if d["step"] == step and d["input_type"] == input_type]


# ---------------------------------------------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------------------------------------------

def _inputs_dir(run_dir: Path, t: str) -> Path:
    """The type's B4 input folder: what an entry sees as /inputs/."""
    return Path(run_dir) / "inputs" / t


def _manifest_path(run_dir: Path, t: str) -> Path:
    """The type's input manifest, kept outside /inputs/ (it is the runner's record, not an entry input)."""
    return Path(run_dir) / "manifests" / f"{t}.json"


def _refuse_old_layout(run_dir: Path, t: str) -> None:
    """Refuse a type frozen in the kit layout before ruling 58c (schema evalkit-forward-inputs/1).

    A `manifest.json` inside a B4 folder is not called the old layout when the type has a new manifest
    (manifests/<type>.json) and the file does not carry the old schema (B4-LAYOUT review 1, N3): it is an added file,
    and `load_frozen` refuses it as one."""
    old = _inputs_dir(run_dir, t) / "manifest.json"
    if old.exists():
        try:
            schema = json.loads(old.read_text(encoding="utf-8")).get("schema")
        except (OSError, ValueError, AttributeError):
            schema = "unreadable"
        if schema != SCHEMA_INPUTS_OLD and _manifest_path(run_dir, t).exists():
            return
        raise RunRefused(
            f"type {t}: {old} is an input manifest in the old kit layout ({schema}: shakemap_grid.xml and "
            f"cpt_list.csv, manifest inside the input folder). Since ruling 58c the forward runner reads only the B4 "
            f"/inputs/ layout ({SCHEMA_INPUTS}: event.json, shakemap/grid.xml, sites.csv, package/). The old layout "
            f"is refused, not read: freeze the inputs again into a new run folder. A package table it wrote stays "
            f"readable by 'python -m scoreboard.evalkit.package hash' and 'check'")


def _utc(s: str | None) -> datetime | None:
    """A UTC time from the register, watcher or ShakeMap text (Z, offset, fraction or none); None if absent."""
    if not s:
        return None
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _shakemap_24h(grid_header: dict, origin_utc: str) -> str:
    """Does this ShakeMap version meet R21's choice (updated at least 24 h after the origin)? Recorded, not enforced."""
    try:
        ts, t0 = _utc(grid_header.get("process_timestamp")), _utc(origin_utc)
    except ValueError:
        return "unknown"
    if ts is None or t0 is None:
        return "unknown"
    return "met" if (ts - t0).total_seconds() >= 24 * 3600 else "not met"


def freeze_inputs(run_dir: str | Path, flag_record: dict, shakemap_grid: str | Path, cpt_list: str | Path, *,
                  input_type: str, register_path: str | Path, shakemap_note: str = "",
                  override_reason: str = "") -> dict:
    """Write the type's B4 input folder, write the input manifest and log its sha256.

    Type A takes the first ShakeMap version; type B the final one. A type is frozen once. The CPT list keeps only
    sites inside the ShakeMap grid (the count left out is in the manifest). The register file's sha256 is frozen in
    the manifest; `predict` refuses any other register. A record the watcher did not flag (`is_flagged`) is refused
    unless `override_reason` says why; the reason goes into the manifest and the log. A record with no `depth_km` or
    no `mag_type` is refused (B4-LAYOUT review 1, F2): event.json never carries a made-up depth or a null mw_type.
    `event_of` itself is unchanged for its other callers. Returns the manifest and hash.
    """
    t = _type(input_type)
    run_dir = Path(run_dir)
    flagged = is_flagged(flag_record)
    if not flagged and not override_reason.strip():
        raise RunRefused("the watcher record is not flagged (is_flagged); pass override_reason to freeze it anyway")
    for k, field_ in (("depth_km", "depth_km"), ("mag_type", "mw_type")):     # B4-LAYOUT review 1, F2
        v = flag_record.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            raise RunRefused(f"the watcher record has no {k!r}: event.json would carry a made-up {field_} (B4 step 2 "
                             f"freezes it from the census read). Add the census-read value to the record and freeze "
                             f"again; nothing was written")
    run_dir.mkdir(parents=True, exist_ok=True)
    for u in INPUT_TYPES:                                 # one layout per run folder
        _refuse_old_layout(run_dir, u)
    if _steps(read_log(run_dir), "inputs_frozen", t):
        raise RunRefused(f"type {t} inputs are already frozen; a frozen input is never replaced")
    ev = event_of(flag_record)
    src = Path(shakemap_grid).read_bytes()
    gz = src[:2] == b"\x1f\x8b"
    grid = read_grid(shakemap_grid)
    if grid.header.get("event_id") and grid.header["event_id"] != ev["id"]:
        raise RunRefused(f"ShakeMap event id {grid.header['event_id']!r} is not the flagged {ev['id']!r}")
    cpt = read_cpt_list(cpt_list)
    pga = grid.sample(cpt["lat"], cpt["lon"])
    keep = np.isfinite(pga)
    kept_ids = tuple(s for s, k in zip(cpt["site_ids"], keep) if k)
    if not kept_ids:
        raise RunRefused("no CPT lies inside the ShakeMap grid")
    d = _inputs_dir(run_dir, t)
    if d.exists() and any(d.iterdir()):
        raise RunRefused(f"type {t}: {d} is not empty but no freeze is logged (a stopped freeze?). The /inputs/ "
                         f"folder is written once; move it aside by hand and freeze again")
    (d / "shakemap").mkdir(parents=True, exist_ok=True)
    (d / "package").mkdir(exist_ok=True)
    (d / B4_EVENT).write_bytes(event_json_bytes(ev))
    (d / B4_GRID).write_bytes(gzip.decompress(src) if gz else src)
    write_sites(d / B4_SITES, kept_ids, cpt["lon"][keep], cpt["lat"][keep])
    write_cpt_list(d / B4_CPT, list(kept_ids), *(cpt[c][keep] for c in ("lat", "lon", "x_m", "y_m")))
    rows = _package.folder_rows(d)
    if tuple(rows) != B4_FILES:
        raise RunRefused(f"type {t}: {d} holds {sorted(rows)}, not the B4 files {list(B4_FILES)}")
    manifest = {
        "schema": SCHEMA_INPUTS, "rule": "R21", "layout": "B4 (ruling 58c)", "input_type": t,
        "input_type_meaning": INPUT_TYPES[t],
        "event": ev, "flag": {"flagged": flagged, "override_reason": override_reason.strip() or None},
        "inputs": {"root": "/inputs/", "files": rows, "package_sha256": _package.table_sha256_of_rows(rows),
                   "table_form": "rule book v1 amendment 2, section 4 (package.TABLE_RECIPE)"},
        "event_json": {"file": B4_EVENT, "fields_from_event": EVENT_JSON_FIELDS},
        "shakemap": {"file": B4_GRID, "source_sha256": _sha_bytes(src), "source_gzip": gz, **grid.header,
                     "note": shakemap_note,
                     "r21_updated_24h_after_origin": _shakemap_24h(grid.header, ev["time_utc"])},
        "sites": {"file": B4_SITES, "n_sites": len(kept_ids), "n_outside_grid": int((~keep).sum()),
                  "site_list_sha256": site_list_sha256(kept_ids)},
        "cpt_list": {"file": B4_CPT, "columns": CPT_HEADER, "why": "the kit's own entries read x_m and y_m"},
        "register": {"sha256": _sha_file(Path(register_path)), "n_bytes": Path(register_path).stat().st_size},
        "frozen_utc": _now(),
    }
    mh = _sha_json(manifest)
    mp = _manifest_path(run_dir, t)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    extra = {} if flagged else {"override_reason": override_reason.strip()}
    _append(run_dir, "inputs_frozen", t, mh, **extra)
    return {"manifest": manifest, "manifest_sha256": mh}


def load_frozen(run_dir: str | Path, input_type: str) -> tuple[dict, str]:
    """The frozen manifest of one type, after checking it against the log and every /inputs/ file against it.

    The folder's table (package.folder_rows: the same walk as `package write`) must equal the frozen one exactly:
    a changed, missing or added file is refused. An added hidden file is refused too (B4-LAYOUT review 1, N4): the
    runner alone writes /inputs/. An old-layout type is refused (`_refuse_old_layout`)."""
    t = _type(input_type)
    run_dir = Path(run_dir)
    _refuse_old_layout(run_dir, t)
    log = _steps(read_log(run_dir), "inputs_frozen", t)
    if not log:
        raise RunRefused(f"type {t}: no inputs frozen")
    manifest = json.loads(_manifest_path(run_dir, t).read_text(encoding="utf-8"))
    mh = _sha_json(manifest)
    if mh != log[0]["sha256"]:
        raise RunRefused(f"type {t}: the input manifest changed after the freeze")
    if manifest.get("schema") != SCHEMA_INPUTS:
        raise RunRefused(f"type {t}: input manifest schema {manifest.get('schema')!r}, this kit reads {SCHEMA_INPUTS}")
    want = manifest["inputs"]["files"]
    d = _inputs_dir(run_dir, t)
    added_hidden = [f for f in _package.hidden_ids(d) if f not in want]    # B4-LAYOUT review 1, N4
    if added_hidden:
        raise RunRefused(f"type {t}: /inputs/ holds hidden files the freeze did not write ({', '.join(added_hidden)}). "
                         f"An entry sees every file in /inputs/, so the runner's folder takes no added file, hidden "
                         f"or not (a Finder .DS_Store too). Remove them and run again")
    try:
        have = _package.folder_rows(d)
    except SpecError as e:
        raise RunRefused(f"type {t}: the /inputs/ folder cannot be read as a package: {e}") from None
    if have != want:
        diff = sorted(f for f in set(want) | set(have) if want.get(f) != have.get(f))
        raise RunRefused(f"type {t}: /inputs/ changed after the freeze ({', '.join(diff)}: a file differs, "
                         f"appeared or went missing)")
    return manifest, mh


def publish_input_hash(run_dir: str | Path, input_type: str, *, run: bool = False, go: str = "") -> dict:
    """Write the anchor text for one frozen input hash and return the OpenTimestamps command.

    Wired, not run: with run=False (the default, and the only value the kit uses) nothing leaves this machine.
    Running needs run=True and a non-empty `go` naming the founder go (ruling 45, PUBLISH)."""
    t = _type(input_type)
    manifest, mh = load_frozen(run_dir, t)
    d = Path(run_dir) / "publish" / t
    d.mkdir(parents=True, exist_ok=True)
    anchor = d / "INPUT-HASH.txt"
    anchor.write_text(f"evalkit forward runner (R21) input manifest\nevent {manifest['event']['id']}\n"
                      f"input type {t}\nsha256 {mh}\n", encoding="utf-8")
    cmd = ["ots", "stamp", str(anchor)]
    ran = False
    if run:
        if not go.strip():
            raise RunRefused("publishing the input hash needs a founder go")
        subprocess.run(cmd, check=True)
        ran = True
    _append(Path(run_dir), "publish_planned" if not ran else "published", t, mh, anchor_sha256=_sha_file(anchor))
    return {"anchor": str(anchor), "command": cmd, "ran": ran, "manifest_sha256": mh}


# ---------------------------------------------------------------------------------------------------------------
# Register (D3 shape)
# ---------------------------------------------------------------------------------------------------------------

def register_line(seq: int, entry: dict, prev_sha256: str, submitted_utc: str) -> dict:
    """One register line of the D3 shape (for tests and replays; D3 owns the real intake)."""
    body = {"seq": seq, "submitted_utc": submitted_utc, "entry": entry, "entry_sha256": _sha_json(entry),
            "prev_sha256": prev_sha256}
    return {**body, "line_sha256": _sha_json(body)}


def write_register(path: str | Path, entries: list[dict], submitted_utc: str = "2026-11-23T00:00:00Z") -> None:
    prev, lines = ZERO, []
    for k, e in enumerate(entries):
        ln = register_line(k + 1, e, prev, submitted_utc)
        lines.append(json.dumps(ln, sort_keys=True))
        prev = ln["line_sha256"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def recipe_sha256(entry: dict) -> str:
    """Hash of one runnable entry recipe (name, manifest, runner): sha256 of its canonical JSON."""
    return _sha_json(entry)


BUILTINS = ("shaking_score", "pga_logistic")


def check_recipe(e: object) -> str:
    """Check one forward recipe at intake and return its recipe_sha256. SpecError on any breach.

    The checks `predict` makes that need no frozen input: name, manifest and runner present; the manifest is a valid
    kit EntryManifest once "*" is read as an earthquake id; a builtin runner is known (pga_logistic needs finite a
    and b); a python runner names 'package.module:function' and its code_sha256 equals callable_sha256 now."""
    if not isinstance(e, dict) or not {"name", "manifest", "runner"} <= set(e):
        raise SpecError("recipe: needs name, manifest and runner")
    if not isinstance(e["name"], str) or not e["name"].strip():
        raise SpecError("recipe.name: must be a non-empty string")
    if not isinstance(e["manifest"], dict) or not isinstance(e["manifest"].get("exposure"), dict):
        raise SpecError(f"recipe {e['name']!r}: manifest must be an object with an exposure object")
    _manifest_for(e, "intake-check")
    r = e["runner"]
    if not isinstance(r, dict):
        raise SpecError(f"recipe {e['name']!r}: runner must be an object")
    params = r.get("params") or {}
    if not isinstance(params, dict):
        raise SpecError(f"recipe {e['name']!r}: runner.params must be an object")
    if r.get("kind") == "builtin":
        if r.get("name") not in BUILTINS:
            raise SpecError(f"recipe {e['name']!r}: builtin runner must be one of {list(BUILTINS)}")
        if r["name"] == "pga_logistic":
            try:
                ok = all(math.isfinite(float(params[k])) for k in ("a", "b"))
            except (KeyError, TypeError, ValueError):
                ok = False
            if not ok:
                raise SpecError(f"recipe {e['name']!r}: pga_logistic needs finite params a and b")
    elif r.get("kind") == "python":
        mod, _, fn = str(r.get("callable", "")).partition(":")
        if not mod or not fn:
            raise SpecError(f"recipe {e['name']!r}: python runner needs 'package.module:function'")
        if not r.get("code_sha256") or callable_sha256(r["callable"]) != r["code_sha256"]:
            raise SpecError(f"recipe {e['name']!r}: code_sha256 must equal callable_sha256 of its module now")
    else:
        raise SpecError(f"recipe {e['name']!r}: runner kind must be builtin or python")
    try:
        return recipe_sha256(e)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"recipe {e['name']!r}: not canonical JSON ({exc})") from None


def submit_recipe(register_path: str | Path, recipe: dict, *, entrant: str | None = None, now=None,
                  recipes_dir: str | Path | None = None) -> dict:
    """Recipe intake on a D3 register: check_recipe, then D3 append_item(register, "entry", recipe_sha256), and
    return the D3 receipt. Nothing is appended when a check fails. `now` is D3's intake clock (tests fix it).

    Refused (SpecError) as well: a recipe whose recipe_sha256 is already on an entry line (an entrant retry gets
    no second line), and, with `recipes_dir`, a recipe whose name is already taken by a registered recipe of
    another hash. The register holds hashes only, so without `recipes_dir` the name check waits for `predict`,
    which runs the first recipe of a name and lists a later one under "register_skipped" (it never refuses the
    run for it). These reads come before D3's lock; `predict` repeats both checks, so a race cannot block a run."""
    h = check_recipe(recipe)
    on = {d.get("item_sha256") for d in _register.check_register(register_path).lines if d.get("item_type") == "entry"}
    if h in on:
        raise SpecError(f"recipe {recipe['name']!r}: this recipe_sha256 is already registered; no second line")
    if recipes_dir is not None:
        for f in sorted(Path(recipes_dir).glob("*.json")):
            e = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(e, dict) and e.get("name") == recipe["name"] and recipe_sha256(e) in on:
                raise SpecError(f"recipe name {recipe['name']!r} is already registered by another recipe "
                                f"({f.name}); choose another name")
    kw = {} if now is None else {"now": now}
    return _register.append_item(register_path, "entry", h, entrant=entrant, **kw)


def callable_sha256(spec: str) -> str:
    """sha256 of the bytes of the module file that holds a 'package.module:function' callable.

    Imports nothing: the file is found folder by folder (sandbox.module_file), so no parent package `__init__.py`
    runs in the kit process (B4 sandbox review 1, B1). Only a `.py` source file is accepted."""
    mod = str(spec).partition(":")[0]
    found = _sandbox.module_file(mod) if mod else None
    if found is None:
        raise SpecError(f"python runner {spec!r}: module source file not found")
    return _sha_file(found)


def _d3_scan(lines: list[str], recipes_dir: Path | None, later: list[dict] = ()) -> dict:
    """Forward recipes from a register written by D3 (evalkit.register, branch eval/d3-register-receipt-2026-11-23).

    A D3 line holds only a hash: {v, seq, intake_utc, item_type, item_sha256, prev_sha256, entrant}, with
    prev_sha256 the sha256 of the line text before it. A forward recipe counts when an "entry" line's item_sha256
    equals the recipe's `recipe_sha256` (put there by D3 `append_item` through `submit_recipe`, not by
    `submit_entry`). Other item types are skipped. A recipe file in `recipes_dir` whose hash is on no line is
    refused: it was changed, or never registered.

    Nothing an entrant can register refuses the run. Each skipped line is kept in the result, so a reader sees it:
      - `after_freeze`: a recipe whose hash is on an entry line after the frozen prefix (`later`, D3 line dicts).
        It is not run; `predict` lists it as LATE with its seq and intake time.
      - `skipped.duplicate_hash`: a second entry line with a hash already seen (an entrant retry). Only the first
        line counts.
      - `skipped.name_taken`: a recipe whose name an earlier line already took (by seq). The first one runs.
      - `skipped.unmatched_entry_lines`: entry lines that match no recipe file. Ordinary prediction entries land
        here by design; so does a registered recipe whose file is missing from `recipes_dir`. Count and seqs only.
    """
    if recipes_dir is None:
        raise SpecError("a D3 register holds hashes only; pass recipes_dir with the runnable entry recipes")
    by_hash = {}
    for f in sorted(Path(recipes_dir).glob("*.json")):
        e = json.loads(f.read_text(encoding="utf-8"))
        by_hash[recipe_sha256(e)] = (f.name, e)
    out, prev, used, names = [], ZERO, set(), set()
    dup, taken, unmatched = [], [], []
    for k, ln in enumerate(lines):          # the D3 check ran first (register_entries); these links stay as a guard
        d = json.loads(ln)
        if d.get("seq") != k + 1 or d.get("prev_sha256") != prev:
            raise SpecError(f"D3 register line {k + 1}: seq or prev_sha256 breaks the chain")
        prev = _sha_bytes(ln.encode("utf-8"))
        h = d.get("item_sha256")
        if d.get("item_type") != "entry":
            continue
        if h not in by_hash:
            unmatched.append(d["seq"])
            continue
        if h in used:
            dup.append({"seq": d["seq"], "register_entry_sha256": h, "intake_utc": d.get("intake_utc")})
            continue
        used.add(h)
        e = by_hash[h][1]
        if e.get("name") in names:
            taken.append({"seq": d["seq"], "name": e.get("name"), "register_entry_sha256": h,
                          "intake_utc": d.get("intake_utc")})
            continue
        names.add(e.get("name"))
        out.append({**e, "entry_sha256": h, "seq": d["seq"], "intake_utc": d.get("intake_utc")})
    after, seen_after = [], set()
    for d in later:
        h = d.get("item_sha256")
        if d.get("item_type") == "entry" and h in by_hash and h not in used and h not in seen_after:
            seen_after.add(h)
            after.append({"name": by_hash[h][1].get("name"), "register_entry_sha256": h,
                          "intake_utc": d.get("intake_utc"), "seq": d.get("seq"),
                          "reason": "registered after the freeze"})
    stray = sorted(by_hash[h][0] for h in by_hash if h not in used and h not in seen_after)
    if stray:
        raise SpecError(f"recipe files whose hash is on no register line (changed or never registered): {stray}")
    return {"entries": out, "after_freeze": after,
            "skipped": {"duplicate_hash": dup, "name_taken": taken,
                        "unmatched_entry_lines": {"n": len(unmatched), "seqs": unmatched}}}


def register_scan(path: str | Path, recipes_dir: str | Path | None = None, *, n_bytes: int | None = None) -> dict:
    """`register_entries` with the skipped lines kept: {"entries", "after_freeze", "skipped"} (see `_d3_scan`).

    Two layouts: the D4 interim JSONL (entry inline, see the module docstring) and a D3 register (hashes only, with
    the recipes in `recipes_dir`). A D3 register must pass D3's own `check_register` (the whole file). With
    `n_bytes`, only the first n_bytes are read (the prefix frozen at the freeze). The interim layout has no skipped
    lines: a repeated name there is refused, as before."""
    data = Path(path).read_bytes()
    text = (data if n_bytes is None else data[:n_bytes]).decode("utf-8")
    raw = [ln for ln in text.splitlines() if ln.strip()]
    if raw and "item_type" in json.loads(raw[0]):
        chk = _register.check_register(path)
        if chk.faults:
            f = chk.faults[0]
            raise SpecError(f"{_register.REGISTER_FAULT}: {path} does not check (D3), so no entry is read "
                            f"(first fault: line {f['line']}, {f['fault']}; {len(chk.faults)} in all)")
        scan = _d3_scan(raw, Path(recipes_dir) if recipes_dir else None, chk.lines[len(raw):])
        if not scan["entries"]:
            raise SpecError(f"{path}: the register holds no entry")
        return scan
    return {"entries": _interim_entries(path, raw), "after_freeze": [], "skipped": None}


def register_entries(path: str | Path, recipes_dir: str | Path | None = None, *,
                     n_bytes: int | None = None) -> list[dict]:
    """Every entry in a register, after checking hashes and the chain where present (see `register_scan`)."""
    return register_scan(path, recipes_dir, n_bytes=n_bytes)["entries"]


def _interim_entries(path: str | Path, raw: list[str]) -> list[dict]:
    out, prev, names = [], ZERO, set()
    for k, ln in enumerate(raw):
        d = json.loads(ln)
        e = d.get("entry")
        if not isinstance(e, dict) or not {"name", "manifest", "runner"} <= set(e):
            raise SpecError(f"register line {k + 1}: entry needs name, manifest and runner")
        if "entry_sha256" in d and d["entry_sha256"] != _sha_json(e):
            raise SpecError(f"register line {k + 1}: entry_sha256 does not match the entry")
        if "prev_sha256" in d and d["prev_sha256"] != prev:
            raise SpecError(f"register line {k + 1}: prev_sha256 breaks the chain")
        if "line_sha256" in d:
            if _sha_json({a: b for a, b in d.items() if a != "line_sha256"}) != d["line_sha256"]:
                raise SpecError(f"register line {k + 1}: line_sha256 does not match")
            prev = d["line_sha256"]
        if e["name"] in names:
            raise SpecError(f"register line {k + 1}: entry name {e['name']!r} is registered twice")
        names.add(e["name"])
        out.append({**e, "entry_sha256": _sha_json(e), "seq": d.get("seq", k + 1),
                    "intake_utc": d.get("submitted_utc")})
    if not out:
        raise SpecError(f"{path}: the register holds no entry")
    return out


def split_late(entries: list[dict], origin_utc: str) -> tuple[list[dict], list[dict]]:
    """(counted, late): an entry counts only if its intake time is at least 24 h before the origin (R5, R21).

    An entry with no intake time cannot show it was in time, so it is LATE. A LATE entry is not run."""
    t0 = _utc(origin_utc)
    counted, late = [], []
    for e in entries:
        try:
            ti = _utc(e.get("intake_utc"))
        except ValueError:
            ti = None
        ok = ti is not None and (t0 - ti).total_seconds() >= 24 * 3600
        (counted if ok else late).append(e)
    return counted, late


# ---------------------------------------------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FrozenInputs:
    """What an entry sees: the frozen event, CPT list and ShakeMap PGA at each CPT. No label field exists."""
    input_type: str
    event: dict
    site_ids: tuple[str, ...]
    lat: np.ndarray = field(repr=False)
    lon: np.ndarray = field(repr=False)
    x_m: np.ndarray = field(repr=False)
    y_m: np.ndarray = field(repr=False)
    pga_g: np.ndarray = field(repr=False)
    manifest_sha256: str = ""


def frozen_inputs(run_dir: str | Path, input_type: str) -> FrozenInputs:
    manifest, mh = load_frozen(run_dir, input_type)
    d = _inputs_dir(Path(run_dir), input_type)
    cpt = read_cpt_list(d / B4_CPT)
    if read_sites(d / B4_SITES) != cpt["site_ids"]:
        raise RunRefused(f"type {input_type}: sites.csv and package/cpt_list.csv name other sites")
    pga = read_grid(d / B4_GRID).sample(cpt["lat"], cpt["lon"])
    return FrozenInputs(input_type, manifest["event"], cpt["site_ids"], cpt["lat"], cpt["lon"], cpt["x_m"],
                        cpt["y_m"], pga, mh)


def inputs_from_folder(folder: str | Path, input_type: str, manifest_sha256: str = "") -> FrozenInputs:
    """What a sandboxed entry sees (ruling 59a): FrozenInputs built from one B4 /inputs/ folder and nothing else.

    The event comes from /inputs/event.json, with the B4 names mapped back to the kit names (EVENT_JSON_FIELDS:
    comcat_id -> id, origin_utc -> time_utc, mw -> mag, mw_type -> mag_type; lon, lat, depth_km). The watcher
    fields of the input manifest (watch_status, watch_class_b) are not entry inputs and are absent here. The sandbox
    child (evalkit.sandbox_child) calls this before the entry runs; the parent checks the folder against the frozen
    table first (evalkit.sandbox)."""
    d = Path(folder)
    ej = json.loads((d / B4_EVENT).read_text(encoding="utf-8"))
    event = {src: ej.get(b4) for b4, src in EVENT_JSON_FIELDS.items()}
    cpt = read_cpt_list(d / B4_CPT)
    if read_sites(d / B4_SITES) != cpt["site_ids"]:
        raise RunRefused(f"type {input_type}: sites.csv and package/cpt_list.csv name other sites")
    pga = read_grid(d / B4_GRID).sample(cpt["lat"], cpt["lon"])
    return FrozenInputs(input_type, event, cpt["site_ids"], cpt["lat"], cpt["lon"], cpt["x_m"], cpt["y_m"], pga,
                        manifest_sha256)


def _builtin(name: str, inp: FrozenInputs, params: dict) -> np.ndarray:
    mw = float(params.get("mw", inp.event["mag"]))
    if name == "shaking_score":
        return shaking.score(inp.pga_g, mw)
    if name == "pga_logistic":
        a, b = float(params["a"]), float(params["b"])
        z = a + b * np.log(np.asarray(inp.pga_g) / shaking.msf(mw))
        return 1.0 / (1.0 + np.exp(-z))
    raise SpecError(f"unknown builtin runner {name!r}")


def _run_entry(e: dict, inp: FrozenInputs, run_dir: Path, manifest: dict, *, sandbox: str,
               sandbox_opts: dict) -> np.ndarray:
    """One run of one entry, in a new sandboxed process that sees only a copy of /inputs/ (ruling 59a; sandbox.py).

    Raises sandbox.SandboxRefused when the entry tried to read outside /inputs/, write outside /outputs/, open a
    socket or start a process; SpecError for a bad runner, a changed module file or a wrong result shape."""
    p = _sandbox.run_entry(e, _inputs_dir(run_dir, inp.input_type), manifest["inputs"]["files"],
                           input_type=inp.input_type, manifest_sha256=inp.manifest_sha256, mode=sandbox,
                           **sandbox_opts)
    p = np.asarray(p, dtype=float)
    if p.shape != (len(inp.site_ids),):
        raise SpecError(f"entry {e['name']!r}: returned {p.shape}, needs one value per frozen site")
    return p


def _manifest_for(e: dict, event_id: str) -> EntryManifest:
    m = dict(e["manifest"])
    m["exposure"] = {(event_id if k == "*" else k): v for k, v in dict(m["exposure"]).items()}
    return EntryManifest.from_dict(m)


def _pred_dir(run_dir: Path, t: str) -> Path:
    return Path(run_dir) / "predictions" / t


def predict(run_dir: str | Path, register_path: str | Path, input_type: str, *,
            recipes_dir: str | Path | None = None, sandbox: str = "process",
            sandbox_opts: dict | None = None) -> dict:
    """Run every registered entry on one type's frozen inputs and log the predictions hash. Takes no labels.

    Refused when the type has predictions already, when its labels were already opened, when the register is not
    the one frozen in the input manifest, or when the other type already ran a different set of counted entries.
    LATE entries (intake less than 24 h before the origin) are not run. Each counted entry runs twice; one whose two
    runs differ by more than RERUN_TOL is RUN FAILED and is not saved. The predictions hash is sha256 of the
    canonical JSON {input manifest hash, frozen register prefix hash, {entry name: R4 entry hash}, late, run failed,
    sandbox refused, sandbox}.

    B4 sandbox (ruling 59a): each run is a new process (`sandbox="process"`, or `"container"` with
    sandbox_opts={"image": ...}) that sees only a read-only copy of /inputs/ and writes only /outputs/. An entry that
    tries to read outside /inputs/, write outside /outputs/, open a socket or start a process is SANDBOX REFUSED: it
    is not run again, not saved and not scored, and the record lists it with the reason. There is no in-process mode.
    `sandbox_opts` passes timeout_s, image, tmp_root and allow_audit_only to sandbox.run_entry. Fail closed (review
    1, S1): with no kernel layer (sandbox-exec cannot start) process mode is refused before anything is written,
    unless sandbox_opts has allow_audit_only=True; the record then says the audit layer ran alone."""
    t = _type(input_type)
    opts = dict(sandbox_opts or {})
    lay = _sandbox.layers(sandbox, allow_audit_only=bool(opts.get("allow_audit_only", False)))
    run_dir = Path(run_dir)
    log = read_log(run_dir)
    if any(d["step"] == "labels_opened" for d in log):
        raise RunRefused(f"type {t}: labels were opened before these predictions; the run is not blind")
    if _steps(log, "predictions_written", t):
        raise RunRefused(f"type {t}: predictions already written; an entry is run once per type")
    inp = frozen_inputs(run_dir, t)
    manifest, _ = load_frozen(run_dir, t)
    reg_raw = Path(register_path).read_bytes()
    n = manifest["register"].get("n_bytes", len(reg_raw))      # the prefix frozen at the freeze (D3 is append only)
    reg_sha = _sha_bytes(reg_raw[:n])
    if len(reg_raw) < n or reg_sha != manifest["register"]["sha256"]:
        raise RunRefused(f"type {t}: the register is not the one frozen in the input manifest")
    scan = register_scan(register_path, recipes_dir, n_bytes=n)
    counted, late = split_late(scan["entries"], inp.event["time_utc"])
    other = [u for u in INPUT_TYPES if u != t and _steps(log, "predictions_written", u)]
    if other:
        prev = json.loads((_pred_dir(run_dir, other[0]) / "predictions.json").read_text(encoding="utf-8"))
        if sorted(prev["counted_register_entries"]) != sorted(e["entry_sha256"] for e in counted):
            raise RunRefused(f"types {other[0]} and {t} must run the same counted entries")
    d = _pred_dir(run_dir, t)
    d.mkdir(parents=True, exist_ok=True)
    hashes, failed, refused = {}, [], []
    for e in counted:
        try:
            p1 = _run_entry(e, inp, run_dir, manifest, sandbox=sandbox, sandbox_opts=opts)
            p2 = _run_entry(e, inp, run_dir, manifest, sandbox=sandbox, sandbox_opts=opts)
        except _sandbox.SandboxRefused as exc:
            refused.append({**exc.record(), "register_entry_sha256": e["entry_sha256"]})
            continue
        same_nan = np.array_equal(np.isnan(p1), np.isnan(p2))
        diff = np.nanmax(np.abs(p1 - p2)) if same_nan and np.isfinite(p1).any() else 0.0
        if not same_nan or diff > RERUN_TOL:
            failed.append({"name": e["name"], "register_entry_sha256": e["entry_sha256"],
                           "reason": "RUN FAILED: a second run differs by more than 1e-9"})
            continue
        entry = Entry(_manifest_for(e, inp.event["id"]), inp.site_ids, p1)
        stem = _safe(e["name"])
        save_entry(entry, d / f"{stem}.json", d / f"{stem}.csv")
        hashes[e["name"]] = {"file": stem, "entry_sha256": entry_sha256(load_entry(d / f"{stem}.json",
                                                                                   d / f"{stem}.csv")),
                             "register_entry_sha256": e["entry_sha256"], "intake_utc": e["intake_utc"]}
    if not hashes:
        raise RunRefused(f"type {t}: no entry counts for this earthquake (LATE {len(late)}, RUN FAILED {len(failed)}, "
                         f"SANDBOX REFUSED {len(refused)})")
    rec = {"schema": SCHEMA_PREDICTIONS, "input_type": t, "input_manifest_sha256": inp.manifest_sha256,
           "register_sha256": reg_sha, "origin_utc": inp.event["time_utc"], "entries": hashes,
           "counted_register_entries": sorted(e["entry_sha256"] for e in counted),
           "late": [{"name": e["name"], "register_entry_sha256": e["entry_sha256"], "intake_utc": e["intake_utc"]}
                    for e in late] + scan["after_freeze"],
           "run_failed": failed, "sandbox_refused": refused, "sandbox": lay.record()}
    if scan["skipped"] is not None:      # D3 layout only, so an interim record keeps its old shape
        rec["register_skipped"] = scan["skipped"]
    ph = _sha_json(rec)
    (d / "predictions.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    _append(run_dir, "predictions_written", t, ph, n_entries=len(hashes), n_sites=len(inp.site_ids),
            n_late=len(late), n_run_failed=len(failed), n_sandbox_refused=len(refused))
    return {"predictions_sha256": ph, "record": rec}


RERUN_TOL = 1e-9   # R21: a second run must match or differ by at most 1e-9


def _safe(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    return s or "entry"


def _check_predictions(run_dir: Path, t: str) -> tuple[dict, list[Entry]]:
    """Predictions of one type, re-hashed against the record and the log. Refuses on any mismatch."""
    log = _steps(read_log(run_dir), "predictions_written", t)
    if not log:
        raise RunRefused(f"type {t}: no predictions hash in the run log; labels stay closed")
    d = _pred_dir(run_dir, t)
    rec = json.loads((d / "predictions.json").read_text(encoding="utf-8"))
    if _sha_json(rec) != log[0]["sha256"]:
        raise RunRefused(f"type {t}: the predictions record changed after its hash was logged")
    if rec.get("input_type") != t:
        raise RunRefused(f"type {t}: the predictions record is for type {rec.get('input_type')!r}")
    _, mh = load_frozen(run_dir, t)
    if rec["input_manifest_sha256"] != mh:
        raise RunRefused(f"type {t}: predictions were made on other inputs")
    entries = []
    for name, h in rec["entries"].items():
        e = load_entry(d / f"{h['file']}.json", d / f"{h['file']}.csv")
        if entry_sha256(e) != h["entry_sha256"]:
            raise RunRefused(f"type {t}: entry {name!r} changed after the predictions hash")
        entries.append(e)
    return rec, entries


# ---------------------------------------------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------------------------------------------

TASK_FIELDS_FROM_CALLER = ("target", "label_source", "label_rule_sha256", "tile_m", "clip", "seed", "label_caveats")


def score(run_dir: str | Path, labels_path: str | Path, input_type: str, task_params: dict, *,
          card_kwargs: dict | None = None, card_version: str = "later") -> dict:
    """Score one type's predictions on the arrived labels. The label file is opened only after every check passes.

    `task_params` gives the task fields the runner cannot know (TASK_FIELDS_FROM_CALLER, plus optional Task fields
    such as n_boot or min_sites); event id, site list hash and track ("forward") come from the frozen inputs.
    On the closed division (the default) the package hash comes from the frozen inputs too (R2, R15; D5 follow-ups,
    wave 10; B4 layout, ruling 58c): the package table (amendment 2 byte form) lists every file of this type's B4
    /inputs/ folder (event.json, package/cpt_list.csv, shakemap/grid.xml, sites.csv; FILE_ID is the path inside
    /inputs/), with the sha256 the input manifest froze; package_sha256 is the sha256 of that table. The table is
    written as `scores/<type>/package_table.csv`, so `python -m scoreboard.evalkit.package check TABLE inputs/<type>`
    checks the frozen inputs against it with nothing left over. An open task names no package hash.
    Pin check (amended R2; ruling 59a, wave 13): clip, tile_m, seed and tile_rule must equal task_pins.json for
    `card_version` (default "later": a forward card is a later card version); a task off its pins is refused before
    the label file is opened (task_builder.check_pins). The caller still gives them; they are checked, not filled.
    Returns the card (evalkit.card.build_card) with a `forward` block naming the input type and every hash."""
    from .card import build_card  # late import: the card module is heavy and not needed to freeze or predict

    t = _type(input_type)
    run_dir = Path(run_dir)
    out = run_dir / "scores" / t
    if _steps(read_log(run_dir), "scored", t) or (out / "card.json").exists():
        raise RunRefused(f"type {t} is already scored; a card is written once and never replaced")
    rec, entries = _check_predictions(run_dir, t)       # refuses before any label byte is read
    manifest, mh = load_frozen(run_dir, t)
    missing = [k for k in TASK_FIELDS_FROM_CALLER if k not in task_params]
    if missing:
        raise RunRefused(f"task_params: missing {missing}")
    for k in ("event_id", "site_list_sha256", "track", "package_sha256"):
        if k in task_params:
            raise RunRefused(f"task_params: {k} comes from the frozen inputs, not the caller")
    division = task_params.get("division", "closed")
    pkg = {}
    if division == "closed":                             # R2, R15: a closed task names its package hash
        pkg_rows = manifest["inputs"]["files"]
        pkg_text = _package.table_text(pkg_rows)
        pkg = {"package_sha256": _package.table_sha256_of_rows(pkg_rows)}
        if pkg["package_sha256"] != manifest["inputs"]["package_sha256"]:
            raise RunRefused(f"type {t}: the frozen package hash does not match its own rows")
    task = Task.from_dict({**task_params, "event_id": manifest["event"]["id"],
                           "site_list_sha256": manifest["sites"]["site_list_sha256"], "track": "forward",
                           "division": division, **pkg})
    try:                                                 # scorer-side pin check, before any label byte
        _task_builder.check_pins(task, card_version)
    except SpecError as exc:
        raise RunRefused(f"type {t}: {exc}") from None
    raw = Path(labels_path).read_bytes()                 # first open of the label file
    lh = _sha_bytes(raw)
    first = [d for d in read_log(run_dir) if d["step"] == "labels_opened"]
    _append(run_dir, "labels_opened", t, lh)            # every open is logged, a refused one too
    if first and first[0]["sha256"] != lh:
        same = first[0]["input_type"] == t
        raise RunRefused(f"type {t} was already opened on another label file; one label file per run"
                         if same else "types A and B must be scored on the same label file")
    labels = load_labels(labels_path)
    card = build_card(task, entries, labels, **(card_kwargs or {}))
    card["forward"] = {"rule": "R21", "input_type": t, "input_type_meaning": INPUT_TYPES[t],
                       "input_manifest_sha256": mh, "predictions_sha256": _sha_json(rec),
                       "late": rec.get("late", []), "run_failed": rec.get("run_failed", []),
                       "sandbox_refused": rec.get("sandbox_refused", []), "sandbox": rec.get("sandbox"),
                       "pins_checked": {"card_version": card_version,
                                        "pins": _task_builder.load_pins(card_version)},
                       "register_skipped": rec.get("register_skipped"),
                       "labels_file_sha256": lh, "scored_apart": "type A and type B are separate cards, never pooled",
                       "package_sha256": task.package_sha256,
                       "package_table": "package_table.csv" if task.package_sha256 else None}
    out.mkdir(parents=True, exist_ok=True)
    if task.package_sha256:
        tp = out / "package_table.csv"
        try:
            with open(tp, "x", encoding="utf-8", newline="") as f:
                f.write(pkg_text)
        except FileExistsError:                          # left by a run stopped before its card: same bytes only
            if tp.read_bytes() != pkg_text.encode("utf-8"):
                raise RunRefused(f"type {t}: {tp} exists with other bytes; a package table is never replaced")
    with open(out / "card.json", "x", encoding="utf-8") as f:   # "x": never overwrite a card
        f.write(json.dumps(card, indent=1, sort_keys=True, default=_json_default) + "\n")
    _append(run_dir, "scored", t, _sha_file(out / "card.json"))
    return card


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and math.isnan(o):
        return None
    raise TypeError(type(o).__name__)


def blind_order(run_dir: str | Path) -> dict:
    """For each type: was "predictions_written" logged before "labels_opened"? The record a reader checks."""
    log = read_log(run_dir)
    out = {}
    for t in INPUT_TYPES:
        p = _steps(log, "predictions_written", t)
        lab = _steps(log, "labels_opened", t)
        if p or lab:
            out[t] = {"predictions_seq": p[0]["seq"] if p else None, "labels_seq": lab[0]["seq"] if lab else None,
                      "blind": bool(p) and (not lab or p[0]["seq"] < lab[0]["seq"])}
    return out

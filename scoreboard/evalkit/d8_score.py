"""D8 scoring of predictions v3 on world set v2, one registered run (eval M5, ruling 55b). G1, G5, G6. No em dashes.

Registration: pillar `docs/evals/D8-SCORING-REGISTRATION-2026-11-24.md` (frozen, ecca48f) with its amendment 1
(`docs/evals/D8-SCORING-REGISTRATION-AMENDMENT-1-2026-11-24.md`). Section numbers below are the registration's;
"A1-n" is amendment 1 item n.

```
 check hashes (sections 1, 2; no label file)  ->  STARTED.txt  ->  hash label file (stop on mismatch, 2.1)
     -> parse labels (3.2; duplicate stops, A1-4 missing and out-of-scope rows counted)
     -> per group, per PGA variant: one tile bootstrap for every row, pair and reference (5.1, 5.2, A1-8)
     -> F1 lines: R1 (b) to (e) applied HERE first (A1-2, A1-3); only lines left go to interval_policy.pooled_lines
     -> descriptive lines (7.3) -> word check outside F1 -> aggregate scan, groups keyed by name (A1-7) -> write
```

This module opens no file. The driver `scripts/d8_score_2026_11_24.py` reads files, checks hashes and writes outputs.
Tests use planted labels only.
"""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np

from ..scorer import auc
from . import card as C
from . import interval_policy as IP
from . import metrics as M
from . import pool as P
from . import resample as RS
from . import world_inputs as W

SCHEMA = "evalkit-d8-scores/1"
KIT_BASE = "bacbdd1"
ADDED_FILES = ("scoreboard/evalkit/d8_score.py", "scripts/d8_score_2026_11_24.py", "tests/test_evalkit_d8_score.py")
REGISTRATION = "docs/evals/D8-SCORING-REGISTRATION-2026-11-24.md"
REGISTRATION_SHA256 = "52bfa7174f144c9378213dabe4d9d2ac5600d510343c22a7494e674caf17076b"
AMENDMENT = "docs/evals/D8-SCORING-REGISTRATION-AMENDMENT-1-2026-11-24.md"

DATA = "data/world-inputs-v2-2026-11-23"
INPUTS_DIR = f"{DATA}/inputs"
PRED_DIR = f"{DATA}/predictions"
LABEL_FILE = f"{DATA}/labels-sealed/labels.parquet"
LABEL_SHA256 = "563f7b6421f26671f30771e302780275fd207a1cb02232793ec614625eb8bc4f"
PRED_MANIFEST = "PREDICTIONS-MANIFEST-v3.json"
PRED_MANIFEST_SHA256 = "40024814fb7f055a31a994414191eab6940097a6995cdbb2d86867b4057633a2"
PROOF_FILE = "PREDICTIONS-MANIFEST-v3.json.ots"
PROOF_SHA256 = "fc45d876c13e92de52e66c4bd83cda4675bc976c8d6b1a6621d07079d5e141c8"
INPUTS_MANIFEST = "INPUTS-MANIFEST-v2.json"
INPUTS_MANIFEST_SHA256 = "199ad71e942260f134014bd5eba26dcef3b2f8405eae8f0e276bbee451852579"
CASES_FILE = "cases-v2-elbistan-pga-2026-11-24.parquet"
CASES_SHA256 = "683cdc6483401d6271d8fe364cae7233e0e1312ffd899e573272cea699e41d7b"

# Section 1: the 18 files. name -> (row name, PGA variant, sha256)
FILES = {
    "level-0__pga_primary.csv": ("Level 0 (HOUSE)", "pga_primary",
                                 "b6c9b5e397bb5d730a4761516326f354919874c1cae286977a4f4439afcd6e2a"),
    "level-0__pga_prj4022.csv": ("Level 0 (HOUSE)", "pga_prj4022",
                                 "def1c8ae0fb474f148aff539c5aeb3c0ad551ebd0c9c77c1083f93367d425bd3"),
    "bi14-lpi--default-2026-11__pga_primary.csv": ("BI14-LPI [default-2026-11]", "pga_primary",
                                                   "1ef00284de85222228fb1fb90d988c0b55bfdfb98e5ea37527db3cfa54ac0bee"),
    "bi14-lpi--default-2026-11__pga_prj4022.csv": ("BI14-LPI [default-2026-11]", "pga_prj4022",
                                                   "3e3bccef271432307b490988b51acec9e4f38411e944739e5ccfc07460e70462"),
    "bi14-lsn--default-2026-11__pga_primary.csv": ("BI14-LSN [default-2026-11]", "pga_primary",
                                                   "c0ab0d406fd08e4b49c8f313958db69cc5f4a58c1897c577532cbd5b748f279b"),
    "bi14-lsn--default-2026-11__pga_prj4022.csv": ("BI14-LSN [default-2026-11]", "pga_prj4022",
                                                   "baad4cc158840c69bd890ff485c730b5b6e4363d77a1df2d097f208c565cf686"),
    "bi14-lpi--authors-maurer-2026-11-19__pga_primary.csv": (
        "BI14-LPI [authors-maurer-2026-11-19]", "pga_primary",
        "d5b4f696887c8987c375438eab4d5cc220129cf780a25bf71aacb3c6fc307a95"),
    "bi14-lpi--authors-maurer-2026-11-19__pga_prj4022.csv": (
        "BI14-LPI [authors-maurer-2026-11-19]", "pga_prj4022",
        "b99bdd9c29843285bee69cf86378684101aa241b85ebb1d299f921c77286554d"),
    "bi14-lsn--authors-maurer-2026-11-19__pga_primary.csv": (
        "BI14-LSN [authors-maurer-2026-11-19]", "pga_primary",
        "8a4ae8c0468759a81140b024b988464ac25c4d24a44f1106623f8aa65452b2f0"),
    "bi14-lsn--authors-maurer-2026-11-19__pga_prj4022.csv": (
        "BI14-LSN [authors-maurer-2026-11-19]", "pga_prj4022",
        "e0d6feccde2b41ae88b37bf4f71daa63d3738e0837b8727d95adf329a0e75feb"),
    "bi14-lpi--peer-global-2026-11-23__pga_primary.csv": (
        "BI14-LPI [peer-global-2026-11-23]", "pga_primary",
        "2662a043a66d8159a893a57ce2de4db0b2fa60f8a3463b40de3f8214a7c56f2f"),
    "bi14-lpi--peer-global-2026-11-23__pga_prj4022.csv": (
        "BI14-LPI [peer-global-2026-11-23]", "pga_prj4022",
        "01bb750003c04034aa0f039ae19648250a8a2cd73377d4f1b42d995f515cd3a4"),
    "bi14-lsn--peer-global-2026-11-23__pga_primary.csv": (
        "BI14-LSN [peer-global-2026-11-23]", "pga_primary",
        "bf72b6f0cd5ecedaa9f0cee85fa51fbdbd4e57e2ba458c59f8037478e733fd2e"),
    "bi14-lsn--peer-global-2026-11-23__pga_prj4022.csv": (
        "BI14-LSN [peer-global-2026-11-23]", "pga_prj4022",
        "61170e7966f5c867c96b1b10642ac740872de15b1cece48b74d07570299c5347"),
    "baseline--shaking-score__pga_primary-inputs-v2-2026-11-24.csv": (
        "baseline: shaking score", "pga_primary",
        "5379e8730370d40f81f1d8b347abe39d3581bc254501554852d7e737085d68c0"),
    "baseline--shaking-score__pga_prj4022.csv": ("baseline: shaking score", "pga_prj4022",
                                                 "b018dc183322d25de27990684b5d2b6b72b56c5e0dcb82a57ec7a36e64b27eed"),
    "usgs-zhu-2017__usgs-grid-georef-2026-11-23-ruling46.csv": (
        "USGS Zhu 2017 (ruling 46)", "pga_primary",
        "9b5789c00fbd117359d701e6187b48c1a4da6a7e439d35d8a51af2f8c254a1fb"),
    "usgs-zhu-2017__usgs-grid-georef-2026-11-23.csv": (
        "USGS Zhu 2017 (raw read, beside)", "pga_primary",
        "6cc1df7f7a40ac287e01e7fc0781e4a36d10505cdf69ecb366fa627e98872b24"),
}
LEVEL0 = "Level 0 (HOUSE)"
SHAKING = "baseline: shaking score"
LPI = "BI14-LPI [default-2026-11]"
ZHU46 = "USGS Zhu 2017 (ruling 46)"
ZHU_RAW = "USGS Zhu 2017 (raw read, beside)"
BI14_ROWS = tuple(f"BI14-{o} [{c}]" for c in ("default-2026-11", "authors-maurer-2026-11-19",
                                              "peer-global-2026-11-23") for o in ("LPI", "LSN"))
VARIANTS = ("pga_primary", "pga_prj4022")
# Rows per variant, Level 0 first (7.2 point 7). Zhu rows take the pga_primary bootstrap (A1-9).
ROWS = {"pga_primary": (LEVEL0, *BI14_ROWS, SHAKING, ZHU46, ZHU_RAW),
        "pga_prj4022": (LEVEL0, *BI14_ROWS, SHAKING)}
MAIN = {("pga_primary", LEVEL0), ("pga_primary", "BI14-LPI [default-2026-11]"),
        ("pga_primary", "BI14-LSN [default-2026-11]"), ("pga_primary", SHAKING), ("pga_primary", ZHU46)}
EXPOSURE_COL = {LEVEL0: "kit_exp_bi14", SHAKING: "kit_exp_shaking", ZHU46: "kit_exp_zhu2017",
                ZHU_RAW: "kit_exp_zhu2017", **{r: "kit_exp_bi14" for r in BI14_ROWS}}

N_BOOT = 2000
SEED = 20261106
TILE_M = 1000.0
MAIN_CLIP = 1e-6
RECORD_GATE = {"yes_sites": 3, "no_sites": 3, "tiles": 5}             # 5.3 point 1
R27_GATE = {"sites": 200, "yes_sites": 30, "no_sites": 30, "tiles": 20}  # 5.3 point 2
PRJ_UNLABELLED_STOP = 0.01                                              # 3.2
NGL = "NGL-API-2026-11-20"
PRJ = "PRJ-6544"
SOURCE_SHORT = {NGL: "NGL", PRJ: "PRJ-6544"}

# 9.3: the R1 (d) list, matched by exact `earthquake` name (precheck N2).
R1D_LIST = frozenset((
    "Achaia-Ilia", "Baja", "Borah Peak", "Chi-Chi", "Edgecumbe", "Elmore Ranch", "Emilia", "Haicheng",
    "Hyogoken-Nambu", "Imperial Valley", "Kocaeli", "Loma Prieta", "Near East Coast of Honshu", "Nihonkai-Chubu",
    "Niigata", "Northridge", "San Fernando", "Superstition Hills", "Tangshan", "Tohoku", "Victoria (Mexicali)",
    "Westmorland"))
# 7.2: the named pooled lines (A, B), paired AUC, pga_primary, NGL groups.
F1_LINES = {"L1": (LEVEL0, SHAKING), "L2": (LPI, SHAKING), "L3": (LEVEL0, LPI)}
DL_OF = {"DL1": "L1", "DL2": "L2", "DL3": "L3"}
# A1-3: R1 (c) is not shown for this registration (no register stamp; sealed cards were released for set
# earthquakes before it). Fixed False for the registered run; tests may set True to reach the live F1 path.
R1C_SHOWN = False

NGL_YES = 1
NGL_NO = 0
PRJ_YES = frozenset(("1", "1.0", "yes", "y", "true", "liquefaction", "liquefied"))
PRJ_NO = frozenset(("0", "0.0", "no", "n", "false", "no liquefaction", "non-liquefaction", "nonliquefaction",
                    "not liquefied"))
NOT_LABELLED = None

NOTE_INTERVAL = ("descriptive; not certified under models fixed; no world interval was measured (C4 measured "
                 "Canterbury tile4 only, docs/evals/C4-CERTIFICATION-2026-11-23.md)")
NOTE_LOGLOSS = "log loss and Brier here measure agreement with this case set, not calibration on the ground"
NOTE_LABELS = "label source not a census; error rates NOT MEASURED (R7, R8)"
MARKS = ("RETROSPECTIVE", "method proposal, not yet calibrated", "task fixed after the organiser saw labels")
HELD_BACK_TEXT = "1 case held back (ruling 50)"
ELBISTAN_ZHU_NOTE = "ruling 50 limit: with the input PGA, 1 of the 3 MODEL ZERO cases would be OFF-GRID"
FORBIDDEN_KEYS = frozenset(("lat", "lon", "latitude", "longitude"))


class D8ScoreStop(RuntimeError):
    """The run stops: nothing past this point is computed or written."""


# ---------------------------------------------------------------------------------------------------------------
# Classes (3.2)
# ---------------------------------------------------------------------------------------------------------------

def _as_number(v):
    """3.2: an integer, a float or a numeric string; JSON true or false count as 1 or 0. Else None."""
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, str):
        try:
            x = float(v.strip())
        except ValueError:
            return None
        return x if math.isfinite(x) else None
    return None


def classify_ngl(fldm_json) -> tuple[str | None, bool]:
    """(YES, NO or None, holds both 1 and 0) from the FLDM rows of one observation. Never truncates (precheck N1)."""
    if fldm_json is None or (isinstance(fldm_json, float) and math.isnan(fldm_json)):
        return NOT_LABELLED, False
    try:
        rows = json.loads(fldm_json) if isinstance(fldm_json, str) else fldm_json
    except (ValueError, TypeError):
        return NOT_LABELLED, False
    if not isinstance(rows, list):
        return NOT_LABELLED, False
    vals = [_as_number(r.get("FLDM_SFEV")) for r in rows if isinstance(r, dict)]
    has1, has0 = any(v == NGL_YES for v in vals), any(v == NGL_NO for v in vals)
    if has1:
        return "YES", has0
    if has0:
        return "NO", False
    return NOT_LABELLED, False


def classify_prj(raw) -> str | None:
    """YES, NO or None from PRJ-6544 `Observation`, on the value stripped and lower-cased (3.2)."""
    if isinstance(raw, bool):
        return "YES" if raw else "NO"
    if isinstance(raw, (int, float)):
        if not math.isfinite(float(raw)):
            return NOT_LABELLED
        return "YES" if float(raw) == 1.0 else "NO" if float(raw) == 0.0 else NOT_LABELLED
    if isinstance(raw, str):
        t = raw.strip().lower()
        return "YES" if t in PRJ_YES else "NO" if t in PRJ_NO else NOT_LABELLED
    return NOT_LABELLED


def parse_labels(rows, case_source: dict, held_ids=frozenset()) -> dict:
    """rows: iterable of dicts with case_id, fldm_json, observation_raw. case_source: in-scope case id -> source.

    Returns {"cls": case_id -> YES/NO/None, "both": set of NGL case ids holding 1 and 0, "raw_prj": case_id -> raw
    text key for distinct counts, "out_of_scope_rows": n}. A duplicate case id stops the run (2.3, R7). A case in
    scope with no row is NOT LABELLED (A1-4); a row whose case id is out of scope is ignored and counted (A1-4).
    """
    seen, cls, both, raw_prj, out, held = set(), {}, set(), {}, 0, 0
    for r in rows:
        cid = r["case_id"]
        if cid in seen:
            raise D8ScoreStop("a duplicate case id in the label file (registration 2.3, R7)")
        seen.add(cid)
        src = case_source.get(cid)
        if cid in held_ids:
            held += 1
            continue
        if src is None:
            out += 1
            continue
        if src == NGL:
            c, b = classify_ngl(r.get("fldm_json"))
            if b:
                both.add(cid)
        else:
            raw = r.get("observation_raw")
            c = classify_prj(raw)
            raw_prj[cid] = repr(raw.strip().lower() if isinstance(raw, str) else raw)
        cls[cid] = c
    missing = [cid for cid in case_source if cid not in seen]
    return {"cls": cls, "both": both, "raw_prj": raw_prj, "out_of_scope_rows": out, "held_back_rows": held,
            "missing": set(missing)}


# ---------------------------------------------------------------------------------------------------------------
# Scope (3.1)
# ---------------------------------------------------------------------------------------------------------------

def group_name(earthquake: str, source: str) -> str:
    return f"{earthquake} [{SOURCE_SHORT.get(source, source)}]"


def scope(cases):
    """Cases in scope: Kaikoura and Canterbury refused, the one held-back Elbistan case dropped (3.1)."""
    names = cases["earthquake"].unique()
    W.check_no_kaikoura(names)
    if any(W.is_canterbury(n) for n in names):
        raise D8ScoreStop("a Canterbury event is in the inputs (3.1 point 2)")
    held = cases["pga_primary_source"].astype(str) == W.WITHHELD_SOURCE
    if int(held.sum()) != 1 or set(cases.loc[held, "earthquake"]) != {"Elbistan"}:
        raise D8ScoreStop("inputs v2 must hold exactly 1 held-back case, in Elbistan (3.1 point 3)")
    return cases.loc[~held].copy()


# ---------------------------------------------------------------------------------------------------------------
# Per group
# ---------------------------------------------------------------------------------------------------------------

def _xy(lat, lon):
    x = 6371000.0 * np.cos(np.deg2rad(lat.mean())) * np.deg2rad(lon - lon.mean())
    y = 6371000.0 * np.deg2rad(lat - lat.mean())
    return x, y


def _counts(y, sites, tiles, mask) -> dict:
    """Case, site and tile counts on the scored cases `mask` (5.3 point 3: sites counted, a site in a class if any
    scored case there has it)."""
    yy, ss, tt = y[mask], sites[mask], tiles[mask]
    return {"cases": int(mask.sum()), "cases_yes": int(yy.sum()), "cases_no": int((~yy).sum()),
            "sites": int(np.unique(ss).size), "yes_sites": int(np.unique(ss[yy]).size),
            "no_sites": int(np.unique(ss[~yy]).size), "tiles": int(np.unique(tt).size)}


def passes_record_gate(c: dict) -> bool:
    return (c["yes_sites"] >= RECORD_GATE["yes_sites"] and c["no_sites"] >= RECORD_GATE["no_sites"]
            and c["tiles"] >= RECORD_GATE["tiles"])


def passes_r27(c: dict) -> bool:
    return (c["sites"] >= R27_GATE["sites"] and c["yes_sites"] >= R27_GATE["yes_sites"]
            and c["no_sites"] >= R27_GATE["no_sites"] and c["tiles"] >= R27_GATE["tiles"])


def _pairs(variant: str, rows_present) -> list[tuple[str, str]]:
    """5.1 point 4, in print order (Level 0 pairs first)."""
    out = [(LEVEL0, SHAKING)] + [(LEVEL0, b) for b in BI14_ROWS] + [(b, SHAKING) for b in BI14_ROWS]
    if variant == "pga_primary":
        out.append((ZHU46, SHAKING))
    return [p for p in out if p[0] in rows_present and p[1] in rows_present]


def _ll_brier(yi, p):
    out = {f"log_loss@{M.clip_key(c)}": M.log_loss(yi, p, c) for c in M.CLIPS}
    out["brier"] = M.brier(yi, p)
    return out


def _ref_stats(yi, pl0):
    """Level 0, coin and in-sample base rate on Level 0's scored cases of one draw (5.1 point 3, A1-9)."""
    if yi.size == 0:
        return {}
    base = float(yi.mean())
    out = {}
    for name, p in (("level0", pl0), ("coin", np.full(yi.size, 0.5)), ("base_rate", np.full(yi.size, base))):
        for k, v in _ll_brier(yi.astype(float), p).items():
            out[f"{name}|{k}"] = v
    return out


def score_variant(g, y, sites, tiles, values: dict, variant: str, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """One bootstrap for every row, pair and reference of one group and variant (5.2 point 3, A1-9: the population
    is every labelled in-scope case of the group; each row and pair is computed on its own subset in each draw)."""
    rows = [r for r in ROWS[variant] if r in values]
    masks = {r: np.isfinite(values[r]) for r in rows}
    pairs = _pairs(variant, rows)
    counts = {r: _counts(y, sites, tiles, masks[r]) for r in rows}
    pcounts = {(a, b): _counts(y, sites, tiles, masks[a] & masks[b]) for a, b in pairs}
    l0 = LEVEL0 in rows

    def stat(idx):
        out = {}
        for r in rows:
            sel = idx[masks[r][idx]]
            out[f"row|{r}"] = auc(y[sel], values[r][sel]) if sel.size else float("nan")
        for a, b in pairs:
            sel = idx[(masks[a] & masks[b])[idx]]
            ya = y[sel]
            out[f"pair|{a}|{b}|a"] = auc(ya, values[a][sel]) if sel.size else float("nan")
            out[f"pair|{a}|{b}|b"] = auc(ya, values[b][sel]) if sel.size else float("nan")
        if l0:
            sel = idx[masks[LEVEL0][idx]]
            ref = _ref_stats(y[sel], values[LEVEL0][sel])
            for k in _ref_keys():
                out[k] = ref.get(k, float("nan"))
        return out

    pt = stat(np.arange(y.size))
    dr = RS.bootstrap(stat, tiles, n_boot, seed) if y.size else {}
    res = {"rows": {}, "pairs": {}, "references": {}}
    for r in rows:
        c = counts[r]
        rec = {"main_or_beside": "main" if (variant, r) in MAIN else "beside", "counts": c,
               "exposure": g["exposure"].get(r, {})}
        if passes_record_gate(c):
            rec["auc"] = RS.interval(pt[f"row|{r}"], dr[f"row|{r}"])
            rec["read"] = "DESCRIPTIVE"
        else:
            rec["read"] = "COUNTS ONLY (under the world record gate)"
        res["rows"][r] = rec
    for a, b in pairs:
        c = pcounts[(a, b)]
        rec = {"metric": "paired AUC", "counts": c}
        if passes_record_gate(c):
            pr = RS.paired(pt[f"pair|{a}|{b}|a"], pt[f"pair|{a}|{b}|b"], dr[f"pair|{a}|{b}|a"],
                           dr[f"pair|{a}|{b}|b"])
            rec.update({k: _num(pr[k]) for k in ("diff", "lo", "hi", "se", "n_draws")})
            rec["read"] = "DESCRIPTIVE"
        else:
            rec["read"] = "COUNTS ONLY (under the world record gate)"
        res["pairs"][f"{a} minus {b}"] = rec
    if l0 and passes_record_gate(counts[LEVEL0]):
        sel = masks[LEVEL0]
        res["references"] = _references(pt, dr, values[LEVEL0][sel])
        res["references"]["note"] = NOTE_LOGLOSS
    elif l0:
        res["references"] = {"read": "COUNTS ONLY (under the world record gate)"}
    res["_pair_points"] = {(a, b): (res["pairs"][f"{a} minus {b}"], pcounts[(a, b)]) for a, b in pairs}
    return res


def _ref_keys():
    return [f"{n}|{k}" for n in ("level0", "coin", "base_rate")
            for k in [f"log_loss@{M.clip_key(c)}" for c in M.CLIPS] + ["brier"]]


def _num(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return v if math.isfinite(v) else None


def _references(pt, dr, pl0) -> dict:
    out = {"scored_on": "Level 0's scored cases", "exact_shares_level0": M.exact_shares(pl0), "metrics": {}}
    for k in [f"log_loss@{M.clip_key(c)}" for c in M.CLIPS] + ["brier"]:
        m = {}
        for n in ("level0", "coin", "base_rate"):
            m[n] = {kk: _num(vv) for kk, vv in RS.interval(pt[f"{n}|{k}"], dr[f"{n}|{k}"]).items()}
        for ref in ("coin", "base_rate"):
            pr = RS.paired(pt[f"level0|{k}"], pt[f"{ref}|{k}"], dr[f"level0|{k}"], dr[f"{ref}|{k}"])
            m[f"level0 minus {ref}"] = {kk: _num(pr[kk]) for kk in ("diff", "lo", "hi", "se", "n_draws")}
        out["metrics"][k] = m
    out["main_clip"] = M.clip_key(MAIN_CLIP)
    return out


def _exposure(g_cases) -> dict:
    out = {}
    for r, col in EXPOSURE_COL.items():
        vc = Counter(g_cases[col].astype(str))
        out[r] = {C.EXPOSURE_MARKS.get(k, k.upper()): int(v) for k, v in sorted(vc.items())}
    return out


def _zhu_counts(reasons, earthquake: str, split: dict) -> dict:
    vc = Counter(str(x) if isinstance(x, str) and x else "EMPTY" for x in reasons)
    rec = {"by_reason": {k: int(v) for k, v in sorted(vc.items())}}
    s = split.get(earthquake)
    if s is None:
        rec["read"] = "NOT RUN (no Zhu grid on disk for this earthquake)"
    else:
        rec["model_zero_by_limit"] = {k: int(v) for k, v in (s.get("model_zero_by_limit") or {}).items()}
    if earthquake == "Elbistan":
        rec["ruling_50_note"] = ELBISTAN_ZHU_NOTE
    return rec


# ---------------------------------------------------------------------------------------------------------------
# The whole run, in memory
# ---------------------------------------------------------------------------------------------------------------

def score(cases, preds: dict, zhu_reasons: dict, zhu_split: dict, label_rows, *, n_boot: int = N_BOOT,
          seed: int = SEED, r1c_shown: bool = R1C_SHOWN) -> dict:
    """cases: inputs v2 table (all 22,528 rows). preds: row name -> variant -> {case_id: value}. zhu_reasons: case_id
    -> reason text of the ruling 46 file. zhu_split: the v3 manifest `zhu_ruling46_split`. label_rows: parsed label
    rows (dicts). Returns the result JSON object (aggregates only)."""
    sc = scope(cases)
    case_source = dict(zip(sc["case_id"], sc["source"]))
    held_ids = frozenset(set(cases["case_id"]) - set(sc["case_id"]))
    lab = parse_labels(label_rows, case_source, held_ids)
    cls = lab["cls"]
    sc["cls"] = [NOT_LABELLED if cid in lab["missing"] else cls.get(cid) for cid in sc["case_id"]]

    prj = sc[sc["source"] == PRJ]
    prj_unlab = int(prj["cls"].isna().sum() - prj["case_id"].isin(lab["missing"]).sum())
    prj_stop = len(prj) > 0 and prj_unlab / len(prj) > PRJ_UNLABELLED_STOP

    out = {"schema": SCHEMA, "registration": REGISTRATION, "amendment": AMENDMENT,
           "settings": {"n_boot": n_boot, "seed": seed, "tile_m": TILE_M, "interval": "percentile 95%",
                        "main_clip": M.clip_key(MAIN_CLIP), "record_gate": RECORD_GATE, "r27_gate": R27_GATE},
           "notes": [NOTE_INTERVAL, NOTE_LABELS, MARKS[2], HELD_BACK_TEXT, NOTE_LOGLOSS,
                     "scored by the organiser, who also enters models"],
           "scope": {"cases_in_inputs": int(len(cases)), "cases_in_scope": int(len(sc)), "held_back": HELD_BACK_TEXT,
                     "groups": int(sc.groupby(["source", "event_id"]).ngroups),
                     "label_rows_out_of_scope_ignored": int(lab["out_of_scope_rows"]),
                     "label_rows_of_the_held_back_case_ignored": int(lab["held_back_rows"]),
                     "cases_with_no_label_row": int(len(lab["missing"])),
                     "prj6544_not_labelled_share": (prj_unlab / len(prj)) if len(prj) else None,
                     "prj6544_coding_check": ("FAILED: counts only for every PRJ-6544 group" if prj_stop
                                              else "passed (at most 1% NOT LABELLED)")},
           "groups": {}}

    pair_items = {ln: {} for ln in F1_LINES}
    for (src, _eid), gc in sorted(sc.groupby(["source", "event_id"]), key=lambda t: group_name(
            t[1]["earthquake"].iloc[0], t[0][0])):
        eq = str(gc["earthquake"].iloc[0])
        name = group_name(eq, src)
        W.check_no_kaikoura([eq])
        rec = {"source": SOURCE_SHORT.get(src, src), "earthquake": eq, "cases_in_scope": int(len(gc)),
               "in_r1d_list": eq in R1D_LIST}
        cl = gc["cls"]
        rec["labels"] = {"YES": int((cl == "YES").sum()), "NO": int((cl == "NO").sum()),
                         "NOT LABELLED": int(cl.isna().sum()),
                         "NOT LABELLED, no row": int(gc["case_id"].isin(lab["missing"]).sum())}
        if src == NGL:
            rec["labels"]["cases whose observation holds both 1 and 0 (read YES)"] = int(
                gc["case_id"].isin(lab["both"]).sum())
        g = {"exposure": _exposure(gc)}
        rec["exposure"] = g["exposure"]
        if src == PRJ and prj_stop:
            rec["read"] = "COUNTS ONLY: PRJ-6544 coding check failed (3.2)"
            rec["labels"]["distinct raw values"] = len({lab["raw_prj"].get(c) for c in gc["case_id"]
                                                        if c in lab["raw_prj"]})
            out["groups"][name] = rec
            continue
        lg = gc[cl.notna()]
        rec["zhu"] = _zhu_counts([zhu_reasons.get(c) for c in gc["case_id"]], eq, zhu_split)
        if lg.empty:
            rec["read"] = "COUNTS ONLY (no labelled case)"
            out["groups"][name] = rec
            continue
        y = (lg["cls"] == "YES").to_numpy()
        x, yy = _xy(lg["lat"].to_numpy(float), lg["lon"].to_numpy(float))
        tiles = RS.tiles_for(x, yy, TILE_M)
        sites = lg["site_id_hash"].to_numpy()
        ids = lg["case_id"].to_numpy()
        rec["variants"] = {}
        for v in VARIANTS:
            vals = {r: np.array([preds[r][v].get(c, np.nan) for c in ids], dtype=float)
                    for r in ROWS[v] if r in preds and v in preds[r]}
            res = score_variant(g, y, sites, tiles, vals, v, n_boot=n_boot, seed=seed)
            pts = res.pop("_pair_points")
            rec["variants"][v] = res
            if src == NGL and v == "pga_primary":
                for ln, (a, b) in F1_LINES.items():
                    if (a, b) in pts:
                        pr, pc = pts[(a, b)]
                        fitted = any(gc[EXPOSURE_COL[e]].astype(str).eq("fitted_on").any() for e in (a, b))
                        pair_items[ln][name] = {"earthquake": eq, "counts": pc, "diff": pr.get("diff"),
                                                "se": pr.get("se"), "fitted": fitted}
        out["groups"][name] = rec

    out["f1"] = f1_lines(pair_items, r1c_shown=r1c_shown)
    out["descriptive_lines"] = descriptive_lines(pair_items)
    out["losses_first"] = losses_first(out)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Pooled lines (7.2, 7.3; A1-2, A1-3, A1-5)
# ---------------------------------------------------------------------------------------------------------------

def _item(name, it, a, b):
    return {"event_id": name, "entry_a": a, "entry_b": b, "metric": "auc", "diff": it["diff"], "se": it["se"],
            "passes_gate": True}


def _finite_pos(v):
    return v is not None and math.isfinite(v) and v > 0


def f1_lines(pair_items: dict, *, r1c_shown: bool = R1C_SHOWN) -> dict:
    """R1 (b) to (e) applied here, before the kit call; only lines left enter the Holm family (A1-2)."""
    out, live = {}, []
    for ln, (a, b) in F1_LINES.items():
        items = pair_items.get(ln, {})
        sset = {n: it for n, it in items.items() if passes_r27(it["counts"])}
        reasons = []
        if len(sset) < P.MIN_EVENTS:
            reasons.append(f"R1 (b) and (e): {len(sset)} earthquake(s) pass the R27 count gate on the pair "
                           f"intersection (R1 e), {P.MIN_EVENTS} needed (R1 b)")
        if not r1c_shown:
            reasons.append("R1 (c): not shown; this registration has no register block time, and sealed cards "
                           "were released for set earthquakes before it (amendment 1)")
        in_d = sorted(n for n, it in sset.items() if it["earthquake"] in R1D_LIST)
        if in_d:
            reasons.append(f"R1 (d): {len(in_d)} set earthquake(s) in the section 9.3 list")
        bad_se = sorted(n for n, it in sset.items() if not _finite_pos(it["se"]) or it["diff"] is None)
        if bad_se:
            reasons.append(f"R1 (e): {len(bad_se)} set earthquake(s) with no finite positive s_i")
        rec = {"a": a, "b": b, "metric": "paired AUC", "pga_variant": "pga_primary", "cases": "NGL groups",
               "label": f"across these {len(sset)} earthquakes",
               "ngl_groups_with_both_entries": len(items),
               "ngl_groups_passing_r27_gate": len(sset),
               "per_earthquake": {n: {"d_i": it["diff"], "s_i": it["se"]} for n, it in sorted(sset.items())},
               "set_earthquakes_fitted_on": sum(1 for it in sset.values() if it["fitted"]),
               "marks": list(MARKS), "notes": [IP.pooled_note(), NOTE_INTERVAL, NOTE_LABELS]}
        if reasons:
            rec.update(word="NOT ALLOWED", reasons=reasons)
        else:
            live.append((ln, [_item(n, it, a, b) for n, it in sorted(sset.items())]))
        out[ln] = rec
    if live:
        res = IP.pooled_lines([items for _, items in live], registered=True, card_version="v3")
        for (ln, _), r in zip(live, res):
            r = _keyed(r)
            out[ln].update({k: r[k] for k in ("word", "pooled", "holm", "leave_one_out", "certified")
                            if k in r})
            if "reason" in r:
                out[ln]["reasons"] = [r["reason"]]
    return out


def _keyed(r: dict) -> dict:
    """A1-7: no per-earthquake list in the output. Earthquake lists become counts or dicts keyed by name."""
    r = json.loads(json.dumps(r, default=_num))
    for k in ("earthquakes_pooled", "earthquakes_excluded"):
        if k in r:
            r[f"n_{k}"] = len(r.pop(k))
    r.pop("per_earthquake", None)
    if isinstance(r.get("leave_one_out"), list):
        r["leave_one_out"] = {x.pop("left_out"): x for x in r["leave_one_out"]}
    return r


def _strip_words(r: dict) -> dict:
    r = _keyed(r)
    r.pop("holm", None)
    r["word"] = "DESCRIPTIVE" if r.get("word") != "NOT ALLOWED" else "NOT ALLOWED"
    for blk in [r.get("pooled") or {}] + list((r.get("leave_one_out") or {}).values()):
        blk.pop("p", None)
    return r


def descriptive_lines(pair_items: dict) -> dict:
    """7.3: equal weight (primary) and kit random effects with Hartung and Knapp (beside). No word, no Holm, no p."""
    out = {}
    for dl, ln in DL_OF.items():
        a, b = F1_LINES[ln]
        items = pair_items.get(ln, {})
        sset = {n: it for n, it in items.items() if passes_record_gate(it["counts"])}
        bad = sorted(n for n, it in sset.items() if not _finite_pos(it["se"]) or it["diff"] is None)
        use = {n: it for n, it in sset.items() if n not in bad}
        rec = {"a": a, "b": b, "of": ln, "label": f"across these {len(use)} earthquakes",
               "ngl_groups_passing_record_gate": len(sset), "left_out_no_finite_s_i": len(bad),
               "in_r1d_list": sum(1 for it in use.values() if it["earthquake"] in R1D_LIST),
               "per_earthquake": {n: {"d_i": it["diff"], "s_i": it["se"]} for n, it in sorted(use.items())},
               "notes": [IP.pooled_note(), NOTE_INTERVAL, NOTE_LABELS], "marks": list(MARKS)}
        if len(use) < P.MIN_EVENTS:
            rec.update(word="NOT ALLOWED", reason=f"{len(use)} earthquake(s) pass the world record gate, "
                                                  f"{P.MIN_EVENTS} needed")
        else:
            its = [_item(n, it, a, b) for n, it in sorted(use.items())]
            rec["equal_weight"] = _strip_words(P.pooled_line(its, registered=True, weighting="equal"))
            rec["random_effects_hk"] = _strip_words(P.pooled_line(its, registered=True, weighting="re"))
            rec["word"] = "DESCRIPTIVE"
        out[dl] = rec
    return out


def losses_first(res: dict) -> list[str]:
    """7.2 point 7: lines where Level 0's point difference is below zero, named first."""
    out = []
    for blk in ("f1", "descriptive_lines"):
        for ln, r in res.get(blk, {}).items():
            if LEVEL0 not in (r["a"], r["b"]):
                continue
            pm = ((r.get("pooled") or (r.get("equal_weight") or {}).get("pooled") or {}).get("mean"))
            if pm is None:
                continue
            l0_diff = pm if r["a"] == LEVEL0 else -pm
            if l0_diff < 0:
                out.append(f"{blk}.{ln}")
    return out


# ---------------------------------------------------------------------------------------------------------------
# Checks before writing (7.1, 8.4; A1-7)
# ---------------------------------------------------------------------------------------------------------------

def _bad_keys(obj, path="out"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).strip().lower() in FORBIDDEN_KEYS:
                yield f"{path}.{k}"
            yield from _bad_keys(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _bad_keys(v, f"{path}[{i}]")


def check_output(res: dict, known_ids) -> dict:
    """Word check outside F1, the lat/lon key refusal and card.scan_aggregates against every in-scope case id and
    site id hash. Returns the JSON-clean object. Raises D8ScoreStop before anything is written."""
    clean = json.loads(json.dumps(res, default=_num, allow_nan=True), parse_constant=lambda c: None)
    clean = _nan_to_none(clean)
    outside = {k: v for k, v in clean.items() if k != "f1"}
    left = IP.words_left(outside, "out")
    if left:
        raise D8ScoreStop(f"a verdict word outside F1 at {left[:5]} (ruling 47)")
    bad = list(_bad_keys(clean))
    if bad:
        raise D8ScoreStop(f"a coordinate key in the output at {bad[:5]}")
    try:
        C.scan_aggregates(clean, frozenset(known_ids), "out")
    except C.CardRefused as e:
        raise D8ScoreStop(f"aggregate scan refused the output: {e}") from e
    json.dumps(clean, allow_nan=False)
    return clean


def _nan_to_none(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_to_none(v) for v in o]
    return o


# ---------------------------------------------------------------------------------------------------------------
# Rendering (8.4: the .md is rendered from the JSON only)
# ---------------------------------------------------------------------------------------------------------------

def _f(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render_md(j: dict) -> str:
    L = [f"{j.get('goal_line', 'G1+G5+G6 (eval M5)')}: D8 scores, one registered run.", "",
         f"# D8 scores: world set v2, predictions v3 ({j.get('date', '2026-11-24')})", ""]
    if j.get("planted"):
        L += ["**PLANTED LABELS: a test run. These numbers are not D8 results.**", ""]
    L += [f"Registration `{j['registration']}`, amendment `{j['amendment']}`. Kit commit `{j.get('kit_commit')}`.",
          "Aggregates only. Every per-group number is DESCRIPTIVE (ruling 47).", "", "## Notes", ""]
    L += [f"- {n}" for n in j["notes"]] + ["", "## Scope", ""]
    L += [f"- {k}: {_f(v)}" for k, v in j["scope"].items()] + [""]
    L += ["## Losses first", ""] + ([f"- {x}" for x in j["losses_first"]] or ["- none"]) + [""]
    L += ["## F1: the named pooled lines", ""]
    for ln, r in j["f1"].items():
        L += [f"### {ln}: {r['a']} minus {r['b']} ({r['label']})", "", f"Word: **{r['word']}**", ""]
        L += [f"- reason: {x}" for x in r.get("reasons", [])]
        L += [f"- NGL groups with both entries: {r['ngl_groups_with_both_entries']}; passing the R27 gate: "
              f"{r['ngl_groups_passing_r27_gate']}; set earthquakes fitted on: {r['set_earthquakes_fitted_on']}"]
        for n, d in r["per_earthquake"].items():
            L.append(f"- {n}: d_i {_f(d['d_i'])}, s_i {_f(d['s_i'])}")
        if r.get("pooled"):
            p = r["pooled"]
            L.append(f"- pooled mean {_f(p.get('mean'))}, interval {_f(p.get('lo'))} to {_f(p.get('hi'))}, "
                     f"tau^2 {_f(p.get('tau2'), 5)}")
        L += [f"- mark: {m}" for m in r["marks"]] + [f"- note: {n}" for n in r["notes"]] + [""]
    L += ["## Descriptive pooled lines (no word)", ""]
    for dl, r in j["descriptive_lines"].items():
        L += [f"### {dl} (of {r['of']}): {r['a']} minus {r['b']} ({r['label']})", "", f"Read: {r['word']}", ""]
        if r.get("reason"):
            L.append(f"- reason: {r['reason']}")
        L.append(f"- earthquakes in the section 9.3 list: {r['in_r1d_list']}")
        for n, d in r["per_earthquake"].items():
            L.append(f"- {n}: d_i {_f(d['d_i'])}, s_i {_f(d['s_i'])}")
        for k in ("equal_weight", "random_effects_hk"):
            if r.get(k, {}).get("pooled"):
                p = r[k]["pooled"]
                L.append(f"- {k}: mean {_f(p.get('mean'))}, interval {_f(p.get('lo'))} to {_f(p.get('hi'))}, "
                         f"tau^2 {_f(p.get('tau2'), 5)}")
        L.append("")
    L += ["## Per group (DESCRIPTIVE)", ""]
    for name, g in j["groups"].items():
        lab = g["labels"]
        L += [f"### {name}", "", f"Cases in scope {g['cases_in_scope']}; YES {lab['YES']}, NO {lab['NO']}, "
              f"NOT LABELLED {lab['NOT LABELLED']} (no row {lab['NOT LABELLED, no row']}). "
              f"In the 9.3 list: {'yes' if g['in_r1d_list'] else 'no'}.", ""]
        if g.get("read"):
            L += [g["read"], ""]
        if g.get("zhu"):
            L += [f"Zhu cells by reason: {g['zhu']['by_reason']}" + (f"; {g['zhu']['read']}" if g['zhu'].get('read')
                                                                    else ""), ""]
            if g["zhu"].get("ruling_50_note"):
                L += [g["zhu"]["ruling_50_note"], ""]
        for v, res in (g.get("variants") or {}).items():
            L += [f"{v}:", "", "| Row | Main | Cases | YES sites | NO sites | Tiles | AUC | 95% |", "|---|---|---|---|"
                  "---|---|---|---|"]
            for r, rr in res["rows"].items():
                c, a = rr["counts"], rr.get("auc") or {}
                L.append(f"| {r} | {rr['main_or_beside']} | {c['cases']} | {c['yes_sites']} | {c['no_sites']} | "
                         f"{c['tiles']} | {_f(a.get('value'))} | {_f(a.get('lo'))} to {_f(a.get('hi'))} |")
            L += ["", "| Pair | Diff | 95% | SE | Read |", "|---|---|---|---|---|"]
            for pn, pr in res["pairs"].items():
                L.append(f"| {pn} | {_f(pr.get('diff'))} | {_f(pr.get('lo'))} to {_f(pr.get('hi'))} | "
                         f"{_f(pr.get('se'))} | {pr['read']} |")
            L.append("")
    L += ["No em or en dashes. Aggregates only."]
    return "\n".join(L) + "\n"

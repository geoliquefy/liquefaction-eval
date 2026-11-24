"""World set v2 inputs and frozen predictions (item D8, eval M5, first slice). G1, G5, G6. No em dashes.

```
 sources on disk (licence line each)            labels (FLDM, PRJ-6544 Observation)
        |                                                |
        v                                                v
 build_inputs --> inputs/cases.parquet + traces.npz     write_labels --> labels-sealed/labels.parquet
        |         (no label column, refused if one       (written once; nothing in this slice reads it)
        |          appears; Kaikoura refused)
        v
 predict(inputs_dir, manifest=...) --> predictions/<model>__<pga>.csv + PREDICTIONS-MANIFEST.json (sha256 per file)
        |   opens only inputs/ files; tests watch every file open with an audit hook; refuses a superseded inputs
        |   manifest
        v
 stamp_plan(manifest, inputs_dir) --> the OpenTimestamps command; refuses predictions built on superseded inputs;
                                      runs only with run=True and a non-empty go (never used)
```

Rules bound here (founder ruling 45, corrections 1 to 10):
  - correction 6: no label is attached to an input row, and the prediction step never opens the label file.
    `INPUT_FORBIDDEN_COLUMNS` lists column names refused in an input table.
  - Kaikoura stays out: `check_no_kaikoura` refuses any event name that names it.
  - Canterbury events (Darfield, Christchurch, Lyttelton and their aftershocks) stay out of the world set, as in
    world set v1 (correction 1 keeps new rows off real Canterbury labels).
  - correction 2: chain rows run under the existing chain ids only; none is changed.
  - No scoring. This module has no metric import.

Coordinates: `inputs/cases.parquet` keeps the site lat and lon because the Zhu 2017 grid read in `predict` needs them.
The file is local and gitignored; no coordinate goes into a committed file (clause 6.2).

USGS Zhu 2017 (review 1 fix): `predict` reads the nearest grid cell for every case whose earthquake has a
`zhu_2017_general.hdf5` on disk (`ZHU_GRIDS`), with `zhu_cells.read_cells`. An empty cell stays empty (NaN) in the
frozen file. Ruling 46 splits empty cells at scoring time (0 only where a published PGA, PGV or Vs30 limit fails,
else OFF-GRID and dropped); B8 registers that reader, so this slice freezes the raw read only. The grids are loaded by
the driver before `predict`, so `predict` itself still opens files under the inputs folder only.

The exposure tags use the world set v1 rules (pillar `docs/evals/world_case_histories.py`, 2026-11-19): event-level
membership in the training tables of each model. Event sets below are copied from that file with its sources.
Each v1 word has a kit mark beside it (`kit_mark`, vocabulary `card.EXPOSURE_MARKS`): an earthquake in a model's
training table gives `fitted_on`; one outside it gives `unseen`; the Zhu family on PRJ-6544 rows outside the Zhu table
gives `unknown` (PRJ-6544 is a Zhu-family compilation; overlap not checked); the shaking score gives `unseen`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import unicodedata
from pathlib import Path

import numpy as np

SCHEMA_INPUTS = "evalkit-world-inputs/2"
SCHEMA_PREDICTIONS = "evalkit-world-predictions/2"

# --------------------------------------------------------------------------------------------------------------
# Licence lines (one per source). Name, URL, attribution text.
# --------------------------------------------------------------------------------------------------------------
LICENCES = {
    "NGL-API-2026-11-20": {
        "licence": "NGL terms of use (the name the 2026-11-20 pull README gives; the NGL home page links no terms "
                   "page, checked 2026-11-23, so no terms URL is cited)",
        "url": "https://nextgenerationliquefaction.org",
        "attribution": "Next Generation Liquefaction (NGL) database, https://nextgenerationliquefaction.org. "
                       "Site pages: https://nextgenerationliquefaction.org/sites/{SITE_ID}",
        "redistribution": "not settled (ruling 19: ask first); NGL ids, rows and readings stay local",
    },
    "PRJ-4022": {
        "licence": "Open Data Commons Attribution (ODC-By; version as DesignSafe states: no version given)",
        "url": "https://doi.org/10.17603/ds2-6vj1-t096",
        "attribution": "Pretell, R., Brandenberg, S. J. and Stewart, J. P. Consistently computed ground motion "
                       "intensity measures for liquefaction triggering assessment. DesignSafe-CI, "
                       "doi:10.17603/ds2-6vj1-t096 (version 3 files, GMIM_PGA 2026-02-22).",
        "redistribution": "allowed with attribution; joined to NGL site rows, so the joined rows stay local",
    },
    "PRJ-6544": {
        "licence": "Open Data Commons Attribution (ODC-By; version as DesignSafe states: no version given)",
        "url": "https://doi.org/10.17603/ds2-x0t1-c556",
        "attribution": "Liquefaction Observations in California. DesignSafe-CI PRJ-6544, doi:10.17603/ds2-x0t1-c556.",
        "redistribution": "allowed with attribution",
    },
    "PRJ-2440": {
        "licence": "Open Data Commons Attribution (ODC-By; version as DesignSafe states: no version given)",
        "url": "https://doi.org/10.17603/ds2-vpmv-5b34",
        "attribution": "GEER Ridgecrest 2019 field reconnaissance, DesignSafe-CI PRJ-2440, "
                       "doi:10.17603/ds2-vpmv-5b34 and doi:10.17603/ds2-9wp5-rp81; Zimmaro et al. 2020, BSSA "
                       "110(4), 1549-1566.",
        "redistribution": "allowed with attribution",
    },
}

# --------------------------------------------------------------------------------------------------------------
# Event names and exposure sets (copied from pillar docs/evals/world_case_histories.py, world set v1)
# --------------------------------------------------------------------------------------------------------------
NGL_TO_CANON = {
    "Loma Prieta": "Loma Prieta", "Kocaeli, Turkey": "Kocaeli", "Northridge-01": "Northridge",
    "Chi-Chi, Taiwan": "Chi-Chi", "Superstition Hills-02": "Superstition Hills",
    "Superstition Hills-01": "Elmore Ranch", "New Zealand-02 (Edgecumbe)": "Edgecumbe",
    "Westmorland": "Westmorland", "Kobe, Japan": "Hyogoken-Nambu", "Imperial Valley-06": "Imperial Valley",
    "El Mayor-Cucapah": "Baja", "Tangshan": "Tangshan", "Tohoku-oki": "Tohoku", "San Fernando": "San Fernando",
    "Victoria, Mexico": "Victoria (Mexicali)", "Haicheng (Liaoning), China": "Haicheng", "Niigata": "Niigata",
    "Nihonkai-Chubu - near the west coast of Honshu": "Nihonkai-Chubu", "Emilia, Italy": "Emilia",
    "Near East Coast of Honshu": "Near East Coast of Honshu", "Mid-Chiba, Japan": "Mid-Chiba",
    "Borah Peak, ID-01": "Borah Peak",
}
# PRJ-6544 'EQ' strings (Database with Predictor Values; spaces dropped in the source) to canonical names. A string
# not listed keeps its own form (1865, 1868, 1892, 1957, NorCal and the others: no other source names them).
PRJ6544_TO_CANON = {
    "LomaPrieta": "Loma Prieta", "Northridge": "Northridge", "SanFernando": "San Fernando",
    "ImperialValley": "Imperial Valley", "SuperstitionHills": "Superstition Hills", "BajaCalifornia": "Baja",
    "1906": "1906 San Francisco", "SanSimeon": "San Simeon", "CapeMendocino": "Cape Mendocino",
    "Chinohills": "Chino Hills", "Hectormine": "Hector Mine", "Ridgecrest": "Ridgecrest",
}
# B&I 2014 Table 3.1 earthquakes (canonical), the set world_case_histories.BI14_EQ_KEYS maps to.
BI14 = {"Niigata", "Inangahua", "Haicheng", "Tangshan", "Imperial Valley", "Victoria (Mexicali)", "Westmorland",
        "Borah Peak", "Nihonkai-Chubu", "Elmore Ranch", "Superstition Hills", "Edgecumbe", "Loma Prieta",
        "Northridge", "Hyogoken-Nambu", "Kocaeli", "Chi-Chi", "Darfield", "Tohoku", "Christchurch"}
MOSS06 = {"Niigata", "Inangahua", "Haicheng", "Tangshan", "Vrancea", "Imperial Valley", "Victoria (Mexicali)",
          "Westmorland", "Nihonkai-Chubu", "Borah Peak", "Edgecumbe", "Elmore Ranch", "Superstition Hills",
          "Loma Prieta", "Northridge", "Hyogoken-Nambu", "Kocaeli", "Chi-Chi"}
ZHU17 = {"Hyogoken-Nambu", "Loma Prieta", "Northridge", "Niigata", "Nihonkai-Chubu", "Tohoku", "Chi-Chi"}
FRAG_GLOBAL = {"Niigata", "San Fernando", "Haicheng", "Tangshan", "Imperial Valley", "Victoria (Mexicali)",
               "Westmorland", "Nihonkai-Chubu", "Borah Peak", "Edgecumbe", "Superstition Hills", "Loma Prieta",
               "Northridge", "Hyogoken-Nambu", "Kocaeli", "Chi-Chi", "Achaia-Ilia", "Baja", "Tohoku", "Emilia"}

# Zhu 2017 grids on disk (pillar data/, read only), by canonical earthquake. Event identity checked by origin date
# (NGL EVNT_YR/EVNT_MD and the ground-failure info.json). Kaikoura and Canterbury grids are never listed.
ZHU_GRIDS = {
    "Hyogoken-Nambu": "comcat-shakemap-nonus-2026-11-07/raw/usp0006rew/ground-failure-preferred/"
                      "us-p0006rew-v1-1597402383402/zhu_2017_general.hdf5",
    "Chi-Chi": "comcat-shakemap-nonus-2026-11-07/raw/usp0009eq0/ground-failure-preferred/"
               "us-p0009eq0-v1-1597409918955/zhu_2017_general.hdf5",
    "Kocaeli": "comcat-shakemap-nonus-2026-11-07/raw/usp0009d4z/ground-failure-preferred/"
               "us-p0009d4z-v1-1597437666702/zhu_2017_general.hdf5",
    "Pazarcik": "comcat-shakemap-nonus-2026-11-07/raw/us6000jllz/ground-failure-preferred/"
                "us-6000jllz_cloud-v14-1681496767701/zhu_2017_general.hdf5",
    "Elbistan": "comcat-shakemap-nonus-2026-11-07/raw/us6000jlqa/ground-failure-preferred/"
                "us-6000jlqa_cloud-v7-1681476225432/zhu_2017_general.hdf5",
    "Tohoku": "usgs-ground-failure-blind-2026-11-05/raw/official20110311054624120_30/ground-failure/"
              "us-20110311054624120_30-1597444114170/zhu_2017_general.hdf5",
    "Loma Prieta": "usgs-ground-failure-blind-2026-11-05/raw/nc216859/ground-failure/"
                   "us-216859-1597388821114/zhu_2017_general.hdf5",
    "Emilia": "usgs-ground-failure-blind-2026-11-05/raw/usp000jkn8/ground-failure/"
              "us-p000jkn8-1609812841613/zhu_2017_general.hdf5",
}
ZHU_MODEL = "USGS Zhu 2017"
ZHU_VARIANT = "usgs-grid"

KAIKOURA_TOKENS = ("kaikoura",)
CANTERBURY_TOKENS = ("christchurch", "darfield", "lyttelton", "lyttleton")
# Every column an input table may carry (review 1 fix): a column outside this list is refused, so a label joined
# under a new name cannot reach the prediction step.
INPUT_ALLOWED_COLUMNS = frozenset((
    "case_id", "source", "earthquake", "event_id", "site_id_hash", "cpt_ref_hash", "trace_row", "n_readings", "mw",
    "mw_type", "pga_primary", "pga_primary_source", "pga_prj4022", "pga_prj4022_ln_sd", "pga_ngl_gmim", "gwt_m",
    "gwt_source", "ngl_review", "ngl_quality_class", "critical_layer_top_m", "critical_layer_base_m", "lat", "lon",
    "cti", "vs30", "wtd_model", "precip", "pgv_raw",
    *(f"{p}exp_{f}" for p in ("", "kit_") for f in ("bi14", "moss06", "zhu2017", "lsn_global_fragility", "shaking"))))
INPUT_FORBIDDEN_COLUMNS = ("label", "observation", "liq", "manif", "fldm", "fldo_desc", "yes", "outcome",
                           "derived_liq_observed", "near_dist")

# Models frozen in this slice. kind "cpt" needs a CPT trace, water table, PGA and Mw; kind "shaking" needs PGA, Mw.
MODELS = {
    "Level 0": {"kind": "cpt", "house": True, "output_kind": "triggering_probability", "exposure_col": "exp_bi14",
                "rule": "p = g5.p_level0(ln k*, sigma_e = 0), ln k* at LPI 5 (run_g6_jun2011.lnk_star), as the "
                        "scorecard v2 world cells"},
    "BI14-LPI [default-2026-11]": {"kind": "cpt", "chain": "default-2026-11", "out": "lpi"},
    "BI14-LSN [default-2026-11]": {"kind": "cpt", "chain": "default-2026-11", "out": "lsn"},
    "BI14-LPI [authors-maurer-2026-11-19]": {"kind": "cpt", "chain": "authors-maurer-2026-11-19", "out": "lpi"},
    "BI14-LSN [authors-maurer-2026-11-19]": {"kind": "cpt", "chain": "authors-maurer-2026-11-19", "out": "lsn"},
    "BI14-LPI [peer-global-2026-11-23]": {"kind": "cpt", "chain": "peer-global-2026-11-23", "out": "lpi"},
    "BI14-LSN [peer-global-2026-11-23]": {"kind": "cpt", "chain": "peer-global-2026-11-23", "out": "lsn"},
    "baseline: shaking score": {"kind": "shaking", "output_kind": "score", "exposure_col": "exp_shaking",
                                "rule": "evalkit.shaking.score(PGA, Mw), MSF-M-ONLY-1.3"},
    ZHU_MODEL: {"kind": "zhu", "output_kind": "probability_of_liquefaction_cell", "exposure_col": "exp_zhu2017",
                "rule": "nearest cell of the USGS ground-failure zhu_2017_general.hdf5 (ZHU_GRIDS) at the site, read "
                        "with evalkit.zhu_cells.read_cells; empty cell or outside the grid = empty value (NaN), never "
                        "0; the ruling 46 split (0 only where a published PGA, PGV or Vs30 limit fails) is applied at "
                        "scoring by the reader B8 registers; earthquakes with no grid on disk = empty"},
}
for _n, _m in MODELS.items():
    _m.setdefault("house", False)
    _m.setdefault("output_kind", "score")
    _m.setdefault("exposure_col", "exp_bi14")
    _m.setdefault("rule", f"evalkit.chain_settings.run(..., {_m.get('chain')!r})[{_m.get('out')!r}]")
    if _m.get("chain") == "authors-maurer-2026-11-19":
        _m["rule"] += "; u2 not passed: tip = qc (NGL SCPT_PWP not used in this slice)"
PGA_MODELS = [n for n, m in MODELS.items() if m["kind"] != "zhu"]
NOT_RUN = {
    "baseline: shaking only (probability)": "needs labels of earlier earthquakes (a fit); not allowed in a "
                                            "label-free slice",
    "M0 (Level 0 + earthquake band)": "band quantiles not loaded in this slice",
    "USGS Zhu 2017 (earthquakes with no grid on disk)": "no zhu_2017_general.hdf5 on disk for that earthquake; "
                                                        "frozen as empty values",
    "GEM TS22": "inputs not on disk",
    "site memory, geographic null": "need labels of other earthquakes",
}
PGA_VARIANTS = ("pga_primary", "pga_prj4022")


class D8Refused(RuntimeError):
    """A D8 step was refused (label column, Kaikoura, order or hash)."""


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(d) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def id_hash(source: str, key: str) -> str:
    """A short stable id hash. The raw source id never enters an input file."""
    return sha_bytes(f"d8-world-v2|{source}|{key}".encode())[:20]


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def is_kaikoura(name: str) -> bool:
    return any(t in _fold(name) for t in KAIKOURA_TOKENS)


def is_canterbury(name: str) -> bool:
    return any(t in _fold(name) for t in CANTERBURY_TOKENS)


def check_no_kaikoura(names) -> None:
    bad = sorted({n for n in names if is_kaikoura(n)})
    if bad:
        raise D8Refused(f"Kaikoura stays out of world set v2: {bad}")


def check_input_columns(columns) -> None:
    """Refuse an input table that carries any label-like column (correction 6)."""
    bad = [c for c in columns if any(t in str(c).lower() for t in INPUT_FORBIDDEN_COLUMNS)]
    if bad:
        raise D8Refused(f"an input table carries label-like columns {bad} (correction 6)")
    extra = sorted(set(map(str, columns)) - INPUT_ALLOWED_COLUMNS)
    if extra:
        raise D8Refused(f"an input table carries columns outside the allowlist {extra} (correction 6)")


KIT_MARK_RULES = {"POSSIBLE": "fitted_on", "EVENT IN TRAINING": "fitted_on", "UNSEEN": "unseen",
                  "UNSEEN (no fitted parameter)": "unseen",
                  "UNSEEN, Zhu family on PRJ-6544": "unknown"}


def kit_mark(v1_word: str) -> str:
    """The kit exposure mark (card.EXPOSURE_MARKS key) for a world set v1 word."""
    if v1_word.startswith("UNSEEN") and "Zhu-family" in v1_word:
        return KIT_MARK_RULES["UNSEEN, Zhu family on PRJ-6544"]
    return KIT_MARK_RULES[v1_word.split(" (PRJ-6544")[0]]


def exposure_tags(canon: str, source: str) -> dict:
    """Exposure per case: the world set v1 word (event level) and the kit mark beside it (`kit_mark`)."""
    v1 = {
        "exp_bi14": "POSSIBLE" if canon in BI14 else "UNSEEN",
        "exp_moss06": "POSSIBLE" if canon in MOSS06 else "UNSEEN",
        "exp_zhu2017": ("EVENT IN TRAINING" if canon in ZHU17 else "UNSEEN")
        + (" (PRJ-6544 is a Zhu-family compilation)" if source == "PRJ-6544" else ""),
        "exp_lsn_global_fragility": "POSSIBLE" if canon in FRAG_GLOBAL else "UNSEEN",
        "exp_shaking": "UNSEEN (no fitted parameter)",
    }
    return {**v1, **{f"kit_{k}": kit_mark(w) for k, w in v1.items()}}


# --------------------------------------------------------------------------------------------------------------
# Prediction step: opens only the inputs folder
# --------------------------------------------------------------------------------------------------------------

def load_inputs(inputs_dir: str | Path, manifest: str = "INPUTS-MANIFEST.json"):
    """Read the frozen input files, check their hashes against the inputs manifest, refuse label columns.

    manifest: the first INPUTS-MANIFEST.json (default) or a later version such as INPUTS-MANIFEST-v2.json, which
    names its own cases file (`cases_file`). Every file a manifest lists is checked by its sha256."""
    import pandas as pd
    d = Path(inputs_dir)
    man = json.loads((d / manifest).read_text())
    if man.get("schema") != SCHEMA_INPUTS:
        raise D8Refused("not a world set v2 input manifest")
    cases_file = man.get("cases_file", "cases.parquet")
    if cases_file not in man["files"]:
        raise D8Refused(f"{manifest}: its cases file {cases_file} has no frozen sha256")
    for name, h in man["files"].items():
        if sha_file(d / name) != h:
            raise D8Refused(f"input file {name} differs from its frozen sha256")
    cases = pd.read_parquet(d / cases_file)
    check_input_columns(cases.columns)
    check_no_kaikoura(cases["earthquake"].unique())
    traces = dict(np.load(d / "traces.npz"))
    return man, cases, traces


def run_models(cases, traces, pga_col: str, P=None, Cq=None, lnk_star=None, to_grid=None, Z=None) -> dict:
    """Every model in MODELS on one PGA column. Returns {model: np.ndarray} in case order (NaN = not runnable).

    P, Cq: the pinned procedures and consequence modules; lnk_star and to_grid: the v2 world-cell helpers. When P is
    None the CPT models are skipped (left NaN) and only the shaking score runs (used by the tests)."""
    from . import chain_settings as CS
    from . import shaking
    from ..g5 import p_level0
    n = len(cases)
    pga = cases[pga_col].to_numpy(float)
    mw = cases["mw"].to_numpy(float)
    out = {m: np.full(n, np.nan) for m in PGA_MODELS}
    ok_s = np.isfinite(pga) & (pga > 0) & np.isfinite(mw)
    for i in np.flatnonzero(ok_s):
        out["baseline: shaking score"][i] = float(shaking.score(np.array([pga[i]]), float(mw[i]))[0])
    if P is None:
        return out
    gwt = cases["gwt_m"].to_numpy(float)
    rows = cases["trace_row"].to_numpy()
    cpt_ok = ok_s & np.isfinite(gwt) & (rows >= 0)
    idx, qcs, fss = [], [], []
    off = traces["offsets"]
    for i in np.flatnonzero(cpt_ok):
        a, b = int(off[rows[i]]), int(off[rows[i] + 1])
        g = to_grid(traces["depth_m"][a:b], traces["qc_kpa"][a:b], traces["fs_kpa"][a:b])
        if g is None:
            continue
        idx.append(i)
        qcs.append(g[0])
        fss.append(g[1])
    if not idx:
        return out
    idx = np.array(idx)
    qc, fs = np.stack(qcs), np.stack(fss)
    g_, a_, m_ = np.clip(gwt[idx], 0, None), pga[idx], mw[idx]
    lnk = lnk_star(P, Z, qc, fs, np.arange(idx.size), g_, a_, m_)
    out["Level 0"][idx] = p_level0(lnk, np.zeros(lnk.size))
    for sid in (CS.DEFAULT_ID, CS.AUTHORS_ID, CS.PEER_GLOBAL_ID):
        r = CS.run(P, Cq, Z, qc, fs, g_, a_, m_, sid)
        out[f"BI14-LPI [{sid}]"][idx] = r["lpi"]
        out[f"BI14-LSN [{sid}]"][idx] = r["lsn"]
    return out


def run_zhu(cases, zhu_grids: dict) -> tuple[np.ndarray, dict]:
    """Zhu 2017 nearest-cell value per case (NaN = no grid, empty cell or outside). zhu_grids: {canonical earthquake:
    {"x", "y", "model", "sha256", "path"}}, loaded before predict with evalkit.zhu_grid.load(...).as_dict(), which adds
    "row_order" when the read is not the default as-stored one. Returns values and counts per earthquake."""
    from . import zhu_cells
    n = len(cases)
    v = np.full(n, np.nan)
    rec = {}
    eq = cases["earthquake"].to_numpy()
    check_no_kaikoura(zhu_grids)
    for name, g in sorted(zhu_grids.items()):
        k = np.flatnonzero(eq == name)
        if not k.size:
            continue
        lat = cases["lat"].to_numpy(float)[k]
        lon = cases["lon"].to_numpy(float)[k]
        ok = np.isfinite(lat) & np.isfinite(lon)
        r = zhu_cells.read_cells(g["x"], g["y"], g["model"], lon[ok], lat[ok])
        v[k[ok]] = r.values
        rec[name] = {"cases": int(k.size), "no_coordinates": int((~ok).sum()), "on_grid": int(r.on_grid.sum()),
                     "empty_cell": int(r.empty_cell.sum()), "outside_extent": int(r.outside_extent.sum()),
                     "grid_sha256": g["sha256"], "grid_path": g["path"]}
        if "row_order" in g:            # evalkit.zhu_grid names the row order only when it is not the default
            rec[name]["row_order"] = g["row_order"]
            rec[name]["orientation"] = g.get("orientation")
    return v, rec


def _fmt(v: float) -> str:
    return "" if not np.isfinite(v) else repr(round(float(v), 12))


def prediction_file_name(model: str, pga_variant: str) -> str:
    """The file name write_predictions gives one model on one PGA variant."""
    slug = "".join(c if c.isalnum() else "-" for c in model).strip("-").lower()
    return f"{slug}__{pga_variant}.csv"


def prediction_texts(case_ids, preds: dict, pga_variant: str) -> dict:
    """{file name: (text, manifest row)} for each model, in memory: the exact bytes write_predictions writes."""
    out = {}
    for model, v in preds.items():
        text = "case_id,value\n" + "".join(f"{c},{_fmt(x)}\n" for c, x in zip(case_ids, v))
        spec = MODELS[model]
        out[prediction_file_name(model, pga_variant)] = (text, {
            "model": model, "pga_variant": pga_variant, "sha256": sha_bytes(text.encode()),
            "n_rows": int(len(v)), "n_values": int(np.isfinite(v).sum()), "house": spec["house"],
            "output_kind": spec["output_kind"], "rule": spec["rule"],
            "exposure_column": spec["exposure_col"], "chain_id": spec.get("chain")})
    return out


def write_predictions(out_dir: str | Path, case_ids, preds: dict, pga_variant: str) -> dict:
    """One CSV per model (case_id,value). Returns the manifest rows for these files."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, (text, row) in prediction_texts(case_ids, preds, pga_variant).items():
        (d / name).write_text(text, encoding="utf-8")
        rows[name] = row
    return rows


def predict(inputs_dir: str | Path, out_dir: str | Path, helpers: dict | None = None, kit_commit: str = "",
            zhu_grids: dict | None = None, manifest: str = "INPUTS-MANIFEST.json") -> dict:
    """Freeze every model's predictions on the frozen inputs. Opens only files under inputs_dir. No scoring.

    zhu_grids: the Zhu 2017 grids already in memory (the driver loads them); None leaves Zhu out.
    manifest: the inputs manifest to predict on (default the first one; INPUTS-MANIFEST-v2.json for inputs v2). A
    manifest that a later inputs version supersedes is refused (D8 Elbistan review 1, S1)."""
    helpers = helpers or {}
    mh = sha_file(Path(inputs_dir) / manifest)
    if mh in superseded_inputs_sha256(inputs_dir):
        raise D8Refused(f"{manifest} (sha256 {mh}) is superseded; predict on the inputs that replace it")
    man, cases, traces = load_inputs(inputs_dir, manifest)
    files = {}
    for var in PGA_VARIANTS:
        preds = run_models(cases, traces, var, **helpers)
        files.update(write_predictions(out_dir, cases["case_id"].tolist(), preds, var))
    zhu_rec = None
    if zhu_grids is not None:
        zv, zhu_rec = run_zhu(cases, zhu_grids)
        files.update(write_predictions(out_dir, cases["case_id"].tolist(), {ZHU_MODEL: zv}, ZHU_VARIANT))
    pm = {"schema": SCHEMA_PREDICTIONS, "item": "D8 first slice (eval M5)", "inputs_manifest_sha256": mh,
          "kit_commit": kit_commit, "n_cases": int(len(cases)), "files": files, "not_run": NOT_RUN,
          "scored": False, "labels_opened": False, "zhu_grid_read": zhu_rec,
          "exposure_kit_mark_rules": KIT_MARK_RULES,
          **({} if manifest == "INPUTS-MANIFEST.json" else {"inputs_manifest": manifest}),
          "pga_variants": {"pga_primary": "PRJ-4022 Parent_MEAN where joined, else the NGL GMIM PGA, else the "
                                          "PRJ-6544 USGS PGA (see the input manifest)",
                           "pga_prj4022": "PRJ-4022 Parent_MEAN only; NaN where not joined",
                           ZHU_VARIANT: "no PGA input: the USGS ground-failure grid carries its own ShakeMap"}}
    p = Path(out_dir) / "PREDICTIONS-MANIFEST.json"
    p.write_text(json.dumps(pm, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": str(p), "sha256": sha_file(p), "n_files": len(files)}


# --------------------------------------------------------------------------------------------------------------
# Stamp step: wired, never run
# --------------------------------------------------------------------------------------------------------------

def superseded_inputs_sha256(inputs_dir: str | Path) -> set[str]:
    """sha256 of every inputs file named superseded in inputs_dir: the hashes SUPERSEDED-INPUTS.txt names before
    "SUPERSEDED by", and each INPUTS-MANIFEST*.json's `supersedes` entries. Empty when nothing is superseded."""
    import re
    d = Path(inputs_dir)
    out: set[str] = set()
    note = d / "SUPERSEDED-INPUTS.txt"
    if note.exists():
        head = note.read_text(encoding="utf-8").split("SUPERSEDED by", 1)[0]
        out |= set(re.findall(r"\b[0-9a-f]{64}\b", head))
    for mp in sorted(d.glob("INPUTS-MANIFEST*.json")):
        sup = json.loads(mp.read_text()).get("supersedes") or {}
        out |= {v["sha256"] for v in sup.values() if isinstance(v, dict) and "sha256" in v}
    return out


def superseded_predictions_sha256(pred_dir: str | Path) -> set[str]:
    """sha256 of every predictions manifest named superseded in pred_dir: the hashes each SUPERSEDED*.txt note names
    before "SUPERSEDED by", and each PREDICTIONS-MANIFEST*.json's `supersedes` entries (`manifest` and `manifests`).
    Empty when nothing is superseded."""
    import re
    d = Path(pred_dir)
    out: set[str] = set()
    for note in sorted(d.glob("SUPERSEDED*.txt")):
        if note.name == "SUPERSEDED-INPUTS.txt":
            continue
        head = note.read_text(encoding="utf-8").split("SUPERSEDED by", 1)[0]
        out |= set(re.findall(r"\b[0-9a-f]{64}\b", head))
    for mp in sorted(d.glob("PREDICTIONS-MANIFEST*.json")):
        try:
            sup = json.loads(mp.read_text()).get("supersedes") or {}
        except (ValueError, AttributeError):
            continue
        m = sup.get("manifest")
        if isinstance(m, dict) and "sha256" in m:
            out.add(m["sha256"])
        ms = sup.get("manifests")
        if isinstance(ms, dict):
            out |= {v for v in ms.values() if isinstance(v, str)}
    return out


def stamp_plan(manifest_path: str | Path, inputs_dir: str | Path, *, run: bool = False, go: str = "",
               runner=subprocess.run) -> dict:
    """Return the OpenTimestamps command for the predictions manifest. Runs it only with run=True and a non-empty
    go naming the founder go (ruling 45: stamping waits for a go). Nothing in the kit or its tests passes both.

    inputs_dir: the inputs folder the predictions were built on. The plan is refused when the predictions manifest
    names no inputs manifest, names one not in inputs_dir, or names one a later inputs version supersedes (D8
    Elbistan review 1, S1: such predictions still hold a withheld value). A predictions manifest that a later one in
    the same folder supersedes is refused too (predictions v3, ruling 52a)."""
    p = Path(manifest_path)
    ph = sha_file(p)
    if ph in superseded_predictions_sha256(p.parent):
        raise D8Refused(f"predictions manifest {p.name} (sha256 {ph}) is superseded by a later predictions manifest "
                        f"in {p.parent.name}/; stamp the manifest that replaces it")
    ih = json.loads(p.read_text()).get("inputs_manifest_sha256")
    if not ih:
        raise D8Refused("the predictions manifest names no inputs manifest; not stamped")
    if ih in superseded_inputs_sha256(inputs_dir):
        raise D8Refused(f"the predictions were built on inputs manifest {ih}, which is superseded "
                        f"(SUPERSEDED-INPUTS.txt); rerun the predictions on the new inputs before any stamp")
    if ih not in {sha_file(m) for m in Path(inputs_dir).glob("INPUTS-MANIFEST*.json")}:
        raise D8Refused(f"inputs manifest {ih} is not in {inputs_dir}; not stamped")
    cmd = ["ots", "stamp", str(p)]
    if run:
        if not go.strip():
            raise D8Refused("stamping needs a founder go; not run")
        runner(cmd, check=True)
        return {"command": cmd, "ran": True, "sha256": sha_file(p)}
    return {"command": cmd, "ran": False, "sha256": sha_file(p)}


# --------------------------------------------------------------------------------------------------------------
# Zhu refreeze (2026-11-23): the first-slice Zhu file read the grids mirrored (pillar
# docs/evals/ERRATUM-ZHU-ROW-ORDER-2026-11-23.md). Only USGS Zhu 2017 is rerun, with evalkit.zhu_grid
# ROW_ORDER_GEOREF. New files under new names and a new manifest version; the first manifest and the first Zhu file
# stay byte for byte as they were and are named superseded in the new manifest. Opens inputs/ and predictions/ only.
# --------------------------------------------------------------------------------------------------------------
ZHU_REFREEZE_ROW_ORDER = "georef-2026-11-23"
ZHU_V1_FILE = f"usgs-zhu-2017__{ZHU_VARIANT}.csv"
ZHU_REFREEZE_FILE = "usgs-zhu-2017__usgs-grid-georef-2026-11-23.csv"
ZHU_RULED_FILE = "usgs-zhu-2017__usgs-grid-georef-2026-11-23-ruling46.csv"
PREDICTIONS_MANIFEST_V1 = "PREDICTIONS-MANIFEST.json"
PREDICTIONS_MANIFEST_V2 = "PREDICTIONS-MANIFEST-v2.json"
SUPERSEDED_NOTE = "SUPERSEDED.txt"
SCHEMA_PREDICTIONS_V2 = "evalkit-world-predictions/3"

# The ShakeMap grid.xml of each Zhu earthquake (pillar data/, read only), for the ruling 46 limits (PGA, PGV, SVEL at
# the nearest node, as the scorecard v3 world reader). Version notes: the ground-failure info.json names the ShakeMap
# version its Zhu grid was run on; Pazarcik and Elbistan have only a later version on disk.
ZHU_SHAKEMAPS = {
    "Hyogoken-Nambu": "comcat-shakemap-nonus-2026-11-07/raw/usp0006rew/shakemap-preferred/"
                      "atlas-usp0006rew-v1-1594167475099/download/grid.xml",
    "Chi-Chi": "comcat-shakemap-nonus-2026-11-07/raw/usp0009eq0/shakemap-preferred/"
               "atlas-usp0009eq0-v1-1594169383237/download/grid.xml",
    "Kocaeli": "comcat-shakemap-nonus-2026-11-07/raw/usp0009d4z/shakemap-preferred/"
               "atlas-usp0009d4z-v1-1594169307058/download/grid.xml",
    "Pazarcik": "comcat-shakemap-nonus-2026-11-07/raw/us6000jllz/shakemap-preferred/"
                "us-us6000jllz-v19-1756921940993/download/grid.xml",
    "Elbistan": "comcat-shakemap-nonus-2026-11-07/raw/us6000jlqa/shakemap-preferred/"
                "us-us6000jlqa-v12-1756575631263/download/grid.xml",
    "Tohoku": "usgs-ground-failure-blind-2026-11-05/raw/official20110311054624120_30/shakemap-of-product/"
              "atlas-official20110311054624120_30-v1-1594161746661/download/grid.xml",
    "Loma Prieta": "usgs-ground-failure-blind-2026-11-05/raw/nc216859/shakemap-of-product/"
                   "atlas-nc216859-v1-1594161215615/download/grid.xml",
    "Emilia": "usgs-ground-failure-blind-2026-11-05/raw/usp000jkn8/shakemap-of-product/"
              "atlas-usp000jkn8-v1-1600467352092/download/grid.xml",
}
ZHU_SHAKEMAP_VERSION_NOTE = {
    "Pazarcik": "ground-failure product ran on ShakeMap v17; v19 (the preferred on disk) is used for the limits",
    "Elbistan": "ground-failure product ran on ShakeMap v11; v12 (the preferred on disk) is used for the limits",
}
ZHU_REFREEZE_RULE = ("nearest cell of the USGS ground-failure zhu_2017_general.hdf5 (ZHU_GRIDS) at the site, the grid "
                     "read with evalkit.zhu_grid.load(path, 'georef-2026-11-23') (rows aligned to ascending y from "
                     "the grid's own metadata), cells read with evalkit.zhu_cells.read_cells; empty cell or outside "
                     "the grid = empty value (NaN), never 0; earthquakes with no grid on disk = empty")
ZHU_RULED_RULE = ("as " + ZHU_REFREEZE_FILE + ", then the ruling 46 split (evalkit.zhu_cells.split_by_reason, the "
                  "masks of cell_input_ruled): an empty cell is 0 (MODEL ZERO) only where the site fails PGA < 0.10 g, "
                  "PGV < 3 cm/s or Vs30 > 620 m/s, read from that earthquake's ShakeMap (ZHU_SHAKEMAPS: PGA/100, PGV, "
                  "SVEL at the nearest node); other empty cells and sites outside the grid are OFF-GRID, an empty "
                  "cell with no failing limit and a missing input is NOT CHECKABLE; both stay empty (dropped at "
                  "scoring); the reason column names the class")


def _read_value_csv(p: Path) -> tuple[list, list]:
    ids, vals = [], []
    for line in p.read_text(encoding="utf-8").splitlines()[1:]:
        c, v = line.split(",", 1)
        ids.append(c)
        vals.append(v)
    return ids, vals


def zhu_ruling46_text(cases, zhu_grids: dict, shakemaps: dict, zv: np.ndarray) -> tuple[str, dict]:
    """The ruling 46 split file text (case_id,value,reason) and its counts per earthquake, from the georef Zhu read
    zv (run_zhu) and each earthquake's ShakeMap (PGA, PGV, SVEL at the nearest node). Refused when the split
    disagrees with zv. Used by refreeze_zhu (predictions v2) and predict_v3, so both run the same code."""
    from . import pgv as PGV
    from . import zhu_cells
    ids = cases["case_id"].tolist()
    eq = cases["earthquake"].to_numpy()
    rv = np.full(len(ids), np.nan)
    reason = np.full(len(ids), "", dtype=object)
    ruled = {}
    lat_all, lon_all = cases["lat"].to_numpy(float), cases["lon"].to_numpy(float)
    for name, g in sorted(zhu_grids.items()):
        k = np.flatnonzero(eq == name)
        if not k.size:
            continue
        ok = np.isfinite(lat_all[k]) & np.isfinite(lon_all[k])
        kk = k[ok]
        lo, la = lon_all[kk], lat_all[kk]
        sm = shakemaps[name]
        pga = PGV.sample_nearest(sm, "PGA", lo, la) / 100.0
        pgv = PGV.sample_nearest(sm, "PGV", lo, la)
        vs = PGV.sample_nearest(sm, "SVEL", lo, la)
        v, why = zhu_cells.split_by_reason(g["x"], g["y"], g["model"], lo, la, pga, pgv, vs)
        rv[kk], reason[kk] = v, why
        reason[k[~ok]] = "NO COORDINATES"
        ruled[name] = {"cases": int(k.size), "no_coordinates": int((~ok).sum()),
                       **{r_: int((why == r_).sum()) for r_ in zhu_cells.REASONS},
                       "model_zero_by_limit": {
                           "pga": int(((why == zhu_cells.MODEL_ZERO) & np.isfinite(pga)
                                       & (pga < zhu_cells.LIMITS["pga_g_below"])).sum()),
                           "pgv": int(((why == zhu_cells.MODEL_ZERO) & np.isfinite(pgv)
                                       & (pgv < zhu_cells.LIMITS["pgv_cms_below"])).sum()),
                           "vs30": int(((why == zhu_cells.MODEL_ZERO) & np.isfinite(vs)
                                        & (vs > zhu_cells.LIMITS["vs30_ms_above"])).sum())},
                       "shakemap_sha256": sm.sha256,
                       "shakemap_version_note": ZHU_SHAKEMAP_VERSION_NOTE.get(name, "same version as the "
                                                                                   "ground-failure product")}
        if not np.array_equal(np.isfinite(zv[kk]), why == zhu_cells.ON_GRID_VALUE) or not np.array_equal(
                zv[kk][np.isfinite(zv[kk])], v[why == zhu_cells.ON_GRID_VALUE]):
            raise D8Refused(f"{name}: the split disagrees with the refrozen read")
    rtext = "case_id,value,reason\n" + "".join(f"{c},{_fmt(x)},{w}\n" for c, x, w in zip(ids, rv, reason))
    return rtext, ruled


def refreeze_zhu(inputs_dir: str | Path, pred_dir: str | Path, zhu_grids: dict, shakemaps: dict | None = None,
                 kit_commit: str = "") -> dict:
    """Rerun only USGS Zhu 2017 on the frozen inputs with georeferenced grids. Writes new files under new names and
    PREDICTIONS-MANIFEST-v2.json; never rewrites a file that exists. The 16 other files are checked byte for byte
    against the first manifest and carried over. shakemaps: {earthquake: evalkit.pgv.Grid} for the ruling 46 split
    (None skips the split file). Returns the comparison with the first Zhu file and the split counts (counts only)."""
    from . import zhu_cells
    d = Path(pred_dir)
    m1p = d / PREDICTIONS_MANIFEST_V1
    for name in (PREDICTIONS_MANIFEST_V2, ZHU_REFREEZE_FILE, ZHU_RULED_FILE, SUPERSEDED_NOTE):
        if (d / name).exists():
            raise D8Refused(f"{name} exists; the refreeze never rewrites a file")
    bad = sorted(k for k, g in zhu_grids.items() if g.get("row_order") != ZHU_REFREEZE_ROW_ORDER)
    if bad:
        raise D8Refused(f"the refreeze reads every grid with {ZHU_REFREEZE_ROW_ORDER}; not so for {bad}")
    man, cases, _ = load_inputs(inputs_dir)
    mh = sha_file(Path(inputs_dir) / "INPUTS-MANIFEST.json")
    m1 = json.loads(m1p.read_text())
    m1_sha = sha_file(m1p)
    if m1.get("inputs_manifest_sha256") != mh:
        raise D8Refused("the first predictions manifest was built on other inputs")
    for name, row in m1["files"].items():
        if sha_file(d / name) != row["sha256"]:
            raise D8Refused(f"{name} differs from the first manifest; stop")
    if ZHU_V1_FILE not in m1["files"]:
        raise D8Refused("the first manifest holds no Zhu file to supersede")
    if shakemaps is not None:
        missing = sorted(set(zhu_grids) - set(shakemaps))
        if missing:
            raise D8Refused(f"no ShakeMap for the ruling 46 split: {missing}")
    ids = cases["case_id"].tolist()
    old_ids, old_txt = _read_value_csv(d / ZHU_V1_FILE)
    if old_ids != ids:
        raise D8Refused("the first Zhu file is not in input case order")
    zv, zrec = run_zhu(cases, zhu_grids)
    new_txt = [_fmt(x) for x in zv]
    files = {k: v for k, v in m1["files"].items() if k != ZHU_V1_FILE}
    text = "case_id,value\n" + "".join(f"{c},{t}\n" for c, t in zip(ids, new_txt))
    # Every check (ShakeMap coverage, split agreement) runs before the first write, so a refusal leaves no new file
    # behind that would block the next run (review 1 nit, wave 4).
    writes = {ZHU_REFREEZE_FILE: text}
    base = {"model": ZHU_MODEL, "house": False, "output_kind": MODELS[ZHU_MODEL]["output_kind"],
            "exposure_column": "exp_zhu2017", "chain_id": None, "n_rows": len(ids)}
    files[ZHU_REFREEZE_FILE] = {**base, "pga_variant": "usgs-grid-georef-2026-11-23", "sha256": sha_bytes(text.encode()),
                                "n_values": int(np.isfinite(zv).sum()), "rule": ZHU_REFREEZE_RULE,
                                "zhu_row_order": ZHU_REFREEZE_ROW_ORDER}
    # comparison with the first (mirrored) Zhu file, per earthquake, counts only
    eq = cases["earthquake"].to_numpy()
    cmp = {}
    for name in sorted(zrec):
        k = np.flatnonzero(eq == name)
        o = [old_txt[i] for i in k]
        nw = [new_txt[i] for i in k]
        ov = [x != "" for x in o]
        nv = [x != "" for x in nw]
        cmp[name] = {"cases": int(k.size), "values_first": int(sum(ov)), "values_refrozen": int(sum(nv)),
                     "first_values_changed": int(sum(a and x != y for a, x, y in zip(ov, o, nw))),
                     "value_to_empty": int(sum(a and not b for a, b in zip(ov, nv))),
                     "empty_to_value": int(sum(b and not a for a, b in zip(ov, nv))),
                     "value_to_other_value": int(sum(a and b and x != y for a, b, x, y in zip(ov, nv, o, nw))),
                     "unchanged": int(sum(x == y for x, y in zip(o, nw)))}
    tot = {key: int(sum(r[key] for r in cmp.values())) for key in next(iter(cmp.values()))} if cmp else {}
    ruled = None
    if shakemaps is not None:
        rtext, ruled = zhu_ruling46_text(cases, zhu_grids, shakemaps, zv)
        writes[ZHU_RULED_FILE] = rtext
        files[ZHU_RULED_FILE] = {**base, "pga_variant": "usgs-grid-georef-2026-11-23-ruling46",
                                 "sha256": sha_bytes(rtext.encode()),
                                 "n_values": sum(1 for ln in rtext.splitlines()[1:] if _cell(ln) != ""),
                                 "rule": ZHU_RULED_RULE, "zhu_row_order": ZHU_REFREEZE_ROW_ORDER,
                                 "limits": zhu_cells.LIMITS, "limits_source": zhu_cells.LIMITS_SOURCE}
    not_run = dict(m1["not_run"])
    if shakemaps is None:
        not_run["USGS Zhu 2017 ruling 46 split"] = "no ShakeMap grids passed"
    pm = {"schema": SCHEMA_PREDICTIONS_V2, "item": "D8 Zhu refreeze (eval M5)", "manifest_version": 2,
          "inputs_manifest_sha256": mh, "kit_commit": kit_commit, "n_cases": int(len(cases)), "files": files,
          "not_run": not_run, "scored": False, "labels_opened": False,
          "supersedes": {"manifest": {"name": PREDICTIONS_MANIFEST_V1, "sha256": m1_sha},
                         "files": {ZHU_V1_FILE: {"sha256": m1["files"][ZHU_V1_FILE]["sha256"],
                                                 "why": "grid rows read as stored, which is mirrored (erratum "
                                                        "ERRATUM-ZHU-ROW-ORDER-2026-11-23)",
                                                 "replaced_by": ZHU_REFREEZE_FILE}},
                         "carried_over_unchanged": sorted(k for k in m1["files"] if k != ZHU_V1_FILE)},
          "zhu_grid_read": zrec, "zhu_compare_with_first": {"per_earthquake": cmp, "total": tot},
          "zhu_ruling46_split": ruled, "zhu_row_order": ZHU_REFREEZE_ROW_ORDER,
          "exposure_kit_mark_rules": m1.get("exposure_kit_mark_rules"), "pga_variants": m1.get("pga_variants")}
    for name, body in writes.items():
        (d / name).write_text(body, encoding="utf-8")
    p2 = d / PREDICTIONS_MANIFEST_V2
    p2.write_text(json.dumps(pm, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (d / SUPERSEDED_NOTE).write_text(
        f"{PREDICTIONS_MANIFEST_V1} (sha256 {m1_sha}) and {ZHU_V1_FILE} are SUPERSEDED by "
        f"{PREDICTIONS_MANIFEST_V2} (sha256 {sha_file(p2)}). They are kept byte for byte as first frozen. "
        f"Reason: the first Zhu read was mirrored (pillar docs/evals/ERRATUM-ZHU-ROW-ORDER-2026-11-23.md). "
        f"The other {len(files) - (2 if ruled is not None else 1)} files are unchanged and listed in both "
        f"manifests.\n", encoding="utf-8")
    return {"manifest": str(p2), "sha256": sha_file(p2), "n_files": len(files), "compare": cmp, "total": tot,
            "ruling46": ruled, "zhu_grid_read": zrec}


# --------------------------------------------------------------------------------------------------------------
# Inputs v2 (founder ruling 50 check, 2026-11-24): withhold an NGL station PGA that belongs to the other event
# --------------------------------------------------------------------------------------------------------------
# The 2023 Kahramanmaras doublet: NGL lists Pazarcik (Mw 7.8) and Elbistan (Mw 7.5) as two events. An NGL GMIM row
# "Ground Motion Station <net>-<code>" gives a station PGA. When that station is in no ShakeMap station list of the
# row's own event, but is in the other event's list with the same PGA (within STATION_MATCH_REL_TOL), the value is
# the other event's record. Such a value is withheld (NaN) in a NEW cases file and manifest; the first ones stay.
# Case ids do not change, so the sealed label file is not touched. Nothing here opens or takes a label.

INPUTS_MANIFEST_V1 = "INPUTS-MANIFEST.json"
INPUTS_MANIFEST_V2 = "INPUTS-MANIFEST-v2.json"
CASES_V2_FILE = "cases-v2-elbistan-pga-2026-11-24.parquet"
INPUTS_SUPERSEDED_NOTE = "SUPERSEDED-INPUTS.txt"
STATION_MATCH_REL_TOL = 0.05
DOUBLET = {"Pazarcik": "Elbistan", "Elbistan": "Pazarcik"}
STATION_CHECK_WORDS = ("OWN EVENT LIST", "WRONG EVENT", "OTHER EVENT, VALUE DIFFERS", "IN NO LIST", "NOT A STATION")
WITHHELD_SOURCE = ("withheld 2026-11-24 (ruling 50 check): the NGL GMIM station PGA is the other doublet event's "
                   "record (station in no ShakeMap list of this event; same PGA within 5% in the other event's list)")
# Columns a v2 inputs file may change. Every other column must stay identical, value for value.
INPUTS_V2_CHANGEABLE = ("pga_primary", "pga_primary_source")


def ngl_station(method: str | None) -> tuple[str, str] | None:
    """(network, code) from an NGL GMIM_METHOD such as "Ground Motion Station TK-2718"; None if it names no
    station."""
    pre = "Ground Motion Station "
    if not method or not method.startswith(pre):
        return None
    net, _, code = method[len(pre):].strip().partition("-")
    return (net, code) if net and code else None


def station_event_check(value_g: float, station: tuple[str, str] | None, own: dict, other: dict,
                        rel_tol: float = STATION_MATCH_REL_TOL) -> str:
    """One word from STATION_CHECK_WORDS for an NGL station PGA. own and other: {(network, code): PGA in g, or a
    list of PGA in g with one value per station list} from the ShakeMap station lists of the row's own event and of
    the other doublet event. WRONG EVENT needs the value to match every list that holds the station (review 1, N3).

    Weight of evidence (review 1, N2): the value match at the same station carries the word; absence from the own
    event's list only supports it, because the own list may be sparse. The record also checks the ShakeMap gap at
    the site."""
    if station is None:
        return "NOT A STATION"
    if station in own:
        return "OWN EVENT LIST"
    if station in other:
        o = np.atleast_1d(np.asarray(other[station], float))
        if (np.isfinite(value_g) and o.size and np.all(o > 0)
                and np.all(np.abs(float(value_g) / o - 1.0) <= rel_tol)):
            return "WRONG EVENT"
        return "OTHER EVENT, VALUE DIFFERS"
    return "IN NO LIST"


def supersede_inputs(inputs_dir: str | Path, withheld: dict, *, check: dict, kit_commit: str = "") -> dict:
    """Write CASES_V2_FILE and INPUTS_MANIFEST_V2 beside the first inputs, with pga_primary withheld (NaN) for the
    case ids in `withheld` ({case_id: reason word}). The first manifest and cases file are kept byte for byte and
    named in SUPERSEDED-INPUTS.txt. Every check runs before the first write; a file that exists is never rewritten.
    check: counts-only record of the station check, copied into the manifest."""
    d = Path(inputs_dir)
    new = [d / CASES_V2_FILE, d / INPUTS_MANIFEST_V2, d / INPUTS_SUPERSEDED_NOTE]
    if any(p.exists() for p in new):
        raise D8Refused("inputs v2 exist already; they are never rewritten")
    man1, cases, _ = load_inputs(d, INPUTS_MANIFEST_V1)
    m1_sha = sha_file(d / INPUTS_MANIFEST_V1)
    if not withheld:
        raise D8Refused("nothing to withhold: no inputs v2 is written")
    if set(withheld.values()) - {"WRONG EVENT"}:
        raise D8Refused("only a WRONG EVENT station PGA is withheld")
    unknown = set(withheld) - set(cases["case_id"])
    if unknown:
        raise D8Refused(f"{len(unknown)} withheld case ids are not in the inputs")
    src = cases.loc[cases["case_id"].isin(list(withheld)), "pga_primary_source"].astype(str)
    if not src.str.startswith("NGL GMIM PGA").all():
        raise D8Refused("only an NGL GMIM PGA can be withheld (a PRJ-4022 or PRJ-6544 PGA is not a station row)")
    v2 = cases.copy()
    k = v2["case_id"].isin(list(withheld)).to_numpy()
    v2.loc[k, "pga_primary"] = np.nan
    v2.loc[k, "pga_primary_source"] = WITHHELD_SOURCE
    same = [c for c in cases.columns if c not in INPUTS_V2_CHANGEABLE]
    if not v2[same].equals(cases[same]) or list(v2.columns) != list(cases.columns):
        raise D8Refused("inputs v2 would change a column outside INPUTS_V2_CHANGEABLE")
    check_input_columns(v2.columns)
    per_eq = {str(e): int(n) for e, n in v2.loc[k, "earthquake"].value_counts().sort_index().items()}
    # writes (after every check)
    v2.to_parquet(d / CASES_V2_FILE, index=False)
    files = {CASES_V2_FILE: sha_file(d / CASES_V2_FILE), "traces.npz": man1["files"]["traces.npz"]}
    man = {k_: v_ for k_, v_ in man1.items() if k_ not in ("files", "built_utc")}
    man.update({"schema": SCHEMA_INPUTS, "manifest_version": 2, "cases_file": CASES_V2_FILE, "files": files,
                "item": "D8 ruling 50 Elbistan PGA check (eval M5)", "kit_commit": kit_commit,
                "supersedes": {"manifest": {"name": INPUTS_MANIFEST_V1, "sha256": m1_sha},
                               "cases_file": {"name": "cases.parquet", "sha256": man1["files"]["cases.parquet"]},
                               "why": "an NGL station PGA of the other doublet event entered pga_primary"},
                "changed": {"columns": list(INPUTS_V2_CHANGEABLE), "cases_withheld": int(k.sum()),
                            "per_earthquake": per_eq, "rule": WITHHELD_SOURCE,
                            "rel_tol": STATION_MATCH_REL_TOL},
                "station_check": check, "labels_opened": False, "scored": False})
    man.setdefault("rules", {})
    man["rules"] = {**man["rules"], "pga_primary": man1.get("rules", {}).get("pga_primary", "") +
                    "; v2: " + WITHHELD_SOURCE}
    p2 = d / INPUTS_MANIFEST_V2
    p2.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (d / INPUTS_SUPERSEDED_NOTE).write_text(
        f"{INPUTS_MANIFEST_V1} (sha256 {m1_sha}) and cases.parquet (sha256 {man1['files']['cases.parquet']}) are "
        f"SUPERSEDED by {INPUTS_MANIFEST_V2} (sha256 {sha_file(p2)}) and {CASES_V2_FILE}. They are kept byte for "
        f"byte as first frozen. Reason: {int(k.sum())} pga_primary values were the other doublet event's station "
        f"record (ruling 50 check, 2026-11-24). traces.npz is unchanged and listed in both manifests. Predictions "
        f"frozen on the first inputs are not rerun by this step.\n", encoding="utf-8")
    return {"manifest": str(p2), "sha256": sha_file(p2), "cases_file_sha256": files[CASES_V2_FILE],
            "withheld": int(k.sum()), "per_earthquake": per_eq, "supersedes": m1_sha}


# --------------------------------------------------------------------------------------------------------------
# Predictions v3 (founder ruling 52a, 2026-11-24): every model of predictions manifest v2 rerun on inputs v2, with
# the same model list, options and code. Only the inputs change. A file whose bytes are unchanged is carried over
# under its v2 name; a file that changed is written under a new name (PRED_V3_TAG). PREDICTIONS-MANIFEST-v3.json
# names inputs v2 and supersedes the v1 and v2 manifests by name and sha256; SUPERSEDED-v3.txt says so. Nothing is
# rewritten. A value may change only on a case whose model inputs moved between inputs v1 and v2; any other change
# refuses the run before the first write (it would be a finding). Opens inputs/ and predictions/ only.
# --------------------------------------------------------------------------------------------------------------
PREDICTIONS_MANIFEST_V3 = "PREDICTIONS-MANIFEST-v3.json"
SUPERSEDED_NOTE_V3 = "SUPERSEDED-v3.txt"
PRED_V3_TAG = "inputs-v2-2026-11-24"
SCHEMA_PREDICTIONS_V3 = "evalkit-world-predictions/3"     # the v2 layout; manifest_version tells them apart
# Input columns each frozen file reads, per model (run_models, run_zhu, zhu_ruling46_text). A changed value is
# expected only on a case where one of the columns of that file's model moved. Keyed per model, not per PGA variant
# (predictions v3 review 1, N1): the shaking score reads only its PGA column and mw, so a move in gwt_m or trace_row
# alone never excuses a shaking score change. "{pga}" stands for the file's PGA variant.
PRED_V3_INPUT_COLUMNS = {
    "shaking": ("{pga}", "mw"),
    "cpt": ("{pga}", "mw", "gwt_m", "trace_row"),
    "zhu": ("earthquake", "lat", "lon"),
}


def pred_v3_input_columns(model: str, pga_variant: str) -> tuple[str, ...]:
    """The input columns one frozen predictions file reads (see PRED_V3_INPUT_COLUMNS). Refused for a model the
    kit does not know, so a new model never runs under a guess."""
    kind = (MODELS.get(model) or {}).get("kind")
    if kind == "zhu":
        return PRED_V3_INPUT_COLUMNS["zhu"]
    if kind not in PRED_V3_INPUT_COLUMNS or pga_variant not in PGA_VARIANTS:
        raise D8Refused(f"no input column list for model {model!r} on {pga_variant!r}")
    return tuple(pga_variant if c == "{pga}" else c for c in PRED_V3_INPUT_COLUMNS[kind])


def v3_file_name(v2_name: str) -> str:
    """The new name of a file that predictions v3 change: <v2 stem>-inputs-v2-2026-11-24.csv."""
    stem = v2_name[:-4] if v2_name.endswith(".csv") else v2_name
    return f"{stem}-{PRED_V3_TAG}.csv"


def _moved_rows(old, new, columns) -> np.ndarray:
    """True where any of `columns` differs between two input tables in the same case order (NaN equals NaN)."""
    import pandas as pd
    m = np.zeros(len(new), bool)
    for c in columns:
        if c not in new.columns or c not in old.columns:
            m[:] = True                 # a column added or dropped moves every row
            continue
        a, b = old[c], new[c]
        same = (a == b) | (pd.isna(a) & pd.isna(b))
        m |= ~same.to_numpy(bool)
    return m


def _cell(line: str) -> str:
    return line.split(",", 1)[1].split(",", 1)[0]


def predict_v3(inputs_dir: str | Path, pred_dir: str | Path, helpers: dict | None = None,
               zhu_grids: dict | None = None, shakemaps: dict | None = None, kit_commit: str = "") -> dict:
    """Rerun every model of PREDICTIONS-MANIFEST-v2.json on inputs v2 (INPUTS-MANIFEST-v2.json). Writes the files
    that change under new names, PREDICTIONS-MANIFEST-v3.json and SUPERSEDED-v3.txt; never rewrites a file. Every
    check runs before the first write. Returns counts only (per file and per earthquake)."""
    import pandas as pd
    helpers = helpers or {}
    d, di = Path(pred_dir), Path(inputs_dir)
    for name in (PREDICTIONS_MANIFEST_V3, SUPERSEDED_NOTE_V3):
        if (d / name).exists():
            raise D8Refused(f"{name} exists; predictions v3 are never rewritten")
    if zhu_grids is None or shakemaps is None:
        raise D8Refused("predictions v3 rerun every v2 model: the Zhu grids and their ShakeMaps are needed")
    bad = sorted(k for k, g in zhu_grids.items() if g.get("row_order") != ZHU_REFREEZE_ROW_ORDER)
    if bad:
        raise D8Refused(f"predictions v3 read every Zhu grid with {ZHU_REFREEZE_ROW_ORDER}, as v2; not so for {bad}")
    missing = sorted(set(zhu_grids) - set(shakemaps))
    if missing:
        raise D8Refused(f"no ShakeMap for the ruling 46 split: {missing}")
    # predictions v2, byte for byte
    m2p = d / PREDICTIONS_MANIFEST_V2
    if not m2p.exists():
        raise D8Refused(f"no {PREDICTIONS_MANIFEST_V2} in {d.name}/: predictions v3 rerun the v2 model list")
    m2, m2_sha = json.loads(m2p.read_text()), sha_file(m2p)
    for name, row in m2["files"].items():
        if sha_file(d / name) != row["sha256"]:
            raise D8Refused(f"{name} differs from {PREDICTIONS_MANIFEST_V2}; stop")
    m1 = (m2.get("supersedes") or {}).get("manifest") or {}
    m1p = d / m1.get("name", PREDICTIONS_MANIFEST_V1)
    if not m1p.exists() or sha_file(m1p) != m1.get("sha256"):
        raise D8Refused(f"{PREDICTIONS_MANIFEST_V1} differs from the sha256 that {PREDICTIONS_MANIFEST_V2} names")
    # inputs v2: not superseded, and the inputs it supersedes are the ones v2 predictions were built on
    i2 = di / INPUTS_MANIFEST_V2
    if not i2.exists():
        raise D8Refused(f"no {INPUTS_MANIFEST_V2}: predictions v3 run on inputs v2")
    ih2 = sha_file(i2)
    if ih2 in superseded_inputs_sha256(di):
        raise D8Refused(f"{INPUTS_MANIFEST_V2} (sha256 {ih2}) is superseded; predict on the inputs that replace it")
    man2, cases, traces = load_inputs(di, INPUTS_MANIFEST_V2)
    ih1 = ((man2.get("supersedes") or {}).get("manifest") or {}).get("sha256")
    if not ih1 or m2.get("inputs_manifest_sha256") != ih1:
        raise D8Refused("the v2 predictions were not built on the inputs that inputs v2 supersede")
    _, cases1, _ = load_inputs(di, ((man2.get("supersedes") or {}).get("manifest") or {}).get("name",
                                                                                            INPUTS_MANIFEST_V1))
    ids = cases["case_id"].tolist()
    if cases1["case_id"].tolist() != ids:
        raise D8Refused("inputs v2 change the case ids or their order")
    moved_any = _moved_rows(cases1, cases, sorted(set(cases.columns) | set(cases1.columns)))
    moved_cols = sorted(c for c in cases.columns if _moved_rows(cases1, cases, [c]).any())
    # rerun every model, in memory
    texts, rows = {}, {}
    for var in PGA_VARIANTS:
        for name, (t, r) in prediction_texts(ids, run_models(cases, traces, var, **helpers), var).items():
            texts[name], rows[name] = t, r
    zv, zrec = run_zhu(cases, zhu_grids)
    texts[ZHU_REFREEZE_FILE] = "case_id,value\n" + "".join(f"{c},{_fmt(x)}\n" for c, x in zip(ids, zv))
    texts[ZHU_RULED_FILE], ruled = zhu_ruling46_text(cases, zhu_grids, shakemaps, zv)
    if set(texts) != set(m2["files"]):
        raise D8Refused(f"the v3 file set differs from v2: {sorted(set(texts) ^ set(m2['files']))}")
    keep_keys = ("model", "pga_variant", "house", "output_kind", "rule", "exposure_column", "chain_id")
    for name, r in rows.items():
        if {k: r.get(k) for k in keep_keys} != {k: m2["files"][name].get(k) for k in keep_keys}:
            raise D8Refused(f"{name}: the model options differ from {PREDICTIONS_MANIFEST_V2}")
    # compare with v2, per file and per earthquake (counts only)
    eq = cases["earthquake"].to_numpy()
    eqs = sorted(set(eq.tolist()))
    zero = {"values_changed": 0, "value_to_empty": 0, "empty_to_value": 0, "value_to_other_value": 0,
            "reason_changed": 0}
    per_file, per_eq_total, unexpected = {}, {e: dict(zero, files_changed=0) for e in eqs}, {}
    changed_cases = np.zeros(len(ids), bool)
    for name in sorted(texts):
        old = (d / name).read_text(encoding="utf-8").splitlines()
        new = texts[name].splitlines()
        if old[0] != new[0] or len(old) != len(new) or [x.split(",", 1)[0] for x in old[1:]] != ids:
            raise D8Refused(f"{name}: the v2 file is not in input case order")
        diff = np.array([a != b for a, b in zip(old[1:], new[1:])], bool)
        cols = pred_v3_input_columns(m2["files"][name]["model"], m2["files"][name]["pga_variant"])
        bad_rows = diff & ~_moved_rows(cases1, cases, cols)
        if bad_rows.any():
            unexpected[name] = {e: int(bad_rows[eq == e].sum()) for e in eqs if bad_rows[eq == e].any()}
        changed_cases |= diff
        fe = {}
        for e in eqs:
            k = np.flatnonzero((eq == e) & diff)
            if not k.size:
                continue
            o = [_cell(old[i + 1]) for i in k]
            n_ = [_cell(new[i + 1]) for i in k]
            c = {"values_changed": int(k.size),
                 "value_to_empty": int(sum(a != "" and b == "" for a, b in zip(o, n_))),
                 "empty_to_value": int(sum(a == "" and b != "" for a, b in zip(o, n_))),
                 "value_to_other_value": int(sum(a != "" and b != "" and a != b for a, b in zip(o, n_))),
                 "reason_changed": int(sum(a == b for a, b in zip(o, n_)))}
            fe[e] = c
            for key, v in c.items():
                per_eq_total[e][key] += v
            per_eq_total[e]["files_changed"] += 1
        per_file[name] = {"values_changed": int(diff.sum()), "per_earthquake": fe}
    if unexpected:
        raise D8Refused(f"a value changed on a case whose model inputs did not move (a finding; nothing written): "
                        f"{json.dumps(unexpected, sort_keys=True)}")
    changed = sorted(n for n, r in per_file.items() if r["values_changed"])
    files, writes, sup_files = {}, {}, {}
    for name in sorted(texts):
        if name not in changed:
            files[name] = m2["files"][name]
            continue
        nn = v3_file_name(name)
        if (d / nn).exists():
            raise D8Refused(f"{nn} exists; predictions v3 are never rewritten")
        t = texts[name]
        nv = sum(1 for x in t.splitlines()[1:] if _cell(x) != "")
        files[nn] = {**m2["files"][name], "sha256": sha_bytes(t.encode()), "n_values": int(nv),
                     "inputs_manifest": INPUTS_MANIFEST_V2, "replaces": name}
        writes[nn] = t
        sup_files[name] = {"sha256": m2["files"][name]["sha256"], "replaced_by": nn,
                           "why": "rerun on inputs v2: " + ", ".join(moved_cols) + " moved on "
                                  f"{int(moved_any.sum())} case(s) ({(man2.get('supersedes') or {}).get('why', '')})"}
    per_eq_inputs = {str(e): int(n) for e, n in pd.Series(eq[moved_any]).value_counts().sort_index().items()}
    pm = {"schema": SCHEMA_PREDICTIONS_V3, "item": "D8 predictions v3 on inputs v2 (eval M5, ruling 52a)",
          "manifest_version": 3, "inputs_manifest": INPUTS_MANIFEST_V2, "inputs_manifest_sha256": ih2,
          "kit_commit": kit_commit, "n_cases": int(len(ids)), "files": files, "not_run": m2.get("not_run"),
          "scored": False, "labels_opened": False,
          "same_as_v2": "model list, options (rule, chain id, output kind, exposure column) and code paths of "
                        f"{PREDICTIONS_MANIFEST_V2}; only the inputs differ",
          "supersedes": {"manifest": {"name": PREDICTIONS_MANIFEST_V2, "sha256": m2_sha},
                         "manifests": {m1p.name: sha_file(m1p), PREDICTIONS_MANIFEST_V2: m2_sha},
                         "files": sup_files, "carried_over_unchanged": sorted(n for n in texts if n not in changed)},
          "inputs_changed": {"columns": moved_cols, "cases": int(moved_any.sum()), "per_earthquake": per_eq_inputs,
                             "from": ih1, "to": ih2},
          "compare_with_v2": {"per_file": per_file, "per_earthquake": per_eq_total,
                              "cases_changed": int(changed_cases.sum()),
                              "cases_changed_outside_moved_inputs": int((changed_cases & ~moved_any).sum())},
          "zhu_grid_read": zrec, "zhu_ruling46_split": ruled, "zhu_row_order": ZHU_REFREEZE_ROW_ORDER,
          "exposure_kit_mark_rules": m2.get("exposure_kit_mark_rules"), "pga_variants": m2.get("pga_variants")}
    # writes (after every check)
    for name, body in writes.items():
        (d / name).write_text(body, encoding="utf-8")
    p3 = d / PREDICTIONS_MANIFEST_V3
    p3.write_text(json.dumps(pm, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (d / SUPERSEDED_NOTE_V3).write_text(
        f"{m1p.name} (sha256 {sha_file(m1p)}) and {PREDICTIONS_MANIFEST_V2} (sha256 {m2_sha}) are SUPERSEDED by "
        f"{PREDICTIONS_MANIFEST_V3} (sha256 {sha_file(p3)}), built on {INPUTS_MANIFEST_V2} (sha256 {ih2}). So are "
        f"these v2 files, each replaced by a file of the same name plus -{PRED_V3_TAG}: {', '.join(changed)}. "
        f"Everything is kept byte for byte as first frozen. The other {len(texts) - len(changed)} files are "
        f"unchanged and listed in both manifests.\n", encoding="utf-8")
    return {"manifest": str(p3), "sha256": sha_file(p3), "n_files": len(files), "changed_files": changed,
            "per_earthquake": {e: v for e, v in per_eq_total.items() if v["values_changed"]},
            "cases_changed": int(changed_cases.sum()), "inputs_changed": pm["inputs_changed"]}

"""Label sources and label sets for the shared G3 registration (experiments 3, 5 and 9).

Registered by docs/REGISTRATION-G3-LABEL-SOURCE-3-5-9-2026-11-02.md (30171ff), sections 1 to 3, committed before
this file. Serves GOALS.md G3. Not a fresh test: every label here was read against Level 0 before. No em dashes.

Use:
    from scoreboard import label_sets as LS
    rows = LS.load_rows()                       # 4,542 train/val CPTs, npz order; no label array is read
    S = LS.build_label_set("B", rows)           # {event: (labels, mask)}; labels int8 1/0/-1 over all rows
    S25 = LS.build_label_set("B", rows, footprint=True)   # same, restricted to the F25 road footprint
    src = LS.label_sources(rows)                # {(source, event): int8 array}, cached per rows object

Sources (registration section 2): "aerial" (CGD0200, G6b rule via scoreboard/labels.py), "bench" (label_yn),
"road25", "road50" (CGD0300 road, Jun Green = no), "road25_g1", "road50_g1" (Jun Green = yes), "property" (CGD0300
property layers, Sep 2010 and Feb 2011). The rows npz label arrays are never a source: asking for "npz" raises.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from . import labels as L

ROOT = Path(__file__).resolve().parents[1]
PILLAR = ROOT.parents[1]
ROWS_NPZ = ROOT / "results" / "g6-jun2011-rows-2026-10-31.npz"
CGD0300_KML = (PILLAR / "data" / "nzgd-public" / "nzgd-v1" / "NZGDv1_EQC_Liquefaction_and_Lateral_Spreading_Observations"
               / "doc.kml")
CGD0300_SHA = "af7ee04d3519455a27a8330983f41b39f69a5c7117b1f77e6b35d81b54079f35"
EVS = ("sep2010", "feb2011", "jun2011", "dec2011")
BENCH_EV = {"sep2010": "2010-Darfield", "feb2011": "2011-Christchurch"}
KNS = "{http://www.opengis.net/kml/2.2}"

ROAD_FOLDER = {"feb2011": "22 Feb 2011 - Road Observations", "jun2011": "13 Jun 2011 - Road Observations"}
# any-ejecta yes; Feb legend 4 colours, Jun legend 6 colours (two protocols; never compare moderate-or-worse classes)
ROAD_YES = {"feb2011": {"Blue": 0, "Yellow": 1, "Orange": 1, "Magenta": 1},
            "jun2011": {"Blue": 0, "Green": 0, "Light Orange": 1, "Dark Orange": 1, "Red": 1, "Red-2": 1}}
PROP_FOLDER = {"sep2010": "4 Sep 2010 - Property Observations", "feb2011": "22 Feb 2011 - Property Observations"}
PROP_YES = {"#blueP": 0, "#greenP": 0, "#orange1P": 1, "#orange2P": 1, "#red1P": 1, "#red2P": 1}
FOOTPRINT_M = 25.0

# registration section 3
SETS = {
    "A": {"sep2010": "aerial", "feb2011": "aerial", "jun2011": "aerial", "dec2011": "aerial"},
    "B": {"sep2010": "bench", "feb2011": "bench", "jun2011": "road25", "dec2011": "aerial"},
    "C": {"sep2010": "bench", "feb2011": "road25", "jun2011": "road25", "dec2011": "aerial"},
    "D": {"sep2010": "property", "feb2011": "property", "jun2011": "road25", "dec2011": "aerial"},
    "B_G1": {"sep2010": "bench", "feb2011": "bench", "jun2011": "road25_g1", "dec2011": "aerial"},
    "B_50": {"sep2010": "bench", "feb2011": "bench", "jun2011": "road50", "dec2011": "aerial"},
    "A_JR": {"sep2010": "aerial", "feb2011": "aerial", "jun2011": "road25", "dec2011": "aerial"},
}
PRIMARY_SET = "B"
MAIN_SETS = ("A", "B", "C", "D")
BESIDE_SETS = ("B_G1", "B_50", "A_JR")
SOURCES = ("aerial", "bench", "road25", "road50", "road25_g1", "road50_g1", "property")
SOURCE_EVENTS = {"aerial": EVS, "bench": ("sep2010", "feb2011"), "road25": ("feb2011", "jun2011"),
                 "road50": ("feb2011", "jun2011"), "road25_g1": ("jun2011",), "road50_g1": ("jun2011",),
                 "property": ("sep2010", "feb2011")}
# G6b gate 2 counts (yes, no) for set A: parity gate L1
G6B_COUNTS = {"sep2010": (902, 1798), "feb2011": (1309, 120), "jun2011": (2864, 487), "dec2011": (1447, 1818)}
MIN_ROWS, MIN_CLASS = 200, 30


class LabelSetError(ValueError):
    pass


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _g6():
    s = importlib.util.spec_from_file_location("g6_ls", ROOT / "scripts" / "run_g6_jun2011.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


@dataclass(eq=False)
class Rows:
    """The 4,542 train/val CPTs in rows-npz order. x, y are bench NZTM; bench_yn holds label_yn per bench event."""
    cpt: np.ndarray
    tile: np.ndarray
    x: np.ndarray
    y: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    bench_yn: dict
    nztm: object
    cache: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.cpt)


def check_no_test(b: pd.DataFrame) -> None:
    if "split" not in b.columns or (b["split"] == "test").any() or not b["split"].isin(["train", "val"]).all():
        raise LabelSetError("a row outside the train/val splits reached the label builder (GOALS rule 7)")


def load_rows(npz=ROWS_NPZ) -> Rows:
    """Rows from the G6 rows npz (cpt, tile only) and the bench (x, y, label_yn), train/val only."""
    d = np.load(npz, allow_pickle=True)
    cpt, tile = d["cpt"], d["tile"]
    g6 = _g6()
    mod = g6.bench_module()
    b = mod.load_bench_v2()
    b = b[b["split"].isin(["train", "val"])].reset_index(drop=True)
    check_no_test(b)
    fb = b[b["event"] == BENCH_EV["feb2011"]].drop_duplicates("cpt").set_index("cpt")
    if not pd.Index(cpt).isin(fb.index).all():
        raise LabelSetError("a rows-npz CPT is not a train/val 2011-Christchurch bench row")
    fb = fb.loc[cpt]
    yn = {}
    for e, ev in BENCH_EV.items():
        v = b[b["event"] == ev].drop_duplicates("cpt").set_index("cpt")["label_yn"].reindex(cpt).to_numpy(float)
        yn[e] = np.where(np.isfinite(v), v, -1).astype(np.int8)
    return Rows(cpt=cpt, tile=tile, x=fb["x"].to_numpy(float), y=fb["y"].to_numpy(float),
                lat=fb["lat"].to_numpy(float), lon=fb["lon"].to_numpy(float), bench_yn=yn, nztm=mod.nztm)


# ---------------------------------------------------------------------------------------------------------------
# CGD0300
# ---------------------------------------------------------------------------------------------------------------

def _kml_root(path=CGD0300_KML, check_sha=True):
    if check_sha and _sha(path) != CGD0300_SHA:
        raise LabelSetError("CGD0300 doc.kml does not match its registered sha256")
    return ET.parse(path).getroot()


def _folder(root, name):
    for f in root.iter(KNS + "Folder"):
        if f.findtext(KNS + "name") == name:
            return f
    raise LabelSetError(f"CGD0300 folder {name!r} not found")


def road_segments(root, event, nztm):
    """(LineStrings in NZTM, class names) for one road folder. Class = the sub-folder name (legend colour)."""
    geoms, cls = [], []
    for sf in _folder(root, ROAD_FOLDER[event]).findall(KNS + "Folder"):
        name = sf.findtext(KNS + "name")
        for ls in sf.iter(KNS + "LineString"):
            a = np.array([[float(v) for v in c.split(",")[:2]] for c in ls.findtext(KNS + "coordinates").split()])
            if len(a) < 2:
                continue
            X, Y = nztm(a[:, 1], a[:, 0])
            geoms.append(shapely.LineString(np.column_stack([X, Y])))
            cls.append(name)
    return np.array(geoms), np.array(cls)


def nearest_class(geoms, cls, x, y):
    """Per point: distance to the nearest segment and that segment's class."""
    pts = shapely.points(x, y)
    (pi, gi), dist = shapely.STRtree(geoms).query_nearest(pts, return_distance=True, all_matches=False)
    d = np.full(len(x), np.inf)
    c = np.full(len(x), "", dtype=object)
    d[pi], c[pi] = dist, cls[gi]
    return d, c.astype(str)


def road_labels(dist, cls, event, radius, green_yes=False):
    m = dict(ROAD_YES[event])
    if event == "jun2011" and green_yes:
        m["Green"] = 1
    unknown = set(cls[dist <= radius]) - set(m)
    if unknown:
        raise LabelSetError(f"road class not in the legend map: {sorted(unknown)}")
    v = np.array([m.get(s, -1) for s in cls])
    return np.where(dist <= radius, v, -1).astype(np.int8)


def property_polys(root, event, nztm):
    def ring(el):
        p = np.array([[float(v) for v in c.split(",")[:2]] for c in el.text.split()])
        X, Y = nztm(p[:, 1], p[:, 0])
        return np.column_stack([X, Y])
    out = {}
    for p in _folder(root, PROP_FOLDER[event]).findall(KNS + "Placemark"):
        st = p.findtext(KNS + "styleUrl")
        if st not in PROP_YES:
            raise LabelSetError(f"property style not in the legend map: {st}")
        for poly in p.iter(KNS + "Polygon"):
            o = ring(poly.find(f"{KNS}outerBoundaryIs/{KNS}LinearRing/{KNS}coordinates"))
            inner = [ring(r) for r in poly.findall(f"{KNS}innerBoundaryIs/{KNS}LinearRing/{KNS}coordinates")]
            out.setdefault(st, []).append(shapely.Polygon(o, inner))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Sources and sets
# ---------------------------------------------------------------------------------------------------------------

def label_sources(rows: Rows, kml=CGD0300_KML) -> dict:
    """{(source, event): int8 labels}, plus ("roaddist", e) and ("roadclass", e). Cached on the rows object."""
    if "sources" in rows.cache:
        return rows.cache["sources"]
    g6 = _g6()
    out = {}
    for e in EVS:
        polys = L.read_cgd0200(g6.KMZ, g6.KML_FOLDER[e], rows.nztm)
        out[("aerial", e)] = L.label_points(L.build_classes(polys), L.CGD0200_LABEL, rows.x, rows.y)[0].astype(np.int8)
    for e in BENCH_EV:
        out[("bench", e)] = rows.bench_yn[e].astype(np.int8)
    root = _kml_root(kml)
    for e in ROAD_FOLDER:
        geoms, cls = road_segments(root, e, rows.nztm)
        d, c = nearest_class(geoms, cls, rows.x, rows.y)
        out[("roaddist", e)], out[("roadclass", e)] = d, c
        for r in (25, 50):
            out[(f"road{r}", e)] = road_labels(d, c, e, r)
            if e == "jun2011":
                out[(f"road{r}_g1", e)] = road_labels(d, c, e, r, green_yes=True)
    for e in PROP_FOLDER:
        cl = L.build_classes(property_polys(root, e, rows.nztm))
        out[("property", e)] = L.label_points(cl, PROP_YES, rows.x, rows.y, edge_m=0.0)[0].astype(np.int8)
    rows.cache["sources"] = out
    return out


def road_footprint(rows: Rows, radius=FOOTPRINT_M) -> np.ndarray:
    """F25: within radius of a Feb 2011 road segment and of a Jun 2011 road segment, any class."""
    s = label_sources(rows)
    return (s[("roaddist", "feb2011")] <= radius) & (s[("roaddist", "jun2011")] <= radius)


def spec_of(name_or_spec) -> dict:
    if isinstance(name_or_spec, dict):
        spec = dict(name_or_spec)
    elif name_or_spec in SETS:
        spec = dict(SETS[name_or_spec])
    else:
        raise LabelSetError(f"unknown label set {name_or_spec!r}; known: {sorted(SETS)}")
    for e, src in spec.items():
        if src == "npz" or str(src).startswith("npz"):
            raise LabelSetError("the rows-npz label arrays are not a label source (Feb 2011 lacks #style00)")
        if src not in SOURCES or e not in SOURCE_EVENTS[src]:
            raise LabelSetError(f"source {src!r} has no labels for {e}")
    return spec


def build_label_set(name_or_spec, rows: Rows | None = None, footprint: bool = False) -> dict:
    """{event: (labels, mask)}. labels: int8 over all rows (1 yes, 0 no, -1 unlabelled, or outside F25 when
    footprint=True); mask = labels >= 0."""
    spec = spec_of(name_or_spec)
    rows = rows if rows is not None else load_rows()
    s = label_sources(rows)
    keep = road_footprint(rows) if footprint else np.ones(len(rows), bool)
    out = {}
    for e, src in spec.items():
        lab = np.where(keep, s[(src, e)], -1).astype(np.int8)
        out[e] = (lab, lab >= 0)
    return out


def counts(label_set: dict) -> dict:
    return {e: {"yes": int((v[0] == 1).sum()), "no": int((v[0] == 0).sum())} for e, v in label_set.items()}


def count_gate(label_set: dict) -> dict:
    """Gate L2 per event: at least MIN_ROWS labelled and MIN_CLASS per class."""
    c = counts(label_set)
    return {e: bool(v["yes"] + v["no"] >= MIN_ROWS and v["yes"] >= MIN_CLASS and v["no"] >= MIN_CLASS)
            for e, v in c.items()}


def parity_gate(rows: Rows) -> dict:
    """Gate L1: set A equals the G6b gate 2 counts."""
    c = counts(build_label_set("A", rows))
    return {e: (c[e]["yes"], c[e]["no"]) == G6B_COUNTS[e] for e in EVS}


# ---------------------------------------------------------------------------------------------------------------
# Feb 2016 bench source (satellite validation registration db1c83d section 4.1, amendment 1 ac014b7). Added
# 2026-11-05 for G6+G5. Train/val only, through the same load_bench_v2 path and check_no_test as load_rows.
BENCH_FEB2016 = "2016-Valentines"


def feb2016_bench(with_labels: bool = False) -> pd.DataFrame:
    """Feb 2016 train/val bench rows, one per CPT: cpt, x, y (bench NZTM), lat, lon; plus `label` (int8: 1 yes,
    0 no, -1 not graded) only when with_labels=True. Feb 2016 grades are flagged (some "no" inferred from unmapped
    ground). The sealed test split is never read."""
    b = _g6().bench_module().load_bench_v2()
    b = b[b["split"].isin(["train", "val"])].reset_index(drop=True)
    check_no_test(b)
    f = b[b["event"] == BENCH_FEB2016].drop_duplicates("cpt").reset_index(drop=True)
    if f.empty:
        raise LabelSetError(f"no train/val bench rows for {BENCH_FEB2016}")
    out = f[["cpt", "x", "y", "lat", "lon"]].copy()
    if with_labels:
        v = f["label_yn"].to_numpy(float)
        out["label"] = np.where(np.isfinite(v), v, -1).astype(np.int8)
    return out

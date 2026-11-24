"""G5 label rule (registration amendment 2026-11-02, 0b8eab1): the G6b rule, with polygon parts only.

Serves GOALS.md G5. Polygons go through make_valid; only Polygon and MultiPolygon parts are kept; each class is
unioned and closed (buffer +GAP_M, then -GAP_M); a point inside more than one class is dropped ("two_classes"),
inside none is dropped ("unmapped"), inside one keeps that class's label if it lies EDGE_M or more from that class's
own boundary ("kept"), else it is dropped ("edge"). A non-finite distance raises: no row is dropped in silence.

Why: shapely.union_all over make_valid output can return a GeometryCollection, whose boundary is None, so every
distance is NaN and the class vanished (Feb 2011 #style00 in CGD0200). No em dashes.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import shapely

GAP_M = 0.5
EDGE_M = 10.0
KML_NS = {"k": "http://www.opengis.net/kml/2.2"}
CGD0200_LABEL = {"#style00": 1, "#style01": 1, "#style02": 0}


class LabelRuleError(ValueError):
    pass


def polygon_parts(g):
    """The Polygon parts of a geometry, as a list. Lines, points and empty parts are dropped."""
    if g is None or g.is_empty:
        return []
    t = g.geom_type
    if t == "Polygon":
        return [g]
    if t in ("MultiPolygon", "GeometryCollection"):
        out = []
        for part in shapely.get_parts(g):
            out += polygon_parts(part)
        return out
    return []


def class_union(polys, gap_m=GAP_M):
    """make_valid, keep polygon parts, union, close. Returns a prepared Polygon or MultiPolygon."""
    parts = []
    for p in polys:
        parts += polygon_parts(shapely.make_valid(p))
    u = shapely.union_all(parts) if parts else shapely.MultiPolygon()
    if gap_m:
        u = shapely.buffer(shapely.buffer(u, gap_m), -gap_m)
    if u.geom_type not in ("Polygon", "MultiPolygon") and not u.is_empty:
        u = shapely.union_all(polygon_parts(u))
    shapely.prepare(u)
    return u


def build_classes(polys_by_class, gap_m=GAP_M):
    return {c: class_union(ps, gap_m) for c, ps in polys_by_class.items()}


def label_points(classes, class_label, x, y, edge_m=EDGE_M):
    """Per point: label 0/1 or -1 (dropped), and the reason. classes: {class: geometry}; class_label: {class: 0|1}."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    inside = {c: shapely.contains_xy(g, x, y) if not g.is_empty else np.zeros(n, bool) for c, g in classes.items()}
    n_in = sum(v.astype(int) for v in inside.values()) if inside else np.zeros(n, int)
    lab = np.full(n, -1)
    reason = np.array(["unmapped"] * n, dtype=object)
    reason[n_in > 1] = "two_classes"
    pts = shapely.points(x, y)
    for c, g in classes.items():
        one = inside[c] & (n_in == 1)
        if not one.any():
            continue
        b = shapely.boundary(g)
        if b is None:
            raise LabelRuleError(f"class {c!r}: boundary is None ({g.geom_type}); polygon parts were not kept")
        d = shapely.distance(b, pts[one])
        if not np.all(np.isfinite(d)):
            raise LabelRuleError(f"class {c!r}: {int((~np.isfinite(d)).sum())} non-finite edge distances")
        keep = d >= edge_m
        idx = np.flatnonzero(one)
        lab[idx[keep]] = int(class_label[c])
        reason[idx[keep]] = "kept"
        reason[idx[~keep]] = "edge"
    return lab, reason


def _ring(el, proj):
    pts = np.array([[float(v) for v in c.split(",")[:2]] for c in el.text.split()])
    x, y = proj(pts[:, 1], pts[:, 0])
    return np.column_stack([x, y])


def read_cgd0200(kmz, folder, proj):
    """Polygons per CGD0200 style for one KML folder (one earthquake). proj(lat, lon) -> (x, y) in metres."""
    with zipfile.ZipFile(kmz) as zf:
        root = ET.fromstring(zf.read("doc.kml"))
    for f in root.iter("{%s}Folder" % KML_NS["k"]):
        if f.findtext("k:name", namespaces=KML_NS) != folder:
            continue
        out = {st: [] for st in CGD0200_LABEL}
        for p in f.iter("{%s}Placemark" % KML_NS["k"]):
            st = p.findtext("k:styleUrl", namespaces=KML_NS)
            for poly in p.iter("{%s}Polygon" % KML_NS["k"]):
                outer = _ring(poly.find("k:outerBoundaryIs/k:LinearRing/k:coordinates", KML_NS), proj)
                inner = [_ring(r, proj) for r in poly.findall("k:innerBoundaryIs/k:LinearRing/k:coordinates", KML_NS)]
                out[st].append(shapely.Polygon(outer, inner))
        return out
    raise LabelRuleError(f"folder {folder!r} not in {kmz}")

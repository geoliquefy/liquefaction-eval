"""Cut one imagery tile and one LiDAR change tile per queue item.

Imagery: Google Earth image pyramids (KML GroundOverlay tiles with a WGS84 LatLonBox, as NZGD CGD0100 serves
them) are resampled onto a north-up NZTM square, 0.1 m per pixel, 70 m wide, with the 25 m grading disc drawn.
LiDAR: the change in ground level, post-event DEM minus pre-event DEM, 5 m cells, 300 m wide, with the disc drawn
and three numbers (median change in the disc, median change in the 75 to 150 m ring, and their difference).
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import date

import numpy as np
from PIL import Image, ImageDraw

from . import geo

DISC_M = 25.0
IMG_HALF_M = 35.0
IMG_RES_M = 0.1
LIDAR_HALF_M = 150.0
LIDAR_CLIP_M = 0.5
LIDAR_SCALE = 6  # screen pixels per 5 m cell


class _ArrayCache:
    def __init__(self, size=3):
        self.size, self.d = size, OrderedDict()

    def get(self, grid):
        k = str(grid.path)
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]
        arr = geo.read_array(grid)
        self.d[k] = arr
        if len(self.d) > self.size:
            self.d.popitem(last=False)
        return arr


_CACHE = _ArrayCache()
_GRIDS = {}


def _grid(path):
    k = str(path)
    if k not in _GRIDS:
        _GRIDS[k] = geo.open_grid(path)
    return _GRIDS[k]


def _d(s):
    return date.fromisoformat(s[:10])


def _disc(draw, cx, cy, r, width=3):
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 230, 0), width=width)
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 230, 0))


def cut_imagery(src, x: float, y: float, out_path) -> dict | None:
    """Resample the source's tiles onto a 70 m NZTM square around (x, y). None when no tile covers the point."""
    n = int(round(2 * IMG_HALF_M / IMG_RES_M))
    off = (np.arange(n) + 0.5) * IMG_RES_M - IMG_HALF_M
    X, Y = np.meshgrid(x + off, y - off)
    lon, lat = geo.nztm_to_lonlat(X.ravel(), Y.ravel())
    lon, lat = np.asarray(lon), np.asarray(lat)
    lo0, lo1, la0, la1 = lon.min(), lon.max(), lat.min(), lat.max()
    out = np.zeros((n * n, 3), dtype=np.uint8)
    filled = np.zeros(n * n, dtype=bool)
    for path, north, south, east, west in src.tiles:
        if east < lo0 or west > lo1 or north < la0 or south > la1:
            continue
        m = (~filled) & (lon >= west) & (lon < east) & (lat <= north) & (lat > south)
        if not m.any() or not path.exists():
            continue
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB"))
        h, w = a.shape[:2]
        c = np.clip(((lon[m] - west) / (east - west) * w).astype(int), 0, w - 1)
        r = np.clip(((north - lat[m]) / (north - south) * h).astype(int), 0, h - 1)
        out[m] = a[r, c]
        filled |= m
    cover = float(filled.mean())
    centre_ok = bool(filled.reshape(n, n)[n // 2, n // 2])
    if not centre_ok:
        return None
    img = Image.fromarray(out.reshape(n, n, 3))
    _disc(ImageDraw.Draw(img), n / 2, n / 2, DISC_M / IMG_RES_M)
    img.save(out_path, optimize=True)
    return {"cover": round(cover, 3), "metres_per_pixel": IMG_RES_M, "width_m": 2 * IMG_HALF_M}


def pick_imagery(registry, earthquake: dict):
    """Candidate sources per role, in the order to try them: 'before' = sources dated before the event, newest
    first; 'after' = sources inside the event window, earliest first. The build uses the first candidate whose
    tiles cover the point centre, so a newer source that misses the point falls back to an older one."""
    ev, nxt = _d(earthquake["date"]), _d(earthquake["next"])
    before = [s for s in registry.imagery if _d(s.date) < ev]
    after = [s for s in registry.imagery if ev <= _d(s.date) < nxt]
    out = []
    if before:
        out.append(("before", sorted(before, key=lambda s: s.date, reverse=True)))
    if after:
        out.append(("after", sorted(after, key=lambda s: s.date)))
    return out


def _first_cover(src, tags, x, y):
    for t in tags:
        if t not in src.grids:
            continue
        path, flown, flown_end = src.grids[t]
        if not path.exists():
            continue
        g = _grid(path)
        if not g.covers(x, y):
            continue
        w = g.window(x, y, 0.0, _CACHE.get(g))
        if np.isfinite(w).all():
            return t, g, flown, flown_end
    return None


def cut_lidar(src, eq_key: str, x: float, y: float, out_path, window_end: str | None = None) -> dict | None:
    """LiDAR change tile. With `window_end`, a post grid whose last flight day (flown_end) falls on or after the
    window end is flagged: its change then includes ground movement from after the window."""
    pair = src.pairs.get(eq_key)
    if not pair:
        return None
    pre = _first_cover(src, pair["pre"], x, y)
    post = _first_cover(src, pair["post"], x, y)
    if not pre or not post:
        return None
    (tp, gp, fp, _), (tq, gq, fq, fq_end) = pre, post
    a = gp.window(x, y, LIDAR_HALF_M, _CACHE.get(gp))
    b = gq.window(x, y, LIDAR_HALF_M, _CACHE.get(gq))
    d = b - a
    n = d.shape[0]
    rr, cc = np.mgrid[0:n, 0:n]
    dist = np.hypot(rr - n // 2, cc - n // 2) * gp.dx
    disc = d[(dist <= DISC_M) & np.isfinite(d)]
    ring = d[(dist >= 75) & (dist <= 150) & np.isfinite(d)]
    med_disc = float(np.median(disc)) if disc.size else None
    med_ring = float(np.median(ring)) if ring.size else None
    from matplotlib import colormaps
    norm = np.clip((np.nan_to_num(d, nan=0.0) + LIDAR_CLIP_M) / (2 * LIDAR_CLIP_M), 0, 1)
    rgba = (colormaps["RdBu"](norm) * 255).astype(np.uint8)
    rgba[~np.isfinite(d)] = (128, 128, 128, 255)
    img = Image.fromarray(rgba[:, :, :3]).resize((n * LIDAR_SCALE, n * LIDAR_SCALE), Image.NEAREST)
    s = n * LIDAR_SCALE
    _disc(ImageDraw.Draw(img), s / 2, s / 2, DISC_M / gp.dx * LIDAR_SCALE, width=2)
    img.save(out_path, optimize=True)
    r = lambda v: None if v is None else round(v, 3)  # noqa: E731
    late = bool(window_end and fq_end and _d(fq_end) >= _d(window_end))
    q = src.rating(eq_key)
    return {"pre": f"{tp} ({fp})", "post": f"{tq} ({fq})", "quality": q["rating"], "quality_why": q["why"],
            "post_flown_after_window_end": late, "cell_m": gp.dx, "width_m": 2 * LIDAR_HALF_M,
            "clip_m": LIDAR_CLIP_M, "median_change_disc_m": r(med_disc), "median_change_ring_m": r(med_ring),
            "disc_minus_ring_m": r(None if med_disc is None or med_ring is None else med_disc - med_ring)}

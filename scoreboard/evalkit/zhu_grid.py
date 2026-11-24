"""One shared reader for USGS Zhu 2017 world grids (zhu_2017_general.hdf5), with the row order taken from the grid's
own metadata (G1+G5+G6, eval M4/M5; founder answer 2026-11-23 "verify, then erratum and v3.1"). No em dashes.

Why. The USGS ground-failure hdf5 stores the axis `y` ascending (y[0] is the south cell centre), but the writer
(mapio `MultiHazardGrid.save`, named in the file's `history` attribute) stores the `model` rows north first. Reading
row i at y[i] therefore puts every value on the mirror latitude about the grid centre. The Canterbury reader
(`scoreboard/ref_usgs_gf.py` `read_product`) flips the rows; the world readers of scorecard v2 (`build_v2.py world()`),
scorecard v3 (`build_v3.py world()`) and D8 (`scripts/d8_world_inputs_2026_11_23.py`) read the rows as stored. Two
independent verifiers found the stored read mirrored on every world grid they opened (pillar
docs/evals/ERRATUM-ZHU-ROW-ORDER-2026-11-23.md).

The rule here (ROW_ORDER_GEOREF), per grid, in this order:
1. Axis. If `y` decreases, the rows already follow `y`; the reader only reverses both to give `y` ascending.
2. GeoTIFF twin. If the product's own north-up twin (`<stem>_model.tif` beside the hdf5) is present and decodes, its
   ModelTiepoint and pixel scale give the latitude of tif row 0. The hdf5 rows are matched to the tif rows cell for
   cell (as stored, or reversed). The match decides which hdf5 row is north. A twin that decodes but contradicts
   the grid is refused: another shape, a tiepoint at neither end of `y`, or a match with neither reading (unless the
   grid reads the same both ways). Only a twin that does not decode is noted; then step 3 applies.
3. Writer convention. With no usable twin, the `history` attribute must name a known writer (WRITER_ROW0). Its
   documented row order applies. Every twin on disk agrees with it (7 world grids, 2 Canterbury grids).
4. Otherwise the grid is refused (ZhuRowOrderUnknown). Nothing is guessed.

Versioned option. `row_order=ROW_ORDER_AS_STORED` (the default) returns the arrays exactly as stored. That is the
registered read of the sealed v2 and v3 cards and of the D8 first-slice freeze; it stays the default so those
builds reproduce. `row_order=ROW_ORDER_GEOREF` returns `model` with rows aligned to an ascending `y`. A later card
(v3.1) and the D8 Zhu refreeze name ROW_ORDER_GEOREF in their registration.

Reads no label. Writes nothing.
"""
from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROW_ORDER_AS_STORED = "as-stored-v2"
ROW_ORDER_GEOREF = "georef-2026-11-23"
ROW_ORDERS = (ROW_ORDER_AS_STORED, ROW_ORDER_GEOREF)
DEFAULT_ROW_ORDER = ROW_ORDER_AS_STORED

# Known hdf5 writers (a token of the `history` attribute) and where they put row 0.
WRITER_ROW0 = {"MultiHazardGrid.save": "north"}
TWIN_MATCH_MIN = 0.99           # share of cells that must agree for a twin to decide the row order
TWIN_MATCH_TOL = 1e-4           # the tif is a float copy of the same values; the model is rounded to 4 decimals


class ZhuRowOrderUnknown(ValueError):
    """The grid's own metadata does not fix which row is north."""


class TwinContradicts(ValueError):
    """The GeoTIFF twin decodes but does not fit the grid (shape, tiepoint or cell match)."""


@dataclass(frozen=True)
class Orientation:
    row0: str                    # "north" or "south": where the stored row 0 lies
    source: str                  # "y-axis", "tif-twin" or "writer-convention"
    detail: dict = field(default_factory=dict)

    @property
    def reverse(self) -> bool:
        """True when the stored rows must be reversed to align with an ascending y."""
        return self.row0 == "north"


@dataclass(frozen=True, eq=False)
class ZhuGrid:
    x: np.ndarray
    y: np.ndarray
    model: np.ndarray
    sha256: str
    path: str
    row_order: str
    orientation: Orientation | None

    def as_dict(self) -> dict:
        """The {x, y, model, sha256, path} form world_inputs.run_zhu takes; row order named when not the default."""
        d = {"x": self.x, "y": self.y, "model": self.model, "sha256": self.sha256, "path": self.path}
        if self.row_order != DEFAULT_ROW_ORDER:
            d["row_order"] = self.row_order
            d["orientation"] = {"row0": self.orientation.row0, "source": self.orientation.source}
        return d


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def twin_path(path) -> Path:
    p = Path(path)
    return p.with_name(p.stem + "_model.tif")


# ---------------------------------------------------------------------------------------------------------------
# Minimal GeoTIFF reader: little-endian, uncompressed or deflate, strips or tiles, predictor 1, 2 or 3, float32 or
# float64. Enough for the USGS ground-failure twins (PIL cannot open the float64 cloud twins).
# ---------------------------------------------------------------------------------------------------------------
_TYPES = {1: "B", 2: "s", 3: "H", 4: "I", 5: "II", 7: "B", 11: "f", 12: "d", 16: "Q", 17: "b"}
_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 11: 4, 12: 8, 16: 8, 17: 1}


def read_geotiff(path) -> tuple[np.ndarray, dict]:
    """(data with nodata as NaN, georef {tie_x, tie_y, scale_x, scale_y, raster_type}). Raises ValueError on any
    layout it does not handle."""
    b = Path(path).read_bytes()
    if b[:4] != b"II*\x00":
        raise ValueError("only little-endian classic TIFF is handled")
    off = struct.unpack_from("<I", b, 4)[0]
    n = struct.unpack_from("<H", b, off)[0]
    tags = {}
    for k in range(n):
        tag, typ, cnt, val = struct.unpack_from("<HHII", b, off + 2 + 12 * k)
        if typ not in _SIZES:
            continue
        sz = _SIZES[typ] * cnt
        pos = off + 2 + 12 * k + 8 if sz <= 4 else val
        if typ == 2:
            tags[tag] = b[pos:pos + cnt].rstrip(b"\x00").decode("latin-1")
        else:
            fmt = "I" * (2 * cnt) if typ == 5 else _TYPES[typ] * cnt
            tags[tag] = struct.unpack_from("<" + fmt, b, pos)
    if 33922 not in tags or 33550 not in tags:
        raise ValueError("no ModelTiepoint or ModelPixelScale tag")
    W, H = tags[256][0], tags[257][0]
    bps = tags[258][0]
    fmt_ = tags.get(339, (3,))[0]
    if fmt_ != 3 or bps not in (32, 64):
        raise ValueError(f"sample format {fmt_} with {bps} bits is not handled")
    comp, pred = tags.get(259, (1,))[0], tags.get(317, (1,))[0]
    # one band only; with one band, planar configuration 2 is the same layout as 1
    if comp not in (1, 8, 32946) or pred not in (1, 2, 3) or tags.get(277, (1,))[0] != 1:
        raise ValueError("compression, predictor or band layout not handled")
    dt = np.dtype("<f8" if bps == 64 else "<f4")

    def dec(chunk, w, h):
        raw = zlib.decompress(chunk) if comp in (8, 32946) else chunk
        if pred == 3:
            a = np.frombuffer(raw, np.uint8)[:h * w * dt.itemsize].reshape(h, w * dt.itemsize)
            a = np.cumsum(a, axis=1, dtype=np.uint8)
            a = a.reshape(h, dt.itemsize, w).transpose(0, 2, 1)[:, :, ::-1]
            return np.ascontiguousarray(a).view(dt).reshape(h, w)
        if pred == 2:                                  # horizontal differencing of the sample words (libtiff)
            it = np.dtype("<u8" if bps == 64 else "<u4")
            a = np.frombuffer(raw, it)[:h * w].reshape(h, w)
            return np.cumsum(a, axis=1, dtype=it).view(dt)
        return np.frombuffer(raw, dt)[:h * w].reshape(h, w)

    out = np.empty((H, W), dt)
    if 322 in tags:
        tw, th = tags[322][0], tags[323][0]
        nx = (W + tw - 1) // tw
        for t, (o, c) in enumerate(zip(tags[324], tags[325])):
            r0, c0 = (t // nx) * th, (t % nx) * tw
            a = dec(b[o:o + c], tw, th)
            out[r0:r0 + th, c0:c0 + tw] = a[:min(th, H - r0), :min(tw, W - c0)]
    else:
        rps = tags.get(278, (H,))[0]
        for s, (o, c) in enumerate(zip(tags[273], tags[279])):
            r0 = s * rps
            h = min(rps, H - r0)
            out[r0:r0 + h] = dec(b[o:o + c], W, h)
    a = out.astype(float)
    nod = tags.get(42113)
    if nod:
        try:
            a[a == float(nod)] = np.nan
        except ValueError:
            pass
    tp, sc = tags[33922], tags[33550]
    if tp[0] != 0 or tp[1] != 0:
        raise ValueError("the tiepoint is not at raster (0, 0)")
    geokeys = tags.get(34735, ())
    raster_type = 1
    for i in range(4, len(geokeys), 4):
        if geokeys[i] == 1025:                       # GTRasterTypeGeoKey: 1 PixelIsArea, 2 PixelIsPoint
            raster_type = geokeys[i + 3]
    return a, {"tie_x": float(tp[3]), "tie_y": float(tp[4]), "scale_x": float(sc[0]), "scale_y": float(sc[1]),
               "raster_type": int(raster_type)}


def _match(a: np.ndarray, b: np.ndarray) -> float:
    fa, fb = np.isfinite(a), np.isfinite(b)
    same_nan = fa == fb
    both = fa & fb
    close = np.zeros_like(both)
    close[both] = np.abs(a[both] - b[both]) <= TWIN_MATCH_TOL
    return float(np.mean(same_nan & (close | ~both)))


def _orient_from_twin(y: np.ndarray, model: np.ndarray, tif) -> Orientation:
    t, g = read_geotiff(tif)
    if t.shape != model.shape:
        raise TwinContradicts(f"twin shape {t.shape} != model shape {model.shape}")
    dy = float(np.median(np.diff(y)))
    sy = g["scale_y"]
    # GeoTIFF convention: raster row r has latitude tie_y - r * scale_y (edge for PixelIsArea, centre for Point).
    c0 = g["tie_y"] - (0.5 * sy if g["raster_type"] == 1 else 0.0)
    if abs(c0 - float(y.max())) <= 0.5 * abs(dy):
        tif_row0 = "north"
    elif abs(c0 - float(y.min())) <= 0.5 * abs(dy):
        tif_row0 = "south"
    else:
        raise TwinContradicts(f"tif row 0 centre {c0} is at neither end of y ({y.min()}, {y.max()})")
    m = np.asarray(model, float)
    as_stored, reversed_ = _match(m, t), _match(m[::-1], t)
    detail = {"twin": str(Path(tif).name), "tif_row0": tif_row0, "tif_row0_centre_lat": round(c0, 8),
              "match_as_stored": round(as_stored, 4), "match_reversed": round(reversed_, 4)}
    if as_stored >= TWIN_MATCH_MIN and reversed_ < TWIN_MATCH_MIN:
        return Orientation(tif_row0, "tif-twin", detail)
    if reversed_ >= TWIN_MATCH_MIN and as_stored < TWIN_MATCH_MIN:
        return Orientation("south" if tif_row0 == "north" else "north", "tif-twin", detail)
    raise TwinContradicts(f"the twin does not decide the row order: {detail}")


def orient(f, path=None, use_twin: bool = True) -> Orientation:
    """Where the stored row 0 lies, from the grid's own metadata. `f` is an open h5py.File."""
    x, y = np.asarray(f["x"][()], float), np.asarray(f["y"][()], float)
    shape = f["model"].shape
    if shape != (y.size, x.size):
        raise ZhuRowOrderUnknown(f"model shape {shape} does not match the axes ({y.size}, {x.size})")
    dy = np.diff(y)
    if np.all(dy < 0):
        return Orientation("north", "y-axis", {"y": "decreasing"})
    if not np.all(dy > 0):
        raise ZhuRowOrderUnknown("y is not monotonic")
    notes = {}
    if use_twin and path is not None and twin_path(path).exists():
        try:
            return _orient_from_twin(y, f["model"][()], twin_path(path))
        except TwinContradicts as e:
            # A twin that decodes but does not fit the grid is a contradiction: refuse. The one exception: a twin
            # that matches both readings of a grid that reads the same both ways (the order changes no value).
            if not ("does not decide" in str(e) and _symmetric(f["model"][()])):
                raise ZhuRowOrderUnknown(f"{path}: {e}") from None
            notes["twin_not_used"] = str(e)[:300]
        except ValueError as e:
            # A twin that does not decode (a layout read_geotiff does not handle) is only noted.
            notes["twin_not_used"] = str(e)[:300]
    hist = f.attrs.get("history", "")
    hist = hist.decode() if isinstance(hist, bytes) else str(hist)
    for token, row0 in WRITER_ROW0.items():
        if token in hist:
            return Orientation(row0, "writer-convention", {"writer": token, **notes})
    raise ZhuRowOrderUnknown(f"{path}: y ascends and no twin or known writer fixes the row order ({notes or hist!r})")


def _symmetric(model) -> bool:
    m = np.asarray(model, float)
    return _match(m, m[::-1]) >= TWIN_MATCH_MIN


def load(path, row_order: str = DEFAULT_ROW_ORDER, use_twin: bool = True) -> ZhuGrid:
    """Read one Zhu 2017 grid. ROW_ORDER_AS_STORED: arrays exactly as stored (the v2, v3 and D8 first-slice read).
    ROW_ORDER_GEOREF: y ascending and model rows aligned to it, the row order taken from the grid's own metadata."""
    import h5py
    if row_order not in ROW_ORDERS:
        raise ValueError(f"row_order must be one of {ROW_ORDERS}, not {row_order!r}")
    p = Path(path)
    with h5py.File(p, "r") as f:
        x, y, model = f["x"][()], f["y"][()], f["model"][()]
        if row_order == ROW_ORDER_AS_STORED:
            return ZhuGrid(x, y, model, _sha(p), str(p), row_order, None)
        o = orient(f, p, use_twin=use_twin)
    if o.source == "y-axis":
        y, model = y[::-1], model[::-1]
    elif o.reverse:
        model = model[::-1]
    return ZhuGrid(np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(model), _sha(p), str(p),
                   row_order, o)

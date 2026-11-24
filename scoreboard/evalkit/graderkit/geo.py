"""NZTM2000 and WGS84 helpers and a minimal GeoTIFF reader built on PIL (no GDAL or rasterio on this Mac)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # the EQC DEM grids are large but trusted local files

_TF = {}


def _transformer(src: str, dst: str):
    key = (src, dst)
    if key not in _TF:
        from pyproj import Transformer
        _TF[key] = Transformer.from_crs(src, dst, always_xy=True)
    return _TF[key]


def nztm_to_lonlat(x, y):
    """NZTM2000 (EPSG:2193) easting, northing to WGS84 lon, lat."""
    return _transformer("EPSG:2193", "EPSG:4326").transform(x, y)


def lonlat_to_nztm(lon, lat):
    return _transformer("EPSG:4326", "EPSG:2193").transform(lon, lat)


@dataclass
class Grid:
    """A north-up raster: value[r, c] has its cell centre at (x0 + c * dx, y0 - r * dy)."""
    path: Path
    width: int
    height: int
    x0: float  # centre of the top-left cell
    y0: float
    dx: float
    dy: float
    nodata: float | None

    def window(self, x: float, y: float, half: float, arr: np.ndarray | None = None):
        """Cells whose centres lie within `half` metres of (x, y) in a square; NaN outside the grid or at nodata."""
        n = int(round(half / self.dx))
        cc = int(round((x - self.x0) / self.dx))
        rr = int(round((self.y0 - y) / self.dy))
        out = np.full((2 * n + 1, 2 * n + 1), np.nan, dtype=float)
        r0, r1, c0, c1 = rr - n, rr + n + 1, cc - n, cc + n + 1
        sr0, sr1 = max(r0, 0), min(r1, self.height)
        sc0, sc1 = max(c0, 0), min(c1, self.width)
        if sr0 >= sr1 or sc0 >= sc1:
            return out
        if arr is None:
            arr = read_array(self)
        block = arr[sr0:sr1, sc0:sc1].astype(float)
        if self.nodata is not None:
            block[np.isclose(block, self.nodata, rtol=1e-6) | (block < -1e30)] = np.nan
        out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = block
        return out

    def covers(self, x: float, y: float) -> bool:
        c = (x - self.x0) / self.dx
        r = (self.y0 - y) / self.dy
        return -0.5 <= c < self.width - 0.5 and -0.5 <= r < self.height - 0.5


def _raster_is_point(tags) -> bool:
    """GeoKey 1025 GTRasterTypeGeoKey: 1 = PixelIsArea (tie point is the cell corner), 2 = PixelIsPoint."""
    keys = tags.get(34735)
    if not keys:
        return False
    keys = list(keys)
    for i in range(4, len(keys), 4):
        if keys[i] == 1025:
            return keys[i + 3] == 2
    return False


def open_grid(path) -> Grid:
    path = Path(path)
    with Image.open(path) as im:
        t = im.tag_v2
        scale = t.get(33550)
        tie = t.get(33922)
        if not scale or not tie:
            raise ValueError(f"{path.name}: no GeoTIFF scale or tie point tags")
        dx, dy = float(scale[0]), float(scale[1])
        i, j, _, X, Y, _ = (float(v) for v in tie[:6])
        point = _raster_is_point(t)
        x0 = X - i * dx + (0.0 if point else dx / 2)
        y0 = Y + j * dy - (0.0 if point else dy / 2)
        nd = t.get(42113)
        nodata = float(str(nd).strip("\x00 ")) if nd not in (None, "") else None
        return Grid(path, im.size[0], im.size[1], x0, y0, dx, dy, nodata)


def read_array(grid: Grid) -> np.ndarray:
    with Image.open(grid.path) as im:
        return np.asarray(im, dtype=np.float32)


def write_geotiff(path, arr: np.ndarray, x0_centre: float, y0_centre: float, cell: float, nodata=-3.4028230607e38):
    """Write a float32 north-up GeoTIFF with PixelIsPoint geokeys (used by the synthetic demo and tests)."""
    from PIL import TiffImagePlugin
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[33550] = (float(cell), float(cell), 0.0)
    info[33922] = (0.0, 0.0, 0.0, float(x0_centre), float(y0_centre), 0.0)
    info.tagtype[34735] = 3
    info[34735] = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 2, 3072, 0, 1, 2193)
    info[42113] = str(nodata)
    Image.fromarray(arr.astype(np.float32), mode="F").save(path, tiffinfo=info)

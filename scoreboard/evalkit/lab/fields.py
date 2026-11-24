"""Gaussian random fields with exponential covariance exp(-3 h / rho) at site positions, by circulant embedding on a
regular grid (Dietrich and Newsam 1997); negative embedding eigenvalues are clipped to 0. Sites take the value of the
grid cell they fall in. Optional geometric anisotropy: the field is stretched by `aniso` along the bearing
`angle_deg` (degrees east of north). Unit variance. No em dashes.
"""
from __future__ import annotations

import numpy as np


class FieldSampler:
    """Draw unit-variance exponential fields of practical range rho_m at fixed sites. Precomputes the spectrum."""

    def __init__(self, x_m: np.ndarray, y_m: np.ndarray, rho_m: float, *, aniso: float = 1.0,
                 angle_deg: float = 45.0, max_cells: int = 1_200_000):
        x, y = np.asarray(x_m, float), np.asarray(y_m, float)
        x0, y0 = x.min(), y.min()
        span = max(x.max() - x0, y.max() - y0, 1.0)
        cell = float(np.clip(rho_m / 5.0, 50.0, 500.0))
        pad = max(rho_m * aniso, 1.0)
        nx = int(np.ceil((x.max() - x0 + pad) / cell)) + 1
        ny = int(np.ceil((y.max() - y0 + pad) / cell)) + 1
        while 4 * nx * ny > max_cells:              # keep the embedding cheap: coarsen the cell
            cell *= 1.25
            nx = int(np.ceil((x.max() - x0 + pad) / cell)) + 1
            ny = int(np.ceil((y.max() - y0 + pad) / cell)) + 1
        mx, my = 2 * nx, 2 * ny
        lx = np.minimum(np.arange(mx), mx - np.arange(mx)) * cell
        ly = np.minimum(np.arange(my), my - np.arange(my)) * cell
        # periodic lags with sign for anisotropy
        sx = np.where(np.arange(mx) <= mx // 2, 1.0, -1.0) * lx
        sy = np.where(np.arange(my) <= my // 2, 1.0, -1.0) * ly
        DX, DY = np.meshgrid(sx, sy, indexing="ij")
        if aniso != 1.0:
            th = np.deg2rad(angle_deg)
            u = DX * np.sin(th) + DY * np.cos(th)          # along the bearing
            v = DX * np.cos(th) - DY * np.sin(th)
            h = np.sqrt((u / aniso) ** 2 + v ** 2)
        else:
            h = np.hypot(DX, DY)
        c = np.exp(-3.0 * h / rho_m)
        lam = np.real(np.fft.fft2(c))
        self.clipped_share = float(-lam[lam < 0].sum() / np.abs(lam).sum())
        lam[lam < 0] = 0.0
        self._sq = np.sqrt(lam / (mx * my))
        self._ix = np.floor((x - x0) / cell).astype(np.int64)
        self._iy = np.floor((y - y0) / cell).astype(np.int64)
        self.cell_m, self.shape, self.rho_m = cell, (mx, my), rho_m
        self._spare: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the spare draw (call at the start of each world so a world depends on its own seed only)."""
        self._spare = None

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """One field at the sites (the imaginary part of each FFT is kept as the next draw)."""
        if self._spare is not None:
            out, self._spare = self._spare, None
            return out
        w = rng.standard_normal(self.shape) + 1j * rng.standard_normal(self.shape)
        f = np.fft.fft2(self._sq * w)
        self._spare = f.imag[self._ix, self._iy].copy()
        return f.real[self._ix, self._iy].copy()

"""G5 scoring helpers: pure numpy, read no data file.

Implements the scoring rules of docs/REGISTRATION-G5-FROZEN-MODEL-2026-10-31.md (492d033, amended d7e0482),
sections 1 to 4. The I/O lives in scripts/g5_score.py. No em dashes.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.special import expit, logit, ndtr

from .scorer import _draw, _members, auc

SEED = 20261031
N_BOOT = 2000
CLIP = (1e-6, 1 - 1e-6)
MU, SIG_R = 0.2, 0.2
MIN_ROWS, MIN_CLASS, MIN_TILES = 200, 30, 20
MIN_MW = 6.0
EDGE_M = 10.0
TILE_M = 1000.0


# ---------------------------------------------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------------------------------------------

def manifest_identity(M: dict) -> str:
    """sha256 over the manifest's canonical JSON, the identity field itself left out (registration section 5)."""
    body = {k: v for k, v in M.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Site part and M0
# ---------------------------------------------------------------------------------------------------------------

def p_level0(lnk: np.ndarray, sigma_e: np.ndarray) -> np.ndarray:
    """p0_i = Phi((ln k*_i - 0.2) / s_i), s_i^2 = 0.2^2 + sigma_e,i^2 (section 1.1). ln k* may be +/- inf."""
    s = np.sqrt(SIG_R**2 + np.asarray(sigma_e, float) ** 2)
    return ndtr((np.asarray(lnk, float) - MU) / s)


def p_m0(p0: np.ndarray, q: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """p_i = mean over the stored quantiles q of expit(logit(p0_i) + q), p0 clipped (section 1.3)."""
    lp = logit(np.clip(np.asarray(p0, float), *CLIP))
    q = np.asarray(q, float)
    out = np.empty(lp.size)
    for a in range(0, lp.size, chunk):
        out[a:a + chunk] = expit(lp[a:a + chunk, None] + q[None, :]).mean(axis=1)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Qualification (section 2), on counts alone
# ---------------------------------------------------------------------------------------------------------------

def qualification(y: np.ndarray, tiles: np.ndarray) -> dict:
    y = np.asarray(y, int)
    n_yes, n_no = int((y == 1).sum()), int((y == 0).sum())
    n_tiles = int(len(np.unique(np.asarray(tiles))))
    ok = n_yes + n_no >= MIN_ROWS and n_yes >= MIN_CLASS and n_no >= MIN_CLASS and n_tiles >= MIN_TILES
    return {"n": n_yes + n_no, "n_yes": n_yes, "n_no": n_no, "n_tiles": n_tiles, "passed": bool(ok),
            "rule": f">= {MIN_ROWS} labelled CPTs, >= {MIN_CLASS} per class, >= {MIN_TILES} occupied 1 km tiles"}


# ---------------------------------------------------------------------------------------------------------------
# Event word (section 3.1)
# ---------------------------------------------------------------------------------------------------------------

def _lg(x):
    return float(logit(np.clip(x, *CLIP)))


def offset(y, p) -> float:
    return _lg(np.mean(y)) - _lg(np.mean(p))


def band(q: np.ndarray, se_obs: float, lo: float = 0.05, hi: float = 0.95) -> tuple[float, float]:
    """5% and 95% points of u_pred + N(0, se_obs^2), u_pred given by its equal-weight quantiles q (exact mixture)."""
    q = np.asarray(q, float)
    if se_obs <= 0:
        return float(np.quantile(q, lo)), float(np.quantile(q, hi))

    def cdf(x):
        return float(ndtr((x - q) / se_obs).mean())

    def inv(level):
        a, b = q.min() - 10 * se_obs, q.max() + 10 * se_obs
        for _ in range(100):
            m = 0.5 * (a + b)
            if cdf(m) < level:
                a = m
            else:
                b = m
        return 0.5 * (a + b)

    return inv(lo), inv(hi)


def event_word(y, p0, tiles, q, *, n_boot=N_BOOT, seed=SEED) -> dict:
    y, p0 = np.asarray(y, float), np.asarray(p0, float)
    a_obs = offset(y, p0)
    mem = _members(tiles)
    rng = np.random.default_rng(seed)
    B = np.array([offset(y[i], p0[i]) for i in (_draw(mem, rng) for _ in range(n_boot))])
    se = float(B.std(ddof=1))
    lo, hi = band(q, se)
    return {"a_obs": a_obs, "a_obs_ci95": [float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))],
            "se_obs": se, "band90": [lo, hi], "word": "IN BAND" if lo <= a_obs <= hi else "OUT OF BAND"}


# ---------------------------------------------------------------------------------------------------------------
# Site word (section 3.2)
# ---------------------------------------------------------------------------------------------------------------

def log_loss(y, p) -> np.ndarray:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), *CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def site_word(y, p_m0_, p0, tiles, *, n_boot=N_BOOT, seed=SEED) -> dict:
    d = log_loss(y, p_m0_) - log_loss(y, p0)
    mem = _members(tiles)
    rng = np.random.default_rng(seed)
    B = np.array([d[i].mean() for i in (_draw(mem, rng) for _ in range(n_boot))])
    lo, hi = float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))
    word = "M0 BETTER" if hi < 0 else "M0 WORSE" if lo > 0 else "CANNOT TELL"
    return {"mean_logloss_m0": float(log_loss(y, p_m0_).mean()), "mean_logloss_l0": float(log_loss(y, p0).mean()),
            "diff": float(d.mean()), "diff_ci95": [lo, hi], "word": word}


# ---------------------------------------------------------------------------------------------------------------
# Printed beside (section 3.3)
# ---------------------------------------------------------------------------------------------------------------

def auc_interval(y, p, tiles, *, n_boot=N_BOOT, seed=SEED) -> dict:
    y = np.asarray(y, bool)
    mem = _members(tiles)
    rng = np.random.default_rng(seed)
    B = []
    for _ in range(n_boot):
        i = _draw(mem, rng)
        if y[i].all() or not y[i].any():
            continue
        B.append(auc(y[i], np.asarray(p)[i]))
    return {"auc": auc(y, p), "ci95": [float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))]}


def reliability(y, p, n_bins: int = 10) -> list[dict]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    k = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = k == b
        out.append({"bin": [b / n_bins, (b + 1) / n_bins], "n": int(m.sum()),
                    "mean_p": float(p[m].mean()) if m.any() else None,
                    "obs_rate": float(y[m].mean()) if m.any() else None})
    return out


# ---------------------------------------------------------------------------------------------------------------
# Transverse Mercator (Krueger series, as the bench's nztm) for UTM tiles and catalogue coordinates
# ---------------------------------------------------------------------------------------------------------------

_A, _F = 6378137.0, 1 / 298.257222101
_N = _F / (2 - _F)
_AA = _A / (1 + _N) * (1 + _N**2 / 4 + _N**4 / 64)
_ALPHA = (_N / 2 - 2 * _N**2 / 3 + 5 * _N**3 / 16, 13 * _N**2 / 48 - 3 * _N**3 / 5, 61 * _N**3 / 240)
_BETA = (_N / 2 - 2 * _N**2 / 3 + 37 * _N**3 / 96, _N**2 / 48 + _N**3 / 15, 17 * _N**3 / 480)
_DELTA = (2 * _N - 2 * _N**2 / 3 - 2 * _N**3, 7 * _N**2 / 3 - 8 * _N**3 / 5, 56 * _N**3 / 15)


def tm_forward(lat, lon, lon0, k0=0.9996, fe=500000.0, fn=0.0):
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float) - lon0)
    c = 2 * np.sqrt(_N) / (1 + _N)
    t = np.sinh(np.arctanh(np.sin(phi)) - c * np.arctanh(c * np.sin(phi)))
    xi = np.arctan2(t, np.cos(lam))
    eta = np.arctanh(np.sin(lam) / np.sqrt(1 + t**2))
    x = eta + sum(_ALPHA[j] * np.cos(2 * (j + 1) * xi) * np.sinh(2 * (j + 1) * eta) for j in range(3))
    y = xi + sum(_ALPHA[j] * np.sin(2 * (j + 1) * xi) * np.cosh(2 * (j + 1) * eta) for j in range(3))
    return fe + k0 * _AA * x, fn + k0 * _AA * y


def tm_inverse(x, y, lon0, k0=0.9996, fe=500000.0, fn=0.0):
    xi = (np.asarray(y, float) - fn) / (k0 * _AA)
    eta = (np.asarray(x, float) - fe) / (k0 * _AA)
    xp = xi - sum(_BETA[j] * np.sin(2 * (j + 1) * xi) * np.cosh(2 * (j + 1) * eta) for j in range(3))
    ep = eta - sum(_BETA[j] * np.cos(2 * (j + 1) * xi) * np.sinh(2 * (j + 1) * eta) for j in range(3))
    chi = np.arcsin(np.sin(xp) / np.cosh(ep))
    phi = chi + sum(_DELTA[j] * np.sin(2 * (j + 1) * chi) for j in range(3))
    lam = np.arctan2(np.sinh(ep), np.cos(xp))
    return np.degrees(phi), lon0 + np.degrees(lam)


def utm_zone(lon_mean: float) -> int:
    return int(np.floor((lon_mean + 180.0) / 6.0)) % 60 + 1


def utm_xy(lat, lon):
    """UTM coordinates in the zone of the mean longitude (false northing 1e7 if the mean latitude is south)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    z = utm_zone(float(np.mean(lon)))
    fn = 10000000.0 if float(np.mean(lat)) < 0 else 0.0
    return tm_forward(lat, lon, -183.0 + 6.0 * z, fn=fn)


def tiles_1km(x, y, size: float = TILE_M) -> np.ndarray:
    return np.array([f"{int(a // size)}_{int(b // size)}" for a, b in zip(x, y)], dtype=object)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

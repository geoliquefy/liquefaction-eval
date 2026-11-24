"""G5 entry by expected information (rule E-I): pure numpy, reads no data file.

Implements docs/AMENDMENT-G5-ENTRY-BY-INFORMATION-2026-11-06.md (22debbc), sections 1, 2, 3 (transfer) and 5.
Serves GOALS.md G5. The I/O, the GMPE table and the backtest live in scripts/g5_entry_backtest.py. No em dashes.

I = E|T|, T = sum_i [y_i log(pm_i / p0_i) + (1 - y_i) log((1 - pm_i) / (1 - p0_i))], the total log-score difference
of M0 against Level 0 on the predicted CPTs, labels drawn half under "Level 0 right" and half under "M0 right" (one
event shift u from the band's quantiles). It needs only p0 per CPT and the band's quantiles: the predict record.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import expit, logit
from scipy.stats import chi2

from . import g5

BAR = math.log(3.0)
N_SET = 500
SEED = 20261106
KILL_RATE = 2.0
KILL_RATIO = 0.25
MERGE_DAYS, MERGE_KM = 30.0, 50.0
MSFMAX = 1.3
LP_LO, LP_HI, LP_STEP = -14.0, 14.0, 0.002


# ---------------------------------------------------------------------------------------------------------------
# M0 site mixture by lookup (gate EG1 checks it against g5.p_m0)
# ---------------------------------------------------------------------------------------------------------------

class PmTable:
    def __init__(self, q, chunk: int = 500):
        q = np.asarray(q, float)
        self.grid = np.arange(LP_LO, LP_HI + LP_STEP / 2, LP_STEP)
        v = np.empty(self.grid.size)
        for a in range(0, self.grid.size, chunk):
            v[a:a + chunk] = expit(self.grid[a:a + chunk, None] + q[None, :]).mean(axis=1)
        self.val = v

    def __call__(self, p0) -> np.ndarray:
        lp = logit(np.clip(np.asarray(p0, float), *g5.CLIP))
        return np.interp(lp, self.grid, self.val)


def u_set(q, n_set: int = N_SET) -> np.ndarray:
    """The M0 shifts used for the 'M0 right' label sets: q at indices floor((k + 0.5) len(q) / n_set); for the
    10,000 stored quantiles this is q[10::20]."""
    q = np.asarray(q, float)
    idx = np.floor((np.arange(n_set) + 0.5) * q.size / n_set).astype(int)
    return q[idx]


def terms(p0, pm):
    p0c = np.clip(np.asarray(p0, float), *g5.CLIP)
    pmc = np.clip(np.asarray(pm, float), *g5.CLIP)
    a = np.log(pmc) - np.log(p0c)
    b = np.log1p(-pmc) - np.log1p(-p0c)
    return a, b


# ---------------------------------------------------------------------------------------------------------------
# Expected |T|
# ---------------------------------------------------------------------------------------------------------------

def expected_info(p0, q, seed: int, pm=None, n_set: int = N_SET, signed: bool = False) -> dict:
    """Monte Carlo I = E|T| (section 1). signed=True is injected defect ED2 (E[T], no absolute value)."""
    p0 = np.asarray(p0, float)
    if pm is None:
        pm = g5.p_m0(p0, q)
    a, b = terms(p0, pm)
    d, b0 = a - b, float(b.sum())
    lp0 = logit(np.clip(p0, *g5.CLIP))
    rng = np.random.default_rng(seed)
    if p0.size == 0:
        return {"I": 0.0, "mc_se": 0.0, "mean_T_L0": 0.0, "mean_T_M0": 0.0, "n_cpt": 0}
    YL = rng.random((n_set, p0.size)) < np.clip(p0, *g5.CLIP)[None, :]
    TL = YL @ d + b0
    u = u_set(q, n_set)
    YM = rng.random((n_set, p0.size)) < expit(lp0[None, :] + u[:, None])
    TM = YM @ d + b0
    f = (lambda x: x) if signed else np.abs
    I = 0.5 * (f(TL).mean() + f(TM).mean())
    se = 0.5 * math.sqrt(f(TL).var(ddof=1) / n_set + f(TM).var(ddof=1) / n_set)
    return {"I": float(I), "mc_se": float(se), "mean_T_L0": float(TL.mean()), "mean_T_M0": float(TM.mean()),
            "n_cpt": int(p0.size)}


def exact_info(p0, q) -> float:
    """Exact E|T| by enumeration of every label set (small N only; gate EG2)."""
    p0 = np.asarray(p0, float)
    N = p0.size
    if N > 16:
        raise ValueError("enumeration is for N <= 16")
    q = np.asarray(q, float)
    pm = g5.p_m0(p0, q)
    a, b = terms(p0, pm)
    Y = ((np.arange(2 ** N)[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    T = Y @ a + (1 - Y) @ b
    pc = np.clip(p0, *g5.CLIP)
    PL = np.exp(Y @ np.log(pc) + (1 - Y) @ np.log1p(-pc))
    S = expit(logit(pc)[None, :] + q[:, None])                       # (K, N)
    PM = np.exp(np.log(S) @ Y.T + np.log1p(-S) @ (1 - Y).T).mean(0)  # (2^N,)
    return float(0.5 * (PL @ np.abs(T)) + 0.5 * (PM @ np.abs(T)))


# ---------------------------------------------------------------------------------------------------------------
# Capacity transfer and geometry (section 3; experiment 6 forms)
# ---------------------------------------------------------------------------------------------------------------

def msf(m):
    return 1.0 + (MSFMAX - 1.0) * (8.64 * np.exp(-np.asarray(m, float) / 4.0) - 1.325)


def rupture_length_km(m):
    return 10 ** (-3.22 + 0.69 * np.asarray(m, float))


def rjb(epi_km, m):
    return np.maximum(0.0, np.asarray(epi_km, float) - rupture_length_km(m) / 2.0)


def transfer(lnk_t, pga_t, m_t, pga_new, m_new, drop_pga: bool = False) -> np.ndarray:
    """ln k*_i(e) = ln k*_i,t + ln(PGA_e,i / PGA_i,t) - ln(MSF(M_e) / MSF(M_t)). drop_pga is injected defect ED1."""
    lnk_t = np.asarray(lnk_t, float)
    dp = 0.0 if drop_pga else np.log(np.asarray(pga_new, float) / np.asarray(pga_t, float))
    out = lnk_t + dp - np.log(msf(m_new) / msf(m_t))
    return np.where(np.isinf(lnk_t), lnk_t, out)


# ---------------------------------------------------------------------------------------------------------------
# Merging, words, intervals (sections 2 and 5)
# ---------------------------------------------------------------------------------------------------------------

def merge_groups(t_days, lat, lon, days: float = MERGE_DAYS, km: float = MERGE_KM) -> np.ndarray:
    """Group id per admitted earthquake (input in any order). An earthquake within `days` after and `km` of any
    earlier one joins that one's group (chaining allowed)."""
    t = np.asarray(t_days, float)
    order = np.argsort(t, kind="stable")
    gid = np.full(t.size, -1, dtype=np.int64)
    ng = 0
    for k, i in enumerate(order):
        prev = order[:k]
        if prev.size:
            near = (t[i] - t[prev] <= days) & (g5.haversine_km(lat[i], lon[i], np.asarray(lat)[prev],
                                                                np.asarray(lon)[prev]) <= km)
            if near.any():
                gid[i] = gid[prev[near]].min()
                continue
        gid[i] = ng
        ng += 1
    return gid


def word(rate: float, ratio: float) -> dict:
    fails = []
    if not rate >= KILL_RATE:
        fails.append(f"rate {rate:.3f} < {KILL_RATE} a year")
    if not ratio >= KILL_RATIO:
        fails.append(f"median ratio {ratio:.3f} < {KILL_RATIO}")
    return {"word": "KILL" if fails else "PASS", "fails": fails}


def poisson_ci(k: int, years: float) -> list[float]:
    lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return [float(lo / years), float(hi / years)]


def year_bootstrap(years, ev_year, ev_I, grp_year, I_ref: float, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Resample the years with replacement. ev_*: admitted earthquakes (year, I); grp_year: year of each group."""
    years = np.asarray(years)
    ev_year, ev_I, grp_year = map(np.asarray, (ev_year, ev_I, grp_year))
    rng = np.random.default_rng(seed)
    ev_by = {y: ev_I[ev_year == y] for y in years}
    g_by = {y: int((grp_year == y).sum()) for y in years}
    R, M = [], []
    for _ in range(n_boot):
        ys = rng.choice(years, years.size, replace=True)
        R.append(sum(g_by[y] for y in ys) / years.size)
        v = np.concatenate([ev_by[y] for y in ys])
        M.append(float(np.median(v)) / I_ref if v.size else 0.0)
    R, M = np.array(R), np.array(M)
    return {"rate_ci95": [float(np.percentile(R, 2.5)), float(np.percentile(R, 97.5))],
            "median_ratio_ci95": [float(np.percentile(M, 2.5)), float(np.percentile(M, 97.5))],
            "n_boot": n_boot, "seed": seed}

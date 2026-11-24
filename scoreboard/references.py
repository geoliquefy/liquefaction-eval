"""G5 reference suite: reference forecasts, CSEP-style tests and the tile-paired comparator. Pure numpy, reads no file.

Implements docs/REGISTRATION-G5-REFERENCE-SUITE-2026-11-02.md (do-now experiment 4), sections 3, 4 and 8. Imports
scoreboard/g5.py and never edits it. The I/O lives in scripts/run_reference_suite.py. No em dashes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm

from . import g5
from .scorer import _draw, _members

CLIP = g5.CLIP
FLOOR = 0.05
PGA_EDGES = (0.0, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, np.inf)
PGA_MIN_N = 30
SIM_SEED = 20261102
N_SIM = 2000
ALPHA_TAIL = 0.025
LN2 = float(np.log(2.0))


# ---------------------------------------------------------------------------------------------------------------
# Reference forecasts (section 3)
# ---------------------------------------------------------------------------------------------------------------

def coin(n: int) -> np.ndarray:
    return np.full(n, 0.5)


def const_rate(y_earlier: list[np.ndarray]) -> float:
    """Pooled, CPT-weighted yes share over all labelled rows of the earlier earthquakes."""
    return float(np.concatenate([np.asarray(v, float) for v in y_earlier]).mean())


def const_equal_weight(y_earlier: list[np.ndarray]) -> float:
    """Beside: the mean of the earlier earthquakes' yes shares, each earthquake weighted once."""
    return float(np.mean([np.asarray(v, float).mean() for v in y_earlier]))


def pga_bin_rates(pga_earlier, y_earlier, edges=PGA_EDGES, min_n=PGA_MIN_N):
    """Per-bin (k + 0.5) / (n + 1) from the pooled earlier rows; a bin with n < min_n takes the pooled constant."""
    pga, y = np.concatenate([np.asarray(v, float) for v in pga_earlier]), np.concatenate(
        [np.asarray(v, float) for v in y_earlier])
    c = float(y.mean())
    k = np.digitize(pga, edges[1:-1])
    rates, counts = [], []
    for b in range(len(edges) - 1):
        m = k == b
        n = int(m.sum())
        counts.append(n)
        rates.append(float((y[m].sum() + 0.5) / (n + 1)) if n >= min_n else c)
    return np.array(rates), counts


def pga_bin_apply(rates, pga, edges=PGA_EDGES):
    return np.asarray(rates)[np.digitize(np.asarray(pga, float), edges[1:-1])]


def floored(p0, f: float = FLOOR) -> np.ndarray:
    return np.clip(np.asarray(p0, float), f, 1 - f)


def _x(p0):
    return logit(np.clip(np.asarray(p0, float), *CLIP))


def recal_fit(p0_earlier, y_earlier) -> tuple[float, float]:
    """a, b of expit(a + b logit(clip p0)) by maximum likelihood on the pooled earlier rows."""
    X = np.concatenate([_x(v) for v in p0_earlier])
    y = np.concatenate([np.asarray(v, float) for v in y_earlier])

    def nll(t):
        z = t[0] + t[1] * X
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    def grad(t):
        r = expit(t[0] + t[1] * X) - y
        return np.array([r.sum(), (r * X).sum()])

    res = minimize(nll, np.array([0.0, 0.1]), jac=grad, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000})
    return float(res.x[0]), float(res.x[1])


def recal_apply(ab, p0) -> np.ndarray:
    return expit(ab[0] + ab[1] * _x(p0))


def mixture(p0, q, f: float | None = None) -> np.ndarray:
    """mean over q of expit(logit(clip p0) + q); f gives the M1 tail floor on p0 first."""
    base = np.asarray(p0, float) if f is None else floored(p0, f)
    return g5.p_m0(base, np.asarray(q, float))


def point_shift(p0, shift: float) -> np.ndarray:
    return expit(_x(p0) + shift)


def mixture_quantiles(w, mu, var, n_q: int = 10000, n_grid: int = 200001) -> np.ndarray:
    """Equal-weight quantiles (levels (k + 0.5) / n_q) of the normal mixture sum_k w_k N(mu_k, var_k)."""
    w, mu, sd = np.asarray(w, float), np.asarray(mu, float), np.sqrt(np.asarray(var, float))
    keep = w > 1e-14
    w, mu, sd = w[keep] / w[keep].sum(), mu[keep], sd[keep]
    mean = float(np.sum(w * mu))
    x = np.linspace(mean - 60, mean + 60, n_grid)
    F = np.zeros_like(x)
    for wk, mk, sk in zip(w, mu, sd):
        F += wk * norm.cdf(x, mk, sk)
    return np.interp((np.arange(n_q) + 0.5) / n_q, F, x)


# ---------------------------------------------------------------------------------------------------------------
# Scores (section 4)
# ---------------------------------------------------------------------------------------------------------------

def loss(y, p) -> np.ndarray:
    return g5.log_loss(y, p)


def brier(y, p) -> np.ndarray:
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def summary(y, p) -> dict:
    ll = loss(y, p)
    return {"n": int(len(ll)), "logloss_per_site": float(ll.mean()), "joint_loglik": float(-ll.sum()),
            "brier": float(brier(y, p).mean()), "mean_p": float(np.mean(p)), "obs_share": float(np.mean(y)),
            "above_coin": bool(ll.mean() > LN2)}


def paired(d, tiles, *, n_boot: int = g5.N_BOOT, seed: int = g5.SEED) -> dict:
    """Mean of d (model minus reference, per site), 95% tile cluster bootstrap interval, SD, MDD = 2.8 SD."""
    d = np.asarray(d, float)
    mem = _members(tiles)
    rng = np.random.default_rng(seed)
    B = np.array([d[i].mean() for i in (_draw(mem, rng) for _ in range(n_boot))])
    lo, hi = float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))
    sd = float(B.std(ddof=1))
    return {"diff": float(d.mean()), "ci95": [lo, hi], "sd": sd, "mdd": 2.8 * sd,
            "word": "BEATS" if hi < 0 else "LOSES" if lo > 0 else "CANNOT TELL"}


def beats(y, p_model, p_ref, tiles, *, sign: float = 1.0, **kw) -> dict:
    """Paired log-loss comparator. The model beats the reference if the 95% interval of mean d lies below 0.

    sign exists only so the unit test can inject the defect (sign = -1 flips d); the suite always uses 1.
    """
    d = sign * (loss(y, p_model) - loss(y, p_ref))
    r = paired(d, tiles, **kw)
    r["beats"] = r["word"] == "BEATS"
    return r


def brier_paired(y, p_model, p_ref, tiles, **kw) -> dict:
    return paired(brier(y, p_model) - brier(y, p_ref), tiles, **kw)


# ---------------------------------------------------------------------------------------------------------------
# CSEP-style tests (section 4)
# ---------------------------------------------------------------------------------------------------------------

def _loglik(y, lp1, lp0):
    return y @ lp1 + (1 - y) @ lp0


def simulate(p_score, *, p0_mix=None, q=None, f=None, n_sim: int = N_SIM, seed: int = SIM_SEED, chunk: int = 100):
    """Simulated counts N and joint log likelihoods L (scored under p_score).

    Independent Bernoulli(p_score) unless p0_mix and q are given: then one shared offset u is drawn per set from q
    and sites are Bernoulli(expit(logit(clip p0_mix) + u)) (f applies the M1 floor first).
    """
    p = np.clip(np.asarray(p_score, float), *CLIP)
    lp1, lp0 = np.log(p), np.log1p(-p)
    rng = np.random.default_rng(seed)
    N, L = np.empty(n_sim), np.empty(n_sim)
    if p0_mix is not None:
        base = _x(np.asarray(p0_mix, float) if f is None else floored(p0_mix, f))
        q = np.asarray(q, float)
    for a in range(0, n_sim, chunk):
        b = min(a + chunk, n_sim)
        if p0_mix is None:
            ps = np.broadcast_to(p, (b - a, p.size))
        else:
            u = q[rng.integers(0, q.size, size=b - a)]
            ps = expit(base[None, :] + u[:, None])
        Y = (rng.random((b - a, p.size)) < ps).astype(float)
        N[a:b] = Y.sum(axis=1)
        L[a:b] = Y @ lp1 + (1 - Y) @ lp0
    return N, L


def n_test(n_obs: int, N_sim) -> dict:
    d1, d2 = float(np.mean(N_sim >= n_obs)), float(np.mean(N_sim <= n_obs))
    return {"n_obs": int(n_obs), "n_sim_mean": float(np.mean(N_sim)),
            "n_sim_95": [float(np.percentile(N_sim, 2.5)), float(np.percentile(N_sim, 97.5))],
            "delta1": d1, "delta2": d2, "passed": bool(min(d1, d2) >= ALPHA_TAIL)}


def l_test(L_obs: float, L_sim) -> dict:
    g = float(np.mean(L_sim <= L_obs + 1e-9 * max(1.0, abs(L_obs))))
    return {"L_obs": float(L_obs), "L_sim_95": [float(np.percentile(L_sim, 2.5)), float(np.percentile(L_sim, 97.5))],
            "gamma": g, "passed": bool(g >= ALPHA_TAIL)}


def conditional_bernoulli_loglik(p_sample, n_fix: int, p_score, *, n_sim: int = N_SIM, seed: int = SIM_SEED):
    """Joint log likelihoods (under p_score) of label sets drawn exactly from independent Bernoulli(p_sample)
    conditioned on sum = n_fix (backward recursion over the count, then sequential sampling)."""
    ps = np.clip(np.asarray(p_sample, float), *CLIP)
    pc = np.clip(np.asarray(p_score, float), *CLIP)
    l1s, l0s = np.log(ps), np.log1p(-ps)
    l1c, l0c = np.log(pc), np.log1p(-pc)
    n, K = ps.size, int(n_fix)
    if K < 0 or K > n:
        raise ValueError("n_fix out of range")
    # f[i, r] = log P(sum of y_i..y_{n-1} = r)
    f = np.full((n + 1, K + 1), -np.inf)
    f[n, 0] = 0.0
    for i in range(n - 1, -1, -1):
        nxt = f[i + 1]
        cur = l0s[i] + nxt
        cur[1:] = np.logaddexp(cur[1:], l1s[i] + nxt[:-1])
        f[i] = cur
    rng = np.random.default_rng(seed)
    r = np.full(n_sim, K)
    L = np.zeros(n_sim)
    for i in range(n):
        pos = r > 0
        prob1 = np.zeros(n_sim)
        prob1[pos] = np.exp(l1s[i] + f[i + 1, r[pos] - 1] - f[i, r[pos]])
        yi = rng.random(n_sim) < prob1
        L += np.where(yi, l1c[i], l0c[i])
        r -= yi.astype(int)
    if np.any(r != 0):
        raise RuntimeError("conditional sampler did not reach the fixed count")
    return L


def cl_test(L_obs: float, L_sim, degenerate: bool = False) -> dict:
    k = float(np.mean(L_sim <= L_obs + 1e-9 * max(1.0, abs(L_obs))))
    return {"L_obs": float(L_obs), "L_sim_95": [float(np.percentile(L_sim, 2.5)), float(np.percentile(L_sim, 97.5))],
            "kappa": k, "passed": bool(k >= ALPHA_TAIL), "degenerate": bool(degenerate)}


def joint_loglik(y, p) -> float:
    p = np.clip(np.asarray(p, float), *CLIP)
    return float(_loglik(np.asarray(y, float), np.log(p), np.log1p(-p)))

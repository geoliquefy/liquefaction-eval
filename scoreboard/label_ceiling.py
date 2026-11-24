"""G3+G4 run R1: the label-noise ceiling (founder ruling 39, 2026-11-07).

Registered by docs/REGISTRATION-R1-LABEL-CEILING-2026-11-07.md (5f60015), committed alone before this file.
Pure numpy and scipy; reads no data file. No em dashes.

Contents:
    ceiling(pi, se, sp)          AUC ceiling C = 0.5 (1 + PPV + NPV - 1) of the true state against a noisy label
    soft_auc, implied_auc_z      AUC_y = 0.5 + (AUC_z - 0.5) J_r and its inverse
    ll_floor(pi, se, sp)         expected log loss of P(y | z), nats per label
    em_fit(L, ev, ...)           Dawid and Skene (1979) by EM, unit weights, per-event prevalence, fixed raters
    wauc(y, s, w)                weighted Mann-Whitney AUC, ties as one half
    recal_logloss(y, x, w)       in-sample 2-parameter logistic recalibration, weighted mean log loss
    pair_auc_pred                DS-predicted AUC of one binary source scoring another
    twin_pairs, twin_stats       distinct-CPT pairs in a distance band and their agreement
    word, synth                  registered words; synthetic generator for gates G1, D1, D2, D3
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit

EPS = 1e-6
MARGIN_AUC = 0.05
MARGIN_LL = 0.05
MDD_UNDER = 0.10


# ---------------------------------------------------------------------------------------------------------------
# Analytic ceilings
# ---------------------------------------------------------------------------------------------------------------

def ceiling(pi, se, sp):
    """AUC of the true state z against label y (y independent of the predictor given z)."""
    pi, se, sp = (np.asarray(v, float) for v in (pi, se, sp))
    py1 = se * pi + (1 - sp) * (1 - pi)
    py0 = 1 - py1
    ppv = np.where(py1 > 0, se * pi / np.maximum(py1, 1e-300), np.nan)
    npv = np.where(py0 > 0, sp * (1 - pi) / np.maximum(py0, 1e-300), np.nan)
    jr = ppv + npv - 1
    return {"C": 0.5 * (1 + jr), "Jr": jr, "PPV": ppv, "NPV": npv, "q": py1}


def soft_auc(auc_z, jr):
    return 0.5 + (np.asarray(auc_z, float) - 0.5) * np.asarray(jr, float)


def implied_auc_z(auc_y, jr):
    return 0.5 + (np.asarray(auc_y, float) - 0.5) / np.asarray(jr, float)


def h2(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def ll_floor(pi, se, sp):
    pi = np.asarray(pi, float)
    return pi * h2(se) + (1 - pi) * h2(sp)


def pi_moments(q, se, sp):
    """Prevalence from an observed yes share q under one rater with known Se, Sp."""
    return (np.asarray(q, float) - (1 - sp)) / (se + sp - 1)


def pair_auc_pred(pi, se_p, sp_p, se_t, sp_t):
    """DS-predicted AUC of binary source p (predictor) scoring source t (target), both independent given z."""
    p11 = pi * se_p * se_t + (1 - pi) * (1 - sp_p) * (1 - sp_t)
    pt1 = pi * se_t + (1 - pi) * (1 - sp_t)
    p10 = pi * se_p * (1 - se_t) + (1 - pi) * (1 - sp_p) * sp_t
    tpr, fpr = p11 / pt1, p10 / (1 - pt1)
    return 0.5 * (1 + tpr - fpr)


# ---------------------------------------------------------------------------------------------------------------
# Dawid and Skene by EM
# ---------------------------------------------------------------------------------------------------------------

def em_fit(L, ev=None, w=None, fixed=None, init=None, tol=1e-8, max_iter=2000, n_events=None):
    """L: (U, P) int matrix, 1 yes, 0 no, -1 missing; one column per protocol (shared across events if the same
    column). ev: event index per unit (default all 0). w: unit weights (default 1). fixed: {col: (se, sp)} held
    fixed. init: (se, sp, pi) start. Returns dict se, sp (P,), pi (E,), post (U,), loglik, n_iter, converged,
    ok (Se + Sp > 1 for every free column)."""
    L = np.asarray(L)
    U, P = L.shape
    ev = np.zeros(U, int) if ev is None else np.asarray(ev, int)
    E = int(ev.max()) + 1 if n_events is None else n_events
    w = np.ones(U) if w is None else np.asarray(w, float)
    fixed = fixed or {}
    obs = L >= 0
    y1 = (L == 1) & obs
    y0 = (L == 0) & obs
    if init is None:
        se, sp, pi = np.full(P, 0.8), np.full(P, 0.8), np.full(E, 0.5)
    else:
        se, sp, pi = (np.array(v, float).copy() for v in init)
    for c, (a, b) in fixed.items():
        se[c], sp[c] = a, b
    free = np.array([c not in fixed for c in range(P)])
    ll_old = -np.inf
    conv = False
    we = np.bincount(ev, w, minlength=E)
    for it in range(1, max_iter + 1):
        lse, l1se = np.log(se), np.log1p(-se)
        lsp, l1sp = np.log(sp), np.log1p(-sp)
        a1 = np.log(pi)[ev] + y1 @ lse + y0 @ l1se          # log P(z=1, y)
        a0 = np.log1p(-pi)[ev] + y1 @ l1sp + y0 @ lsp        # log P(z=0, y)
        m = np.maximum(a1, a0)
        lse_ = m + np.log(np.exp(a1 - m) + np.exp(a0 - m))
        ll = float((w * lse_).sum())
        post = np.exp(a1 - lse_)
        # M step
        wp, wn = w * post, w * (1 - post)
        pi = np.clip(np.bincount(ev, wp, minlength=E) / np.maximum(we, 1e-300), EPS, 1 - EPS)
        se_n = (wp @ y1) / np.maximum(wp @ obs, 1e-300)
        sp_n = (wn @ y0) / np.maximum(wn @ obs, 1e-300)
        se = np.where(free, np.clip(se_n, EPS, 1 - EPS), se)
        sp = np.where(free, np.clip(sp_n, EPS, 1 - EPS), sp)
        if abs(ll - ll_old) < tol * max(1.0, abs(ll)):
            conv = True
            break
        ll_old = ll
    # final posterior at the final parameters
    a1 = np.log(pi)[ev] + y1 @ np.log(se) + y0 @ np.log1p(-se)
    a0 = np.log1p(-pi)[ev] + y1 @ np.log1p(-sp) + y0 @ np.log(sp)
    post = expit(a1 - a0)
    ok = bool(np.all((se + sp)[free] > 1))
    return {"se": se, "sp": sp, "pi": pi, "post": post, "loglik": ll, "n_iter": it, "converged": conv, "ok": ok}


def posterior_excluding(L, ev, fit, col):
    """P(z = 1 | every label except column col), with the fit's parameters."""
    L = np.asarray(L).copy()
    L[:, col] = -1
    se, sp, pi = fit["se"], fit["sp"], fit["pi"]
    y1, y0 = L == 1, L == 0
    a1 = np.log(pi)[ev] + y1 @ np.log(se) + y0 @ np.log1p(-se)
    a0 = np.log1p(-pi)[ev] + y1 @ np.log1p(-sp) + y0 @ np.log(sp)
    return expit(a1 - a0)


# ---------------------------------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------------------------------

def wauc(y, s, w=None):
    """Weighted Mann-Whitney AUC, ties one half. Equals scorer.auc when w is all ones."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    o = np.argsort(s, kind="mergesort")
    s, y, w = s[o], y[o], w[o]
    wp, wn = np.where(y, w, 0.0), np.where(y, 0.0, w)
    # group ties
    brk = np.flatnonzero(np.diff(s)) + 1
    gp = np.add.reduceat(wp, np.r_[0, brk]) if len(s) else np.zeros(0)
    gn = np.add.reduceat(wn, np.r_[0, brk]) if len(s) else np.zeros(0)
    below = np.cumsum(gn) - gn
    num = float((gp * (below + 0.5 * gn)).sum())
    den = float(gp.sum() * gn.sum())
    return num / den if den > 0 else float("nan")


class SortedAUC:
    """AUC of a fixed (y, s) under many weight vectors: the sort and tie groups are computed once."""

    def __init__(self, y, s):
        y = np.asarray(y, bool)
        s = np.asarray(s, float)
        self.o = np.argsort(s, kind="mergesort")
        ss = s[self.o]
        self.y = y[self.o]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1] if len(ss) else np.zeros(0, int)

    def __call__(self, w):
        w = np.asarray(w, float)[self.o]
        gp = np.add.reduceat(np.where(self.y, w, 0.0), self.starts)
        gn = np.add.reduceat(np.where(self.y, 0.0, w), self.starts)
        below = np.cumsum(gn) - gn
        den = gp.sum() * gn.sum()
        return float((gp * (below + 0.5 * gn)).sum() / den) if den > 0 else float("nan")


def log_loss(y, p, w=None):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ll = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return float((w * ll).sum() / w.sum())


def recal_logloss(y, x, w=None, n_iter=25):
    """Weighted logistic regression of y on (1, x) by Newton; returns the weighted mean log loss in sample."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    X = np.column_stack([np.ones_like(x), (x - x.mean()) / (x.std() + 1e-12)])
    ybar = float((w * y).sum() / w.sum())
    b = np.array([np.log(ybar / (1 - ybar)) if 0 < ybar < 1 else 0.0, 0.0])
    for _ in range(n_iter):
        p = expit(X @ b)
        g = X.T @ (w * (y - p))
        H = (X * (w * p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    return log_loss(y, expit(X @ b), w)


def logit_clip(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def norm_rank(s):
    _, inv, cnt = np.unique(np.asarray(s, float), return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv.ravel()] / len(s)


# ---------------------------------------------------------------------------------------------------------------
# Twins
# ---------------------------------------------------------------------------------------------------------------

def twin_pairs(x, y, rmin=0.5, rmax=10.0):
    """Index pairs (i < j) of distinct rows with separation in (rmin, rmax]; and pairs at rmin or less."""
    xy = np.column_stack([x, y])
    P = cKDTree(xy).query_pairs(rmax, output_type="ndarray")
    if len(P) == 0:
        return np.zeros((0, 2), int), 0
    d = np.hypot(*(xy[P[:, 0]] - xy[P[:, 1]]).T)
    return P[d > rmin], int((d <= rmin).sum())


def twin_stats(lab, P, w=None):
    """Agreement, kappa and twin AUC over ordered pairs. w: per-row weights (pair weight = w_i w_j, for the
    tile bootstrap). Returns dict or None if no pair."""
    if len(P) == 0:
        return None
    a, b = lab[P[:, 0]].astype(float), lab[P[:, 1]].astype(float)
    pw = np.ones(len(P)) if w is None else w[P[:, 0]] * w[P[:, 1]]
    if pw.sum() <= 0:
        return None
    # ordered pairs: (a -> b) and (b -> a)
    t = np.r_[a, b]
    s = np.r_[b, a]
    ww = np.r_[pw, pw]
    agree = float((ww * (t == s)).sum() / ww.sum())
    p1 = float((ww * t).sum() / ww.sum())
    pe = p1 ** 2 + (1 - p1) ** 2
    kappa = (agree - pe) / (1 - pe) if pe < 1 else float("nan")
    n1, n0 = (ww * t).sum(), (ww * (1 - t)).sum()
    tpr = (ww * t * s).sum() / n1 if n1 > 0 else np.nan
    fpr = (ww * (1 - t) * s).sum() / n0 if n0 > 0 else np.nan
    return {"agree": agree, "kappa": float(kappa), "auc": float(0.5 * (1 + tpr - fpr)), "n_pairs": int(len(P))}


# ---------------------------------------------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------------------------------------------

def pct(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if len(v) else [float("nan")] * 2


def word(g_draws, margin=MARGIN_AUC, prefix=""):
    lo, hi = pct(g_draws)
    sd = float(np.nanstd(g_draws, ddof=1))
    w = "CEILING BINDS" if hi <= margin else ("ROOM LEFT" if lo > margin else "CANNOT TELL")
    flags = []
    if hi < 0:
        flags.append("ABOVE CEILING")
    if 2.8 * sd > MDD_UNDER and not prefix:
        flags.append("UNDERPOWERED")
    return {"word": (prefix + w), "ci95": [lo, hi], "mdd": 2.8 * sd, "flags": flags}


def common_word(words):
    """Jun and Dec: the word common to every admissible scenario, else CANNOT TELL."""
    ws = set(words)
    return ws.pop() if len(ws) == 1 else "CANNOT TELL"


# ---------------------------------------------------------------------------------------------------------------
# Synthetic data for the gates
# ---------------------------------------------------------------------------------------------------------------

def synth(n_units, n_tiles, pi, se, sp, cover, rng, tile_sd=1.0, dep_share=0.0):
    """One synthetic earthquake. se, sp, cover: per column. A tile random effect (SD tile_sd) on the logit of the
    truth keeps the marginal prevalence near pi. dep_share: share of units at which every rater copies one shared
    fair-coin draw, independent of z (defect D1: a shared error mode; the truth is unchanged). Returns L, z, tile."""
    tile = rng.integers(0, n_tiles, n_units)
    eff = rng.normal(0, tile_sd, n_tiles)
    lp = np.log(pi / (1 - pi)) + eff[tile]
    z = (rng.random(n_units) < expit(lp)).astype(int)
    P = len(se)
    se, sp, cover = map(np.asarray, (se, sp, cover))
    p_yes = np.where(z[:, None] == 1, se[None, :], 1 - sp[None, :])
    L = (rng.random((n_units, P)) < p_yes).astype(int)
    if dep_share > 0:
        dep = rng.random(n_units) < dep_share
        shared = (rng.random(n_units) < 0.5).astype(int)   # a coin, shared by all raters
        L[dep] = shared[dep][:, None]
    obs = rng.random((n_units, P)) < cover[None, :]
    L = np.where(obs, L, -1)
    keep = obs.any(axis=1)
    return L[keep], z[keep], tile[keep]

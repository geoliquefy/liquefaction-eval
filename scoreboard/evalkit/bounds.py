"""Exact worst-case bounds over the labels of ungraded audit units (AUDIT-DESIGN section 5, item 4; review 2 N2).

No em dashes. Tier 1: this module computes no card score; it is the solver the verdict rule will call once audit
grades exist.

Setting. An earthquake's audit has n units with inclusion weights w_i > 0. Graded units carry a label 1 (YES) or
0 (NO); the K ungraded units (CANNOT TELL and CANNOT GRADE together) carry None / NaN. Each bound is the minimum and
maximum of a statistic's point value over all 2^K YES/NO assignments of the ungraded units, graded labels fixed.

Weighted AUC and paired AUC difference. Both are Num(y) / Den(y):
    Num(y) = sum over i != j of w_i w_j y_i (1 - y_j) h_ij
    Den(y) = sum(w_i y_i) x sum(w_j (1 - y_j)) = sum over i != j of w_i w_j y_i (1 - y_j)   (y binary, so i = j adds 0)
with h_ij = 1 if s_i > s_j, 1/2 if tied, 0 if below (AUC), or h_ij = hA_ij - hB_ij (paired difference of models A
and B). Den(y) > 0 needs at least one YES and at least one NO among graded plus assigned units (review 2 N2): the
solver adds that constraint and refuses a case where no assignment satisfies it.

Dinkelbach iteration: for a given lambda, F(lambda) = max over feasible binary y of Num(y) - lambda Den(y)
           = const + sum_u L_u y_u + sum_{u<v} Q_uv y_u y_v     (u, v ungraded),
a quadratic binary program. The products y_u y_v become z_uv with the exact constraints z <= y_u, z <= y_v,
z >= y_u + y_v - 1, 0 <= z <= 1, and the binary linear program is solved by HiGHS (scipy.optimize.milp, relative
gap 0). Then lambda <- Num(y*) / Den(y*), until F(lambda) is zero to a relative 1e-12. The minimum is minus the
maximum with h replaced by -h. For K <= ENUM_MAX the full enumeration also runs and both must agree to 1e-12.

Log loss and Brier score (and their paired differences). The weighted mean sum(w_i l_i(y_i)) / sum(w_i) has a
denominator free of y, so the bound is exact unit by unit: each ungraded unit takes the label that makes its term
largest (upper bound), then smallest (lower bound). No class constraint applies to these statistics.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ENUM_MAX = 16
AGREE_TOL = 1e-12
_DINKELBACH_MAX = 100


class BoundError(ValueError):
    """The case has no assignment with a defined statistic, or the inputs are malformed."""


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------

def _labels(y: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Graded labels as floats (NaN for ungraded) and the boolean mask of ungraded units."""
    arr = np.array([np.nan if v is None else float(v) for v in y], dtype=float)
    ung = np.isnan(arr)
    if not np.all(np.isin(arr[~ung], (0.0, 1.0))):
        raise BoundError("graded labels must be 0 or 1; ungraded units are None or NaN")
    return arr, ung


def _weights(w: Sequence | None, n: int) -> np.ndarray:
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or not np.all(w > 0):
        raise BoundError("weights must be finite, positive and one per unit")
    return w


def concordance(s: Sequence[float]) -> np.ndarray:
    """h_ij = 1 if s_i > s_j, 1/2 if tied, 0 if below (diagonal 1/2; it never enters, since y_i (1 - y_i) = 0)."""
    s = np.asarray(s, dtype=float)
    return (s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :])


def weighted_auc(y: Sequence, s: Sequence[float], w: Sequence | None = None) -> float:
    """Weighted AUC Num / Den for fully graded labels; NaN if one class is empty. Equals scorer.auc when w = 1."""
    return ratio(concordance(s), np.asarray(y, dtype=float), _weights(w, len(y)))


def ratio(h: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Num(y) / Den(y) for one full binary labelling (module docstring); NaN if Den is zero."""
    a = w * y
    b = w * (1.0 - y)
    den = float(a.sum() * b.sum() - np.sum(a * b))          # i = j terms are zero for binary y; kept exact
    if den <= 0:
        return float("nan")
    num = float(a @ h @ b - np.sum(a * b * np.diag(h)))
    return num / den


def _feasible(arr: np.ndarray, ung: np.ndarray) -> tuple[bool, bool]:
    """(needs one assigned YES, needs one assigned NO), and refuse a case with no feasible assignment."""
    graded = arr[~ung]
    need_yes, need_no = not np.any(graded == 1.0), not np.any(graded == 0.0)
    k = int(ung.sum())
    if (need_yes and need_no and k < 2) or ((need_yes or need_no) and k < 1):
        raise BoundError("no assignment has at least one YES and one NO: the AUC is undefined for every labelling")
    return need_yes, need_no


# ---------------------------------------------------------------------------------------------------------------
# Enumeration (the check)
# ---------------------------------------------------------------------------------------------------------------

def enumerate_ratio_bounds(h: np.ndarray, y: Sequence, w: Sequence | None = None) -> dict:
    """Minimum and maximum of Num / Den over every assignment of the ungraded units (K <= ENUM_MAX)."""
    arr, ung = _labels(y)
    w = _weights(w, arr.size)
    _feasible(arr, ung)
    idx = np.flatnonzero(ung)
    if idx.size > ENUM_MAX:
        raise BoundError(f"enumeration runs only for K <= {ENUM_MAX}; K = {idx.size}")
    lo, hi = np.inf, -np.inf
    full = arr.copy()
    for bits in itertools.product((0.0, 1.0), repeat=idx.size):
        full[idx] = bits
        r = ratio(h, full, w)
        if np.isnan(r):
            continue
        lo, hi = min(lo, r), max(hi, r)
    return {"lo": float(lo), "hi": float(hi), "k": int(idx.size), "method": "enumeration"}


# ---------------------------------------------------------------------------------------------------------------
# Dinkelbach with an exact binary linear program
# ---------------------------------------------------------------------------------------------------------------

def _qbp(c: np.ndarray, arr: np.ndarray, ung: np.ndarray, need_yes: bool, need_no: bool) -> np.ndarray:
    """argmax over feasible binary y_U of sum_{i != j} c_ij y_i (1 - y_j), graded y fixed. Returns the full y."""
    idx = np.flatnonzero(ung)
    gr = np.flatnonzero(~ung)
    k = idx.size
    yg = arr[gr]
    cuu = c[np.ix_(idx, idx)].copy()
    np.fill_diagonal(cuu, 0.0)
    # linear part: c_uv y_u (u != v ungraded), c_ug y_u (1 - y_g), and -c_gu y_g y_u
    lin = cuu.sum(axis=1) + c[np.ix_(idx, gr)] @ (1.0 - yg) - c[np.ix_(gr, idx)].T @ yg
    iu, iv = np.triu_indices(k, 1)
    quad = -(cuu[iu, iv] + cuu[iv, iu])
    m = iu.size
    # variables: y_U (k), z (m); milp minimises, so negate
    obj = -np.concatenate([lin, quad])
    rows, lo_b, hi_b = [], [], []
    for t in range(m):
        r1 = np.zeros(k + m); r1[k + t] = 1.0; r1[iu[t]] = -1.0          # z - y_u <= 0
        r2 = np.zeros(k + m); r2[k + t] = 1.0; r2[iv[t]] = -1.0          # z - y_v <= 0
        r3 = np.zeros(k + m); r3[k + t] = 1.0; r3[iu[t]] = -1.0; r3[iv[t]] = -1.0   # z - y_u - y_v >= -1
        rows += [r1, r2, r3]
        lo_b += [-np.inf, -np.inf, -1.0]
        hi_b += [0.0, 0.0, np.inf]
    if need_yes or need_no:
        r = np.zeros(k + m); r[:k] = 1.0
        rows.append(r)
        lo_b.append(1.0 if need_yes else -np.inf)
        hi_b.append(k - 1.0 if need_no else np.inf)
    cons = [LinearConstraint(np.array(rows), lo_b, hi_b)] if rows else []
    integrality = np.concatenate([np.ones(k), np.zeros(m)])
    res = milp(obj, constraints=cons, integrality=integrality, bounds=Bounds(0.0, 1.0),
               options={"mip_rel_gap": 0.0, "presolve": True})
    if res.status != 0 or res.x is None:
        raise BoundError(f"HiGHS did not solve the binary program: {res.message}")
    full = arr.copy()
    full[idx] = np.round(res.x[:k])
    return full


def _max_ratio(h: np.ndarray, arr: np.ndarray, ung: np.ndarray, w: np.ndarray) -> tuple[float, int]:
    need_yes, need_no = _feasible(arr, ung)
    ww = np.outer(w, w)
    # start from any feasible labelling: ungraded set to NO, then fix the class constraint
    y0 = arr.copy()
    y0[ung] = 0.0
    if need_yes:
        y0[np.flatnonzero(ung)[0]] = 1.0
    if not np.any(y0 == 0.0):
        y0[np.flatnonzero(ung)[-1]] = 0.0
    lam = ratio(h, y0, w)
    for it in range(1, _DINKELBACH_MAX + 1):
        y = _qbp(ww * (h - lam), arr, ung, need_yes, need_no)
        a, b = w * y, w * (1.0 - y)
        f = float(a @ (h - lam) @ b - np.sum(a * b * (np.diag(h) - lam)))
        r = ratio(h, y, w)
        if f <= AGREE_TOL * max(1.0, abs(lam)) * float(a.sum() * b.sum()) or r <= lam:
            return max(lam, r), it
        lam = r
    raise BoundError("Dinkelbach did not converge")


def ratio_bounds(h: np.ndarray, y: Sequence, w: Sequence | None = None, *, check: bool = True) -> dict:
    """Exact minimum and maximum of Num / Den over feasible assignments (Dinkelbach + HiGHS); with K <= ENUM_MAX
    and `check`, the enumeration runs too and must agree to AGREE_TOL."""
    arr, ung = _labels(y)
    w = _weights(w, arr.size)
    h = np.asarray(h, dtype=float)
    if h.shape != (arr.size, arr.size):
        raise BoundError("h must be n x n")
    if not ung.any():
        r = ratio(h, arr, w)
        if np.isnan(r):
            raise BoundError("no assignment has at least one YES and one NO: the AUC is undefined")
        return {"lo": r, "hi": r, "k": 0, "method": "graded only", "iterations": [0, 0]}
    hi, it_hi = _max_ratio(h, arr, ung, w)
    neg_lo, it_lo = _max_ratio(-h, arr, ung, w)
    out = {"lo": -neg_lo, "hi": hi, "k": int(ung.sum()), "method": "dinkelbach-highs", "iterations": [it_lo, it_hi]}
    if check and out["k"] <= ENUM_MAX:
        e = enumerate_ratio_bounds(h, y, w)
        if abs(e["lo"] - out["lo"]) > AGREE_TOL or abs(e["hi"] - out["hi"]) > AGREE_TOL:
            raise BoundError(f"solver and enumeration disagree: {out} vs {e}")
        out["enumeration_agrees"] = True
    return out


def auc_bounds(y: Sequence, s: Sequence[float], w: Sequence | None = None, *, check: bool = True) -> dict:
    """Exact worst-case range of the weighted AUC of score s."""
    return ratio_bounds(concordance(s), y, w, check=check)


def auc_diff_bounds(y: Sequence, s_a: Sequence[float], s_b: Sequence[float], w: Sequence | None = None, *,
                    check: bool = True) -> dict:
    """Exact worst-case range of the weighted paired AUC difference A - B (shared denominator)."""
    return ratio_bounds(concordance(s_a) - concordance(s_b), y, w, check=check)


# ---------------------------------------------------------------------------------------------------------------
# Unit-by-unit bounds: log loss and Brier
# ---------------------------------------------------------------------------------------------------------------

def _unit_loss(kind: str, p: np.ndarray, clip: float) -> tuple[np.ndarray, np.ndarray]:
    """(loss if YES, loss if NO) per unit."""
    if kind == "brier":
        return (1.0 - p) ** 2, p ** 2
    if kind == "log_loss":
        pc = np.clip(p, clip, 1.0 - clip)
        return -np.log(pc), -np.log1p(-pc)
    raise BoundError(f"kind must be 'brier' or 'log_loss', got {kind!r}")


def loss_bounds(kind: str, y: Sequence, p_a: Sequence[float], p_b: Sequence[float] | None = None,
                w: Sequence | None = None, *, clip: float = 1e-6) -> dict:
    """Exact range of the weighted mean loss of p_a (or of the paired difference p_a - p_b) over assignments."""
    arr, ung = _labels(y)
    w = _weights(w, arr.size)
    l1, l0 = _unit_loss(kind, np.asarray(p_a, dtype=float), clip)
    if p_b is not None:
        m1, m0 = _unit_loss(kind, np.asarray(p_b, dtype=float), clip)
        l1, l0 = l1 - m1, l0 - m0
    g = ~ung
    fixed = np.where(arr[g] == 1.0, l1[g], l0[g])
    base = float(np.sum(w[g] * fixed))
    hi = base + float(np.sum(w[ung] * np.maximum(l1[ung], l0[ung])))
    lo = base + float(np.sum(w[ung] * np.minimum(l1[ung], l0[ung])))
    tot = float(w.sum())
    return {"lo": lo / tot, "hi": hi / tot, "k": int(ung.sum()), "method": "unit by unit"}


def enumerate_loss_bounds(kind: str, y: Sequence, p_a: Sequence[float], p_b: Sequence[float] | None = None,
                          w: Sequence | None = None, *, clip: float = 1e-6) -> dict:
    """The same range by full enumeration (K <= ENUM_MAX): the check for loss_bounds."""
    arr, ung = _labels(y)
    w = _weights(w, arr.size)
    idx = np.flatnonzero(ung)
    if idx.size > ENUM_MAX:
        raise BoundError(f"enumeration runs only for K <= {ENUM_MAX}")
    l1, l0 = _unit_loss(kind, np.asarray(p_a, dtype=float), clip)
    if p_b is not None:
        m1, m0 = _unit_loss(kind, np.asarray(p_b, dtype=float), clip)
        l1, l0 = l1 - m1, l0 - m0
    vals = []
    full = arr.copy()
    for bits in itertools.product((0.0, 1.0), repeat=idx.size):
        full[idx] = bits
        vals.append(float(np.sum(w * np.where(full == 1.0, l1, l0)) / w.sum()))
    return {"lo": min(vals), "hi": max(vals), "k": int(idx.size), "method": "enumeration"}

"""G1 scorer: score liquefaction methods one earthquake at a time.

Serves GOALS.md G1 (the instrument) and G2 (event offsets). Pure numpy. It reads no
data file; callers pass arrays. The unit of resampling is the site cluster inside
one earthquake, because neighbouring holes share their outcome (residual range
250 to 600 m in Canterbury, docs/SCALING-RESEARCH-2026-10-29.md).

Verdict words, per earthquake and method against the reference cell:
  SEPARATES     the paired AUC difference is significant after the family correction
                (Holm, applied by `apply_holm` across every cell of a registration)
  EQUIVALENT    the 90% paired interval lies inside +/- margin (two one-sided tests);
                "no difference found" alone never earns this word
  UNDERPOWERED  neither, or the earthquake has fewer clusters than `min_clusters`

`min_clusters` must come from the null calibration (scripts/null_calibration.py): the
percentile cluster bootstrap is too narrow when clusters are few. Below `min_clusters`,
`small_test=True` replaces the bootstrap p-value with a cluster sign-flip permutation test
(`sign_flip_p`). That test can say SEPARATES but never EQUIVALENT: it gives no interval.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from itertools import product

import numpy as np

Z_ALPHA = 1.959963984540054  # two-sided 0.05
Z_POWER = 0.8416212335729143  # power 0.80


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as one half. NaN if one class is empty."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks over ties, vectorised
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    ranks = avg[inv.ravel()]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _norm_rank(s: np.ndarray) -> np.ndarray:
    """Average ranks scaled to (0, 1]. AUC is unchanged; methods on different scales become swappable."""
    _, inv, cnt = np.unique(np.asarray(s, dtype=float), return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv.ravel()] / s.size


def _auc_rows(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """AUC of each row of S (shape P x n) by pairwise comparison, ties as one half."""
    pos, neg = S[:, y], S[:, ~y]
    out = np.empty(S.shape[0])
    step = max(1, 4_000_000 // max(1, pos.shape[1] * neg.shape[1]))
    for i in range(0, S.shape[0], step):
        d = pos[i:i + step, :, None] - neg[i:i + step, None, :]
        out[i:i + step] = ((d > 0) + 0.5 * (d == 0)).mean(axis=(1, 2))
    return out


def sign_flip_p(
    y: np.ndarray,
    s_new: np.ndarray,
    s_ref: np.ndarray,
    clusters: np.ndarray,
    *,
    n_perm: int = 2000,
    exact_max: int = 12,
    seed: int = 0,
) -> float:
    """Two-sided p-value for AUC(new) = AUC(ref) by swapping the two methods inside whole clusters.

    The paired permutation design of Bandos, Rockette and Gur (2005), with the cluster as the
    unit. Scores are first turned into within-earthquake ranks so the two methods share a scale.
    Under the null the two methods are exchangeable inside each cluster. All 2^G sign patterns
    are enumerated when G <= exact_max; otherwise n_perm random patterns (p counts the observed).
    """
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return 1.0
    a, b = _norm_rank(s_new), _norm_rank(s_ref)
    _, cid = np.unique(np.asarray(clusters), return_inverse=True)
    cid = cid.ravel()
    g = int(cid.max()) + 1
    if g <= exact_max:
        flips = np.array(list(product((False, True), repeat=g)), dtype=bool)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        flips = rng.random((n_perm, g)) < 0.5
        exact = False
    rows = flips[:, cid]
    A = np.where(rows, b, a)
    B = np.where(rows, a, b)
    d = _auc_rows(y, A) - _auc_rows(y, B)
    d_obs = _auc_rows(y, np.stack([a, b]))
    d_obs = d_obs[0] - d_obs[1]
    tol = 1e-12
    if exact:
        return float(np.mean(np.abs(d) >= abs(d_obs) - tol))
    return float((1 + np.sum(np.abs(d) >= abs(d_obs) - tol)) / (1 + n_perm))


def grid_clusters(x_m: np.ndarray, y_m: np.ndarray, cell_m: float) -> np.ndarray:
    """Cluster ids from a square grid of side cell_m (metres, projected coordinates)."""
    gx = np.floor(np.asarray(x_m, dtype=float) / cell_m).astype(np.int64)
    gy = np.floor(np.asarray(y_m, dtype=float) / cell_m).astype(np.int64)
    _, ids = np.unique(np.stack([gx, gy], axis=1), axis=0, return_inverse=True)
    return ids.ravel()


def _members(clusters: np.ndarray) -> list[np.ndarray]:
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    return np.split(order, np.cumsum(np.bincount(inv))[:-1])


def _draw(members: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Row indices for one cluster bootstrap draw (clusters drawn with replacement)."""
    pick = rng.integers(0, len(members), size=len(members))
    return np.concatenate([members[k] for k in pick])


@dataclass
class EventResult:
    event: object
    n_pos: int
    n_neg: int
    n_clusters: int
    auc: dict[str, float]
    auc_ci: dict[str, tuple[float, float]]
    diff: dict[str, float] = field(default_factory=dict)
    diff_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    diff_ci90: dict[str, tuple[float, float]] = field(default_factory=dict)
    p: dict[str, float] = field(default_factory=dict)
    mdd: dict[str, float] = field(default_factory=dict)
    verdict: dict[str, str] = field(default_factory=dict)
    note: dict[str, str] = field(default_factory=dict)
    p_method: dict[str, str] = field(default_factory=dict)


def _verdict(p_value, ci90, n_clusters, *, alpha, margin, min_clusters, p_method="bootstrap"):
    if n_clusters < min_clusters:
        if p_method == "signflip" and p_value < alpha:
            return "SEPARATES", f"sign-flip test, {n_clusters} clusters < {min_clusters}"
        return "UNDERPOWERED", f"{n_clusters} clusters < {min_clusters}"
    if p_value < alpha:
        return "SEPARATES", ""
    lo, hi = ci90
    if -margin < lo and hi < margin:
        return "EQUIVALENT", ""
    return "UNDERPOWERED", ""


def score_event(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    clusters: np.ndarray,
    reference: str,
    *,
    n_boot: int = 2000,
    margin: float = 0.05,
    alpha: float = 0.05,
    min_clusters: int = 0,
    small_test: bool = False,
    n_perm: int = 2000,
    seed: int = 0,
    event: object = None,
) -> EventResult:
    """Score every method in one earthquake; paired differences against `reference`.

    The verdict here uses the unadjusted p-value. A registration calls `apply_holm`
    over all cells before any verdict is read. With `small_test`, an earthquake under
    `min_clusters` takes its p-value from `sign_flip_p` instead of the bootstrap.
    """
    y = np.asarray(y, dtype=bool)
    if reference not in scores:
        raise KeyError(f"reference {reference!r} not in scores")
    names = list(scores)
    s = {m: np.asarray(scores[m], dtype=float) for m in names}
    point = {m: auc(y, s[m]) for m in names}
    members = _members(clusters)
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in names}
    for _ in range(n_boot):
        idx = _draw(members, rng)
        yb = y[idx]
        if yb.all() or not yb.any():
            continue
        for m in names:
            boot[m].append(auc(yb, s[m][idx]))
    boot = {m: np.asarray(v) for m, v in boot.items()}
    res = EventResult(
        event=event,
        n_pos=int(y.sum()),
        n_neg=int((~y).sum()),
        n_clusters=len(members),
        auc=point,
        auc_ci={m: _pct(boot[m]) for m in names},
    )
    for m in names:
        if m == reference:
            continue
        d = boot[m] - boot[reference]
        res.diff[m] = point[m] - point[reference]
        res.diff_ci[m] = _pct(d)
        res.diff_ci90[m] = _pct(d, 5.0, 95.0)
        if small_test and res.n_clusters < min_clusters:
            res.p[m] = sign_flip_p(y, s[m], s[reference], clusters, n_perm=n_perm, seed=seed)
            res.p_method[m] = "signflip"
        else:
            res.p[m] = _boot_p(d)
            res.p_method[m] = "bootstrap"
        se = float(np.std(d, ddof=1)) if d.size > 1 else float("inf")
        res.mdd[m] = (Z_ALPHA + Z_POWER) * se
        res.verdict[m], res.note[m] = _verdict(
            res.p[m], res.diff_ci90[m], res.n_clusters,
            alpha=alpha, margin=margin, min_clusters=min_clusters, p_method=res.p_method[m],
        )
    return res


def score_board(
    events: np.ndarray,
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    clusters: np.ndarray,
    reference: str,
    *,
    seed: int = 0,
    **kw,
) -> list[EventResult]:
    """One EventResult per earthquake. Clusters are resampled inside each earthquake only."""
    events = np.asarray(events)
    out = []
    for i, e in enumerate(np.unique(events)):
        m = events == e
        out.append(
            score_event(
                np.asarray(y)[m],
                {k: np.asarray(v)[m] for k, v in scores.items()},
                np.asarray(clusters)[m],
                reference,
                event=e,
                seed=seed + i,
                **kw,
            )
        )
    return out


def apply_holm(
    board: list[EventResult], *, alpha: float = 0.05, margin: float = 0.05, min_clusters: int = 0
) -> list[EventResult]:
    """Holm step-down over every (earthquake, method) cell; rewrites verdicts in place."""
    cells = [(r, m) for r in board for m in r.p]
    order = sorted(range(len(cells)), key=lambda i: cells[i][0].p[cells[i][1]])
    k = len(cells)
    rejected = set()
    for rank, i in enumerate(order):
        r, m = cells[i]
        if r.p[m] < alpha / (k - rank):
            rejected.add(i)
        else:
            break
    for i, (r, m) in enumerate(cells):
        p_eff = 0.0 if i in rejected else 1.0
        r.verdict[m], r.note[m] = _verdict(
            p_eff, r.diff_ci90[m], r.n_clusters, alpha=0.5, margin=margin, min_clusters=min_clusters,
            p_method=r.p_method.get(m, "bootstrap"),
        )
    return board


def event_offset(
    y: np.ndarray, p: np.ndarray, clusters: np.ndarray, *, n_boot: int = 2000, seed: int = 0
) -> tuple[float, tuple[float, float]]:
    """G2: logit(observed rate) minus logit(mean predicted probability), with a cluster interval.

    Positive means the method under-predicted in this earthquake. Valid ONLY on census
    data (every hole in an area graded). On a curated case set the intercept moves by an
    unknown sampling ratio (Prentice and Pyke 1979), so the offset is not identifiable.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    def off(yy, pp):
        o = np.clip(yy.mean(), 1e-6, 1 - 1e-6)
        q = np.clip(pp.mean(), 1e-6, 1 - 1e-6)
        return float(np.log(o / (1 - o)) - np.log(q / (1 - q)))

    members = _members(clusters)
    rng = np.random.default_rng(seed)
    b = [off(y[idx], p[idx]) for idx in (_draw(members, rng) for _ in range(n_boot))]
    return off(y, p), _pct(np.asarray(b))


def _boot_p(d: np.ndarray) -> float:
    """Two-sided bootstrap p-value for a zero difference."""
    if d.size == 0:
        return 1.0
    return float(min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean())))


def _pct(a: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    if a.size == 0:
        return (float("nan"), float("nan"))
    a_lo, a_hi = np.percentile(a, [lo, hi])
    return (float(a_lo), float(a_hi))

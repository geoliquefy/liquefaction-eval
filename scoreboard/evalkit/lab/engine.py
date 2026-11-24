"""Vectorised tile bootstrap for the lab. Equal, draw for draw, to resample.bootstrap with the same tiles and seed.

resample.bootstrap draws len(members) tile indices with rng.integers per draw and concatenates the members. The same
draw is a multiplicity weight per site: w_i = how often site i's tile was picked. AUC on the concatenated rows equals
the weighted Mann-Whitney AUC (ties one half); the mean log loss equals the weighted mean. Proved in
tests/test_evalkit_lab.py against the kit. No em dashes.
"""
from __future__ import annotations

import numpy as np

from ...scorer import _members


def tile_index(tiles: np.ndarray) -> tuple[np.ndarray, int]:
    """Site -> tile position in the kit's member order (np.unique order), and the tile count."""
    _, inv = np.unique(np.asarray(tiles), return_inverse=True)
    inv = inv.ravel()
    return inv, int(inv.max()) + 1


def weights(tiles: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """n_boot x n site multiplicities, using the same rng calls as resample.bootstrap."""
    inv, t = tile_index(tiles)
    assert len(_members(tiles)) == t
    rng = np.random.default_rng(seed)
    counts = np.empty((n_boot, t), dtype=np.float64)
    for b in range(n_boot):
        counts[b] = np.bincount(rng.integers(0, t, size=t), minlength=t)
    return counts[:, inv]


class Ranked:
    """One score vector, sorted once, ready for weighted AUC on many weight rows."""

    def __init__(self, y: np.ndarray, s: np.ndarray):
        y = np.asarray(y, bool)
        s = np.asarray(s, float)
        self.order = np.argsort(s, kind="stable")
        ss = s[self.order]
        self.starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
        self.pos = y[self.order].astype(np.float64)
        self.neg = 1.0 - self.pos

    def auc(self, w: np.ndarray) -> np.ndarray:
        ws = w[:, self.order]
        wp = np.add.reduceat(ws * self.pos, self.starts, axis=1)
        wn = np.add.reduceat(ws * self.neg, self.starts, axis=1)
        below = np.cumsum(wn, axis=1) - wn
        tp, tn = wp.sum(1), wn.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            a = (wp * (below + 0.5 * wn)).sum(1) / (tp * tn)
        a[(tp == 0) | (tn == 0)] = np.nan
        return a


def site_log_loss(y: np.ndarray, p: np.ndarray, clip: float = 1e-6) -> np.ndarray:
    pc = np.clip(p, clip, 1.0 - clip)
    return -(y * np.log(pc) + (1.0 - y) * np.log1p(-pc))


def mean_loss(w: np.ndarray, loss: np.ndarray) -> np.ndarray:
    return (w @ loss) / w.sum(1)


def placement_values(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """DeLong placement value per site: positives, share of negatives below (ties one half); negatives, share of
    positives above. AUC = mean over positives = mean over negatives."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    neg, pos = np.sort(s[~y]), np.sort(s[y])
    out = np.empty(s.size)
    sp, sn = s[y], s[~y]
    out[y] = (np.searchsorted(neg, sp, "left") + 0.5 * (np.searchsorted(neg, sp, "right")
                                                         - np.searchsorted(neg, sp, "left"))) / max(neg.size, 1)
    out[~y] = ((pos.size - np.searchsorted(pos, sn, "right")) + 0.5 * (np.searchsorted(pos, sn, "right")
                                                                       - np.searchsorted(pos, sn, "left"))) / max(pos.size, 1)
    return out

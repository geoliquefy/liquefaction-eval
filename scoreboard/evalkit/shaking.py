"""Shaking-only baselines for the card (own-work list item B4, eval M4). Pure numpy; reads no file. No em dashes.

Three rows, all built from shaking alone (no soil, no site memory):

  baseline: shaking score            score = PGA / MSF(M), a continuous ranking score (output kind "score", AUC
                                     only). It needs no fit and no earlier earthquake, so it exists on every task.
  baseline: shaking only             p = expit(a + b ln(PGA / MSF(M))), a and b fitted by maximum likelihood on the
                                     pooled labelled rows of EARLIER earthquakes only, each earlier earthquake with
                                     its own M. Never fitted on the scored earthquake (refused by event id). This is
                                     the R10 shaking-only probability that replaces the binned row.
  baseline: shaking only (binned,    the scorecard v1 and v2 rule kept beside for one release: pooled yes share per
            old)                     PGA bin (edges 0, 0.1, ..., 0.5 g, open top; a bin under 30 sites merges into its
                                     neighbour), read at this earthquake's PGA. Same code as
                                     docs/scorecard-v1-2026-11-15/build_v1.py shaking_rates and read_bins.

MSF form, named MSF_FORM = "MSF-M-ONLY-1.3" (correction B4, founder check 2026-11-23). Source: Boulanger and
Idriss 2014, CPT and SPT based liquefaction triggering procedures, report UCD/CGM-14/01 (April 2014), section 2.6,
page 14, eq. 2.19 (the numeric form of eq. 2.18):
    MSF    = 1 + (MSFmax - 1) (8.64 exp(-M / 4) - 1.325)          (MSF = 1 at M 7.5; MSF = MSFmax at M 5.25)
In B&I 2014 MSFmax is a site value: eq. 2.20 (page 14) sets MSFmax = 1.09 + (qc1Ncs / 180)^3, capped at 2.2. That
needs a CPT, so the B&I 2014 MSF as published is NOT a shaking-only quantity. MSF-M-ONLY-1.3 is the CPT-free form
used here: eq. 2.19 with MSFmax fixed at one number for every site and every earthquake, MSFMAX = 1.3. The MSF then
depends on M alone. The number 1.3 is what eq. 2.20 gives at qc1Ncs of about 107 (a loose to medium clean sand of
the kind that liquefies); no site value of qc1Ncs enters. It is the same value scoreboard/g5_entry.py uses (MSFMAX),
so the baseline and the house G5 transfer share one MSF. The choice cannot change the ranking inside one earthquake
(M is one number per earthquake); it only sets how earlier earthquakes of other magnitudes pool in the fit.
The v3 registration (item B8) quotes MSF_FORM and MSF_SOURCE. Not used: Idriss (1999), B&I 2014 eq. 2.17 (page 12),
MSF = 6.9 exp(-M / 4) - 0.058 <= 1.8. It is also CPT-free, but it is the sand curve (close to eq. 2.19 with MSFmax
1.8), so it would be a different named form. Earlier B4 text cited "eq. 2.17 and 2.18" for the formula above; that
citation was wrong (eq. 2.17 is the Idriss 1999 form).
The functions take MSFmax only as one scalar in [1.09, 2.2]; an array (a value per site) is refused. A value other
than 1.3 is a variant, not MSF-M-ONLY-1.3, and the row statement says so.

Ties: the score adds no ties of its own (no bins). Two sites tie only when their input PGA values are equal; the
kit counts them (`tie_count`) and does not break them, because any tie-break order would carry information that is
not in the shaking. AUC gives a tied pair half credit.

Fit guards (review 1): the fit is refused, never marked and kept, when the earlier rows have fewer than MIN_CLASS
yes or MIN_CLASS no, when one PGA / MSF threshold splits yes from no (separation: the slope runs off to infinity and
p ties at 0 or 1), when Newton steps do not converge in MAX_ITER, or when the slope b is not above 0 (stronger
shaking must not mean fewer yes). The fit record carries converged, n_iter and separated.
"""
from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .spec import Entry, EntryManifest

MSFMAX = 1.3
MSFMAX_RANGE = (1.09, 2.2)            # B&I 2014 eq. 2.20 (page 14): floor 1.09 at qc1Ncs = 0, cap 2.2
MSF_FORM = "MSF-M-ONLY-1.3"
MSF_SOURCE = ("MSF-M-ONLY-1.3: Boulanger and Idriss 2014 (UCD/CGM-14/01, April 2014) eq. 2.19, page 14, "
              "MSF = 1 + (MSFmax - 1)(8.64 exp(-M/4) - 1.325), with MSFmax fixed at 1.3 for every site (CPT-free; "
              "eq. 2.20 would set it from qc1Ncs and gives 1.3 at qc1Ncs about 107), as scoreboard/g5_entry.py")
PGA_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, np.inf)
MIN_BIN = 30
RIDGE = 1e-6
MAX_ITER = 50
MIN_CLASS = 10                        # fewest earlier yes rows, and fewest earlier no rows, a fit may use
FIT_MARKER = "B4 continuous fit:"     # starts the statement of every probability_entry row
NAME_SCORE = "baseline: shaking score"
NAME_PROB = "baseline: shaking only"
NAME_BINNED = "baseline: shaking only (binned, old)"


@dataclass(frozen=True)
class Earlier:
    """One earlier earthquake's labelled rows for the fit: its id, magnitude, PGA (g) and 0/1 labels."""
    event_id: str
    mw: float
    pga: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "pga", np.asarray(self.pga, float))
        object.__setattr__(self, "y", np.asarray(self.y, float))
        if self.pga.ndim != 1 or self.pga.shape != self.y.shape or self.pga.size == 0:
            raise ValueError(f"earlier {self.event_id!r}: needs one PGA and one label per row")
        if not np.all(np.isin(self.y, (0.0, 1.0))):
            raise ValueError(f"earlier {self.event_id!r}: labels must be 0 or 1")
        _check_pga(self.pga, f"earlier {self.event_id!r}", positive=True)
        _check_mw(self.mw)


# ---------------------------------------------------------------------------------------------------------------
# MSF and the score
# ---------------------------------------------------------------------------------------------------------------

def _check_mw(mw: float) -> None:
    if not (isinstance(mw, numbers.Real) and not isinstance(mw, bool) and np.isfinite(mw)
            and 4.0 <= float(mw) <= 9.5):
        raise ValueError(f"magnitude must be a finite number in [4, 9.5], got {mw!r}")


def _check_msfmax(msfmax: float) -> None:
    if not (isinstance(msfmax, numbers.Real) and not isinstance(msfmax, bool)):
        raise ValueError(f"MSFmax must be one number for every site (a CPT-free MSF), not {type(msfmax).__name__}")
    if not (MSFMAX_RANGE[0] <= float(msfmax) <= MSFMAX_RANGE[1]):
        raise ValueError(f"MSFmax must lie in {list(MSFMAX_RANGE)} (B&I 2014 eq. 2.20), got {msfmax!r}")


def msf_form(msfmax: float = MSFMAX) -> str:
    """The MSF form name for a row statement: MSF-M-ONLY-1.3, or a named variant when MSFmax is not 1.3."""
    _check_msfmax(msfmax)
    if float(msfmax) == MSFMAX:
        return f"MSF form {MSF_FORM} (B&I 2014 eq. 2.19, page 14, MSFmax fixed at 1.3, no CPT input)"
    return (f"MSF form VARIANT of {MSF_FORM} (B&I 2014 eq. 2.19, page 14, MSFmax fixed at {float(msfmax):g}, "
            "no CPT input)")


def _check_pga(pga: np.ndarray, what: str, positive: bool) -> None:
    if not np.all(np.isfinite(pga)):
        raise ValueError(f"{what}: every PGA must be finite")
    if positive and np.any(pga <= 0):
        raise ValueError(f"{what}: every PGA must be above 0 g (the fit uses ln PGA)")
    if not positive and np.any(pga < 0):
        raise ValueError(f"{what}: PGA must not be negative")


def msf(mw: float, msfmax: float = MSFMAX) -> float:
    """MSF at magnitude mw: B&I 2014 eq. 2.19 (page 14) with one fixed MSFmax (MSF-M-ONLY-1.3 at 1.3)."""
    _check_mw(mw)
    _check_msfmax(msfmax)
    return float(1.0 + (msfmax - 1.0) * (8.64 * np.exp(-float(mw) / 4.0) - 1.325))


def score(pga, mw: float, msfmax: float = MSFMAX) -> np.ndarray:
    """PGA / MSF(M): the shaking demand scaled to M 7.5. Continuous; higher means stronger shaking."""
    pga = np.asarray(pga, float)
    _check_pga(pga, "score", positive=False)
    return pga / msf(mw, msfmax)


def tie_count(values) -> dict:
    """Counts only: distinct values and the number of rows that share a value with another row."""
    _, c = np.unique(np.asarray(values, float), return_counts=True)
    return {"n": int(c.sum()), "n_distinct": int(c.size), "n_rows_tied": int(c[c > 1].sum())}


# ---------------------------------------------------------------------------------------------------------------
# The probability version (earlier earthquakes only)
# ---------------------------------------------------------------------------------------------------------------

def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0)))


def _logistic(x: np.ndarray, y: np.ndarray, iters: int) -> tuple[float, float, bool, int]:
    """a, b of expit(a + b x) by Newton steps, plus converged and the number of steps taken."""
    X = np.column_stack([np.ones_like(x), x])
    m = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
    beta = np.array([np.log(m / (1 - m)), 0.0])
    converged, n_iter = False, 0
    for n_iter in range(1, iters + 1):
        p = _expit(X @ beta)
        w = np.clip(p * (1 - p), 1e-12, None)
        g = X.T @ (y - p) - RIDGE * beta
        H = (X * w[:, None]).T @ X + RIDGE * np.eye(2)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            break
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    return float(beta[0]), float(beta[1]), converged, n_iter


def _separated(x: np.ndarray, y: np.ndarray) -> bool:
    """True when one threshold on x splits yes from no (complete or quasi-complete separation, either way)."""
    xy, xn = x[y == 1], x[y == 0]
    return bool(xn.max() <= xy.min() or xy.max() <= xn.min())


def fit(earlier: Sequence[Earlier], scored_event_id: str, msfmax: float = MSFMAX) -> dict:
    """Fit a and b on the pooled rows of the earlier earthquakes (each row weighted once, as the binned rule).

    Refuses an earlier set that names the scored earthquake, a repeated earlier earthquake, or no earlier one.
    Refuses a fit that fails a guard (module docstring): too few yes or no, separation, no convergence, b <= 0.
    Returns aggregates only: a, b, MSFmax, the earlier ids and magnitudes, row counts and the guard results.
    """
    if not earlier:
        raise ValueError("the shaking-only probability needs at least one earlier earthquake")
    ids = [e.event_id for e in earlier]
    if scored_event_id in ids:
        raise ValueError(f"the scored earthquake {scored_event_id!r} is in the fit set; fit on earlier ones only")
    if len(set(ids)) != len(ids):
        raise ValueError("each earlier earthquake may enter the fit once")
    _check_msfmax(msfmax)
    x = np.concatenate([np.log(score(e.pga, e.mw, msfmax)) for e in earlier])
    y = np.concatenate([e.y for e in earlier])
    n_yes, n_no = int(y.sum()), int(y.size - y.sum())
    if n_yes < MIN_CLASS or n_no < MIN_CLASS:
        raise ValueError(f"the earlier rows have {n_yes} yes and {n_no} no; the fit needs at least {MIN_CLASS} of "
                         "each")
    if _separated(x, y):
        raise ValueError("the earlier rows are separated by shaking alone (one PGA / MSF threshold splits yes from "
                         "no); the slope has no finite estimate, so the fit is refused")
    a, b, converged, n_iter = _logistic(x, y, MAX_ITER)
    if not converged:
        raise ValueError(f"the shaking-only fit did not converge in {MAX_ITER} Newton steps; the fit is refused")
    if not b > 0:
        raise ValueError(f"the shaking-only fit has slope b = {b:.3g}, not above 0 (stronger shaking would mean "
                         "fewer yes); the fit is refused")
    return {"form": "p = expit(a + b ln(PGA / MSF(M)))", "a": a, "b": b, "msfmax": float(msfmax),
            "msf_form": msf_form(msfmax), "msf_source": MSF_SOURCE, "fitted_on": ids,
            "fitted_on_mw": [float(e.mw) for e in earlier],
            "n_rows": int(y.size), "n_yes": n_yes, "scored_event_id": scored_event_id,
            "slope_positive": True, "converged": True, "n_iter": n_iter, "separated": False,
            "min_class": MIN_CLASS}


def probability(f: dict, pga, mw: float) -> np.ndarray:
    """Apply a fit to this earthquake's PGA and magnitude."""
    pga = np.asarray(pga, float)
    _check_pga(pga, "probability", positive=True)
    return _expit(f["a"] + f["b"] * np.log(score(pga, mw, f["msfmax"])))


# ---------------------------------------------------------------------------------------------------------------
# The old binned rule (kept beside for one release)
# ---------------------------------------------------------------------------------------------------------------

def binned_rates(earlier_pga, earlier_y) -> list[list[float]]:
    """Scorecard v1/v2 shaking-only bins: [lo, hi, n, yes]; merge under MIN_BIN sites top down, lowest upward."""
    earlier_pga, earlier_y = np.asarray(earlier_pga, float), np.asarray(earlier_y, float)
    bins = []
    for lo, hi in zip(PGA_EDGES[:-1], PGA_EDGES[1:]):
        k = (earlier_pga >= lo) & (earlier_pga < hi)
        bins.append([lo, hi, int(k.sum()), int(earlier_y[k].sum())])
    i = len(bins) - 1
    while i > 0:
        if bins[i][2] < MIN_BIN:
            lo_b = bins[i - 1]
            bins[i - 1] = [lo_b[0], bins[i][1], lo_b[2] + bins[i][2], lo_b[3] + bins[i][3]]
            del bins[i]
        i -= 1
    if len(bins) > 1 and bins[0][2] < MIN_BIN:
        bins[1] = [bins[0][0], bins[1][1], bins[0][2] + bins[1][2], bins[0][3] + bins[1][3]]
        del bins[0]
    return bins


def binned_p(bins: list[list[float]], pga) -> np.ndarray:
    """Read the bins at this earthquake's PGA (scorecard v1/v2 read_bins)."""
    pga = np.asarray(pga, float)
    out = np.empty(pga.size)
    for lo, hi, n, yes in bins:
        k = (pga >= lo) & (pga < hi)
        out[k] = yes / n
    out[pga < bins[0][0]] = bins[0][3] / bins[0][2]
    return out


# ---------------------------------------------------------------------------------------------------------------
# Entries for the card
# ---------------------------------------------------------------------------------------------------------------

def _manifest(name: str, kind: str, statement: str, event_id: str) -> EntryManifest:
    return EntryManifest(model=name, output_kind=kind, statement=statement, exposure={event_id: "unseen"},
                         house=False)


def score_entry(site_ids: Sequence[str], pga, mw: float, event_id: str, msfmax: float = MSFMAX) -> Entry:
    """The continuous ranking row. No fit, so it exists even with no earlier earthquake."""
    stmt = f"PGA / MSF(M) at M {float(mw):g}; {msf_form(msfmax)}; ranking only, no fit"
    return Entry(_manifest(NAME_SCORE, "score", stmt, event_id), tuple(site_ids), score(pga, mw, msfmax))


def probability_entry(site_ids: Sequence[str], pga, mw: float, event_id: str, earlier: Sequence[Earlier],
                      msfmax: float = MSFMAX) -> tuple[Entry, dict]:
    """The R10 shaking-only probability, fitted on earlier earthquakes only, and its fit record (aggregates).

    The statement carries the fit itself (a, b, MSFmax, the earlier ids and magnitudes, n_rows, n_yes) and the scored
    magnitude, so the card documents the earlier-only fit on its own. It starts with FIT_MARKER.
    """
    f = fit(earlier, event_id, msfmax)
    earlier_txt = ", ".join(f"{i} (M {m:g})" for i, m in zip(f["fitted_on"], f["fitted_on_mw"]))
    stmt = (f"{FIT_MARKER} p = expit(a + b ln(PGA / MSF(M))) at M {float(mw):g}, a = {f['a']:.4f}, "
            f"b = {f['b']:.4f}, MSFmax {f['msfmax']:g}; {msf_form(f['msfmax'])}; fitted on the earlier earthquakes "
            f"{earlier_txt} only, never on this one; n_rows {f['n_rows']}, n_yes {f['n_yes']}")
    return Entry(_manifest(NAME_PROB, "target_probability", stmt, event_id), tuple(site_ids),
                 probability(f, pga, mw)), f


def binned_entry(site_ids: Sequence[str], pga, event_id: str, earlier: Sequence[Earlier]) -> tuple[Entry, list]:
    """The old binned row (scorecard v1/v2 rule), kept beside the continuous rows for one release."""
    if not earlier:
        raise ValueError("the binned shaking row needs at least one earlier earthquake")
    if event_id in [e.event_id for e in earlier]:
        raise ValueError(f"the scored earthquake {event_id!r} is in the earlier set")
    bins = binned_rates(np.concatenate([e.pga for e in earlier]), np.concatenate([e.y for e in earlier]))
    stmt = "OLD RULE, kept for one release: pooled yes share per PGA bin over earlier label sets (scorecard v1, v2)"
    return Entry(_manifest(NAME_BINNED, "target_probability", stmt, event_id), tuple(site_ids),
                 binned_p(bins, pga)), bins


def bins_record(bins: list[list[float]]) -> list[dict]:
    """The bins as card-safe aggregates (as scorecard v1/v2 printed them)."""
    return [{"pga_from_g": b[0], "pga_to_g": (None if np.isinf(b[1]) else b[1]), "n": b[2], "yes": b[3]}
            for b in bins]

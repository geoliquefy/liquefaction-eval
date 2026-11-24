"""M3 attempt 3 (item A4, G1+G5+G6, eval M3): parity on the PEER 2021/04 global set (PRJ-3012). No em dashes.

Registration: pillar `docs/evals/PARITY-M3-ROUND3-REGISTRATION-2026-11-23.md` (committed alone, b7bd8c4). This module
holds the pure parts: the gates on the rows, the AUC with its tie variants, the rounding band of a printed value and
the decision word. It reads no file. The runner is `scripts/run_m3_parity_round3_2026_11_23.py`.
Tests: `tests/test_evalkit_parity_round3.py` (planted cases only).
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np

# PEER 2021/04 Table 3.2, global rows = PRJ-3012 ReadMe Table 1 (21 earthquakes, 275 cases). Event names as stored in
# the PRJ-3012 record field `EventName` (the same dict as `run_g1_curated.README_COUNTS`).
PEER_TABLE_3_2 = {
    "Niigata": 3, "San Fernando": 2, "Haicheng": 2, "Tangshan": 10, "Imperial Valley": 7,
    "Victoria (Mexicali)": 5, "Westmorland": 9, "Nihonkai-Chubu": 2, "Borah Peak": 3, "Edgecumbe": 23,
    "Elmore Ranch": 2, "Superstition Hills": 8, "Loma Prieta": 67, "Northridge": 3, "Hyogoken-Nambu": 21,
    "Kocaeli": 16, "Chi-Chi": 34, "Achaia-Ilia": 2, "Baja": 3, "Tohoku": 7, "Emilia": 46,
}
N_CASES = 275
YES_SHARE_PRINTED = "0.58"  # PEER 2021/04 p. 33: "58% are 'Manifestation'"
MIN_READINGS = 20
FLOAT_GUARD = 1e-12


class GateError(RuntimeError):
    """A registered gate failed: the run stops and prints no word."""


def rounding_band(printed: str) -> tuple[float, float]:
    """Values that print as `printed` at its own number of decimals: [p - h, p + h], h = half a unit of the last digit.

    Taken from the string, so "0.77" gives (0.765, 0.775) and "0.770" gives (0.7695, 0.7705).
    """
    d = Decimal(printed)
    exp = d.as_tuple().exponent
    if not isinstance(exp, int) or exp >= 0:
        raise ValueError(f"printed value {printed!r} has no decimals")
    h = Decimal(1).scaleb(exp) / 2
    return float(d - h), float(d + h)


def decide(a: float, printed: str) -> str:
    """Registration section 4: AGREES if a lies in the rounding band (ends included), else DISAGREES."""
    if not np.isfinite(a):
        raise GateError("AUC is not finite")
    lo, hi = rounding_band(printed)
    return "AGREES" if lo - FLOAT_GUARD <= a <= hi + FLOAT_GUARD else "DISAGREES"


def auc_variants(y, s) -> dict:
    """Mann-Whitney AUC over all yes/no pairs: ties count 1/2 (decides), 0 (strict) and 1 (beside)."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    if not np.all(np.isfinite(s)):
        raise GateError("a score is not finite")
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        raise GateError("one class is empty")
    neg_sorted = np.sort(neg)
    below = np.searchsorted(neg_sorted, pos, side="left").sum()
    at_or_below = np.searchsorted(neg_sorted, pos, side="right").sum()
    ties = at_or_below - below
    pairs = float(pos.size) * float(neg.size)
    return {"half": (below + 0.5 * ties) / pairs, "strict": below / pairs, "ties_one": at_or_below / pairs,
            "tied_pairs": int(ties), "pairs": int(pairs), "n_yes": int(pos.size), "n_no": int(neg.size)}


def pair_units(a: float, printed: str, n_yes: int, n_no: int) -> float:
    """|a - printed| in units of one yes/no pair (1 / (yes x no)). Beside only."""
    return abs(a - float(Decimal(printed))) * n_yes * n_no


def gate_rows(events, manifestation) -> dict:
    """G-rows: 275 records, counts per earthquake equal Table 3.2, labels 0 or 2, yes share prints as 0.58."""
    events = [str(e) for e in events]
    m = np.asarray(manifestation, float)
    if len(events) != N_CASES or m.size != N_CASES:
        raise GateError(f"G-rows: {len(events)} records, expected {N_CASES}")
    counts: dict[str, int] = {}
    for e in events:
        counts[e] = counts.get(e, 0) + 1
    if counts != PEER_TABLE_3_2:
        raise GateError("G-rows: counts per earthquake differ from PEER Table 3.2")
    if not np.all(np.isin(m, (0.0, 2.0))):
        raise GateError("G-rows: a Manifestation value is not 0 or 2")
    y = m == 2.0
    share = y.mean()
    lo, hi = rounding_band(YES_SHARE_PRINTED)
    if not (lo - FLOAT_GUARD <= share <= hi + FLOAT_GUARD):
        raise GateError("G-rows: the yes share does not print as 0.58")
    return {"n": int(y.size), "yes": int(y.sum()), "no": int((~y).sum()), "counts": counts}


def gate_inputs(mw, pga, gwt, n_readings) -> None:
    """G-inputs: finite Mw, PGA and GWT, and at least 20 finite readings, for every case."""
    for name, v in (("Mw", mw), ("PGA", pga), ("GWT", gwt)):
        if not np.all(np.isfinite(np.asarray(v, float))):
            raise GateError(f"G-inputs: a case lacks a finite {name}")
    if np.any(np.asarray(n_readings) < MIN_READINGS):
        raise GateError(f"G-inputs: a case has fewer than {MIN_READINGS} finite readings")


def gate_lpi(lpi) -> None:
    """G-lpi: every case has a finite LPI; no case is dropped."""
    lpi = np.asarray(lpi, float)
    if lpi.size != N_CASES or not np.all(np.isfinite(lpi)):
        raise GateError("G-lpi: a case has no finite LPI")


def gate_chain(lpi_settings, lpi_reference, tol: float = 1e-5) -> float:
    """G-chain: the default id reproduces the registered G1 curated path. Returns the largest difference."""
    a, b = np.asarray(lpi_settings, float), np.asarray(lpi_reference, float)
    if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise GateError("G-chain: shapes or finite patterns differ")
    d = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0
    if d > tol:
        raise GateError(f"G-chain: default id differs from the G1 curated path by {d:.3g} > {tol}")
    return d

"""Scorer self-checks that must pass before any card is made (rule book R12 (a) to (c)), and the duplicate detector.

Each gate runs the kit's own machinery (tile bootstrap, paired difference, join, eligibility) on a seeded synthetic
earthquake from synth.py:
  planted_better      an entry of known extra skill separates from a weaker one, in AUC and in log loss (R12 a)
  null_silent         two exchangeable entries: at most NULL_MAX_FALSE of NULL_REPS replicates separate (R12 b)
  defect_one_tile     labels shuffled inside one tile change the AUC: the scorer reads labels row by row (R12 c)
  defect_all_labels   labels shuffled everywhere: the skilled entry's AUC interval covers 0.5 (R12 c)
  defect_join_order   entry rows permuted: the id join gives the same AUC, a row-order join would not (R12 c)
  defect_monotone_copy  a monotone copy of an entry is flagged as the same model (v0 counted one model as 8 dots)
  defect_kind         an other_probability or score entry without a conversion is refused probability metrics (R3)

Duplicate rule (a proposal that needs calibration, see README): Spearman rank correlation of two entries on the
card's labelled sites above DUPLICATE_RHO marks them as one model. v0's M0 against Level 0 was 0.9995.
No em dashes.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from scipy.special import expit

from ..scorer import _norm_rank, auc
from . import metrics, resample, synth
from .spec import Entry, align, probability_eligible

GATE_SEED = 20261115
DUPLICATE_RHO = 0.99
NULL_REPS = 20
NULL_MAX_FALSE = 3          # P(X >= 4 | 20, 0.05) = 0.016
_N, _N_BOOT, _TILE_M = 800, 200, 2000.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties. NaN if either is constant."""
    ra, rb = _norm_rank(np.asarray(a, float)), _norm_rank(np.asarray(b, float))
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def duplicate_pairs(values: dict[str, np.ndarray], threshold: float = DUPLICATE_RHO) -> list[dict]:
    """Pairs of entries whose rank correlation exceeds `threshold`: the card counts each such pair as one model."""
    names = list(values)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = spearman(values[a], values[b])
            if rho > threshold:
                out.append({"a": a, "b": b, "spearman": rho})
    return out


def distinct_models(names: list[str], pairs: list[dict]) -> int:
    """Number of groups after joining every duplicate pair (union-find on names)."""
    parent = {n: n for n in names}

    def root(n: str) -> str:
        while parent[n] != n:
            n = parent[n]
        return n

    for pr in pairs:
        parent[root(pr["a"])] = root(pr["b"])
    return len({root(n) for n in names})


# ---------------------------------------------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------------------------------------------

def _event(seed: int) -> dict:
    ev = synth.synthetic_event(_N, seed)
    ev["y"] = ev["label"].astype(bool)
    ev["tiles"] = resample.tiles_for(ev["x_m"], ev["y_m"], _TILE_M)
    return ev


def _model(ev: dict, noise_sd: float, rng: np.random.Generator) -> np.ndarray:
    """A target probability from the susceptibility seen with noise (the field is unseen)."""
    s_seen = ev["s"] + rng.normal(0.0, noise_sd, ev["s"].size)
    return ev["fragility"] * expit(ev["intercept"] + ev["slope"] * s_seen)


def _paired_word(ev: dict, a: np.ndarray, b: np.ndarray, key: str, seed: int) -> dict:
    y = ev["y"]

    def stats(v: np.ndarray):
        return lambda idx: {"auc": auc(y[idx], v[idx]), "log_loss": metrics.log_loss(y[idx], v[idx], 1e-6)}

    da = resample.bootstrap(stats(a), ev["tiles"], _N_BOOT, seed)[key]
    db = resample.bootstrap(stats(b), ev["tiles"], _N_BOOT, seed)[key]
    full = np.arange(y.size)
    return resample.paired(stats(a)(full)[key], stats(b)(full)[key], da, db)


def gate_planted_better(seed: int) -> dict:
    ev = _event(seed)
    rng = np.random.default_rng(seed + 1)
    strong, weak = ev["p_target"], _model(ev, 1.0, rng)
    r_auc = _paired_word(ev, strong, weak, "auc", seed)
    r_ll = _paired_word(ev, strong, weak, "log_loss", seed)
    ok = r_auc["lo"] > 0 and r_ll["hi"] < 0
    return {"pass": bool(ok), "auc_diff": r_auc, "log_loss_diff": r_ll}


def gate_null_silent(seed: int) -> dict:
    ev = _event(seed)
    rng = np.random.default_rng(seed + 2)
    false = 0
    for k in range(NULL_REPS):
        r = _paired_word(ev, _model(ev, 0.6, rng), _model(ev, 0.6, rng), "auc", seed + 10 + k)
        false += int(r["p"] < 0.05)
    return {"pass": false <= NULL_MAX_FALSE, "false_separations": false, "replicates": NULL_REPS,
            "max_allowed": NULL_MAX_FALSE}


def gate_defect_one_tile(seed: int) -> dict:
    ev = _event(seed)
    p = _model(ev, 0.6, np.random.default_rng(seed + 3))
    y, t = ev["y"], ev["tiles"]
    mixed = [k for k in np.unique(t) if 0 < y[t == k].sum() < (t == k).sum()]
    big = max(mixed, key=lambda k: (t == k).sum())
    rows = np.flatnonzero(t == big)
    y_bad = y.copy()
    y_bad[rows] = y[rows][::-1] if not np.array_equal(y[rows], y[rows][::-1]) else np.roll(y[rows], 1)
    a_clean, a_bad = auc(y, p), auc(y_bad, p)
    return {"pass": a_clean != a_bad, "auc_clean": a_clean, "auc_defect": a_bad, "tile_rows": int(rows.size)}


def gate_defect_all_labels(seed: int) -> dict:
    ev = _event(seed)
    rng = np.random.default_rng(seed + 4)
    p = ev["p_target"]
    y_bad = rng.permutation(ev["y"])
    draws = resample.bootstrap(lambda idx: {"auc": auc(y_bad[idx], p[idx])}, ev["tiles"], _N_BOOT, seed)["auc"]
    iv = resample.interval(auc(y_bad, p), draws)
    return {"pass": iv["lo"] <= 0.5 <= iv["hi"], "auc_shuffled": iv}


def gate_defect_join_order(seed: int) -> dict:
    ev = _event(seed)
    rng = np.random.default_rng(seed + 5)
    p = _model(ev, 0.6, rng)
    perm = rng.permutation(p.size)
    entry = Entry(synth.manifest("gate entry", "target_probability", "gate"),
                  tuple(np.array(ev["site_ids"])[perm]), p[perm])
    by_id = auc(ev["y"], align(entry, ev["site_ids"]))
    by_row = auc(ev["y"], entry.p)
    ok = by_id == auc(ev["y"], p) and by_row != by_id
    return {"pass": bool(ok), "auc_by_id": by_id, "auc_by_row_order": by_row}


def gate_defect_monotone_copy(seed: int) -> dict:
    ev = _event(seed)
    rng = np.random.default_rng(seed + 6)
    a, b = _model(ev, 0.6, rng), _model(ev, 0.6, rng)
    copy = expit(3.0 * np.log(a / (1.0 - a)))
    pairs = duplicate_pairs({"a": a, "copy": copy, "b": b})
    ok = [(pr["a"], pr["b"]) for pr in pairs] == [("a", "copy")]
    return {"pass": ok, "flagged": pairs}


def gate_defect_kind(seed: int) -> dict:
    kinds = {k: probability_eligible(synth.manifest("g", k, "gate"))
             for k in ("target_probability", "other_probability", "score")}
    converted = probability_eligible(synth.manifest("g", "other_probability", "gate", conversion_id="a" * 64))
    ok = kinds == {"target_probability": True, "other_probability": False, "score": False} and converted
    return {"pass": ok, "eligible": kinds, "eligible_with_conversion": converted}


GATES = {"planted_better": gate_planted_better, "null_silent": gate_null_silent,
         "defect_one_tile": gate_defect_one_tile, "defect_all_labels": gate_defect_all_labels,
         "defect_join_order": gate_defect_join_order, "defect_monotone_copy": gate_defect_monotone_copy,
         "defect_kind": gate_defect_kind}


@lru_cache(maxsize=4)
def _run_gates_json(seed: int) -> str:
    res = {name: fn(seed) for name, fn in GATES.items()}
    return json.dumps({"pass": all(r["pass"] for r in res.values()), "seed": seed, "gates": res})


def run_gates(seed: int = GATE_SEED) -> dict:
    """Every gate; `pass` is True only if all pass. Deterministic in `seed`, so the result is cached per seed."""
    return json.loads(_run_gates_json(seed))

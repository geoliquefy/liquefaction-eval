"""Level 0 recalibration map (founder ruling 56b).

Registration: pillar docs/evals/L0-RECAL-REGISTRATION-2026-11-24.md (committed alone, before any fit). Level 0 itself
is not refit. This module holds a separate two-number map applied to Level 0's world-form output:

    q = expit(a + b x),   x = logit(clip(p, CLIP, 1 - CLIP))

The fit is the kit's own R9 level 2 logistic fit (metrics._logistic_fit), done once on the registered data and frozen
in a map file named by sha256. A future frozen set is scored with the frozen map unchanged (registration section 7).
Add-only module. No em or en dashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from . import metrics as M

MAP_ID = "l0-recal-2026-11-24"
ROW_NAME = "Level 0 + recal [l0-recal-2026-11-24]"
FAMILY = "logistic recalibration, 2 numbers (Platt): q = expit(a + b * logit(clip(p, clip, 1 - clip)))"
CLIP = 1e-6
INPUT_FORM = ("g5.p_level0(ln k*, 0): Level 0 world form, sigma_e = 0, ln k* at LPI 5 (run_g6_jun2011.lnk_star); "
              "not the Canterbury card form with sigma_e = sdlnpga")
MIN_CASES, MIN_CLASS = 200, 30
MAP_KEYS = ("map_id", "row_name", "family", "clip", "input_form", "a", "b")


class RecalStop(RuntimeError):
    """A registered stop rule fired (registration section 6). No map is frozen."""


def _check_inputs(p, y) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, float)
    y = np.asarray(y)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("p and y must be 1-d arrays of the same length")
    if not np.all(np.isfinite(p)) or p.min(initial=0.0) < 0.0 or p.max(initial=1.0) > 1.0:
        raise ValueError("p must be finite and in [0, 1]")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must hold only 0 and 1")
    return p, y.astype(float)


def fit(p, y, clip: float = CLIP) -> dict:
    """Maximum likelihood (a, b), every case weight 1. Raises RecalStop on S3, S4 or S5."""
    p, y = _check_inputs(p, y)
    n_yes, n_no = int(y.sum()), int(y.size - y.sum())
    if n_yes + n_no < MIN_CASES or n_yes < MIN_CLASS or n_no < MIN_CLASS:
        raise RecalStop(f"S3: {n_yes + n_no} cases ({n_yes} yes, {n_no} no); need {MIN_CASES} and {MIN_CLASS} each")
    x = M.clipped_logit(p, clip)
    if np.ptp(x) == 0.0:
        raise RecalStop("S4: logit(p) is constant; the slope is not identified")
    w = M._logistic_fit(x, y, slope_free=True)
    if w is None:
        raise RecalStop("S4: the fit did not converge")
    a, b = float(w[0]), float(w[1])
    if not (np.isfinite(a) and np.isfinite(b)):
        raise RecalStop("S4: a or b is not finite")
    if b <= 0.0:
        raise RecalStop(f"S5: slope b = {b:.6g} <= 0; the map would flatten or invert the ranking")
    z = a + b * x
    mu = expit(z)
    v = mu * (1.0 - mu)
    h = np.array([[v.sum(), (v * x).sum()], [(v * x).sum(), (v * x * x).sum()]])
    cov = np.linalg.inv(h)
    return {"a": a, "b": b, "se_a_wald_in_sample": float(np.sqrt(cov[0, 0])),
            "se_b_wald_in_sample": float(np.sqrt(cov[1, 1])), "n": n_yes + n_no, "n_yes": n_yes, "n_no": n_no,
            "loglik": float(np.sum(y * z - np.logaddexp(0.0, z))), "clip": clip}


def apply(p, a: float, b: float, clip: float = CLIP) -> np.ndarray:
    """The map: q = expit(a + b logit(clip(p))). Monotone increasing in p when b > 0."""
    return expit(a + b * M.clipped_logit(np.asarray(p, float), clip))


def summary(y, p, clip: float = CLIP) -> dict:
    """Aggregates for one case set: count, yes share, log loss, Brier, calibration intercept and slope."""
    p, y = _check_inputs(p, y)
    return {"n": int(y.size), "yes_share": float(y.mean()), f"log_loss@{M.clip_key(clip)}": M.log_loss(y, p, clip),
            "brier": M.brier(y, p), "citl": M.citl(y, p, clip), "slope": M.recal_slope(y, p, clip),
            **M.exact_shares(p)}


def map_record(a: float, b: float, **meta) -> dict:
    rec = {"map_id": MAP_ID, "row_name": ROW_NAME, "family": FAMILY, "clip": CLIP, "input_form": INPUT_FORM,
           "a": float(a), "b": float(b)}
    clash = set(meta) & set(rec)
    if clash:
        raise ValueError(f"meta may not override {sorted(clash)}")
    rec.update(meta)
    return rec


def canonical_bytes(rec: dict) -> bytes:
    return (json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_map(path: Path, rec: dict) -> str:
    """Write the frozen map once. Refuses to overwrite (S9). Returns the file's sha256."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; a frozen map is never overwritten")
    for k in MAP_KEYS:
        if k not in rec:
            raise ValueError(f"map record lacks {k}")
    data = canonical_bytes(rec)
    with open(path, "xb") as f:
        f.write(data)
    return sha256_bytes(data)


def load_map(path: Path, expected_sha256: str) -> dict:
    """Read a frozen map and refuse it unless its bytes hash to expected_sha256."""
    data = Path(path).read_bytes()
    got = sha256_bytes(data)
    if got != expected_sha256:
        raise ValueError(f"map sha256 {got} differs from the frozen {expected_sha256}")
    rec = json.loads(data)
    if data != canonical_bytes(rec):
        raise ValueError("map file is not in canonical form")
    if rec.get("map_id") != MAP_ID or rec.get("family") != FAMILY or rec.get("clip") != CLIP:
        raise ValueError("map record does not match this module's map id, family or clip")
    if not rec["b"] > 0:
        raise ValueError("frozen map has b <= 0")
    return rec


def apply_map(p, rec: dict) -> np.ndarray:
    return apply(p, rec["a"], rec["b"], rec["clip"])

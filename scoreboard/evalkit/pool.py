"""The pooled line across earthquakes (rule book R1). No em dashes.

R1 allows a line that combines earthquakes only if: (a) the registration names it before the join (two entries, one
metric); (b) it pools per-earthquake paired differences with random-effects weights and prints their spread;
(c) at least 3 earthquakes pass the count gate; (d) the word holds with each earthquake left out in turn;
(e) it prints under the full cards as "across these N earthquakes".

Estimator (R1 P1, a proposal; calibrated on synthetic data in tests/test_evalkit_pool.py). Per-earthquake differences
d_i with tile-bootstrap standard errors s_i:
    w_i = 1 / s_i^2, Q = sum w_i (d_i - d_FE)^2, tau^2 = max(0, (Q - (k - 1)) / (sum w - sum w^2 / sum w))
    (DerSimonian and Laird 1986); w*_i = 1 / (s_i^2 + tau^2); mean m = sum w*_i d_i / sum w*_i.
    Hartung-Knapp variance V_HK = sum w*_i (d_i - m)^2 / ((k - 1) sum w*_i); the interval and p use Student t on
    k - 1 degrees of freedom (Hartung and Knapp 2001; IntHout, Ioannidis and Borm 2014).
    Default: plain HK, as R1 states. Calibration (tests/test_evalkit_pool.py, 4,000 synthetic meta-analyses per
    shape, k = 3 to 5, s_i 0.01 to 0.03, tau 0 to 0.03): plain HK covers 0 at 0.94 to 0.95; the normal DL interval
    falls to 0.85 to 0.90 when tau > 0. Option `hk_floor=True`: V = max(V_HK, 1 / sum w*_i) (Knapp and Hartung 2003;
    Roever, Knapp and Friede 2015) covers 0.96 to 1.00, too wide, so it is not the default. Known edge: plain HK gives
    V = 0 when every d_i is equal; then p = 0 for a non-zero mean.
Words (R1, C2): NOT ALLOWED (not registered, or fewer than MIN_EVENTS earthquakes pass the gate); then Holm across
every pooled line of one release: with L lines, the line with the j-th smallest p is tested at
alpha_j = 0.05 / (L - j + 1), step-down. POOLED CANNOT TELL: Holm does not reject. HOLDS: Holm rejects, and every
leave-one-out interval at level 1 - alpha_j excludes 0 with the sign of m. FRAGILE: Holm rejects, and some
leave-one-out interval covers 0 or has the other sign.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from ..scorer import Z_ALPHA

MIN_EVENTS = 3
ALPHA = 0.05


def equal_weight(diffs: Sequence[float], level: float = 0.95) -> dict:
    """Equal weight per earthquake (design R2.1; scorecard v2 registration K1): mean of d_i, variance
    sum (d_i - mean)^2 / (k (k - 1)), Student t on k - 1 degrees of freedom. The per-earthquake SEs do not enter.
    Not lab-certified (LAB-CERTIFICATES section 5 covers the random-effects weights only)."""
    d = np.asarray(diffs, dtype=float)
    if d.size < 2 or not np.all(np.isfinite(d)):
        raise ValueError("equal weight needs at least two finite differences")
    mean = float(d.mean())
    se = float(np.sqrt(np.sum((d - mean) ** 2) / (d.size * (d.size - 1))))
    df = d.size - 1
    tq = float(stats.t.ppf(0.5 + level / 2, df))
    p = float(2.0 * stats.t.sf(abs(mean) / se, df)) if se > 0 else (0.0 if mean != 0 else 1.0)
    return {"mean": mean, "tau2": None, "q": None, "k": int(d.size), "method": "equal", "level": level, "se": se,
            "lo": mean - tq * se, "hi": mean + tq * se, "p": p, "df": df}


def random_effects(diffs: Sequence[float], ses: Sequence[float], *, method: str = "hk", hk_floor: bool = False,
                   level: float = 0.95) -> dict:
    """Pooled mean, SE, interval at `level`, two-sided p and tau^2 (module docstring).

    method "hk": Hartung-Knapp t interval on k - 1 degrees of freedom (needs k >= 2); "dl": the normal
    DerSimonian-Laird interval (kept only to show, in the tests, why R1 moved off it).
    """
    d, s = np.asarray(diffs, dtype=float), np.asarray(ses, dtype=float)
    if d.size == 0 or not np.all(s > 0) or not np.all(np.isfinite(d)):
        raise ValueError("random_effects needs at least one finite difference and positive standard errors")
    if method == "equal":
        return equal_weight(d, level)
    if method not in ("hk", "dl"):
        raise ValueError(f"method must be 'hk', 'dl' or 'equal', got {method!r}")
    w = 1.0 / s ** 2
    d_fe = float(np.sum(w * d) / np.sum(w))
    q = float(np.sum(w * (d - d_fe) ** 2))
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (q - (d.size - 1)) / c) if d.size > 1 else 0.0
    ws = 1.0 / (s ** 2 + tau2)
    mean = float(np.sum(ws * d) / np.sum(ws))
    var_dl = float(1.0 / np.sum(ws))
    out = {"mean": mean, "tau2": tau2, "q": q, "k": int(d.size), "method": method, "level": level}
    if method == "dl":
        se = float(np.sqrt(var_dl))
        z = float(stats.norm.ppf(0.5 + level / 2)) if level != 0.95 else Z_ALPHA
        p = float(2.0 * stats.norm.sf(abs(mean) / se))
        return {**out, "se": se, "lo": mean - z * se, "hi": mean + z * se, "p": p, "df": None}
    if d.size < 2:
        raise ValueError("Hartung-Knapp needs at least two earthquakes")
    var_hk = float(np.sum(ws * (d - mean) ** 2) / ((d.size - 1) * np.sum(ws)))
    var = max(var_hk, var_dl) if hk_floor else var_hk
    se = float(np.sqrt(var))
    df = d.size - 1
    tq = float(stats.t.ppf(0.5 + level / 2, df))
    if se > 0:
        p = float(2.0 * stats.t.sf(abs(mean) / se, df))
    else:
        p = 0.0 if mean != 0 else 1.0
    return {**out, "se": se, "se_hk_plain": float(np.sqrt(var_hk)), "hk_floor": hk_floor, "lo": mean - tq * se,
            "hi": mean + tq * se, "p": p, "df": df}


def _sign(r: dict) -> int:
    """+1 or -1 if the interval excludes 0, else 0."""
    return 1 if r["lo"] > 0 else -1 if r["hi"] < 0 else 0


def _prepare(items: Sequence[dict], registered: bool, min_events: int) -> dict:
    keys = {(it["entry_a"], it["entry_b"], it["metric"]) for it in items}
    if len(keys) > 1:
        raise ValueError(f"a pooled line compares two entries on one metric; items name {sorted(keys)}")
    ids = [it["event_id"] for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError("each earthquake enters a pooled line once")
    used = [it for it in items if it["passes_gate"]]
    out = {"comparison": dict(zip(("entry_a", "entry_b", "metric"), next(iter(keys)))) if keys else None,
           "earthquakes_pooled": [it["event_id"] for it in used],
           "earthquakes_excluded": [it["event_id"] for it in items if not it["passes_gate"]],
           "per_earthquake": [{"event_id": it["event_id"], "diff": it["diff"], "se": it["se"]} for it in used]}
    if not registered:
        return {**out, "word": "NOT ALLOWED", "reason": "R1 (a): the pooled line was not registered before the join"}
    if len(used) < min_events:
        return {**out, "word": "NOT ALLOWED",
                "reason": f"R1 (c): {len(used)} earthquake(s) pass the count gate, {min_events} needed"}
    out["_used"] = used
    return out


def pooled_lines(lines: Sequence[Sequence[dict]], *, registered: bool, min_events: int = MIN_EVENTS,
                 alpha: float = ALPHA, hk_floor: bool = False, weighting: str = "re") -> list[dict]:
    """R1 words for every pooled line of one release, Holm across them.

    Each line is a list of per-earthquake items {event_id, entry_a, entry_b, metric, diff, se, passes_gate}; every item
    of a line names the same two entries and metric. Items that fail the count gate are listed, not pooled. A line
    that is NOT ALLOWED takes no place in the Holm family. weighting "re" (default): random effects, Hartung-Knapp;
    "equal": equal weight per earthquake (`equal_weight`, not lab-certified).
    """
    if weighting not in ("re", "equal"):
        raise ValueError(f"weighting must be 're' or 'equal', got {weighting!r}")
    meth = "hk" if weighting == "re" else "equal"
    prepared = [_prepare(items, registered, min_events) for items in lines]
    live = [k for k, p in enumerate(prepared) if "_used" in p]
    for k in live:
        used = prepared[k]["_used"]
        prepared[k]["pooled"] = random_effects([it["diff"] for it in used], [it["se"] for it in used],
                                               hk_floor=hk_floor, method=meth)
    order = sorted(live, key=lambda k: prepared[k]["pooled"]["p"])
    n_live, stopped = len(live), False
    for rank, k in enumerate(order):
        p = prepared[k]
        used = p.pop("_used")
        alpha_j = alpha / (n_live - rank)
        p["holm"] = {"family_size": n_live, "rank": rank + 1, "alpha_j": alpha_j}
        p["label"] = f"across these {len(used)} earthquakes"
        loo = []
        for i, it in enumerate(used):
            rest = used[:i] + used[i + 1:]
            r = random_effects([x["diff"] for x in rest], [x["se"] for x in rest], hk_floor=hk_floor,
                               level=1.0 - alpha_j, method=meth)
            loo.append({"left_out": it["event_id"], **r})
        p["leave_one_out"] = loo
        reject = (not stopped) and p["pooled"]["p"] < alpha_j
        if not reject:
            stopped = True
            p["word"] = "POOLED CANNOT TELL"
            continue
        sign = 1 if p["pooled"]["mean"] > 0 else -1
        p["word"] = "HOLDS" if all(_sign(r) == sign for r in loo) else "FRAGILE"
    return prepared


def pooled_line(items: Sequence[dict], *, registered: bool, min_events: int = MIN_EVENTS,
                hk_floor: bool = False, weighting: str = "re") -> dict:
    """One pooled line alone (a Holm family of one)."""
    return pooled_lines([items], registered=registered, min_events=min_events, hk_floor=hk_floor,
                        weighting=weighting)[0]


def leave_one_group_out(items: Sequence[dict], groups: dict, *, weighting: str = "re", level: float = 0.95,
                        hk_floor: bool = False) -> list[dict]:
    """The pooled mean with each group (for example a region) left out in turn (scorecard v2 registration K1).

    items: per-earthquake items as for pooled_lines (only gate-passing items with a diff enter); groups: event_id ->
    group name. A group whose removal leaves fewer than 2 earthquakes prints NOT ALLOWED.
    """
    if weighting not in ("re", "equal"):
        raise ValueError(f"weighting must be 're' or 'equal', got {weighting!r}")
    meth = "hk" if weighting == "re" else "equal"
    used = [it for it in items if it["passes_gate"] and it["diff"] is not None]
    missing = [it["event_id"] for it in used if it["event_id"] not in groups]
    if missing:
        raise ValueError(f"no group for {missing}")
    out = []
    for g in sorted({groups[it["event_id"]] for it in used}):
        rest = [it for it in used if groups[it["event_id"]] != g]
        if len(rest) < 2:
            out.append({"left_out": g, "k": len(rest), "word": "NOT ALLOWED"})
            continue
        r = random_effects([x["diff"] for x in rest], [x["se"] for x in rest], method=meth, level=level,
                           hk_floor=hk_floor)
        out.append({"left_out": g, **r})
    return out


def card_item(card: dict, entry_a: str, entry_b: str, metric: str) -> dict:
    """One pooled-line item from a card: the paired difference entry_a - entry_b on `metric` ('auc' or 'log_loss')."""
    item = {"event_id": card["task"]["event_id"], "entry_a": entry_a, "entry_b": entry_b, "metric": metric,
            "passes_gate": card["status"] == "SCORED", "diff": None, "se": None}
    if not item["passes_gate"]:
        return item
    for c in card["comparisons"]:
        if c["metric"] != metric or {c["a"], c["b"]} != {entry_a, entry_b}:
            continue
        flip = 1.0 if c["a"] == entry_a else -1.0
        return {**item, "diff": flip * c["diff"], "se": c["se"]}
    raise KeyError(f"card {item['event_id']!r} has no {metric} comparison of {entry_a!r} and {entry_b!r}")

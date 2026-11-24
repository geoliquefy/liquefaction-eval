"""G5 amendment of 2026-11-05: the adopted rulings 25, 26 and 28 in code. Pure numpy and scipy plus shapely geometry.

Implements docs/AMENDMENT-G5-RULINGS-2026-11-05.md (f083587, committed alone before this file). Serves GOALS.md G5.
Imports scoreboard/g5.py and scoreboard/references.py and edits neither. The I/O lives in scripts/g5_score.py.
Reads no data file. No em dashes.

  section 3  tile CRS, mapped land per tile, land weights, land-weighted offset and domain
  section 4  earlier-earthquake constant (the G5 record), reference comparisons, the amended site word
  section 5  class (b) per-earthquake N-test z and information gain, and the pooled test with K80 and power
  section 7  canonical predictions text and hash (absolute 1e-9), and the tolerance comparison
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import binom, nct, norm
from scipy.stats import t as tdist

from . import g5
from . import references as R
from .g5_class_b import MIN_CPT as MIN_CPT_B, MIN_MAG as MIN_MW_B, is_mw_family
from .scorer import _draw, _members

AMENDMENT = "docs/AMENDMENT-G5-RULINGS-2026-11-05.md"
AMENDMENT_COMMIT = "f083587"

# ---------------------------------------------------------------------------------------------------------------
# Section 7: canonical predictions hash
# ---------------------------------------------------------------------------------------------------------------

STR_COLS = frozenset({"cpt_id", "tile", "subarea", "c1_tag"})
CANON_FMT = ".9f"
CANON_ZERO = format(0.0, CANON_FMT)


def canon_cell(s: str) -> str:
    """One float cell: Python float(), then format(x, '.9f'); inf, -inf, nan as such; a rounded zero has no sign."""
    x = float(s)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    t = format(x, CANON_FMT)
    return CANON_ZERO if float(t) == 0.0 else t


def canonical_text(path, str_cols=STR_COLS) -> str:
    """Canonical text of a predictions CSV (amendment section 7.2)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    keep = [c in str_cols for c in head]
    out = [",".join(head)]
    for r in body:
        out.append(",".join(v if k else canon_cell(v) for v, k in zip(r, keep)))
    return "\n".join(out) + "\n"


def canonical_sha256(path, str_cols=STR_COLS) -> str:
    return hashlib.sha256(canonical_text(path, str_cols).encode("utf-8")).hexdigest()


def compare_csv(a, b, str_cols=STR_COLS, tol: float = 1e-9) -> dict:
    """Tolerance check of two predictions CSVs (section 7.4): same rows, columns and strings; largest float gap."""
    def rd(p):
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.reader(f))
    A, B = rd(a), rd(b)
    same = len(A) == len(B) and A[0] == B[0]
    worst, where, n_cells_diff = 0.0, None, 0
    if same:
        keep = [c in str_cols for c in A[0]]
        for i, (ra, rb) in enumerate(zip(A[1:], B[1:])):
            for j, (va, vb, k) in enumerate(zip(ra, rb, keep)):
                if va == vb:
                    continue
                n_cells_diff += 1
                if k:
                    same = False
                    continue
                xa, xb = float(va), float(vb)
                d = 0.0 if (xa == xb or (math.isnan(xa) and math.isnan(xb))) else abs(xa - xb)
                if not math.isfinite(d):
                    d = math.inf
                if d > worst:
                    worst, where = d, {"row": i, "column": A[0][j], "a": va, "b": vb}
    return {"same_rows_columns_strings": bool(same), "cells_differing": n_cells_diff, "max_abs_diff": worst,
            "at": where, "same_prediction": bool(same and worst <= tol), "tol": tol}


# ---------------------------------------------------------------------------------------------------------------
# Section 3: tiles, mapped land, land weights, land-weighted offset
# ---------------------------------------------------------------------------------------------------------------

NZTM = {"lon0": 173.0, "k0": 0.9996, "fe": 1600000.0, "fn": 10000000.0}
CRS_MIN_SHARE = 0.99


def crs_candidates(lat, lon) -> list[dict]:
    """In order: the runner's UTM zone of the mean longitude, then NZTM2000 (section 3.1.3)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    z = g5.utm_zone(float(np.mean(lon)))
    fn = 10000000.0 if float(np.mean(lat)) < 0 else 0.0
    lon0 = -183.0 + 6.0 * z
    return [{"name": f"UTM zone {z}{'S' if fn else 'N'}",
             "proj": lambda la, lo, _l=lon0, _f=fn: g5.tm_forward(la, lo, _l, fn=_f)},
            {"name": "NZTM2000",
             "proj": lambda la, lo: g5.tm_forward(la, lo, NZTM["lon0"], k0=NZTM["k0"], fe=NZTM["fe"], fn=NZTM["fn"])}]


def _tile_ij(tiles):
    out = []
    for t in tiles:
        a, _, b = str(t).partition("_")
        try:
            out.append((int(a), int(b)))
        except ValueError:
            return None
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def tile_crs(tiles, lat, lon, min_share: float = CRS_MIN_SHARE) -> dict | None:
    """The first candidate CRS in which at least min_share of the points lie inside the square their tile names."""
    ij = _tile_ij(tiles)
    if ij is None or len(ij) == 0:
        return None
    for c in crs_candidates(lat, lon):
        x, y = c["proj"](np.asarray(lat, float), np.asarray(lon, float))
        inside = (np.floor(x / g5.TILE_M).astype(np.int64) == ij[:, 0]) & \
                 (np.floor(y / g5.TILE_M).astype(np.int64) == ij[:, 1])
        share = float(inside.mean())
        if share >= min_share:
            return {"name": c["name"], "proj": c["proj"], "share_inside": share}
    return None


def tile_box(t: str):
    import shapely
    a, b = (int(v) for v in str(t).split("_"))
    s = g5.TILE_M
    return shapely.box(a * s, b * s, a * s + s, b * s + s)


def tile_land_km2(domain, tiles) -> dict:
    """{tile: km2 of the domain geometry inside the tile square}; domain None gives the full square (1 km2)."""
    import shapely
    out = {}
    for t in sorted(set(map(str, tiles))):
        bx = tile_box(t)
        if domain is None:
            out[t] = bx.area / 1e6
            continue
        try:
            g = shapely.intersection(domain, bx)
        except shapely.errors.GEOSException:
            g = shapely.intersection(shapely.make_valid(domain), bx, grid_size=0.01)
        out[t] = g.area / 1e6
    return out


def land_weights(tiles, area_km2: dict) -> np.ndarray:
    """w_i = A_t / n_t (experiment 2, W2): tile land split equally among the tile's rows."""
    tiles = np.asarray([str(t) for t in tiles], dtype=object)
    _, inv, cnt = np.unique(tiles, return_inverse=True, return_counts=True)
    a = np.array([area_km2.get(t, 0.0) for t in tiles], float)
    return a / cnt[inv]


def w_offset(y, p, w) -> float:
    y, p, w = (np.asarray(v, float) for v in (y, p, w))
    sw = w.sum()
    if not sw > 0:
        return float("nan")
    return g5._lg((w * y).sum() / sw) - g5._lg((w * p).sum() / sw)


def boot_stats(tiles, fns: dict, *, n_boot: int = g5.N_BOOT, seed: int = g5.SEED) -> dict:
    """Per statistic, the values over the registered tile bootstrap draws (same draws for every statistic, and the
    same draws as g5.event_word: one rng from the seed, _draw over the tile members)."""
    mem = _members(tiles)
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_boot) for k in fns}
    for b in range(n_boot):
        i = _draw(mem, rng)
        for k, f in fns.items():
            out[k][b] = f(i)
    return out


def summarize(point: float, B: np.ndarray) -> dict:
    B = np.asarray(B, float)
    ok = B[np.isfinite(B)]
    sd = float(ok.std(ddof=1)) if ok.size > 1 else float("nan")
    return {"value": float(point), "ci95": [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))]
            if ok.size else [float("nan")] * 2, "se": sd, "mdd": 2.8 * sd, "n_draws_finite": int(ok.size)}


def land_block(y, p0, tiles, w, *, a_obs: float, n_boot: int = g5.N_BOOT, seed: int = g5.SEED) -> dict:
    """Land-weighted offset, its interval and SE, and a_obs - a_land paired on the same draws (section 3.1, 3.2)."""
    y, p0, w = (np.asarray(v, float) for v in (y, p0, w))
    a_land = w_offset(y, p0, w)
    B = boot_stats(tiles, {"cpt": lambda i: g5.offset(y[i], p0[i]), "land": lambda i: w_offset(y[i], p0[i], w[i])},
                   n_boot=n_boot, seed=seed)
    return {"a_land": summarize(a_land, B["land"]), "a_obs_same_draws": summarize(a_obs, B["cpt"]),
            "cpt_minus_land": summarize(a_obs - a_land, B["cpt"] - B["land"]),
            "land_share_obs": float((w * y).sum() / w.sum()), "land_share_p0": float((w * p0).sum() / w.sum()),
            "cpt_share_obs": float(y.mean())}


# ---------------------------------------------------------------------------------------------------------------
# Section 4: the G5 record, the earlier-earthquake constant, the reference comparisons and the site word
# ---------------------------------------------------------------------------------------------------------------

# Frozen by the amendment (section 4.2): label set A, train/val rows, counts from experiment 4 gate 2 (6fccee8).
RECORD_TRAIN = (
    {"event": "sep2010", "origin_utc": "2010-09-03T16:35:47Z", "n_yes": 902, "n": 2700, "source": "M0 training, set A"},
    {"event": "feb2011", "origin_utc": "2011-02-21T23:51:42Z", "n_yes": 1309, "n": 1429, "source": "M0 training, set A"},
    {"event": "jun2011", "origin_utc": "2011-06-13T02:20:49Z", "n_yes": 2864, "n": 3351, "source": "M0 training, set A"},
    {"event": "dec2011", "origin_utc": "2011-12-23T02:18:03Z", "n_yes": 1447, "n": 3265, "source": "M0 training, set A"},
)
RECORD_TAGS = ("PROSPECTIVE", "RETROSPECTIVE")


def record_from_scores(score_jsons: list[dict]) -> list[dict]:
    """Record rows from committed G5 score JSONs: PROSPECTIVE and RETROSPECTIVE only; DRY RUN never enters."""
    out = []
    for J in score_jsons:
        if J.get("tag") not in RECORD_TAGS:
            continue
        Q = J.get("gate4_qualification", {})
        if not Q.get("n"):
            continue
        out.append({"event": J["event"]["event_id"], "origin_utc": J["event"]["date_utc"], "n_yes": int(Q["n_yes"]),
                    "n": int(Q["n"]), "source": f"G5 score {J['tag']}"})
    return out


def earlier_constant(origin_utc: str, extra: list[dict] | None = None) -> dict:
    """Pooled CPT-weighted yes share over every record earthquake with an origin strictly before origin_utc."""
    rows = [r for r in list(RECORD_TRAIN) + list(extra or []) if r["origin_utc"] < origin_utc]
    n = sum(r["n"] for r in rows)
    k = sum(r["n_yes"] for r in rows)
    return {"c": (k / n) if n else None, "n_yes": k, "n": n, "earthquakes": [r["event"] for r in rows]}


def compare_refs(y, p_model, tiles, c: float | None) -> dict:
    """Paired log-loss comparisons of one model with the coin and with the constant (references.beats)."""
    y = np.asarray(y, float)
    out = {"logloss": float(R.loss(y, p_model).mean()), "coin": R.beats(y, p_model, R.coin(len(y)), tiles)}
    out["const"] = R.beats(y, p_model, np.full(len(y), c), tiles) if c is not None else None
    return out


def site_word(name: str, refs: dict, l0_word: str | None) -> str:
    """Amended site word (section 4.4)."""
    w = {k: (refs[k]["word"] if refs.get(k) else None) for k in ("coin", "const")}
    if w["coin"] == "LOSES" or w["const"] == "LOSES":
        word = f"{name} LOSES TO A REFERENCE"
    elif w["coin"] == "BEATS" and w["const"] == "BEATS":
        word = f"{name} BEATS BOTH"
    else:
        word = "CANNOT TELL"
    return word


# ---------------------------------------------------------------------------------------------------------------
# Section 5: class (b) per-earthquake numbers and the pooled test
# ---------------------------------------------------------------------------------------------------------------

SE_FALLBACK = 0.27
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(9)
GH_W = GH_W / GH_W.sum()
# MIN_MW_B, MIN_CPT_B and the Mw family come from scoreboard/g5_class_b.py (the watcher rule, mwp left out)
STANDING_PROTOCOL = None  # named by a later founder ruling (section 5.1.3); until then no earthquake enters

# Experiment 6 class (b) (results/G5-TEST-CLASS-2026-11-02.json, power.b): K80 and the per-earthquake mean and SD
# behind the power statement. N-test under S1, T-test under S0. A unit test checks these against the JSON.
K80 = {"M0": {"n": 29, "t": 10}, "M1": {"n": 25, "t": 9}, "M0-land": {"n": 21, "t": 15}, "M2": {"n": None, "t": None}}
POWER_PARAMS = {
    "M0": {"mean_z": -0.5136933735620087, "sd_z": 0.9672166101499233,
           "mean_ig": 0.44498653196832527, "sd_ig": 0.4777864039498242},
    "M1": {"mean_z": -0.5631474332553589, "sd_z": 0.9437299277211116,
           "mean_ig": 1.2169410271548304, "sd_ig": 1.2329309921436429},
    "M0-land": {"mean_z": -0.610713195498822, "sd_z": 0.9268201879297465,
                "mean_ig": 0.17666665990041672, "sd_ig": 0.252413533402441},
}


def watcher_radius_km(mw: float) -> float:
    """scripts/g5_watch.py radius_km, copied (the watcher module reads files at import)."""
    return float(min(200.0, 30.0 * 10 ** (0.5 * (mw - 6.0))))


def class_b_entry(event: dict, comcat: dict | None, lat, lon) -> dict:
    """Section 5.1 entry, from event.json, package.json's ComCat row and the package CPT locations."""
    reasons = []
    mw = comcat.get("mag") if comcat else None
    mtype = str(comcat.get("mag_type", "")).lower() if comcat else ""
    if comcat is None:
        reasons.append("no ComCat row in package.json")
    elif not (is_mw_family(mtype) and mw is not None and float(mw) >= MIN_MW_B):
        reasons.append(f"ComCat {mtype} {mw} is not Mw-family >= {MIN_MW_B}")
    if "mw" in (event.get("deviations") or []):
        reasons.append("declared mw deviation")
    n_r, R_km = None, None
    if comcat is not None and mw is not None:
        R_km = watcher_radius_km(float(mw))
        d = g5.haversine_km(float(comcat["lat"]), float(comcat["lon"]), np.asarray(lat, float), np.asarray(lon, float))
        n_r = int((np.hypot(d, max(float(comcat.get("depth_km") or 0.0), 0.0)) <= R_km).sum())
        if n_r < MIN_CPT_B:
            reasons.append(f"{n_r} package CPTs within {R_km:.1f} km (< {MIN_CPT_B})")
    if STANDING_PROTOCOL is None:
        reasons.append("no standing grading protocol named yet (section 5.1.3)")
    elif event.get("label_protocol") != STANDING_PROTOCOL:
        reasons.append("label protocol is not the standing protocol")
    if event.get("tag") != "PROSPECTIVE":
        reasons.append(f"tag {event.get('tag')} (headline needs PROSPECTIVE)")
    return {"entry": not reasons, "reasons": reasons, "comcat_mw": mw, "comcat_mag_type": mtype or None,
            "radius_km": R_km, "n_package_cpt_within_radius": n_r}


def v_from_hash(h: str) -> float:
    """V for the randomized PIT, fixed by the canonical predictions hash before labels (section 5.2)."""
    return float(np.random.default_rng(int(h[:16], 16)).random())


def pit_z(n_yes: int, n: int, mean_p0b: float, q, s: float, v: float) -> dict:
    """Randomized PIT of n_yes under Binomial(n, expit(logit(mean p0_B) + a + e)), a over q[25::50], e ~ N(0, s^2)
    on the 9-point Gauss-Hermite grid (experiment 6, registration 67a52e8 section 7.3)."""
    q200 = np.asarray(q, float)[25::50]
    L = float(logit(np.clip(mean_p0b, *g5.CLIP)))
    r = expit(L + q200[:, None] + s * GH_X[None, :])
    w = np.broadcast_to(GH_W[None, :] / len(q200), r.shape)
    lt = float((w * binom.cdf(n_yes - 1, n, r)).sum())
    eq = float((w * binom.pmf(n_yes, n, r)).sum())
    pit = float(np.clip(lt + v * eq, 1e-12, 1 - 1e-12))
    return {"pit": pit, "z": float(norm.ppf(pit)), "p_lt": lt, "p_eq": eq, "s": s, "n_q": int(q200.size)}


def info_gain(y, p0, p_band) -> float:
    """mean over rows of [log loss(Level 0) - log loss(band)]; positive = band better."""
    return float((g5.log_loss(y, p0) - g5.log_loss(y, p_band)).mean())


def stouffer_power(mean_z: float, sd_z: float, K: int, crit: float = 1.959963984540054) -> float:
    m = mean_z * math.sqrt(K)
    return float(norm.cdf((-crit - m) / sd_z) + 1.0 - norm.cdf((crit - m) / sd_z))


def t_power(mean_ig: float, sd_ig: float, K: int) -> float | None:
    if K < 2:
        return None
    tc = tdist.ppf(0.975, K - 1)
    d = mean_ig / sd_ig * math.sqrt(K)
    return float(1 - nct.cdf(tc, K - 1, d) + nct.cdf(-tc, K - 1, d))


def stouffer(z) -> dict:
    z = np.asarray(z, float)
    K = z.size
    Z = float(z.sum() / math.sqrt(K)) if K else float("nan")
    return {"K": int(K), "Z": Z, "p_two_sided": float(2 * (1 - norm.cdf(abs(Z)))) if K else None}


def ig_ttest(ig) -> dict:
    ig = np.asarray(ig, float)
    K = ig.size
    if K < 2:
        return {"K": int(K), "mean": float(ig.mean()) if K else None, "sd": None, "t": None, "t_crit": None, "p": None}
    m, s = float(ig.mean()), float(ig.std(ddof=1))
    t = (m / (s / math.sqrt(K))) if s > 0 else (math.inf if m > 0 else -math.inf if m < 0 else 0.0)
    tc = float(tdist.ppf(0.975, K - 1))
    p = float(2 * (1 - tdist.cdf(abs(t), K - 1))) if math.isfinite(t) else 0.0
    return {"K": int(K), "mean": m, "sd": s, "t": t, "t_crit": tc, "p": p}


def n_word(Z: float, band: str) -> str:
    return f"N-TEST REJECTS {band}" if abs(Z) > 1.959963984540054 else "N-TEST CONSISTENT"


def t_word(T: dict) -> str:
    if T["t"] is None:
        return "CANNOT TELL"
    return "GAINS OVER LEVEL 0" if T["t"] > T["t_crit"] else "LOSES TO LEVEL 0" if T["t"] < -T["t_crit"] else \
        "CANNOT TELL"


POOL_MODES = {"headline": ("PROSPECTIVE",), "beside": ("RETROSPECTIVE",), "demo": ("DRY RUN",)}


def pool(score_jsons: list[dict], mode: str = "headline", bands=("M0", "M1", "M0-land", "M2")) -> dict:
    """Section 5.3. Per band: Stouffer N-test and IG t-test over the pooled earthquakes (origin-time order), K80,
    power at K, and the verdict read when K first reaches K80. headline: PROSPECTIVE with class (b) entry only;
    beside: RETROSPECTIVE; demo: DRY RUN (says it counts for nothing). Other tags are refused, never pooled."""
    if mode not in POOL_MODES:
        raise ValueError(f"mode must be one of {sorted(POOL_MODES)}")
    tags = POOL_MODES[mode]
    used, refused = [], []
    for J in score_jsons:
        cb = J.get("class_b")
        name = J.get("event", {}).get("event_id")
        if J.get("tag") not in tags:
            refused.append({"event": name, "tag": J.get("tag"), "why": f"tag not pooled in mode {mode}"})
            continue
        if cb is None:
            refused.append({"event": name, "tag": J.get("tag"), "why": "no class (b) numbers"})
            continue
        if mode == "headline" and not cb["entry"]["entry"]:
            refused.append({"event": name, "tag": J.get("tag"), "why": "; ".join(cb["entry"]["reasons"])})
            continue
        used.append(J)
    used.sort(key=lambda J: J["event"]["date_utc"])
    out = {"mode": mode, "note": "DRY RUN, counts for nothing" if mode == "demo" else None,
           "earthquakes": [J["event"]["event_id"] for J in used], "refused": refused, "bands": {}}
    for b in bands:
        zs = [J["class_b"]["bands"][b]["z"] for J in used if J["class_b"]["bands"].get(b, {}).get("z") is not None]
        igs = [J["class_b"]["bands"][b]["ig"] for J in used if b in J["class_b"]["bands"]]
        k80 = K80.get(b, {"n": None, "t": None})
        P = POWER_PARAMS.get(b)
        S, T = stouffer(zs), ig_ttest(igs)
        rec = {"n_test": S, "t_test": T, "k80": k80,
               "power_at_K": {"n": stouffer_power(P["mean_z"], P["sd_z"], S["K"]) if P and S["K"] else None,
                              "t": t_power(P["mean_ig"], P["sd_ig"], T["K"]) if P else None}}
        if k80["n"] is None:
            rec["n_verdict"] = "NO K80 REGISTERED" if b == "M2" else "BELOW K80: NO VERDICT"
        elif S["K"] < k80["n"]:
            rec["n_verdict"] = "BELOW K80: NO VERDICT"
        else:
            rec["n_verdict"] = n_word(stouffer(zs[:k80["n"]])["Z"], b)
            rec["n_read_at_K80"] = stouffer(zs[:k80["n"]])
        if k80["t"] is None:
            rec["t_verdict"] = "NO K80 REGISTERED" if b == "M2" else "BELOW K80: NO VERDICT"
        elif T["K"] < k80["t"]:
            rec["t_verdict"] = "BELOW K80: NO VERDICT"
        else:
            T80 = ig_ttest(igs[:k80["t"]])
            rec["t_verdict"] = t_word(T80)
            rec["t_read_at_K80"] = T80
        out["bands"][b] = rec
    return out


def read_json_files(paths) -> list[dict]:
    import json
    return [json.loads(Path(p).read_text()) for p in paths]


def _buf_csv(rows: list[list[str]]) -> str:
    """Helper for tests: CSV text with the csv module's default dialect."""
    s = io.StringIO()
    csv.writer(s, lineterminator="\n").writerows(rows)
    return s.getvalue()

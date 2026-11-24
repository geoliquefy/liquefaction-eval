"""Standard baseline "baseline: effort only" (eval plan step 1c; design R2.2 items 1 and 3). No em dashes.

Registered by `docs/evals/EFFORT-BASELINE-REGISTRATION-2026-11-19.md` (pillar root), committed before this file.
A model built only from where people looked: it must never see soil, water, geology, shaking or the scored
earthquake's labels. Three guards, each tested:

1. Allowlist: `check_features` refuses any feature name outside EFFORT_FEATURES (pre-event only). The old
   2d1bfc6 set lives on as PROXY_FEATURES, "diagnostic: investigation-density proxy", never a verdict baseline.
2. Leave one earthquake out: `fit_predict_loeo` drops the scored earthquake's labels before anything else reads
   the label mapping, and fits on the other earthquakes only.
3. Fixed model: L2 logistic regression on standardised features, lambda fixed; no search, no selection. A feature
   with zero variance in the training rows is dropped (fixed rule, reported).

Pure numpy and scipy.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from . import metrics as M

NAME = "baseline: effort only"
PROXY_NAME = "diagnostic: investigation-density proxy"
ALLOWLIST_NAME = "pre-event"          # recorded in the entry manifest (review 2 N3); the card slot requires it
PROXY_ALLOWLIST_NAME = "proxy"
LAMBDA = 1.0
TOL = 1e-10
EXPOSURE = "seen_place"          # the LOEO fit sees other earthquakes' labels at the same sites (PLACE exposure)
# The baseline: only features measured without reference to the damage (pre-event registration, review 1 B1).
EFFORT_FEATURES = {
    "builtup_share_250m": "share of the 250 m disc in a pre-event built-up land-cover class; 0 to 1",
    "tree_share_250m": "share of the 250 m disc in pre-event woody land-cover classes; 0 to 1",
    "in_aerial_footprint": "inside the union of all aerial-map classes of the earthquake; 0 / 1",
    "pre_cpt_count_1000m": "CPTs pushed before the first earthquake within 1000 m; log(1 + n)",
    "pre_cpt_nn_dist_m": "distance to the nearest CPT pushed before the first earthquake; log(1 + min(d, 5000))",
}
# The 2d1bfc6 set: post-event CPT density and the pushed-after flag descend from the outcome. Diagnostic only
# (verification bias); never a verdict baseline, and the card's effort slot refuses it.
PROXY_FEATURES = {
    "recon_road_dist_m": "distance to the nearest reconnaissance road-observation segment; log(1 + min(d, 5000))",
    "builtup_share_250m": "share of the 250 m disc in a pre-event built-up land-cover class; 0 to 1",
    "in_aerial_footprint": "inside the union of all aerial-map classes of the earthquake; 0 / 1",
    "cpt_nn_dist_m": "distance to the nearest other CPT in the national index; log(1 + min(d, 5000))",
    "cpt_count_250m": "other CPTs within 250 m; log(1 + n)",
    "cpt_count_1000m": "other CPTs within 1000 m; log(1 + n)",
    "pushed_after_event": "the CPT was pushed after the earthquake; 0 / 1",
}
# B7 (eval M4), known limit N4: the same pre-event set with the footprint taken from the imagery extent, not the
# union of the mapped classes (evalkit.placement.imagery_footprint_feature builds it). Manifest name
# "pre-event-imagery"; the card's effort slot takes it only with effort_imagery_footprint=True.
FOOTPRINT_ALLOWLIST_NAME = "pre-event-imagery"
FOOTPRINT_FEATURES = {
    **{k: v for k, v in EFFORT_FEATURES.items() if k != "in_aerial_footprint"},
    "in_imagery_footprint": "inside the extent of the earthquake's post-event aerial imagery; 0 / 1",
}


class EffortRefused(ValueError):
    pass


def allowlist_name(allowlist: Mapping[str, str]) -> str:
    """The manifest name of one of the three allowlists, matched by identity."""
    for a, name in ((EFFORT_FEATURES, ALLOWLIST_NAME), (PROXY_FEATURES, PROXY_ALLOWLIST_NAME),
                    (FOOTPRINT_FEATURES, FOOTPRINT_ALLOWLIST_NAME)):
        if allowlist is a:
            return name
    raise EffortRefused("allowlist must be EFFORT_FEATURES or PROXY_FEATURES (or FOOTPRINT_FEATURES, B7)")


def check_features(names: Sequence[str], allowlist: Mapping[str, str] = EFFORT_FEATURES) -> tuple[str, ...]:
    """Refuse any feature that is not on the allowlist (default: the pre-event effort allowlist)."""
    allowlist_name(allowlist)
    names = tuple(names)
    if not names:
        raise EffortRefused("no features")
    bad = [n for n in names if n not in allowlist]
    if bad:
        raise EffortRefused(f"not an effort feature (allowlist): {bad}")
    if len(set(names)) != len(names):
        raise EffortRefused("a feature is named twice")
    return names


@dataclass(frozen=True)
class Fit:
    features: tuple[str, ...]      # used (after the zero-variance rule)
    dropped: tuple[str, ...]
    mean: np.ndarray
    sd: np.ndarray
    intercept: float
    coef: np.ndarray               # standardised
    n_train: int
    n_yes_train: int

    def predict(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        Z = np.column_stack([(np.asarray(X[f], float) - m) / s for f, m, s in zip(self.features, self.mean, self.sd)]
                            ) if self.features else np.zeros((len(next(iter(X.values()))), 0))
        return 1.0 / (1.0 + np.exp(-(self.intercept + Z @ self.coef)))

    def summary(self) -> dict:
        return {"features": list(self.features), "dropped_zero_variance": list(self.dropped),
                "intercept": self.intercept, "coef_standardised": dict(zip(self.features, self.coef.tolist())),
                "n_train": self.n_train, "n_yes_train": self.n_yes_train, "lambda": LAMBDA}


def fit(X: Mapping[str, np.ndarray], y: np.ndarray, allowlist: Mapping[str, str] = EFFORT_FEATURES) -> Fit:
    """Fixed L2 logistic regression: mean log loss + LAMBDA / n * sum(beta^2), standardised features."""
    names = check_features(list(X), allowlist)
    y = np.asarray(y, float)
    if y.ndim != 1 or not np.isin(y, (0.0, 1.0)).all():
        raise EffortRefused("training labels must be 0 / 1")
    if y.size == 0 or y.min() == y.max():
        raise EffortRefused("training labels need both classes")
    cols, used, dropped, mu, sd = [], [], [], [], []
    for f in names:
        v = np.asarray(X[f], float)
        if v.shape != y.shape or not np.all(np.isfinite(v)):
            raise EffortRefused(f"{f}: shape or non-finite values")
        s = float(v.std())
        if s == 0.0:
            dropped.append(f)
            continue
        m = float(v.mean())
        cols.append((v - m) / s); used.append(f); mu.append(m); sd.append(s)
    Z = np.column_stack(cols) if cols else np.zeros((y.size, 0))
    n = y.size

    def obj(w):
        b, beta = w[0], w[1:]
        eta = b + Z @ beta
        ll = np.mean(np.logaddexp(0.0, eta) - y * eta)
        p = 1.0 / (1.0 + np.exp(-eta))
        g0 = np.mean(p - y)
        g = Z.T @ (p - y) / n + 2.0 * LAMBDA / n * beta
        return ll + LAMBDA / n * float(beta @ beta), np.concatenate([[g0], g])

    w0 = np.zeros(1 + Z.shape[1])
    w0[0] = np.log(y.mean() / (1.0 - y.mean()))
    r = minimize(obj, w0, jac=True, method="L-BFGS-B", options={"gtol": TOL, "ftol": TOL, "maxiter": 10000})
    if not r.success:
        raise EffortRefused(f"fit did not converge: {r.message}")
    return Fit(tuple(used), tuple(dropped), np.array(mu), np.array(sd), float(r.x[0]), r.x[1:].copy(), n,
               int(y.sum()))


def fit_predict_loeo(features: Mapping[str, Mapping[str, np.ndarray]], labels: Mapping[str, np.ndarray],
                     scored: str, allowlist: Mapping[str, str] = EFFORT_FEATURES) -> tuple[np.ndarray, Fit]:
    """Leave one earthquake out. features[e][name] and labels[e] are arrays over the same sites for earthquake e
    (labels 1, 0, or -1 for unlabelled). The scored earthquake's labels are removed before any other use; the fit
    sees only the other earthquakes' labelled rows. Returns p for every site of `scored` and the fit."""
    train_events = [e for e in labels if e != scored]          # the scored labels go no further than this line
    if scored not in features:
        raise EffortRefused(f"no features for the scored earthquake {scored!r}")
    if not train_events:
        raise EffortRefused("leave one earthquake out needs at least one other earthquake")
    names = check_features(list(features[scored]), allowlist)
    Xtr = {f: [] for f in names}
    ytr = []
    for e in train_events:
        if set(features[e]) != set(names):
            raise EffortRefused(f"{e}: feature names differ from the scored earthquake's")
        lab = np.asarray(labels[e])
        k = lab >= 0
        ytr.append(lab[k].astype(float))
        for f in names:
            Xtr[f].append(np.asarray(features[e][f], float)[k])
    model = fit({f: np.concatenate(v) for f, v in Xtr.items()}, np.concatenate(ytr), allowlist)
    return model.predict(features[scored]), model


def stratified_auc(y: np.ndarray, s: np.ndarray, strata: np.ndarray) -> dict:
    """AUC inside strata, combined with weights n_yes,k * n_no,k (pairs inside a stratum). Strata lacking a class
    are skipped and counted. Design R2.2 item 3."""
    y = np.asarray(y).astype(bool)
    num, den, used, skipped = 0.0, 0.0, 0, 0
    for k in np.unique(strata):
        m = strata == k
        ny, nn = int(y[m].sum()), int((~y[m]).sum())
        if ny == 0 or nn == 0:
            skipped += 1
            continue
        w = float(ny) * nn
        num += w * M.auc(y[m], s[m]); den += w; used += 1
    return {"auc": num / den if den else float("nan"), "strata_used": used, "strata_skipped": skipped}


def entry(site_ids: Sequence[str], p: np.ndarray, event_id: str, statement: str,
          allowlist: Mapping[str, str] = EFFORT_FEATURES):
    """The baseline as a kit Entry for card.build_card(..., effort_only=...): ranking only (other_probability).

    The manifest records the allowlist the predictions were fitted from (review 2 N3): "pre-event" for
    EFFORT_FEATURES, "proxy" for PROXY_FEATURES, "pre-event-imagery" for FOOTPRINT_FEATURES (B7). The card's
    effort slot accepts "pre-event" by default, and only "pre-event-imagery" with effort_imagery_footprint=True.
    """
    from .spec import Entry, EntryManifest
    name = allowlist_name(allowlist)
    m = EntryManifest(model=NAME, output_kind="other_probability", statement=statement,
                      exposure={event_id: EXPOSURE}, house=False, feature_allowlist=name)
    return Entry(m, tuple(site_ids), np.asarray(p, float))

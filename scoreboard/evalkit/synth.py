"""Synthetic earthquakes for the eval kit: gates, tests and the demo card. Reads no data file. No em dashes.

The generator has two stages, so the kit can show what goes wrong when a probability of one event is scored against
labels of another (rule book R3):

    triggering      eta_t = a + b s + field(x, y),   p_trigger = expit(eta_t)
    manifestation   y = 1 only if triggered and a fragility draw succeeds,   p_target = fragility * p_trigger

`s` is a site susceptibility that models see with noise; `field` is a smooth spatial term (bilinear on a coarse grid
of side `field_cell_m`) that no model sees, so residuals are spatially correlated over about that distance.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from .spec import Entry, EntryManifest, Labels, Task, site_list_sha256

DEMO_SEED = 20261115
DEMO_EVENT = "synthetic-demo-1"


def spatial_field(x_m: np.ndarray, y_m: np.ndarray, rng: np.random.Generator, *, cell_m: float, sd: float,
                  extent_m: float) -> np.ndarray:
    """Gaussian values on a square grid of side cell_m, bilinearly interpolated at the sites."""
    n = int(np.ceil(extent_m / cell_m)) + 2
    g = rng.normal(0.0, sd, (n, n))
    fx, fy = np.asarray(x_m) / cell_m, np.asarray(y_m) / cell_m
    i, j = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - i, fy - j
    return (g[i, j] * (1 - tx) * (1 - ty) + g[i + 1, j] * tx * (1 - ty)
            + g[i, j + 1] * (1 - tx) * ty + g[i + 1, j + 1] * tx * ty)


def synthetic_event(n: int, seed: int, *, extent_m: float = 20000.0, field_cell_m: float = 3000.0,
                    field_sd: float = 0.8, intercept: float = -0.5, slope: float = 1.2, fragility: float = 0.6,
                    unlabelled_share: float = 0.0) -> dict:
    """One synthetic earthquake: site ids, coordinates, the true terms and labels (nan = not labelled)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, extent_m, n)
    y = rng.uniform(0.0, extent_m, n)
    s = rng.normal(0.0, 1.0, n)
    field = spatial_field(x, y, rng, cell_m=field_cell_m, sd=field_sd, extent_m=extent_m)
    p_trigger = expit(intercept + slope * s + field)
    triggered = rng.random(n) < p_trigger
    manifest = triggered & (rng.random(n) < fragility)
    label = manifest.astype(float)
    label[rng.random(n) < unlabelled_share] = np.nan
    return {"site_ids": tuple(f"syn:{k:05d}" for k in range(n)), "x_m": x, "y_m": y, "s": s, "field": field,
            "p_trigger": p_trigger, "p_target": fragility * p_trigger, "label": label,
            "intercept": intercept, "slope": slope, "fragility": fragility}


def labels_of(ev: dict) -> Labels:
    return Labels(site_ids=ev["site_ids"], label=ev["label"], x_m=ev["x_m"], y_m=ev["y_m"])


def manifest(model: str, kind: str, event_id: str, *, house: bool = False, exposure: str = "unseen",
             conversion_id: str | None = None, statement: str | None = None) -> EntryManifest:
    """A short manifest for synthetic entries."""
    return EntryManifest(model=model, output_kind=kind, statement=statement or f"synthetic {kind} for tests",
                         exposure={event_id: exposure}, house=house, conversion_id=conversion_id)


def demo_inputs(seed: int = DEMO_SEED, n: int = 1500, n_boot: int = 2000) -> tuple[Task, list[Entry], Labels, Entry]:
    """The built-in synthetic earthquake behind `python -m scoreboard.evalkit demo`.

    Four entries and a shaking-only baseline. The house model has exact zeros (as Level 0 does), the house copy is a
    monotone transform of it (the duplicate detector must join them), the published triggering model is a probability
    of another event (ranking only), and the published index is a score (ranking only).
    """
    ev = synthetic_event(n, seed, unlabelled_share=0.1)
    rng = np.random.default_rng(seed + 1)
    ids, s = ev["site_ids"], ev["s"]
    s_house = s + rng.normal(0.0, 0.6, n)
    p_house = ev["fragility"] * expit(ev["intercept"] + ev["slope"] * s_house)
    p_house[s_house < -1.5] = 0.0
    p_copy = expit(2.0 * np.log(np.clip(p_house, 1e-6, 1) / np.clip(1 - p_house, 1e-6, 1)))
    p_copy[p_house == 0.0] = 0.0
    p_trig = expit(ev["intercept"] + ev["slope"] * (s + rng.normal(0.0, 0.8, n)))
    index = s + rng.normal(0.0, 0.7, n)
    shaking = np.clip(0.18 + 0.12 * (ev["field"] + rng.normal(0.0, 0.5, n)), 0.02, 0.9)
    e = DEMO_EVENT
    entries = [
        Entry(manifest("House manifestation model", "target_probability", e, house=True,
                       statement="probability of minor or worse ejecta at the site"), ids, p_house),
        Entry(manifest("House model, steeper copy", "target_probability", e, house=True, exposure="fitted_on",
                       statement="the house model with its logit doubled"), ids, p_copy),
        Entry(manifest("Published triggering model", "other_probability", e, exposure="unknown",
                       statement="probability of triggering at depth, not of surface ejecta"), ids, p_trig),
        Entry(manifest("Published index", "score", e, exposure="seen_inputs",
                       statement="a severity index; higher means more likely"), ids, index),
    ]
    shaking_only = Entry(manifest("baseline: shaking only", "target_probability", e,
                                  statement="yes share by shaking bin from earlier earthquakes"), ids, shaking)
    task = Task(event_id=e, target="minor or worse ejecta seen at the site after the synthetic earthquake",
                label_source="synthetic label source v1", label_rule_sha256="0" * 64,
                site_list_sha256=site_list_sha256(ids), track="forward", division="closed", tile_m=2000.0,
                clip=1e-6, seed=seed, label_caveats="synthetic labels drawn from the generator; no survey",
                n_boot=n_boot, census=True, earlier_base_rate=0.2, earlier_label_sets=1)
    return task, entries, labels_of(ev), shaking_only

"""Blind queue: from a local site list to the only fields a grader may see.

Allow-list, not deny-list: a queue item keeps queue_id, x, y and earthquake. Every other field (CPT name, row,
stratum, score band, weight, inclusion probability, seed flag, site code, any label) is dropped. A packet check
scans the built folder for the dropped field names and for known label words and stops the build if one appears.
"""
from __future__ import annotations

import json
from pathlib import Path

ALLOWED = ("queue_id", "x", "y", "earthquake")

# Field names that must never reach a packet (from the audit sample and seed files, and the inventory sources).
FORBIDDEN_KEYS = (
    "cpt", "row", "row_id", "stratum", "placement", "density", "score_band", "band", "weight",
    "inclusion_probability", "N_h", "n_h", "pi", "in_first_24", "seed_order", "site_code", "is_seed",
    "label", "labels", "y_true", "liquefied", "ejecta_observed", "severity_observed", "answer", "answer_key",
    "lsn", "m0", "level0", "zhu", "gem", "score", "model", "prediction",
)
# Inventory source names that must not appear in packet text (section 4.1 item 3 of the audit design).
FORBIDDEN_WORDS = ("CGD0200", "CGD0300", "CGD0400", "PRJ-2937", "liq_polygons", "Green et al")

# One window per earthquake: grade only ejecta that first appears after this earthquake and before the next one.
EARTHQUAKES = {
    "sep2010": {"name": "Darfield, 4 Sep 2010", "date": "2010-09-04", "next": "2011-02-22"},
    "feb2011": {"name": "Christchurch, 22 Feb 2011", "date": "2011-02-22", "next": "2011-06-13"},
    "jun2011": {"name": "13 Jun 2011", "date": "2011-06-13", "next": "2011-12-23"},
    # Dec 2011 has no later earthquake in the frame; its window closes at the 2 Jan 2012 aftershocks (M5.5).
    "dec2011": {"name": "23 Dec 2011", "date": "2011-12-23", "next": "2012-01-02"},
}


class BlindError(ValueError):
    pass


def blind_item(unit: dict) -> dict:
    missing = [k for k in ALLOWED if k not in unit]
    if missing:
        raise BlindError(f"site list item lacks {missing}")
    if unit["earthquake"] not in EARTHQUAKES:
        raise BlindError(f"unknown earthquake {unit['earthquake']!r}")
    return {"queue_id": str(unit["queue_id"]), "x": float(unit["x"]), "y": float(unit["y"]),
            "earthquake": unit["earthquake"]}


def blind_queue(units: list[dict], seeds: list[dict] | None = None) -> list[dict]:
    """One queue of audit units plus the flagged seed units, sorted by queue id (the registered shuffled order).

    Seeds count only when `in_first_24` is true (audit design 4.5). The seed flag itself is dropped, so a
    seed looks exactly like an audit unit.
    """
    items = [blind_item(u) for u in units]
    for s in seeds or []:
        if s.get("in_first_24", True):
            items.append(blind_item(s))
    ids = [i["queue_id"] for i in items]
    if len(set(ids)) != len(ids):
        raise BlindError("duplicate queue id")
    return sorted(items, key=lambda i: i["queue_id"])


def load_site_list(path) -> list[dict]:
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict):
        d = d.get("units") or d.get("sites") or d.get("items")
    if not isinstance(d, list):
        raise BlindError(f"{path}: expected a list of site items")
    return d


def check_packet(folder) -> list[str]:
    """Return every problem found; an empty list means the packet folder is blind."""
    problems = []
    for p in Path(folder).rglob("*.json"):
        text = p.read_text()
        try:
            obj = json.loads(text)
        except ValueError:
            problems.append(f"{p.name}: not JSON")
            continue
        for key in _keys(obj):
            if key in FORBIDDEN_KEYS:
                problems.append(f"{p.name}: forbidden key {key!r}")
        for w in FORBIDDEN_WORDS:
            if w.lower() in text.lower():
                problems.append(f"{p.name}: forbidden word {w!r}")
    return problems


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)

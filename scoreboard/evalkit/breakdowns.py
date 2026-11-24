"""B6 (eval M4): breakdowns the field reads, as card tables that name no winner. No em dashes.

The optional card section `breakdowns` splits the card's scored sites four ways that an engineer reads:

  pga       by PGA bin (g), cells [lo, hi): below 0.10, 0.10 to 0.20, ... , 0.50 and above (PGA_EDGES_G)
  ic        by soil behaviour type index Ic class, Robertson SBTn zone edges 1.31, 2.05, 2.60, 2.95, 3.60
            (Robertson 1990; Robertson 2009, Can. Geotech. J. 46: zones 7 to 2)
  crust     by non-liquefiable crust thickness (m), cells below 1, 1 to 2, 2 to 3, 3 to 5, 5 and above
  severity  by severity class: the card's sites relabelled as "any ejecta" (class 1 or worse) and as
            "moderate or worse" (class `moderate_at` or worse)

Each splitting value comes in as a SiteValues object: one value per site, joined to the card by site id only (R7),
with a statement saying how it was made. A blank (nan) value goes to the cell "no value".

What a cell prints. Every cell prints n (labelled sites in it). A cell under the floor (fewer than `floor_n` sites, or
fewer than `floor_per_class` yes or no sites) prints TOO_FEW for everything else: no yes count and no metric. A cell
with fewer than `floor_tiles` tiles (default: the task's count gate, task.min_tiles) prints TOO_FEW_TILES with n and
its tile count only, since a bootstrap over one or two tiles gives intervals of near zero width. A cell at or above
the floor prints n_yes, n_no, n_tiles, the observed yes share, and for every scored card row the AUC;
rows with probability metrics (R3) also get log loss at the card's clip, Brier and mean p. Each value carries a 95%
tile-bootstrap interval (R11) drawn inside the cell with the card's tile size and seed.

The site and class floor is far below the card's count gate (task.min_sites 200, task.min_per_class 30). That is on
purpose: a cell is a subset read beside the card, not a card. The v3 registration (B8) must state the gap.

Count gate (R11 b). A card that fails its count gate reads UNDERPOWERED and prints counts only. Its breakdowns do the
same: every cell prints n and the status UNDERPOWERED, and no cell carries `rows` (build_card passes scored=False).

Severity and the card label. The severity table prints, as aggregate counts, how often the severity class disagrees
with the card label (label yes with class 0; label no with class 1 or worse). Such a site is relabelled by its class
in the severity cells; it is not dropped.

Severity cells are ranking only. An entry's probability is a probability of the card's target, not of a severity
class (R3), so the severity table prints AUC and no proper score.

No winner (B6 done-when). Rows in each cell keep the card's own row order. No cell carries a best row, a rank, a
difference or a word, and `check_no_winner` refuses a block that does. A cell value is a point on a subset, not a
comparison.

Off by default: build_card(..., breakdowns=None) adds nothing, so every earlier card (scorecard v2 among them) keeps
its bytes. Correction 1 (ruling 45): until the v3 registration (B8) is committed alone, this section runs only on
synthetic and planted data, never on real Canterbury labels.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from . import metrics, resample

PGA_EDGES_G = (0.1, 0.2, 0.3, 0.4, 0.5)
IC_EDGES = (1.31, 2.05, 2.60, 2.95, 3.60)
CRUST_EDGES_M = (1.0, 2.0, 3.0, 5.0)
FLOOR_N = 30
FLOOR_PER_CLASS = 5
MODERATE_AT = 2
TOO_FEW = "too few"
TOO_FEW_TILES = "too few tiles"
UNDERPOWERED = "UNDERPOWERED (R11): counts only"
NO_VALUE = "no value"
KINDS = ("pga", "ic", "crust", "severity")
UNITS = {"pga": "g", "ic": "", "crust": "m"}
EDGES = {"pga": PGA_EDGES_G, "ic": IC_EDGES, "crust": CRUST_EDGES_M}
IC_SOURCE = "Robertson 1990; Robertson 2009 (Can. Geotech. J. 46): SBTn zone edges"
NO_WINNER = ("NO WINNER (B6): each cell prints every scored row's value on that subset; no cell names a best row, "
             "and no difference between rows or cells is a word")
SEVERITY_NOTE = ("severity cells are ranking only (R3): an entry's probability is of the card's target, not of a "
                 "severity class")
FORBIDDEN_KEYS = frozenset({"best", "word", "words", "rank", "winner", "favours", "a_is", "diff"})


class BreakdownError(ValueError):
    """A breakdown input breaks a rule; the card is not built."""


@dataclass(frozen=True, eq=False)
class SiteValues:
    """One splitting value per site (nan = no value), joined to the card by site id only (R7)."""
    kind: str
    statement: str
    site_ids: tuple[str, ...]
    v: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_ids", tuple(self.site_ids))
        object.__setattr__(self, "v", np.asarray(self.v, dtype=float))
        if self.kind not in KINDS:
            raise BreakdownError(f"breakdowns: kind must be one of {list(KINDS)}, got {self.kind!r}")
        if not isinstance(self.statement, str) or not self.statement.strip():
            raise BreakdownError(f"breakdowns {self.kind}: needs a statement saying how the values were made")
        if self.v.ndim != 1 or self.v.size != len(self.site_ids) or self.v.size == 0:
            raise BreakdownError(f"breakdowns {self.kind}: needs one value per site id")
        if len(set(self.site_ids)) != len(self.site_ids):
            raise BreakdownError(f"breakdowns {self.kind}: site ids repeat")
        fin = self.v[np.isfinite(self.v)]
        if np.isinf(self.v).any():
            raise BreakdownError(f"breakdowns {self.kind}: values must be finite or blank (nan)")
        if self.kind == "severity" and not np.all((fin >= 0) & (fin == np.round(fin))):
            raise BreakdownError("breakdowns severity: classes must be whole numbers 0 (none), 1, 2, ...")
        if self.kind in ("pga", "crust") and np.any(fin < 0):
            raise BreakdownError(f"breakdowns {self.kind}: values must not be negative")

    def aligned(self, ids: Sequence[str]) -> np.ndarray:
        pos = {s: k for k, s in enumerate(self.site_ids)}
        missing = sum(1 for s in ids if s not in pos)
        if missing:
            raise BreakdownError(f"breakdowns {self.kind}: no entry for {missing} labelled site(s); give nan for "
                                 f"no value")
        return self.v[np.array([pos[s] for s in ids], dtype=int)]


@dataclass(frozen=True, eq=False)
class Breakdowns:
    """The splitting values the card's breakdown section uses. Any of the four may be absent."""
    values: tuple[SiteValues, ...]
    moderate_at: int = MODERATE_AT
    floor_n: int = FLOOR_N
    floor_per_class: int = FLOOR_PER_CLASS
    floor_tiles: int | None = None          # None: the task's count gate, task.min_tiles

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        kinds = [s.kind for s in self.values]
        if not kinds:
            raise BreakdownError("breakdowns: give at least one of " + ", ".join(KINDS))
        if len(set(kinds)) != len(kinds):
            raise BreakdownError("breakdowns: each kind at most once")
        for k, lo in (("moderate_at", 2), ("floor_n", 2), ("floor_per_class", 1)):
            v = getattr(self, k)
            if not isinstance(v, int) or isinstance(v, bool) or v < lo:
                raise BreakdownError(f"breakdowns: {k} must be a whole number of at least {lo}")
        ft = self.floor_tiles
        if ft is not None and (not isinstance(ft, int) or isinstance(ft, bool) or ft < 1):
            raise BreakdownError("breakdowns: floor_tiles must be None (the task's min_tiles) or a whole number of at "
                                 "least 1")


def cell_names(kind: str) -> tuple[str, ...]:
    """Cell names for a binned kind, lowest first, then NO_VALUE. Half-open cells [lo, hi)."""
    e, u = EDGES[kind], UNITS[kind]
    unit = f" {u}" if u else ""
    names = [f"below {e[0]:.2f}{unit}"]
    names += [f"{a:.2f} to {b:.2f}{unit}" for a, b in zip(e[:-1], e[1:])]
    names.append(f"{e[-1]:.2f}{unit} and above")
    return tuple(names) + (NO_VALUE,)


def assign(kind: str, v: np.ndarray) -> np.ndarray:
    """Cell index per site: 0 .. len(edges) by np.digitize (right-open), len(edges) + 1 for a blank value."""
    e = EDGES[kind]
    out = np.full(v.size, len(e) + 1, dtype=int)
    fin = np.isfinite(v)
    out[fin] = np.digitize(v[fin], e, right=False)
    return out


def block(bd: Breakdowns, task, rows: Sequence[dict], ids: Sequence[str], y: np.ndarray, x: np.ndarray,
          yy: np.ndarray, *, scored: bool) -> dict:
    """The `breakdowns` card section. `rows` are the card's scored rows (card._row dicts), values on `ids`.

    `scored` is the card's count gate (R11 b). When it is False every cell prints n and UNDERPOWERED only.
    """
    ft = task.min_tiles if bd.floor_tiles is None else bd.floor_tiles
    out = {"rule": (f"cells print n; under {bd.floor_n} sites or {bd.floor_per_class} per class a cell prints "
                    f"'{TOO_FEW}'; under {ft} tiles it prints '{TOO_FEW_TILES}'; 95% tile-bootstrap intervals "
                    f"inside each cell at the card's tile and seed; an UNDERPOWERED card prints counts only"),
           "floor_n": bd.floor_n, "floor_per_class": bd.floor_per_class, "floor_tiles": ft, "notes": [NO_WINNER]}
    if not scored:
        out["notes"].append(UNDERPOWERED)
    for sv in bd.values:
        v = sv.aligned(ids)
        if sv.kind == "severity":
            out["severity"] = _severity_table(sv, v, bd, task, rows, y, x, yy, ft, scored)
            out["notes"].append(SEVERITY_NOTE)
            continue
        names = cell_names(sv.kind)
        idx = assign(sv.kind, v)
        cells = {name: _cell(idx == k, y, rows, bd, task, x, yy, ft, scored, ranking_only=False)
                 for k, name in enumerate(names)}
        tab = {"statement": sv.statement, "edges": list(EDGES[sv.kind]), "unit": UNITS[sv.kind] or "none",
               "cells": cells}
        if sv.kind == "ic":
            tab["edges_source"] = IC_SOURCE
        out[sv.kind] = tab
    check_no_winner(out)
    return out


def _severity_table(sv: SiteValues, v: np.ndarray, bd: Breakdowns, task, rows, y, x, yy, ft: int,
                    scored: bool) -> dict:
    has = np.isfinite(v)
    cells = {}
    for name, cut in (("any ejecta", 1), ("moderate or worse", bd.moderate_at)):
        ys = np.where(has, (v >= cut).astype(float), np.nan)
        cells[name] = _cell(has, ys, rows, bd, task, x, yy, ft, scored, ranking_only=True)
    disagree = {"label_yes_class_0": int(np.sum(has & (y == 1) & (v == 0))),
                "label_no_class_1_or_worse": int(np.sum(has & (y == 0) & (v >= 1)))}
    return {"statement": sv.statement, "moderate_at": bd.moderate_at, "n_no_value": int((~has).sum()),
            "label_disagreements": disagree,
            "classes": {"any ejecta": "class 1 or worse is yes, class 0 is no",
                        "moderate or worse": f"class {bd.moderate_at} or worse is yes, lower classes are no"},
            "cells": cells}


def _cell(mask: np.ndarray, y: np.ndarray, rows, bd: Breakdowns, task, x, yy, ft: int, scored: bool, *,
          ranking_only: bool) -> dict:
    n = int(mask.sum())
    if not scored:
        return {"n": n, "status": UNDERPOWERED}
    yc = y[mask]
    n_yes = int(np.nansum(yc)) if n else 0
    if n < bd.floor_n or min(n_yes, n - n_yes) < bd.floor_per_class:
        return {"n": n, "status": TOO_FEW}
    xc, yyc = x[mask], yy[mask]
    tiles = resample.tiles_for(xc, yyc, task.tile_m)
    n_tiles = int(np.unique(tiles).size)
    if n_tiles < ft:
        return {"n": n, "status": TOO_FEW_TILES, "n_tiles": n_tiles}
    ll_key = f"log_loss@{metrics.clip_key(task.clip)}"
    vals = {r["name"]: r["v"][mask] for r in rows}
    prob = {r["name"]: (r["eligible"] and not ranking_only) for r in rows}

    def stats(i: np.ndarray) -> dict:
        yi = yc[i]
        s = {"observed_share": float(yi.mean())}
        for name, vv in vals.items():
            vi = vv[i]
            s[(name, "auc")] = metrics.auc(yi.astype(bool), vi)
            if prob[name]:
                s[(name, "log_loss")] = metrics.log_loss(yi, vi, task.clip)
                s[(name, "brier")] = metrics.brier(yi, vi)
                s[(name, "mean_p")] = float(np.mean(vi))
        return s

    point = stats(np.arange(n))
    draws = resample.bootstrap(stats, tiles, task.n_boot, task.seed)

    def iv(k) -> dict:
        return resample.interval(point[k], draws[k])

    out_rows = {}
    for r in rows:
        name = r["name"]
        m = {"auc": iv((name, "auc"))}
        if prob[name]:
            m.update({ll_key: iv((name, "log_loss")), "brier": iv((name, "brier")), "mean_p": iv((name, "mean_p"))})
        elif not ranking_only:
            m["probability_metrics"] = "NOT APPLICABLE (R3): ranking only"
        out_rows[name] = m
    return {"n": n, "status": "SHOWN", "n_yes": n_yes, "n_no": n - n_yes, "n_tiles": n_tiles,
            "observed_share": iv("observed_share"), "rows": out_rows}


def check_no_winner(obj: object, path: str = "breakdowns", *, names: bool = False) -> None:
    """Refuse a breakdown block that names a winner: a key such as best, rank, word, favours or diff.

    The keys of a cell's `rows` are model names, not structure, so a model named "best" or "rank" is not refused; the
    metric blocks under each name are still checked.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not names and isinstance(k, str) and k.lower() in FORBIDDEN_KEYS:
                raise BreakdownError(f"B6: {path} carries {k!r}; a breakdown names no winner")
            check_no_winner(v, f"{path}.{k}", names=(k == "rows" and not names and isinstance(v, dict)))
    elif isinstance(obj, (list, tuple)):
        for k, v in enumerate(obj):
            check_no_winner(v, f"{path}[{k}]")


def is_too_few(cell: dict) -> bool:
    """True for a cell under the site or class floor: it prints n and TOO_FEW only."""
    return cell.get("status") == TOO_FEW and set(cell) == {"n", "status"}


def is_counts_only(cell: dict) -> bool:
    """True for a cell that prints counts only: under a floor, short of tiles, or on an UNDERPOWERED card."""
    return cell.get("status") in (TOO_FEW, TOO_FEW_TILES, UNDERPOWERED) and set(cell) <= {"n", "status", "n_tiles"}

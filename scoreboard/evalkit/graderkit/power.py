"""Grader hours and the power of a smaller audit sample.

Inputs are the committed planning aggregates of the audit design (docs/evals/AUDIT-DESIGN-2026-11-19.md,
section 2, revision 2 draw) and the effort table of its section 6. No label is read here. The planning values are
per-earthquake Kish effective YES and NO counts at 75 units, from inventory "yes" shares (set A aerial, set B bench
or road). Under a proportional subsample inside every stratum, the Kish design effect does not change, so the
effective counts scale with the kept fraction.

AUC precision uses the Hanley-McNeil (1982) standard error at AUC 0.7, as section 2 does. A paired difference of
two AUCs on the same units uses SE_d = SE * sqrt(2 * (1 - r)), with r the correlation of the two AUC estimates
(Hanley-McNeil 1983; r = 0.5 is the planning value here, a stated assumption). The minimum detectable difference
at 80 % power, two-sided 5 %, is (1.96 + 0.84) * SE_d.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Z, Z80 = 1.96, 0.8416
AUC0 = 0.7
R_PAIRED = 0.5
RANKING_HW = 0.20  # an earthquake whose single-AUC half-width at full size exceeds this carries no ranking verdict
PREV_HW = 0.15     # prevalence half-width the cut earthquake must still reach (set A and set B both)

# Audit design section 2 (revision 2): units, Kish design effect, (effective YES, effective NO) and inventory
# "yes" share per source, at 75 units per earthquake. Strata per earthquake: 6 (Sep), 12 (others).
PLAN = {
    "sep2010": {"units": 75, "deff": 1.44, "strata": 6, "A": (18, 34, 0.33), "B": (13, 39, 0.23)},
    "feb2011": {"units": 75, "deff": 2.35, "strata": 12, "A": (28, 4.0, 0.92), "B": (26, 6.1, 0.83)},
    "jun2011": {"units": 75, "deff": 1.98, "strata": 12, "A": (29, 10, 0.85), "B": (23, 15, 0.64)},
    "dec2011": {"units": 75, "deff": 1.77, "strata": 12, "A": (18, 25, 0.44), "B": None},
}
MIN_PER_STRATUM = 2


@dataclass
class Effort:
    """Audit design section 6. Minutes are per queue item; the range is 8 to 20 minutes and 25 to 35 % disagree."""
    units: int = 300
    seeds: int = 24
    double_fraction: float = 1.0   # share of audit units a second grader also grades (seeds always double)
    minutes: float = 12.0
    disagree: float = 0.30
    adjudication_minutes: float = 15.0
    graders: int = 3
    training_minutes: float = 240.0
    regrade_items: int = 24

    def hours(self) -> dict:
        queue = self.units + self.seeds
        double_items = self.units * self.double_fraction + self.seeds
        primary = (queue + double_items) * self.minutes / 60
        adjud = double_items * self.disagree * self.adjudication_minutes / 60
        train = self.graders * self.training_minutes / 60
        regrade = self.regrade_items * self.minutes / 60
        return {"queue_items": queue, "graded_items": round(queue + double_items), "primary": primary,
                "adjudication": adjud, "training": train, "regrade": regrade,
                "total": primary + adjud + train + regrade}


def hm_se(auc: float, n_yes: float, n_no: float) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    v = (auc * (1 - auc) + (n_yes - 1) * (q1 - auc * auc) + (n_no - 1) * (q2 - auc * auc)) / (n_yes * n_no)
    return math.sqrt(max(v, 0.0))


def precision(eq: str, m: int) -> dict:
    """Half-widths for one earthquake when m of its 75 units are kept (proportional inside strata)."""
    p = PLAN[eq]
    f = m / p["units"]
    out = {"units": m}
    for src in ("A", "B"):
        if p[src] is None:
            continue
        ny, nn, prev = p[src]
        se = hm_se(AUC0, ny * f, nn * f)
        neff = m / p["deff"]
        out[src] = {"eff_yes": ny * f, "eff_no": nn * f, "auc_hw": Z * se,
                    "paired_mdd": (Z + Z80) * se * math.sqrt(2 * (1 - R_PAIRED)),
                    "prev_hw": Z * math.sqrt(prev * (1 - prev) / neff)}
    return out


def propose() -> dict:
    """Keep 75 where the earthquake carries ranking verdicts; cut the others to the smallest size that keeps the
    prevalence half-width at or under PREV_HW for every source, never under 2 units per stratum."""
    out = {}
    for eq, p in PLAN.items():
        full = precision(eq, p["units"])
        srcs = [s for s in ("A", "B") if p[s] is not None]
        ranking = any(full[s]["auc_hw"] <= RANKING_HW for s in srcs)
        if ranking:
            out[eq] = {"keep": p["units"], "why": "ranking verdicts rest here; any cut widens them"}
            continue
        floor = MIN_PER_STRATUM * p["strata"]
        keep = next(m for m in range(floor, p["units"] + 1)
                    if all(precision(eq, m)[s]["prev_hw"] <= PREV_HW for s in srcs))
        out[eq] = {"keep": keep, "why": f"AUC already uninformative at {p['units']} units "
                                        f"(half-width {min(full[s]['auc_hw'] for s in srcs):.2f}); "
                                        f"cut to the prevalence target {PREV_HW}"}
    return out


def subsample(units: list[dict], keep: dict, seed: int) -> list[dict]:
    """Second-phase draw: keep `keep[eq]` units of each earthquake, split over its strata in proportion to the
    drawn n_h (at least MIN_PER_STRATUM, capped at n_h), drawn without replacement with PCG64(seed).
    Inclusion probabilities multiply: pi' = pi * m_h / n_h, weight' = 1 / pi'. A proposal tool only: the frozen
    draw changes only through a new registration (audit design section 3, freeze rule)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    key = lambda u: (u["earthquake"], u["placement"], u["density"], u["score_band"])  # noqa: E731
    strata = {}
    for u in units:
        strata.setdefault(key(u), []).append(u)
    out = []
    for eq in sorted({u["earthquake"] for u in units}):
        hs = sorted(k for k in strata if k[0] == eq)
        n = {h: len(strata[h]) for h in hs}
        target = keep.get(eq, sum(n.values()))
        m = {h: min(n[h], max(MIN_PER_STRATUM, int(round(target * n[h] / sum(n.values()))))) for h in hs}
        # fix rounding so the earthquake total equals the target where the floors allow it
        for h in sorted(hs, key=lambda h: -n[h]):
            diff = target - sum(m.values())
            if diff == 0:
                break
            step = 1 if diff > 0 else -1
            if MIN_PER_STRATUM <= m[h] + step <= n[h]:
                m[h] += step
        for h in hs:
            rows = sorted(strata[h], key=lambda u: u["queue_id"])
            idx = rng.choice(len(rows), size=m[h], replace=False)
            for i in sorted(idx):
                u = dict(rows[i])
                u["inclusion_probability"] = u["inclusion_probability"] * m[h] / n[h]
                u["weight"] = 1 / u["inclusion_probability"]
                u["n_h"] = m[h]
                out.append(u)
    return out


def minutes_from_grades(grades_dir) -> float | None:
    """Median minutes per item from the form's timer, over every grader's latest grade of each item."""
    secs = []
    for p in Path(grades_dir).glob("*/*.jsonl"):
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        if lines:
            s = json.loads(lines[-1]).get("seconds", 0)
            if s > 0:
                secs.append(s)
    return float(np.median(secs)) / 60 if secs else None


def _h(e: Effort) -> str:
    h = e.hours()
    return f"{h['total']:.0f}"


def note(measured_minutes: float | None = None) -> str:
    base = Effort()
    lo = Effort(minutes=8, disagree=0.25)
    hi = Effort(minutes=20, disagree=0.35)
    plus = Effort(seeds=SEEDS_BEFORE_GRADING)
    prop = propose()
    kept = sum(v["keep"] for v in prop.values())
    opts = options(kept)
    L = []
    L.append("## Grader hours, recomputed from the audit design section 6")
    L.append("")
    b = base.hours()
    L.append("| Part | Registered design (300 units + 24 seeds, two graders on every item) |")
    L.append("|---|---|")
    for k in ("primary", "adjudication", "training", "regrade"):
        L.append(f"| {k} | {b[k]:.1f} h |")
    L.append(f"| **total** | **{b['total']:.1f} h** |")
    L.append("")
    L.append(f"Range: {lo.hours()['total']:.0f} h (8 minutes, 25 % disagree) to {hi.hours()['total']:.0f} h "
             "(20 minutes, 35 % disagree). The design says 120 to 220; its upper end is low, because 648 graded "
             "items at 20 minutes is already 216 h before adjudication.")
    L.append(f"With the 12 Jun and Dec seeds that section 4.5 adds before grading (36 seeds): {_h(plus)} h.")
    if measured_minutes:
        L.append(f"With the measured median of {measured_minutes:.1f} minutes per item: "
                 f"{Effort(minutes=measured_minutes).hours()['total']:.0f} h.")
    L.append("")
    L.append("## Power per earthquake (planning values, AUC 0.7)")
    L.append("")
    L.append("| Earthquake | Units | AUC half-width A / B | Paired MDD A / B (r 0.5) | Prevalence half-width A / B |")
    L.append("|---|---|---|---|---|")
    for eq in PLAN:
        for m in sorted({PLAN[eq]["units"], prop[eq]["keep"]}, reverse=True):
            pr = precision(eq, m)
            cell = lambda k: " / ".join(f"{pr[s][k]:.2f}" for s in ("A", "B") if s in pr)  # noqa: E731
            L.append(f"| {eq} | {m} | {cell('auc_hw')} | {cell('paired_mdd')} | {cell('prev_hw')} |")
    L.append("")
    L.append("## Proposal")
    L.append("")
    for eq, v in prop.items():
        L.append(f"- {eq}: keep {v['keep']} ({v['why']}).")
    L.append("")
    n = SEEDS_BEFORE_GRADING
    L.append(f"Every option counts {n} seeds: section 4.5 adds the 12 Jun and Dec seeds before grading starts.")
    L.append("")
    L.append("| Option | Units + seeds | Second grader on | Hours | Single-grader labels |")
    L.append("|---|---|---|---|---|")
    L.append(f"| Registered design | 300 + {n} | all | {_h(opts['registered'])} | none |")
    L.append(f"| S1 smaller sample (proposal above) | {kept} + {n} | all | {_h(opts['S1'])} | none |")
    L.append(f"| P1 second grader on a 30 % stratified subsample, all seeds | 300 + {n} | 30 % + seeds | "
             f"{_h(opts['P1'])} | 210 of 300 units |")
    L.append(f"| S1 and P1 | {kept} + {n} | 30 % + seeds | {_h(opts['S1+P1'])} | "
             f"{kept - round(kept * 0.3)} of {kept} units |")
    L.append("")
    lo_c, hi_c = p1_label_cost(0.25), p1_label_cost(0.35)
    L.append("Cost of P1 to label accuracy: not measured. Under P1, 70 % of units carry one grader's label with no "
             "adjudication. The design expects 25 to 35 % of double-graded items to disagree. If the two graders are "
             "equally good and the adjudicator sides with each half the time, a single label differs from the "
             f"adjudicated one on about {lo_c:.0%} to {hi_c:.0%} of single-graded units. That is an assumption, not "
             "a measurement; the pilot's agreement rate replaces it.")
    return "\n".join(L)


SEEDS_BEFORE_GRADING = 36  # 24 registered seeds plus the 12 Jun and Dec seeds of audit design section 4.5


def options(kept: int, seeds: int = SEEDS_BEFORE_GRADING) -> dict:
    """The four protocol options, each with the seeds that exist when grading starts."""
    return {"registered": Effort(seeds=seeds), "S1": Effort(units=kept, seeds=seeds),
            "P1": Effort(seeds=seeds, double_fraction=0.3),
            "S1+P1": Effort(units=kept, seeds=seeds, double_fraction=0.3)}


def p1_label_cost(disagree: float) -> float:
    """Share of single-graded units whose label would differ from an adjudicated one, under the stated
    assumption that the adjudicator sides with either grader half the time. Not a measurement."""
    return disagree / 2

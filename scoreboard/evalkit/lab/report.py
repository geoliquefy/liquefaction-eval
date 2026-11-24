"""Markdown tables for the lab certificates from the aggregate JSON (lab.aggregate). No em dashes.

  <venv python> -m scoreboard.evalkit.lab.report <agg.json>
"""
from __future__ import annotations

import json
import sys

from .aggregate import RULES
from .run import RUN_ORDER

SHOW = ("variogram", "trend-variogram-v2", "fixed-2km", "fixed-4km", "fixed-8km")


def f(c: dict | None) -> str:
    if not c or c.get("coverage") is None:
        return "n/a"
    return f"{c['coverage']:.3f} ({c['mcse']:.3f}) {c['status'][0]}"


def base_table(a: dict) -> str:
    out = ["| Rule | " + " | ".join(f"{cond} {m}" for cond in ("B-NULL", "B-MED") for m in ("AUC", "log loss"))
           + " | min cell | B-MED AUC width |", "|---" * 7 + "|"]
    for rule in RULES:
        row = [rule]
        for cond in ("B-NULL", "B-MED"):
            cv = a["conditions"][cond]["rules"][rule]["coverage"]
            row += [f(cv["all/auc"]), f(cv["all/log_loss"])]
        t = a["decision"]["table"][rule]
        row += [f"{t['min_coverage']:.3f}", f"{t['b_med_auc_width']:.4f}"]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def cell_table(a: dict, rule: str) -> str:
    out = ["| Cell | B-NULL | B-MED |", "|---|---|---|"]
    keys = [k for k in a["conditions"]["B-NULL"]["rules"][rule]["coverage"] if not k.startswith("all/")]
    for k in keys:
        out.append(f"| {k} | {f(a['conditions']['B-NULL']['rules'][rule]['coverage'][k])} | "
                   f"{f(a['conditions']['B-MED']['rules'][rule]['coverage'][k])} |")
    return "\n".join(out)


def words_table(a: dict) -> str:
    out = ["| Condition | Rule | null rate Holm (upper 95%) | null unadjusted | family-wise (AUC / LL) | power AUC (MDD) | power LL (MDD) |",
           "|---|---|---|---|---|---|---|"]
    for cond in ("B-NULL", "B-MED", "B-SMALL"):
        c = a["conditions"].get(cond)
        if not isinstance(c, dict):
            continue
        for rule in SHOW:
            w = c["rules"][rule]["words"]
            nr = w["null_rate_holm"]
            fw = w["fwer_event_family"]
            pw = w["power"]
            row = [cond, rule, f"{nr['rate']:.3f} ({nr['upper95']:.3f})" if nr else "n/a",
                   f"{w['null_rate_unadjusted']['rate']:.3f}" if w["null_rate_unadjusted"] else "n/a",
                   f"{fw['auc']:.3f} / {fw['log_loss']:.3f}" if fw else "",
                   f"{pw['auc']['rate']:.3f} ({pw['auc']['median_mdd']:.3f})" if "auc" in pw else "",
                   f"{pw['log_loss']['rate']:.3f} ({pw['log_loss']['median_mdd']:.3f})" if "log_loss" in pw else ""]
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def hk_table(a: dict) -> str:
    out = ["| Condition | k | Rule | coverage AUC | coverage LL | allowed share | null rejection AUC |", "|---|---|---|---|---|---|---|"]
    for cond in ("B-NULL", "B-MED", "V-EV8", "X-RM-LONG", "X-LONGRAND"):
        c = a["conditions"].get(cond)
        if not isinstance(c, dict):
            continue
        for rule in ("variogram", "trend-variogram-v2", "fixed-4km"):
            for k, h in c["rules"][rule].get("hk", {}).items():
                out.append(f"| {cond} | {k} | {rule} | {f(h['auc'])} | {f(h['log_loss'])} | "
                           f"{h['auc']['allowed_share']:.3f} | {h['auc']['null_rejection']} |")
    return "\n".join(out)


def adversarial_table(a: dict) -> str:
    out = ["| Condition | worlds | " + " | ".join(f"{r} AUC / LL" for r in SHOW) + " | 8 km + t (explor.) AUC / LL |",
           "|---" * (3 + len(SHOW)) + "|"]
    for cond in RUN_ORDER:
        c = a["conditions"].get(cond)
        if not isinstance(c, dict):
            out.append(f"| {cond} | NOT RUN |" + " |" * (len(SHOW) + 1))
            continue
        row = [cond, str(c["n_worlds"])]
        for rule in SHOW:
            cv = c["rules"][rule]["coverage"]
            row.append(f"{f(cv['all/auc'])} / {f(cv['all/log_loss'])}")
        t = c["exploratory_small_cluster_t"]["8km"]
        row.append(f"{f(t['auc'])} / {f(t['log_loss'])}")
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def gate_table(a: dict) -> str:
    out = ["| Condition | Rule | gate pass | gate fail | tiles < 10 | 10 to 19 | 20 to 39 | 40 or more |", "|---|---|---|---|---|---|---|---|"]
    for cond in ("X-SMALL", "B-NULL", "B-MED"):
        c = a["conditions"].get(cond)
        if not isinstance(c, dict):
            continue
        for rule in ("trend-variogram-v2", "fixed-2km", "fixed-4km", "fixed-8km"):
            g = c["rules"][rule]["count_gate"]
            out.append(f"| {cond} | {rule} | {f(g['gate_pass'])} | {f(g['gate_fail'])} | {f(g['tiles_0_10'])} | "
                       f"{f(g['tiles_10_20'])} | {f(g['tiles_20_40'])} | {f(g['tiles_40_up'])} |")
    return "\n".join(out)


def tile_table(a: dict) -> str:
    c = a["conditions"]["B-NULL"]["tile_choice"]
    out = ["| Slot | variogram picks (0.5/1/2/4/8 km) | trend-variogram-v2 picks | v2 RANGE NOT IDENTIFIED | variogram same tile, other history | tiles at 0.5/1/2/4/8 km |",
           "|---|---|---|---|---|---|"]
    for e, d in c.items():
        v = "/".join(f"{x:.2f}" for x in d["variogram"].values())
        t = "/".join(f"{x:.2f}" for x in d["trend-variogram-v2"].values())
        out.append(f"| {e} | {v} | {t} | {d['trend-variogram-v2 range not identified']:.3f} | "
                   f"{d.get('variogram same tile with another earlier-label history', 'n/a')} | "
                   f"{'/'.join(str(x) for x in d['tiles_at_sizes_median'])} |")
    return "\n".join(out)


def main(path: str) -> None:
    a = json.loads(open(path).read())
    for name, fn in (("BASE", base_table), ("WORDS", words_table), ("HK", hk_table), ("ADV", adversarial_table),
                     ("GATE", gate_table), ("TILES", tile_table)):
        print(f"\n### {name}\n")
        print(fn(a))
    print("\n### CELLS fixed-4km\n")
    print(cell_table(a, "fixed-4km"))
    print("\n### CELLS trend-variogram-v2\n")
    print(cell_table(a, "trend-variogram-v2"))


if __name__ == "__main__":
    main(sys.argv[1])

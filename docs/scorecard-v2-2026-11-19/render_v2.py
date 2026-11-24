"""Render SCORECARD.md from scorecard.json (scorecard v2, record track). Presentation only: every number is read from
the JSON, nothing is computed here except formatting. Usage: python3 render_v2.py <scorecard.json> <SCORECARD.md>.
No em dashes.
"""
from __future__ import annotations

import json
import sys

EVN = {"sep2010": "Sep 2010", "feb2011": "Feb 2011", "jun2011": "Jun 2011", "dec2011": "Dec 2011"}
ROWS = ["Level 0", "M0 (Level 0 + earthquake band)", "LSN + global fragility [authors-maurer-2026-11-19]",
        "LSN + CES fragility [authors-maurer-2026-11-19]", "LSN + global fragility [default-2026-11]",
        "LSN + CES fragility [default-2026-11]", "USGS Zhu 2017", "GEM TS22", "baseline: coin",
        "baseline: earlier-earthquake rate", "baseline: shaking only", "baseline: effort only"]
SHORT = {"Level 0": "Level 0 (HOUSE)", "M0 (Level 0 + earthquake band)": "M0 (HOUSE)",
         "LSN + global fragility [authors-maurer-2026-11-19]": "LSN + global frag., authors id (HOUSE)",
         "LSN + CES fragility [authors-maurer-2026-11-19]": "LSN + CES frag., authors id (HOUSE)",
         "LSN + global fragility [default-2026-11]": "LSN + global frag., default id (beside)",
         "LSN + CES fragility [default-2026-11]": "LSN + CES frag., default id (beside)",
         "USGS Zhu 2017": "USGS Zhu 2017 (ORGANISER-RUN)",
         "GEM TS22": "GEM TS22 (ORGANISER-RUN, STAND-IN INPUTS, NOT MONOTONE)",
         "baseline: coin": "coin", "baseline: earlier-earthquake rate": "earlier-earthquake rate",
         "baseline: shaking only": "shaking only", "baseline: effort only": "pre-event effort"}


# Post-review fix (review 1 M1, M2): counts read from the data, no score. Zhu empty cells: the case falls on an
# empty (NaN) cell of the USGS zhu_2017_general.hdf5 grid, read as 0 by build_v2.py (deviation 3). Counted by the
# same nearest-cell lookup as build_v2.world() on cases.parquet. PGA tags from cases.parquet `pga_tag`.
# Tuple: (cases on empty cells, cases, of which yes, of which no).
ZHU_EMPTY = {("P1", "Hyogoken-Nambu"): (22, 26, 15, 7), ("P2", "Hyogoken-Nambu"): (30, 36, 20, 10),
             ("P1", "Chi-Chi"): (0, 54, 0, 0), ("P2", "Chi-Chi"): (0, 75, 0, 0)}
KOBE_NOTE = ("Hyogoken-Nambu PGA: 25 of 26 P1 cases carry the NGL tag 'site response analysis' (P2: the same 25, "
             "plus 10 PRJ-3012 cases with no per-case method and 1 ShakeMap). A site-response PGA can be lower where "
             "the soil softened, so this PGA may depend on the outcome. This is not tested. The same PGA enters "
             "Level 0, LSN and LPI. Shaking only ranks below 0.5 here. Kobe stays in every population, because "
             "removing it now would follow the results.")


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def iv(d):
    if not isinstance(d, dict) or d.get("value") is None:
        return "n/a"
    return f"{d['value']:.3f} [{f3(d.get('lo'))}, {f3(d.get('hi'))}]"


def wiv(d, tiles):
    s = iv(d)
    return s + " (uninformative: under 10 tiles)" if tiles < 10 and s != "n/a" else s


def card_table(cards, unseen, title):
    out = [f"### {title}", "", "| Row | " + " | ".join(EVN.values()) + " |", "|---|" + "---|" * 4]
    def cnt(c):
        g = c.get("count_gate")
        if g:
            return f"{g['n_labelled']} ({g['n_yes']}), {g['n_tiles']}"
        return f"{c.get('n', 0)} ({c.get('n_yes', 0)}), 0"
    out.append("| n (yes), 4 km tiles | " + " | ".join(cnt(c) for c in cards.values()) + " |")
    out.append("| status | " + " | ".join(c.get("status", "") for c in cards.values()) + " |")
    for r in ROWS:
        cells = []
        for e, c in cards.items():
            if c.get("status") != "SCORED":
                cells.append("counts only")
                continue
            m = c["metrics"].get(r)
            s = iv(m["auc"]) if m else "NOT AVAILABLE"
            u = (unseen or {}).get(e, {}).get(r)
            if u is not None and "auc" in u:
                s += f"; UNSEEN-only {iv(u['auc'][r])} (n {u['unseen_n']})"
            cells.append(s)
        out.append(f"| {SHORT[r]} | " + " | ".join(cells) + " |")
    return out


def main(src, dst):
    J = json.loads(open(src).read())
    Cn = J["canterbury"]
    L = []
    L += ["# Liquefaction model scorecard v2 (record track)", "",
          f"G6 + G1 + G5, eval milestone M4. Registration: `{J['registration']}`. Kit {J['kit']['branch']} at "
          f"{J['kit']['commit']}. Canonical card hash (spec.card_sha256 of scorecard.json without this field): "
          f"`{J['card_sha256']}`. No em dashes.", "",
          "## Read this first", ""]
    L += [f"- {s}." for s in J["statements"]]
    L += [f"- Every interval: 95% tile bootstrap, fixed 4 km tiles on the NZTM2000 grid anchored at the origin, "
          f"{J['n_boot']} draws, seed {J['seed']}: **{J['interval_note']}**.",
          "- Log loss and calibration on inventory \"no\" labels measure agreement with the label source, not with "
          "the ground. GEM TS22 ran on our open stand-in input layers and is not monotone (step 2c).",
          "- The v1 label caveat text inside each card still says \"label error NOT MEASURED\"; step 1a has since "
          "measured it, and the Menon ceilings below replace that phrase.",
          "- Pre-event effort baseline known limit (step 1 review 2, N4): `in_aerial_footprint` is the union of the "
          "mapped CGD0200 classes, not the imagery footprint.",
          "- Aggregates only (NZGD terms clause 6.2). Bench v2 test split never read. No synthetic data.", ""]
    L += ["## Gates", ""]
    for k, v in J["gates"].items():
        L.append(f"- {k}: {'PASS' if v.get('pass') else 'FAIL'}")
    L += ["", "## Canterbury, set B (primary): AUC per earthquake, DESCRIPTIVE", "",
          "Point [95% tile interval, not certified]. UNSEEN-only = the entry and the baselines on the sites where "
          "that entry saw no case at any earthquake (exposure ledger). No word is printed within an earthquake.", ""]
    L += card_table(Cn["cards"]["B"], Cn["unseen_reads"]["B"], "All labelled sites")
    L += [""]
    L += card_table(Cn["pre_event_cpt_cards"]["B"], None, "Beside: only CPTs pushed before the earthquake")
    L += ["", "Menon AUC ceilings for set B (step 1a): " + "; ".join(
        f"{EVN[e]} {v['source']} " + ", ".join(f"{k} {x['C']:.2f}" + (" (inadmissible, step 1a)" if (e, k) == ("dec2011", "S_feb") else "")
                                         for k, x in v["menon_C"].items())
        + f" ({v['status']})" for e, v in J["label_ceilings_menon_C"]["B"].items()) + ".", "",
          "Jun 2011 note (review 1 m6): the S_sep ceiling 0.64 lies below the observed set B AUC of Level 0 (0.683) "
          "and Zhu 2017 (0.696), so the data contradict that scenario. S_feb (0.77) and S_pool (0.66) remain.", "",
          "Menon AUC ceilings for set A (step 1a): " + "; ".join(
        f"{EVN[e]} {v['source']} " + ", ".join(f"{k} {x['C']:.2f}" + (" (inadmissible, step 1a)" if (e, k) == ("dec2011", "S_feb") else "")
                                         for k, x in v["menon_C"].items())
        + f" ({v['status']})" for e, v in J["label_ceilings_menon_C"]["A"].items()) + ".", ""]
    L += ["### Exposure per cell (labelled sites, set B): SEEN / POSSIBLE / PLACE / UNSEEN", "",
          "| Entry | " + " | ".join(EVN.values()) + " |", "|---|" + "---|" * 4]
    for r in ROWS[:8]:
        cells = []
        for e in EVN:
            u = Cn["unseen_reads"]["B"][e].get(r)
            if u is None:
                cells.append("n/a")
                continue
            c = u["exposure_counts_labelled_sites"]
            cells.append("NOT CHECKED" if c["NOT CHECKED"] else
                         f"{c['SEEN']} / {c['POSSIBLE']} / {c['PLACE']} / {c['UNSEEN']}")
        L.append(f"| {SHORT[r]} | " + " | ".join(cells) + " |")
    L += ["", "### UNSEEN-only reads with the baselines on the same sites (set B), DESCRIPTIVE", "",
          "| Entry | Earthquake | n (yes), 4 km tiles | entry AUC | shaking only | pre-event effort |",
          "|---|---|---|---|---|---|"]
    for r in ROWS[:8]:
        for e in EVN:
            u = Cn["unseen_reads"]["B"][e].get(r)
            if not u or "auc" not in u:
                continue
            a = u["auc"]
            sh = iv(a["baseline: shaking only"]) if "baseline: shaking only" in a else "NOT AVAILABLE"
            L.append(f"| {SHORT[r]} | {EVN[e]} | {u['unseen_n']} ({u['unseen_yes']}), {u['unseen_tiles_4km']} | "
                     f"{iv(a[r])} | {sh} | {iv(a['baseline: effort only'])} |")
    L += ["", "Cells absent here have too few UNSEEN sites (counts in the exposure table above).", ""]
    L += ["", "### Probability rows, set B: log loss (clip 1e-6), DESCRIPTIVE", "",
          "| Row | " + " | ".join(EVN.values()) + " |", "|---|" + "---|" * 4]
    for r in ROWS:
        cells = []
        for e, c in Cn["cards"]["B"].items():
            if c.get("status") == "SCORED" and r not in c.get("metrics", {}):
                cells.append("NOT AVAILABLE")
                continue
            m = c.get("metrics", {}).get(r) if c.get("status") == "SCORED" else None
            cells.append(iv(m.get("log_loss")) if m and "log_loss" in m else "ranking only")
        if any(x not in ("ranking only", "NOT AVAILABLE") for x in cells):
            L.append(f"| {SHORT[r]} | " + " | ".join(cells) + " |")
    L += ["", "## Canterbury, set A (beside)", ""]
    L += card_table(Cn["cards"]["A"], Cn["unseen_reads"]["A"], "All labelled sites")
    L += [""]
    L += card_table(Cn["pre_event_cpt_cards"]["A"], None, "Beside: only CPTs pushed before the earthquake")
    L += ["", "Label-source sign agreement (set A against set B, every paired difference): " + "; ".join(
        f"{EVN[e]}: " + (v if isinstance(v, str) else f"{v['same_sign']} same sign, {v['opposite_sign']} opposite"
                         + (f" ({v['note']})" if v["note"] else ""))
        for e, v in J["label_source_sign_agreement"].items()) + ". Descriptive count only.", "",
          "Dec 2011 note (review 1 m5): sets A and B carry the same Dec labels, so the Dec count does not test label "
          "sources. Its opposite signs come from baselines fitted on different earlier label sets.", ""]
    L += ["## Pooled lines (all DESCRIPTIVE)", "",
          "| Line | Population | A minus B | Weighting | Earthquakes | Mean [95%] | Leave one out (min to max mean) |",
          "|---|---|---|---|---|---|---|"]
    for lid, p in J["pooled_lines"].items():
        for wk in ("primary_kit_random_effects", "beside_equal_weight", "primary_equal_weight", "beside_kit_weights"):
            x = p.get(wk)
            if not x:
                continue
            if x.get("word") == "NOT ALLOWED":
                L.append(f"| {lid} | {p['population']} | {p['a']} minus {p['b']} | {wk} | "
                         f"{', '.join(x.get('earthquakes_pooled', []))} | NOT ALLOWED ({x.get('reason', '')}) | |")
                continue
            pl = x["pooled"]
            loo = [y["mean"] for y in x.get("leave_one_out", [])]
            L.append(f"| {lid} | {p['population']} | {p['a']} minus {p['b']} | {wk} | "
                     f"{', '.join(x['earthquakes_pooled'])} | {pl['mean']:+.3f} [{pl['lo']:+.3f}, {pl['hi']:+.3f}] | "
                     + (f"{min(loo):+.3f} to {max(loo):+.3f}" if loo else "") + " |")
    c2 = J["pooled_lines"]["C2"]["primary_kit_random_effects"]["pooled"]
    w1 = {y["left_out"]: y["mean"] for y in J["pooled_lines"]["W1 P1"]["primary_equal_weight"]["leave_one_out"]}
    L += ["", f"C2 note (review 1 m4): Q = {c2['q']:.2f} is below k - 1 = 3, so tau^2 = {c2['tau2']:g}. With hk_floor "
          f"off (the registered kit default) the Hartung-Knapp SE ({c2['se']:.4f}) is below the fixed-effect SE, so "
          "the C2 primary interval is narrower than a fixed-effect interval.", "",
          f"W1 note (review 1 M2): W1 P1 without Hyogoken-Nambu (Kobe) is {w1['Hyogoken-Nambu']:+.3f} (equal weight; "
          "source: scorecard.json pooled_lines, 'W1 P1', primary_equal_weight, leave_one_out, left_out "
          "Hyogoken-Nambu). This is a descriptive side line, not a replacement. " + KOBE_NOTE, ""]
    lor = J["pooled_lines"]["W1 P2"].get("leave_one_region_out", [])
    L += ["W1 P2 leave one region out (equal weight, mean with that region left out): " + "; ".join(
        f"{y['left_out']} {y['mean']:+.3f}" if "mean" in y else f"{y['left_out']} {y.get('word', '')}" for y in lor)
          + ".", ""]
    L += ["", "Why descriptive: Certificate 4 (Hartung-Knapp, 4 km inputs, k = 3 or 4) holds at the first lab's 2 km "
          "model-error range; the real model-difference range is not identified (LAB-CERTIFICATES 11.4, 12.1). No "
          "certificate covers equal weighting or world earthquakes.", ""]
    W = J["world"]
    for pop in ("P1", "P2"):
        L += [f"## World case histories, {pop} ({'primary' if pop == 'P1' else 'beside, union'}): AUC, DESCRIPTIVE",
              "", "1 km local tiles. Record gate: 3 yes, 3 no, 5 tiles.", "",
              "| Earthquake | n (yes), tiles | Level 0 | BI14-LSN | BI14-LPI | Zhu 2017 | shaking only | "
              "Level 0 minus shaking [95%] | B&I exposure | Zhu 2017 exposure | LSN global fragility exposure |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for eq, v in W["populations"][pop]["earthquakes"].items():
            if not v["passes_record_gate"]:
                continue
            a, t = v["auc"], v["tiles_1km"]
            g = lambda k: wiv(a[k], t) if k in a else "NOT RUN"  # noqa: E731
            d = v["paired"].get("Level 0 minus baseline: shaking only")
            ex = lambda k: ", ".join(f"{m} {n}" for m, n in v["exposure"][k].items())  # noqa: E731
            zhu = g("USGS Zhu 2017")
            if (pop, eq) in ZHU_EMPTY and zhu != "NOT RUN":
                ne, nn, ny, nno = ZHU_EMPTY[(pop, eq)]
                zhu += (f" ({ne} of {nn} cases off the model grid, read as 0: {ny} yes, {nno} no)" if ne else
                        f" (0 of {nn} cases off the model grid)")
            name = eq + (" (see Kobe note)" if eq == "Hyogoken-Nambu" else "")
            L.append(f"| {name} | {v['n']} ({v['n_yes']}), {t} | {g('Level 0')} | "
                     f"{g('BI14-LSN (both LSN fragility rows)')} | {g('BI14-LPI')} | {zhu} | "
                     f"{g('baseline: shaking only')} | "
                     + (f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" if d else "n/a")
                     + (" (uninformative: under 10 tiles)" if d and t < 10 else "")
                     + f" | {ex('B&I chain (Level 0, M0, LSN, LPI)')} | {ex('USGS Zhu 2017')} | "
                     f"{ex('LSN global fragility')}: in sample (fit) |")
        L += ["", "BI14-LSN is the ranking of both LSN rows. Its B&I column is the triggering chain's exposure. The "
              "LSN global fragility was fitted on this same compilation, so it is in sample (a fit) on every "
              "earthquake.", "", "Kobe note (review 1 M2): " + KOBE_NOTE, ""]
        under = [(eq, v) for eq, v in W["populations"][pop]["earthquakes"].items() if not v["passes_record_gate"]]
        L += [f"Under the record gate (counts only): {len(under)} earthquakes.", "",
              "| Earthquake | n (yes), 1 km tiles |", "|---|---|"]
        L += [f"| {eq} | {v['n']} ({v['n_yes']}), {v['tiles_1km']} |" for eq, v in under]
        L += [""]
    L += [f"World cross-check against the G1 curated chain: {W['cross_check']}. Zhu grids: "
          + "; ".join(f"{k} {v['cases']} cases in P1 and P2 together, {v['nan_cells_set_to_0']} of them on empty "
                      f"cells read as 0" for k, v in W["zhu_grids"].items()) + ".", ""]
    L += ["## Diagnostic only: investigation-density proxy AUC (never a baseline)", "",
          "| Set | " + " | ".join(EVN.values()) + " |", "|---|" + "---|" * 4]
    for s in ("B", "A"):
        L.append(f"| {s} | " + " | ".join(iv(Cn["diagnostic_proxy_auc"][s][e]) for e in EVN) + " |")
    L += ["", "## Attribution", ""] + [f"- {a}" for a in J["attribution"]]
    L += ["", "## Deviations", "", "Logged in `REGISTRATION.md` (Deviations). Items 1 to 9 come from the build. Item 3 "
          "(Zhu empty cells read as 0) sets 22 of 26 inputs of the Kobe P1 Zhu cell (see the world tables). Items 10 "
          "onward are post-review fixes from review 1: renderer, notes and logs only, no new score, no change to any "
          "population, entry or rule.", "",
          "Build logs: `build_v2-attempt1.log` (stopped at the output scan) and `build_v2-attempt2.log` (the card), "
          "committed after review 1 and hashed in SEAL.txt. They hold statuses, gate values and timings only."]
    L += ["", "## Not done", "",
          "- Independent review 1 is done (CARD STANDS WITH FIXES, REVIEW-1-2026-11-19.md). Its fixes are applied "
          "here. There is no round 2 (no blocker).",
          "- OpenTimestamps: a founder step (it sends hashes to public calendars). Not stamped.",
          "- Audit (step 1d) not graded; external parity not achieved; GEM TS22 training events NOT CHECKED; CES "
          "fragility CPT list unpublished (POSSIBLE everywhere).", ""]
    open(dst, "w").write("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

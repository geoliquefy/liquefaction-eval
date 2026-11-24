"""B1 (eval M4): the house prints its losses first. No em dashes.

Three pieces, all off by default so every earlier card (the sealed scorecard v2 among them) is unchanged:

  1. Declaration. A house `other_probability` entry may set `declared_target` in its manifest (spec.EntryManifest).
     The house then stands behind its number as a probability of the card's target (manifestation), and the row gets
     log loss, Brier and calibration instead of "ranking only" (spec.probability_eligible). The numbers are not
     changed. The card marks the row DECLARED_MARK in its entry list.
  2. LOSS block. `build_card(..., losses_first=True)` puts a `losses` block as the card's FIRST key. Every house row
     with probability metrics whose proper score is higher (worse) than a baseline's (log loss at the card's clip,
     or Brier, strictly higher; a tie prints nothing) gets one LOSS line. The line lists every baseline it loses to,
     with both scores for the row and the baseline, and the paired log loss difference (row minus baseline) with its
     95% interval when the card holds that comparison. A house row with no probability metrics is listed under
     `no_proper_score`.
  3. Proper-score gate. `build_card(..., require_house_proper_score=True)` refuses the card when a house
     `other_probability` entry has neither `declared_target` nor a conversion (`house_rows_without_proper_score`).
     With it on, every house probability row carries a proper score (own-work list B1 done-when).

Correction 8 (ruling 45, 2026-11-23). "LSN + global fragility" and "LSN + CES fragility" carry the SAME mark. Both
are our run of one published table (Maurer, engrXiv 121, Table 3), and our chain fails parity with that paper. So
neither is a house model in the LOSS block: each LOSS line for them reads "loss of our run of a published model",
never a house loss. PUBLISHED_OURS below names both rows, under any chain id. Item B2 (outside.py, branch
eval/b2-not-run-marks-2026-11-23) owns the matching pattern for its NOT RUN BY AUTHOR SETTINGS gate; the integrator
keeps the two rules naming the same rows.

Baselines here are the card's rows with role "baseline" and probability metrics: the coin, the earlier-earthquake
rate and the shaking-only baseline. The effort-only baseline is ranking only and takes no part.

A LOSS line is a point estimate on the labelled sites, not a word: it does not say the row is separably worse.
`loss_block` reads only the card dict, so it also reads an existing card without rebuilding it.
"""
from __future__ import annotations

from . import outside
from .metrics import clip_key

DECLARED_MARK = "DECLARED TARGET PROBABILITY (house)"
LOSS_RULE = ("B1: every house row whose log loss at the card's clip or Brier is higher than a baseline's gets a LOSS "
             "line with both scores; point estimates, not a word; a tie prints nothing; printed first (R16 b)")

# Correction 8: both LSN fragility rows, under any chain id ([default-2026-11], [authors-maurer-2026-11-19], or a new
# chain id from a later chain fix). B2 owns the matching pattern and the wording (outside.PUBLISHED_RUN_ROW,
# outside.PUBLISHED_RUN); this module reads both, so the two modules cannot drift apart (wave 1 integration).
HOUSE_KIND = "house"
PUBLISHED_OURS_KIND = "our run of a published model"
assert outside.PUBLISHED_RUN.startswith(PUBLISHED_OURS_KIND)
PUBLISHED_OURS_MARK = PUBLISHED_OURS_KIND.upper() + outside.PUBLISHED_RUN[len(PUBLISHED_OURS_KIND):]


def is_published_ours(name: str) -> bool:
    """True for either LSN fragility row (correction 8): our run of a published model, not a house model."""
    return outside.is_published_run(name)


def entry_meta_extra(manifest) -> dict:
    """Extra entry-list fields for a declared row; empty for every other row, so older cards keep their bytes."""
    if getattr(manifest, "declared_target", None) is not True:
        return {}
    return {"declared_target": True, "declaration": DECLARED_MARK}


def house_rows_without_proper_score(manifests) -> list[str]:
    """Names of house `other_probability` manifests with no declaration and no conversion (proper-score gate)."""
    return [m.model for m in manifests if m.house and m.output_kind == "other_probability"
            and m.conversion_id is None and m.declared_target is not True]


def _paired(card: dict, house: str, base: str) -> dict | None:
    """The card's paired log loss difference, house minus baseline, with its 95% interval; None if not tested.

    The kit lists entries before baselines, so the house is comparison `a` on cards it builds. The flip covers a card
    dict that holds the pair the other way round (a hand-made or older card)."""
    for c in card.get("comparisons", []):
        if c.get("family") != "probability" or {c.get("a"), c.get("b")} != {house, base}:
            continue
        if c.get("diff") is None:
            return None
        if c["a"] == house:
            return {"diff": c["diff"], "lo": c.get("lo"), "hi": c.get("hi")}
        lo, hi = c.get("lo"), c.get("hi")
        return {"diff": -c["diff"], "lo": None if hi is None else -hi, "hi": None if lo is None else -lo}
    return None


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _text(name: str, kind: str, base: str, worse: list[str], clip: float, h: tuple, b: tuple) -> str:
    who = (f"loss of our run of a published model (Maurer, engrXiv 121, Table 3): {name}"
           if kind == PUBLISHED_OURS_KIND else f"house loss: {name}")
    return (f"LOSS: {who} has a higher {' and a higher '.join(worse)} than {base} (point estimates, not a word): "
            f"log loss at clip {clip_key(clip)} {_fmt(h[0])} vs {_fmt(b[0])}; Brier {_fmt(h[1])} vs {_fmt(b[1])}")


def loss_block(card: dict) -> dict:
    """The LOSS block for one card dict (built by card.build_card, plain or descriptive)."""
    clip = card["task"]["clip"]
    block = {"rule": LOSS_RULE, "clip": clip}
    entries = card.get("entries", [])
    house = [e for e in entries if e.get("house")]
    # A row B2 marks NOT RUN BY AUTHOR SETTINGS has no scores to compare; it is listed apart, not as "no proper score".
    not_run = [e["name"] for e in house if e.get("author_confirmed") is False]
    house = [e for e in house if e.get("author_confirmed") is not False]
    block["no_proper_score"] = [e["name"] for e in house if not e.get("probability_metrics")]
    if not_run:
        block["not_run_by_author_settings"] = not_run
    if card.get("status") != "SCORED" or "metrics" not in card:
        block.update(status="NOT SCORED", why="UNDERPOWERED (R11): counts only; no score, so no LOSS line",
                     n_lines=0, lines=[])
        return block
    mets = card["metrics"]
    bases = [e["name"] for e in entries if e.get("role") == "baseline" and e.get("probability_metrics")]
    lines = []
    for e in house:
        if not e.get("probability_metrics"):
            continue
        name = e["name"]
        kind = PUBLISHED_OURS_KIND if is_published_ours(name) else HOUSE_KIND
        h = mets[name]
        h_ll, h_br = h["log_loss"]["value"], h["brier"]["value"]
        lost = []
        for b in bases:
            m = mets[b]
            b_ll, b_br = m["log_loss"]["value"], m["brier"]["value"]
            worse = [k for k, hv, bv in (("log loss", h_ll, b_ll), ("Brier", h_br, b_br))
                     if hv is not None and bv is not None and hv > bv]
            if not worse:
                continue
            item = {"baseline": b, "worse_on": worse, "log_loss": {"house": h_ll, "baseline": b_ll},
                    "brier": {"house": h_br, "baseline": b_br},
                    "text": _text(name, kind, b, worse, clip, (h_ll, h_br), (b_ll, b_br))}
            paired = _paired(card, name, b)
            if paired is not None:
                item["paired_log_loss_house_minus_baseline"] = paired
            lost.append(item)
        if lost:
            line = {"line": "LOSS", "house": name, "kind": kind,
                    "mark": PUBLISHED_OURS_MARK if kind == PUBLISHED_OURS_KIND else e.get("mark", ""),
                    "declared_target": e.get("declared_target", False), "lost_to": lost}
            for k in ("statement", "conversion_id"):
                if e.get(k) is not None:
                    line[k] = e[k]
            lines.append(line)
    block.update(status="SCORED", n_lines=len(lines), lines=lines)
    return block


def put_first(card: dict) -> dict:
    """The card with its LOSS block as the first key. The input card is not changed."""
    return {"losses": loss_block(card), **{k: v for k, v in card.items() if k != "losses"}}

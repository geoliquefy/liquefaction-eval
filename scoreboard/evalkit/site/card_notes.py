"""Notes that the sealed v2 card prints but that its JSON does not hold.

G1+G5+G6 (eval M6), item D9. Source: the sealed renderer docs/scorecard-v2-2026-11-19/render_v2.py
(pillar; sha256 13303b35... in that folder's SEAL.txt). The text is copied word for word. The site
prints these notes only for the card whose canonical hash is the key below. No em dashes.
"""
from __future__ import annotations

V2_CARD_SHA256 = "cc9f7b744d6bd6b3b4d5ce0956f04e3a17539ef9014de0277223033c08c8847d"

KOBE_NOTE = ("Hyogoken-Nambu PGA: 25 of 26 P1 cases carry the NGL tag 'site response analysis' (P2: the same 25, "
             "plus 10 PRJ-3012 cases with no per-case method and 1 ShakeMap). A site-response PGA can be lower where "
             "the soil softened, so this PGA may depend on the outcome. This is not tested. The same PGA enters "
             "Level 0, LSN and LPI. Shaking only ranks below 0.5 here. Kobe stays in every population, because "
             "removing it now would follow the results.")

# (cases on empty Zhu grid cells read as 0, cases, of which yes, of which no). Counts, not scores.
ZHU_EMPTY = {("P1", "Hyogoken-Nambu"): (22, 26, 15, 7), ("P2", "Hyogoken-Nambu"): (30, 36, 20, 10),
             ("P1", "Chi-Chi"): (0, 54, 0, 0), ("P2", "Chi-Chi"): (0, 75, 0, 0)}

EXTRA_STATEMENTS = [
    "Log loss and calibration on inventory \"no\" labels measure agreement with the label source, not with the "
    "ground. GEM TS22 ran on our open stand-in input layers and is not monotone (step 2c).",
    "The v1 label caveat text inside each card still says \"label error NOT MEASURED\"; step 1a has since measured "
    "it, and the Menon ceilings replace that phrase.",
    "Pre-event effort baseline known limit (step 1 review 2, N4): `in_aerial_footprint` is the union of the mapped "
    "CGD0200 classes, not the imagery footprint.",
    "Aggregates only (NZGD terms clause 6.2). Bench v2 test split never read. No synthetic data.",
]

JUN_NOTE = ("Jun 2011 note (review 1 m6): the S_sep ceiling 0.64 lies below the observed set B AUC of Level 0 (0.683) "
            "and Zhu 2017 (0.696), so the data contradict that scenario. S_feb (0.77) and S_pool (0.66) remain.")

DEC_NOTE = ("Dec 2011 note (review 1 m5): sets A and B carry the same Dec labels, so the Dec count does not test label "
            "sources. Its opposite signs come from baselines fitted on different earlier label sets.")

WHY_DESCRIPTIVE = ("Why descriptive: Certificate 4 (Hartung-Knapp, 4 km inputs, k = 3 or 4) holds at the first lab's "
                   "2 km model-error range; the real model-difference range is not identified (LAB-CERTIFICATES 11.4, "
                   "12.1). No certificate covers equal weighting or world earthquakes.")

NOT_DONE = [
    "Independent review 1 is done (CARD STANDS WITH FIXES). Its fixes are applied. There is no round 2 (no blocker).",
    "OpenTimestamps: a founder step (it sends hashes to public calendars). Not stamped.",
    "Audit (step 1d) not graded; external parity not achieved; GEM TS22 training events NOT CHECKED; CES fragility "
    "CPT list unpublished (POSSIBLE everywhere).",
]


def for_card(card_sha256: str) -> dict | None:
    """The notes for one card, or None when the site has no notes for that card hash."""
    if card_sha256 != V2_CARD_SHA256:
        return None
    return {"kobe": KOBE_NOTE, "zhu_empty": ZHU_EMPTY, "extra_statements": EXTRA_STATEMENTS, "jun": JUN_NOTE,
            "dec": DEC_NOTE, "why_descriptive": WHY_DESCRIPTIVE, "not_done": NOT_DONE}

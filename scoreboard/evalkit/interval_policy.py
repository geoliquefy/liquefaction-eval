"""Which interval a card prints, by card version (C4, ruling 47). No em dashes.

v2 (the default): nothing here changes a v2 card. `apply_per_earthquake(card, "v2")` returns the card as given, and
`pooled_lines(..., card_version="v2")` is `pool.pooled_lines` with the caller's options. The sealed v2 card keeps its
canonical hash (cc9f7b74...), because the v2 path runs no new code.

v3: the C4 outcome is the only interval. The outcome was fixed by a registered lab test
(pillar docs/evals/C4-CERTIFICATION-REGISTRATION-2026-11-23.md, results docs/evals/C4-CERTIFICATION-2026-11-23.md)
and is copied below as a constant. Per line type:
  per-earthquake  the certified method's interval and words; with none certified, every within-earthquake word is
                  withheld (card.descriptive) and the note names the measured coverage
  pooled          the certified method's pooled interval and the R1 word; with none certified, the R1 word stays on
                  the pooled line (ruling 47) with the kit's Hartung-Knapp line, marked not certified

Review 1 (wave 3) fixes on the v3 path only:
  - A v3 card is built on the interval C4 measured: fixed 4 km tiles (task.tile_rule "fixed", task.tile_m 4000).
    `check_task` refuses any other tile, because the note quotes tile4's measured coverage.
  - Two words outside `comparisons` also rest on the per-earthquake tile bootstrap, so they are per-earthquake words
    under ruling 47 and are withheld: the B5 geographic null's `calibrated_auc[...]["word_vs_null"]` (a copy of a
    comparison word) and the B7 placement proxy word (EXPLAINED BY DENSITY STRATA, OPEN BIAS, CANNOT TELL from the
    within-strata tile bootstrap). Each reads DESCRIPTIVE; the numbers and intervals stay. The proxy's
    `residual_excludes_0.5` flag (a test result) and its word rule are removed. A final scan refuses a v3 card that
    still holds any word value anywhere (`PER_EARTHQUAKE_WORDS`).
"""
from __future__ import annotations

import json
from collections.abc import Sequence

from . import pool

CARD_VERSIONS = ("v2", "v3")
DEFAULT_CARD_VERSION = "v2"
REGISTRATION = "docs/evals/C4-CERTIFICATION-REGISTRATION-2026-11-23.md"
RESULTS = "docs/evals/C4-CERTIFICATION-2026-11-23.md"
NO_METHOD_LINE = "No method certifies; words only on the pooled line"

# The registered outcome of the C4 run (RESULTS). None = no method certifies that line type. Result 2026-11-23:
# no candidate certifies either line type (tile4 per-earthquake UNDER, pooled HK and event2 OVER; block AUC under the
# 20-block gate everywhere; cluster NOT TESTABLE).
C4_OUTCOME: dict = {
    "truth": "models fixed",
    "per_earthquake": None,
    "pooled": None,
    "record_line": NO_METHOD_LINE,
    "measured_coverage": {
        # tile4 (the interval a v3 card prints), models-fixed truth, 1,500 worlds in each of 5 conditions
        "per_earthquake": "0.87 to 0.94 (AUC 0.92 to 0.93, log loss 0.87 to 0.94; lab worlds, C4)",
        "pooled": "0.945 to 0.977 (Hartung-Knapp, 3 earthquakes; lab worlds, C4)",
    },
    "run": "1,500 worlds x 5 conditions (B-NULL, B-MED, R-NULL, R-MED, R-HI); kit 92933f0; registration pillar b844a93",
}


class IntervalPolicyError(ValueError):
    """A card version or a certified method this kit cannot print."""


def check_version(card_version: str) -> str:
    if card_version not in CARD_VERSIONS:
        raise IntervalPolicyError(f"card_version must be one of {CARD_VERSIONS}, got {card_version!r}")
    return card_version


V3_TILE = {"rule": "fixed", "tile_m": 4000.0}     # the tile4 interval C4 measured (registration section 7)


def check_task(task, card_version: str = DEFAULT_CARD_VERSION) -> None:
    """v2: no check. v3: refuse a task whose interval is not the plain fixed 4 km tile bootstrap C4 measured, since
    the v3 note quotes that interval's measured coverage (review 1)."""
    if check_version(card_version) == "v2":
        return
    if task.tile_rule != V3_TILE["rule"] or float(task.tile_m) != V3_TILE["tile_m"]:
        raise IntervalPolicyError(
            f"a v3 card prints the tile4 interval only (fixed 4 km tiles, the interval C4 measured); got tile_rule "
            f"{task.tile_rule!r}, tile_m {float(task.tile_m):g}")


# Every word a within-earthquake interval can give on a card: comparison words and directions (card._compare), the
# B7 placement proxy words, and the R1 pooled words (a per-earthquake card holds no pooled line).
PER_EARTHQUAKE_WORDS = frozenset({"SEPARATES", "EQUIVALENT", "CANNOT TELL", "BETTER", "WORSE",
                                  "EXPLAINED BY DENSITY STRATA", "OPEN BIAS",
                                  "HOLDS", "FRAGILE", "POOLED CANNOT TELL"})


def words_left(obj: object, path: str = "card") -> list[str]:
    """Paths of every string value in `obj` that is exactly one of PER_EARTHQUAKE_WORDS."""
    if isinstance(obj, str):
        return [path] if obj in PER_EARTHQUAKE_WORDS else []
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in words_left(v, f"{path}.{k}")]
    if isinstance(obj, (list, tuple)):
        return [p for i, v in enumerate(obj) for p in words_left(v, f"{path}[{i}]")]
    return []


def _withhold_other_words(card: dict, descriptive_word: str) -> dict:
    """v3, no per-earthquake method certified: withhold the words outside `comparisons` (review 1)."""
    out = json.loads(json.dumps(card))
    gn = out.get("geographic_null")
    if isinstance(gn, dict):
        for row in (gn.get("calibrated_auc") or {}).values():
            if "word_vs_null" in row:
                row["word_vs_null"] = descriptive_word
    px = (out.get("placement") or {}).get("proxy")
    if isinstance(px, dict) and px.get("word") not in (None, "NOT SUPPLIED"):
        px["word"] = descriptive_word
        px.pop("residual_excludes_0.5", None)
        px.pop("rule", None)
        px["word_note"] = ("the proxy word rests on the within-earthquake tile bootstrap; withheld on a v3 card "
                           "(ruling 47: words only on the pooled line)")
    left = words_left(out)
    if left:
        raise IntervalPolicyError(f"a v3 card with no certified per-earthquake method still holds words at {left}")
    return out


def policy(card_version: str = DEFAULT_CARD_VERSION) -> dict:
    """What a card of this version prints for each line type."""
    if check_version(card_version) == "v2":
        return {"card_version": "v2", "per_earthquake": "as sealed v2 (scorecard v2 registration K2)",
                "pooled": "as sealed v2 (caller's pool.pooled_lines options)"}
    return {"card_version": "v3", "truth": C4_OUTCOME["truth"],
            "per_earthquake": C4_OUTCOME["per_earthquake"] or "DESCRIPTIVE (no method certifies)",
            "pooled": C4_OUTCOME["pooled"] or "kit Hartung-Knapp, word kept, marked not certified",
            "record_line": C4_OUTCOME["record_line"], "registration": REGISTRATION, "results": RESULTS}


def _coverage_text(kind: str) -> str:
    m = C4_OUTCOME["measured_coverage"].get(kind)
    return f"; measured coverage {m}" if m else ""


def per_earthquake_note() -> str:
    return (f"descriptive; not certified under models fixed{_coverage_text('per_earthquake')} "
            f"(C4, {RESULTS})")


def pooled_note() -> str:
    m = C4_OUTCOME["pooled"]
    if m is None:
        return f"pooled interval not certified under models fixed{_coverage_text('pooled')} (C4, {RESULTS})"
    return f"pooled interval certified under models fixed: {m} (C4, {RESULTS})"


def apply_per_earthquake(card: dict, card_version: str = DEFAULT_CARD_VERSION) -> dict:
    """v2: the card as given (same object). v3: the C4 per-earthquake outcome."""
    if check_version(card_version) == "v2":
        return card
    method = C4_OUTCOME["per_earthquake"]
    if method is None:
        from .card import DESCRIPTIVE, descriptive
        out = _withhold_other_words(descriptive(card, per_earthquake_note()), DESCRIPTIVE)
        out["card_version"] = "v3"
        out["interval_policy"] = policy("v3")
        return out
    raise IntervalPolicyError(f"v3 per-earthquake method {method!r} is certified but not wired into card.py")


def pooled_lines(lines: Sequence[Sequence[dict]], *, registered: bool, card_version: str = DEFAULT_CARD_VERSION,
                 **v2_options) -> list[dict]:
    """v2: pool.pooled_lines with the caller's options, unchanged. v3: the C4 pooled outcome; v2 options are refused,
    because v3 has one interval only."""
    if check_version(card_version) == "v2":
        return pool.pooled_lines(lines, registered=registered, **v2_options)
    if v2_options:
        raise IntervalPolicyError(f"a v3 pooled line takes no interval options; got {sorted(v2_options)}")
    method = C4_OUTCOME["pooled"]
    if method in (None, "tile4"):
        out = pool.pooled_lines(lines, registered=registered, weighting="re")
        note = pooled_note()
        for line in out:
            line["interval_note"] = note
            line["certified"] = method is not None
        return out
    raise IntervalPolicyError(f"v3 pooled method {method!r} is certified but not wired into pool.py")

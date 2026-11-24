"""Outside-model rows and the author_confirmed flag (own-work list B2, eval M4). No em dashes.

An outside-model row is an entry that stands for another author's model: any entry that is not a house entry
(USGS Zhu 2017, GEM TS22), plus a house-built row that claims an author's settings by name: LSN with the CES
fragility or with the global fragility, under an author chain id `[authors-...]`. The organiser ran these rows. Their
authors have not confirmed the settings (E1 is out).

The two LSN rows always get the same mark (ruling 45, correction 8). Both fragilities come from one published table
(Maurer, engrXiv 121, Table 3), and our chain fails parity with that paper. So each LSN row is our run of a published
model, not a house model, and neither one can carry a house LOSS line that the other escapes. The match depends only on
the chain id, never on which of the two fragilities the row uses: a chain fix that takes a new `authors-...` id
(correction 2) is matched for both rows at once. So under the card option `author_confirmed` each such row carries the flag, and an
unconfirmed row prints NOT RUN BY AUTHOR SETTINGS instead of scores. It is left out of scoring altogether: no metric
block, no comparison, no Holm family member, no best candidate.

The option is off by default (`author_confirmed=None` in `card.build_card`). Then the card is byte for byte the card
the kit built before B2, so the sealed scorecard v2 stays reproducible.

`check_card` is the gate: it refuses any card that prints a score for an unconfirmed outside row. `card.build_card`,
`card.descriptive` and `card.write_card` call it on every card.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

NOT_RUN = "NOT RUN BY AUTHOR SETTINGS"
RULE = ("B2: an outside-model row prints scores only when its authors confirm the settings it was run with; "
        "until then it reads NOT RUN BY AUTHOR SETTINGS and is left out of every comparison")
# House-built rows that claim an outside author's settings by name. Own-work list B2 named the CES fragility row;
# ruling 45 correction 8 adds the global fragility row, because both are our run of one published table (Maurer,
# engrXiv 121, Table 3). One pattern covers both, so the two rows cannot drift apart.
LSN_FRAGILITIES = ("CES", "global")
PUBLISHED_RUN = "our run of a published model (Maurer, engrXiv 121, Table 3; our chain fails parity with that paper)"
AUTHOR_ROW_PATTERNS = (re.compile(r"^LSN \+ (?:CES|global) fragility \[authors-[^\]]+\]$"),)
# Wave 1 integration (correction 8 reconciled with B1): the one test for "our run of a published model", under any
# chain id or none. evalkit.losses uses this test and builds its LOSS mark from PUBLISHED_RUN, so the card entry note
# and the LOSS line can never name the two LSN rows differently.
PUBLISHED_RUN_ROW = re.compile(r"LSN \+ (?:CES|global) fragility(?: \[[^\]]+\])?")


def is_published_run(name: str) -> bool:
    """True for either LSN fragility row, under any chain id or none (correction 8)."""
    return PUBLISHED_RUN_ROW.fullmatch(name) is not None


def fragility_twin(name: str) -> str | None:
    """The same LSN row with the other fragility ("CES" <-> "global"), or None for any other name."""
    m = re.fullmatch(r"LSN \+ (CES|global) fragility(.*)", name)
    if not m:
        return None
    other = LSN_FRAGILITIES[1 - LSN_FRAGILITIES.index(m.group(1))]
    return f"LSN + {other} fragility{m.group(2)}"


class AuthorGateError(RuntimeError):
    """A card prints a score for an outside-model row its authors have not confirmed."""


def is_outside(name: str, role: str, house: bool) -> bool:
    """True for an entry that is not house and not a baseline, or a house row that names an author's settings."""
    if role != "entry":
        return False
    return (not house) or any(p.fullmatch(name) for p in AUTHOR_ROW_PATTERNS)


def split_rows(rows: list[dict], author_confirmed: Mapping[str, bool]) -> tuple[list[dict], list[dict]]:
    """(scored rows, not-run rows). A name missing from `author_confirmed` counts as not confirmed.

    A name in `author_confirmed` that is not an outside row is refused, so a typo cannot silently confirm nothing. A map
    that confirms one LSN fragility row and not its twin under the same chain id is refused (correction 8).
    """
    if not isinstance(author_confirmed, Mapping):
        raise TypeError("author_confirmed must map row names to true or false")
    outside = {r["name"] for r in rows if is_outside(r["name"], r["role"], r["manifest"].house)}
    stray = sorted(set(author_confirmed) - outside)
    if stray:
        raise ValueError(f"author_confirmed names rows that are not outside-model rows: {stray}")
    for k, v in author_confirmed.items():
        if not isinstance(v, bool):
            raise TypeError(f"author_confirmed[{k!r}] must be true or false, got {v!r}")
    split = sorted(n for n in outside if fragility_twin(n) in outside
                   and author_confirmed.get(n, False) != author_confirmed.get(fragility_twin(n), False))
    if split:   # correction 8: one published table, so the two LSN rows are confirmed together or not at all
        raise ValueError(f"author_confirmed must give both LSN fragility rows the same value: {split}")
    scored, not_run = [], []
    for r in rows:
        if r["name"] in outside:
            r["author_confirmed"] = bool(author_confirmed.get(r["name"], False))
        (not_run if r.get("author_confirmed") is False else scored).append(r)
    return scored, not_run


def entry_marks(meta: dict, row: dict) -> dict:
    """Add the flag (and, when false, the NOT RUN mark) to one card entry block."""
    if "author_confirmed" not in row:
        return meta
    meta = dict(meta, author_confirmed=row["author_confirmed"])
    if is_published_run(row["name"]):
        meta["published_run"] = PUBLISHED_RUN
    if not row["author_confirmed"]:
        meta["mark"] = NOT_RUN if not meta.get("mark") else f"{meta['mark']}; {NOT_RUN}"
        meta["probability_metrics"] = False
    return meta


def mark_card(card: dict, not_run: list[dict]) -> dict:
    """Put NOT RUN BY AUTHOR SETTINGS where an unconfirmed row's scores would go, and add the author block."""
    names = [r["name"] for r in not_run]
    if "metrics" in card:
        for n in names:
            card["metrics"][n] = NOT_RUN
    card["author_rule"] = {"rule": RULE, "not_run": names,
                           "confirmed": sorted(e["name"] for e in card["entries"] if e.get("author_confirmed"))}
    return card


def unconfirmed(card: dict) -> set[str]:
    """Names of outside rows the card itself marks as not confirmed."""
    out = {e["name"] for e in card.get("entries", []) if e.get("author_confirmed") is False}
    out |= set(card.get("author_rule", {}).get("not_run", []))
    return out


def check_card(card: dict) -> None:
    """The gate. Refuses a card that prints any score for an unconfirmed outside row.

    A score is: a metrics block that is not the NOT RUN text, a comparison naming the row, a best-entry candidate or
    name equal to the row, a duplicates pair naming the row, or a B6 breakdowns cell whose `rows` names the row. The
    unconfirmed set comes from the author block and from every entry flagged `author_confirmed` false, so deleting
    the author block does not open the gate. A card with neither (the default card, the sealed v2 card) has nothing
    unconfirmed and passes unchanged.
    """
    bad = unconfirmed(card)
    if not bad:
        return
    problems = []
    for n in sorted(bad):
        m = card.get("metrics", {}).get(n, NOT_RUN)
        if m != NOT_RUN:
            problems.append(f"{n!r} has a metrics block")
    for c in card.get("comparisons", []):
        hit = {c.get("a"), c.get("b")} & bad
        if hit:
            problems.append(f"comparison names {sorted(hit)}")
    for fam, b in (card.get("best") or {}).items():
        if isinstance(b, dict) and ({b.get("candidate"), b.get("name")} & bad):
            problems.append(f"best {fam} names an unconfirmed row")
    dups = card.get("duplicates", {})
    for d in (dups.get("pairs", []) if isinstance(dups, dict) else []):
        if {d.get("a"), d.get("b")} & bad:
            problems.append("a duplicates pair names an unconfirmed row")
    hit = _breakdown_rows(card.get("breakdowns")) & bad
    if hit:
        problems.append(f"a breakdowns cell scores {sorted(hit)}")
    if problems:
        raise AuthorGateError(f"B2: the card prints a score for an outside-model row its authors have not "
                              f"confirmed: {'; '.join(problems[:5])}")


def _breakdown_rows(obj: object) -> set[str]:
    """Every row name under a `rows` key anywhere in the B6 breakdowns section (empty when there is none)."""
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "rows" and isinstance(v, dict):
                out |= {n for n in v if isinstance(n, str)}
            out |= _breakdown_rows(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _breakdown_rows(v)
    return out

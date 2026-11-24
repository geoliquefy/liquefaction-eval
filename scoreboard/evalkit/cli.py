"""Command line for the eval kit. No em dashes.

  python -m scoreboard.evalkit gates
  python -m scoreboard.evalkit score --task task.json --entries DIR --labels labels.csv --out card.json
        [--shaking-only shaking.csv] [--registered-task-sha256 HEX] [--losses-first] [--require-house-proper-score]
  python -m scoreboard.evalkit demo [--out card.json] [--n-boot N] [--losses-first]
  python -m scoreboard.evalkit hash card.json       (the platform-stable card hash, spec.card_sha256)

An entries DIR holds one `NAME.json` manifest and one `NAME.csv` (`site_id,p`) per entry. Exit codes: 0 done,
1 gates failed, 2 refused (a rule or the spec is broken).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .card import CardRefused, build_card, write_card
from .gates import run_gates
from .spec import (Entry, EntryManifest, SpecError, card_sha256, load_card_sha256, load_entry, load_labels,
                   load_predictions, load_task)
from .synth import demo_inputs


def load_entries(directory: str | Path) -> list[Entry]:
    """Every NAME.json + NAME.csv pair in `directory`, in name order. A manifest without its CSV is refused."""
    d = Path(directory)
    out = []
    for mp in sorted(d.glob("*.json")):
        cp = mp.with_suffix(".csv")
        if not cp.exists():
            raise SpecError(f"{mp.name}: no matching {cp.name}")
        out.append(load_entry(mp, cp))
    if not out:
        raise SpecError(f"{d}: no entries (NAME.json + NAME.csv pairs)")
    return out


def _shaking_only(path: str, event_id: str) -> Entry:
    m = EntryManifest(model="baseline: shaking only", output_kind="target_probability",
                      statement="yes share by shaking bin from earlier earthquakes", exposure={event_id: "unseen"},
                      house=False)
    return Entry(m, *load_predictions(path))


def _summary(card: dict) -> dict:
    """The few lines worth printing to a terminal."""
    out = {}
    if "losses" in card:                                       # B1: the house losses print first
        out["losses"] = [i["text"] for ln in card["losses"]["lines"] for i in ln["lost_to"]]
    out |= {"status": card["status"], "track": card["track"], "count_gate": card["count_gate"],
           "best": {k: v["name"] for k, v in card["best"].items()}, "card_sha256": card_sha256(card)}
    if card["status"] == "SCORED":
        out["auc"] = {k: round(v["auc"]["value"], 3) for k, v in card["metrics"].items()}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scoreboard.evalkit", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gates", help="run the scorer self-checks (R12)")
    sc = sub.add_parser("score", help="build the card for one task")
    sc.add_argument("--task", required=True)
    sc.add_argument("--entries", required=True)
    sc.add_argument("--labels", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--shaking-only")
    sc.add_argument("--registered-task-sha256")
    sc.add_argument("--losses-first", action="store_true", help="put the house LOSS block first (B1)")
    sc.add_argument("--require-house-proper-score", action="store_true",
                    help="refuse a house other_probability row with no declared target and no conversion (B1)")
    dm = sub.add_parser("demo", help="build a card from the built-in synthetic earthquake")
    dm.add_argument("--out")
    dm.add_argument("--n-boot", type=int, default=2000)
    dm.add_argument("--losses-first", action="store_true", help="put the house LOSS block first (B1)")
    hs = sub.add_parser("hash", help="print the platform-stable hash of a card file")
    hs.add_argument("card")
    a = ap.parse_args(argv)

    if a.cmd == "hash":
        print(load_card_sha256(a.card))
        return 0

    if a.cmd == "gates":
        res = run_gates()
        print(json.dumps(res, indent=1))
        return 0 if res["pass"] else 1
    try:
        if a.cmd == "score":
            task = load_task(a.task)
            labels = load_labels(a.labels)
            shaking = _shaking_only(a.shaking_only, task.event_id) if a.shaking_only else None
            card = build_card(task, load_entries(a.entries), labels, shaking_only=shaking,
                              registered_task_sha256=a.registered_task_sha256, losses_first=a.losses_first,
                              require_house_proper_score=a.require_house_proper_score)
            write_card(card, a.out, labels.site_ids)
        else:
            task, entries, labels, shaking = demo_inputs(n_boot=a.n_boot)
            card = build_card(task, entries, labels, shaking_only=shaking, losses_first=a.losses_first)
            if a.out:
                write_card(card, a.out, labels.site_ids)
            else:
                print(json.dumps(card, indent=1))
                return 0
    except (CardRefused, SpecError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_summary(card), indent=1))
    return 0

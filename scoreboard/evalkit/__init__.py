"""evalkit: the liquefaction eval rule book (docs/evals/RULEBOOK-V1-DRAFT-2026-11-15.md) enforced in code.

Serves GOALS.md G1, G5 and G6 (ruling 42 goal B). A library with no side effects: it reads only the files a caller
names and writes only the card a caller asks for. See README.md in this folder for the rule-to-module map.
No em dashes.
"""
from .card import CardRefused, build_card, scan_aggregates, write_card
from .gates import duplicate_pairs, run_gates
from .pool import card_item, pooled_line
from .resample import bootstrap, choose_tile, holm, interval, paired
from .spec import (Entry, EntryManifest, Labels, SpecError, Task, entry_sha256, load_entry, load_labels, load_task,
                   probability_eligible, site_list_sha256)

__all__ = ["CardRefused", "Entry", "EntryManifest", "Labels", "SpecError", "Task", "bootstrap", "build_card",
           "card_item", "choose_tile", "duplicate_pairs", "entry_sha256", "holm", "interval", "load_entry",
           "load_labels", "load_task", "paired", "pooled_line", "probability_eligible", "run_gates",
           "scan_aggregates", "site_list_sha256", "write_card"]

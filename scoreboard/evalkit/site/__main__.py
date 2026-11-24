"""`python -m scoreboard.evalkit.site --card-dir <sealed card folder> --out <site folder>`. No em dashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import build


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scoreboard.evalkit.site")
    ap.add_argument("--card-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--card-name", default=None)
    a = ap.parse_args(argv)
    man = build(a.card_dir, a.out, card_name=a.card_name)
    print(json.dumps({"card_sha256": man["card_sha256"], "files": len(man["files"]),
                      "verdict_grep_hits": len(man["verdict_grep_hits"])}))
    for h in man["verdict_grep_hits"]:
        print("verdict grep:", h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

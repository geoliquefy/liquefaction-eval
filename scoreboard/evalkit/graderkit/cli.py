"""python -m scoreboard.evalkit.graderkit {demo,build,serve,check,power}"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import blind, build, power, serve, sources


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="graderkit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="build the synthetic demo packet and serve the form")
    d.add_argument("--out", default="graderkit-demo")
    d.add_argument("--port", type=int, default=8765)
    d.add_argument("--no-serve", action="store_true")
    b = sub.add_parser("build", help="build a packet from a local site list (stays local)")
    b.add_argument("--sites", required=True)
    b.add_argument("--seeds")
    b.add_argument("--sources", required=True)
    b.add_argument("--base", help="folder the source paths are relative to (default: the TOML's folder)")
    b.add_argument("--out", required=True)
    b.add_argument("--only", type=int, help="build only the first N queue items (a trial)")
    s = sub.add_parser("serve", help="serve the blind form for a packet")
    s.add_argument("--packet", required=True)
    s.add_argument("--grades", required=True)
    s.add_argument("--port", type=int, default=8765)
    c = sub.add_parser("check", help="check that a packet folder is blind")
    c.add_argument("--packet", required=True)
    p = sub.add_parser("power", help="print the grader hours and power note")
    p.add_argument("--grades", help="a grades folder; its timer replaces the 12-minute assumption")
    a = ap.parse_args(argv)

    if a.cmd == "demo":
        packet = __import__("scoreboard.evalkit.graderkit.demo", fromlist=["run_demo"]).run_demo(a.out)
        if not a.no_serve:
            serve.serve(packet, Path(a.out) / "grades", a.port)
        return 0
    if a.cmd == "build":
        reg = sources.load_registry(a.sources, a.base, strict=False)
        for sid, why in reg.refused:
            print(f"refused source {sid}: {why}", file=sys.stderr)
        seeds = blind.load_site_list(a.seeds) if a.seeds else None
        queue = blind.blind_queue(blind.load_site_list(a.sites), seeds)
        if a.only:
            queue = queue[: a.only]
        m = build.build_packets(queue, reg, a.out)
        print(json.dumps(m["counts"]))
        return 0
    if a.cmd == "serve":
        serve.serve(a.packet, a.grades, a.port)
        return 0
    if a.cmd == "check":
        probs = blind.check_packet(a.packet)
        print("\n".join(probs) if probs else "packet is blind")
        return 1 if probs else 0
    if a.cmd == "power":
        mins = power.minutes_from_grades(a.grades) if a.grades else None
        print(power.note(mins))
        return 0
    return 2

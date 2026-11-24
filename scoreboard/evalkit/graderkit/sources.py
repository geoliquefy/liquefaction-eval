"""Imagery and LiDAR source registry with a licence check.

A source must name a licence the kit knows. A source with an unknown licence, or with a licence file that is not
on disk, stops the build. An NZGD source must also carry the EQC "Important notice" (Map_Layer_Descriptions.pdf
p. 3: it must be reproduced wherever the data or derivatives are reproduced). Each tile carries its licence, its
attribution line and its share rule into the packet. The manifest carries every notice. The form prints the
attribution under the tile and the notice below the tiles.

A LiDAR source may rate each earthquake's change: good, caution, unusable or unchecked (the default). The rating
and its reason go into item.json and the tile caption.
"""
from __future__ import annotations

import csv
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# share: "open" = may go to an outside grader with the attribution line;
#        "local" = NZGD terms clause 6.2: no republishing in unaltered form; giving packets to an outside grader
#                  needs a founder ruling (recorded in the pillar note), the kit only marks it.
LICENCES = {
    "CC-BY-4.0": {"share": "open", "needs_file": False,
                  "text": "Creative Commons Attribution 4.0 International"},
    "NZGD-ToU-2024": {"share": "local", "needs_file": True, "needs_notice": True,
                      "text": "NZGD Terms of Use (2024-11-18), clauses 6.1 and 6.2"},
    "synthetic": {"share": "open", "needs_file": False, "text": "synthetic demo data, no licence needed"},
}
QUALITY = ("good", "caution", "unusable", "unchecked")


class LicenceError(ValueError):
    pass


@dataclass
class ImagerySource:
    id: str
    date: str
    tiles: list  # (path, north, south, east, west)
    licence: str
    attribution: str
    share: str
    notice: str = ""


@dataclass
class LidarSource:
    id: str
    grids: dict  # tag -> (path, flown, flown_end ISO date or "")
    pairs: dict  # earthquake -> {"pre": [tags], "post": [tags]}
    licence: str
    attribution: str
    share: str
    notice: str = ""
    quality: dict = field(default_factory=dict)  # earthquake -> {"rating": one of QUALITY, "why": text}

    def rating(self, eq: str) -> dict:
        q = self.quality.get(eq) or {}
        return {"rating": q.get("rating", "unchecked"), "why": q.get("why", "no quality check recorded")}


@dataclass
class Registry:
    imagery: list = field(default_factory=list)
    lidar: list = field(default_factory=list)
    refused: list = field(default_factory=list)  # (id, reason)


def check_licence(src: dict, base: Path) -> str:
    lic = src.get("licence")
    if lic not in LICENCES:
        raise LicenceError(f"source {src.get('id')!r}: licence {lic!r} is not a known licence {sorted(LICENCES)}")
    if LICENCES[lic]["needs_file"]:
        lf = src.get("licence_file")
        if not lf or not (base / lf).exists():
            raise LicenceError(f"source {src.get('id')!r}: licence file {lf!r} not on disk")
    if not src.get("attribution"):
        raise LicenceError(f"source {src.get('id')!r}: no attribution line")
    if LICENCES[lic].get("needs_notice") and not str(src.get("notice", "")).strip():
        raise LicenceError(f"source {src.get('id')!r}: licence {lic} needs the EQC Important notice (field 'notice')")
    return LICENCES[lic]["share"]


def _read_tiles_csv(csv_path: Path, root: Path) -> list:
    out = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rel = r.get("path") or r.get("rel")
            if r.get("saved") not in (None, "", "1", "True", "true"):
                continue
            p = root / rel
            out.append((p, float(r["north"]), float(r["south"]), float(r["east"]), float(r["west"])))
    return out


def load_registry(toml_path, base=None, strict=True) -> Registry:
    """Read a sources TOML. Paths in it are relative to `base` (default: the TOML's folder).

    strict=True stops at the first licence problem; strict=False records it in `refused` and skips the source.
    """
    toml_path = Path(toml_path)
    cfg = tomllib.loads(toml_path.read_text())
    base = Path(base) if base else toml_path.parent
    reg = Registry()
    for src in cfg.get("source", []):
        try:
            share = check_licence(src, base)
        except LicenceError as e:
            if strict:
                raise
            reg.refused.append((src.get("id"), str(e)))
            continue
        kind = src.get("kind")
        if kind == "imagery_kml":
            tiles = []
            for t in src["tiles"]:
                csv_path, root = base / t["csv"], base / t["root"]
                if csv_path.exists():
                    tiles += _read_tiles_csv(csv_path, root)
                else:
                    reg.refused.append((src["id"], f"tile list {t['csv']} not on disk"))
            reg.imagery.append(ImagerySource(src["id"], src["date"], tiles, src["licence"],
                                             src["attribution"], share, str(src.get("notice", "")).strip()))
        elif kind == "lidar_dem":
            grids = {k: (base / v["file"], v.get("flown", ""), v.get("flown_end", ""))
                     for k, v in src["grids"].items()}
            quality = src.get("quality", {})
            for eq, q in quality.items():
                if q.get("rating") not in QUALITY:
                    raise ValueError(f"source {src['id']!r}: quality rating for {eq} must be one of {QUALITY}")
            reg.lidar.append(LidarSource(src["id"], grids, src["pairs"], src["licence"], src["attribution"], share,
                                         str(src.get("notice", "")).strip(), quality))
        else:
            raise ValueError(f"source {src.get('id')!r}: unknown kind {kind!r}")
    return reg

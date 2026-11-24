"""Synthetic demo set: fake points offshore, fake imagery tiles and fake LiDAR grids. No real site, no real label.

The demo site list (demo/sites.json) also carries fields a grader must never see (cpt, score_band, demo_truth),
so the demo shows the blinding at work: the built packet holds none of them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import blind, build, geo, sources

HERE = Path(__file__).resolve().parent
DEMO_SITES = HERE / "demo" / "sites.json"
DEMO_SEEDS = HERE / "demo" / "seeds.json"
SEED = 20261123


def _tile(rng, size, ejecta: bool, after: bool) -> Image.Image:
    base = rng.normal(0, 1, (size // 8, size // 8))
    g = np.clip(110 + 25 * base, 40, 200)
    rgb = np.stack([g * 0.55, g, g * 0.45], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    for _ in range(6):  # roofs and a road
        x, y = rng.integers(0, size - 120, 2)
        d.rectangle((x, y, x + rng.integers(60, 120), y + rng.integers(60, 120)), fill=(150, 70, 60))
    d.rectangle((0, size // 2 - 40, size, size // 2 + 40), fill=(90, 90, 92))
    if ejecta and after:
        for _ in range(int(rng.integers(8, 16))):
            cx, cy = (size / 2 + rng.normal(0, size / 10, 2)).astype(int)
            r = int(rng.integers(12, 40))
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(170, 165, 150))
    return img.filter(ImageFilter.GaussianBlur(1.2))


def make_sources(out: Path) -> Path:
    """Write synthetic imagery tiles, DEM grids and a sources TOML into out/sources. Returns the TOML path."""
    rng = np.random.Generator(np.random.PCG64(SEED))
    sites = blind.load_site_list(DEMO_SITES) + blind.load_site_list(DEMO_SEEDS)
    src = out / "sources"
    (src / "tiles").mkdir(parents=True, exist_ok=True)
    epochs = {"before": "2009-06-01", "sep2010": "2010-09-10", "feb2011": "2011-02-24",
              "jun2011": "2011-06-20", "dec2011": "2011-12-30"}
    rows = {k: [] for k in epochs}
    for s in sites:
        for k in epochs:
            if k not in ("before", s["earthquake"]):
                continue
            lon0, lat0 = geo.nztm_to_lonlat(s["x"] - 60, s["y"] - 60)
            lon1, lat1 = geo.nztm_to_lonlat(s["x"] + 60, s["y"] + 60)
            name = f"tiles/{k}-{s['queue_id']}.jpg"
            _tile(rng, 1200, bool(s.get("demo_truth")), k != "before").save(src / name, quality=85)
            rows[k].append({"path": name, "north": lat1, "south": lat0, "east": lon1, "west": lon0})
    for k, rs in rows.items():
        with open(src / f"tiles-{k}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "north", "south", "east", "west"])
            w.writeheader()
            w.writerows(rs)
    xs = [s["x"] for s in sites]
    ys = [s["y"] for s in sites]
    x0, y1 = min(xs) - 400, max(ys) + 400
    nx, ny = int((max(xs) + 400 - x0) / 5) + 1, int((y1 - (min(ys) - 400)) / 5) + 1
    gx, gy = np.meshgrid(x0 + 5 * np.arange(nx), y1 - 5 * np.arange(ny))
    pre = 3.0 + 0.002 * (gx - x0) + rng.normal(0, 0.03, gx.shape)
    post = pre + rng.normal(0, 0.03, gx.shape) - 0.05
    for s in sites:
        if s.get("demo_truth"):
            post -= 0.35 * np.exp(-((gx - s["x"]) ** 2 + (gy - s["y"]) ** 2) / (2 * 30.0 ** 2))
    geo.write_geotiff(src / "dem-pre.tif", pre, x0, y1, 5.0)
    geo.write_geotiff(src / "dem-post.tif", post, x0, y1, 5.0)
    attr = "SYNTHETIC DEMO TILE, not a real place"
    lines = []
    for k, date in epochs.items():
        lines += ["[[source]]", f'id = "demo-imagery-{k}"', 'kind = "imagery_kml"', f'date = "{date}"',
                  'licence = "synthetic"', f'attribution = "{attr}"',
                  f'tiles = [{{csv = "tiles-{k}.csv", root = "."}}]', ""]
    lines += ["[[source]]", 'id = "demo-lidar"', 'kind = "lidar_dem"', 'licence = "synthetic"',
              f'attribution = "{attr}"', "[source.grids]",
              'pre = {file = "dem-pre.tif", flown = "synthetic pre"}',
              'post = {file = "dem-post.tif", flown = "synthetic post"}', "[source.pairs]"]
    for eq in blind.EARTHQUAKES:
        lines.append(f'{eq} = {{pre = ["pre"], post = ["post"]}}')
    toml = src / "sources.toml"
    toml.write_text("\n".join(lines) + "\n")
    return toml


def run_demo(out_dir, log=print) -> Path:
    out = Path(out_dir)
    toml = make_sources(out)
    reg = sources.load_registry(toml)
    queue = blind.blind_queue(blind.load_site_list(DEMO_SITES), blind.load_site_list(DEMO_SEEDS))
    build.build_packets(queue, reg, out / "packet", log=log)
    return out / "packet"

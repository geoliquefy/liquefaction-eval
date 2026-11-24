"""Build the packet folder the web form reads: one folder per queue item, a manifest, and the rule."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import blind, geo, tiles

HERE = Path(__file__).resolve().parent


def build_packets(queue: list[dict], registry, out_dir, log=print) -> dict:
    out = Path(out_dir)
    items_dir = out / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    shares, notices = set(), {}
    counts = {"items": 0, "with_imagery_after": 0, "with_imagery_before": 0, "with_lidar": 0, "lidar_quality": {}}
    order = sorted(queue, key=lambda q: (q["earthquake"], q["queue_id"]))  # group by earthquake for the DEM cache
    for q in order:
        qid, eqk = q["queue_id"], q["earthquake"]
        eq = blind.EARTHQUAKES[eqk]
        d = items_dir / qid
        d.mkdir(exist_ok=True)
        lon, lat = geo.nztm_to_lonlat(q["x"], q["y"])
        ev = []
        for role, candidates in tiles.pick_imagery(registry, eq):
            name = f"imagery-{role}.png"
            for src in candidates:  # first source that covers the point centre
                meta = tiles.cut_imagery(src, q["x"], q["y"], d / name)
                if meta:
                    ev.append({"file": name, "kind": "imagery", "role": role, "source": src.id, "date": src.date,
                               "licence": src.licence, "attribution": src.attribution, "share": src.share, **meta})
                    counts[f"with_imagery_{role}"] += 1
                    shares.add(src.share)
                    if src.notice:
                        notices[src.id] = src.notice
                    break
        for src in registry.lidar:
            meta = tiles.cut_lidar(src, eqk, q["x"], q["y"], d / "lidar-change.png", window_end=eq["next"])
            if meta:
                ev.append({"file": "lidar-change.png", "kind": "lidar_change", "role": "change",
                           "source": src.id, "licence": src.licence, "attribution": src.attribution,
                           "share": src.share, **meta})
                counts["with_lidar"] += 1
                counts["lidar_quality"][meta["quality"]] = counts["lidar_quality"].get(meta["quality"], 0) + 1
                shares.add(src.share)
                if src.notice:
                    notices[src.id] = src.notice
                break
        item = {"queue_id": qid, "earthquake": eqk, "earthquake_name": eq["name"], "window_start": eq["date"],
                "window_end": eq["next"], "lon": round(float(lon), 6), "lat": round(float(lat), 6),
                "disc_radius_m": tiles.DISC_M, "evidence": ev,
                "missing": [] if ev else ["no imagery and no LiDAR change on disk for this window (reason R1)"]}
        (d / "item.json").write_text(json.dumps(item, indent=1))
        counts["items"] += 1
    manifest = {
        "queue": [q["queue_id"] for q in sorted(queue, key=lambda q: q["queue_id"])],
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "share_rule": "local" if "local" in shares else "open",
        "counts": counts,
        "notices": [{"source": s, "notice": t} for s, t in sorted(notices.items())],
        "refused_sources": [{"source": s, "reason": r} for s, r in registry.refused],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    shutil.copy(HERE / "RULE.md", out / "RULE.md")
    problems = blind.check_packet(out)
    if problems:
        raise blind.BlindError("packet is not blind: " + "; ".join(problems[:10]))
    log(f"built {counts['items']} items in {out}: {counts}; share rule {manifest['share_rule']}")
    return manifest

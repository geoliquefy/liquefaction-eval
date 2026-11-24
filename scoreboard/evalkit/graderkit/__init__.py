"""Grader kit for the probability-sampled label audit (eval plan step 1d, work list item E3).

Parts:
  blind.py    turns a local site list into a blind queue (only queue id, point and earthquake survive)
  sources.py  the imagery and LiDAR source registry, with a licence check per source
  geo.py      NZTM and WGS84 helpers and a small GeoTIFF reader (PIL only, no GDAL)
  tiles.py    cuts an imagery tile and a LiDAR change tile per queue item
  build.py    builds the packet folder that the web form reads
  serve.py    serves the blind form on 127.0.0.1 and writes grades to local JSON
  power.py    grader hours and the power of a smaller sample
  demo.py     a synthetic demo set (no real site, no real label)

Site lists, tiles and grades built from NZGD data stay local (NZGD terms clause 6.2).
Only code, the rule (RULE.md), the power note and the synthetic demo set are committed.
"""

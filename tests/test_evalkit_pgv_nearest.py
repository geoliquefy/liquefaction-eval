"""B8 (eval M4): the nearest-node PGV option that scorecard v3 registers (section 5: "nearest cell, no fill").
Synthetic fixture only; no real grid and no real label is read here. The default (bilinear) must not change.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scoreboard.evalkit import pgv as PGV  # noqa: E402

GRID = Path(__file__).resolve().parent / "fixtures" / "shakemap_grid_synthetic.xml"


def test_nearest_takes_the_nearest_node_value():
    g = PGV.read_grid(GRID)
    # PGV = 10 + 1000 (lon - 172) + 500 (lat + 43.60) at the nodes (lon step 0.01, lat step 0.01)
    lon = np.array([172.004, 172.006, 172.026, 172.0349, 171.9951])
    lat = np.array([-43.596, -43.594, -43.581, -43.5851, -43.6049])
    v = PGV.sample_nearest(g, "PGV", lon, lat)
    node_lon, node_lat = np.round(lon, 2), np.round(lat, 2)
    want = 10 + 1000 * (node_lon - 172.0) + 500 * (node_lat + 43.60)
    assert np.allclose(v, want, atol=1e-9)
    bil = PGV.sample(g, "PGV", lon[:3], lat[:3])
    assert not np.allclose(bil, v[:3])                       # the option changes the value between nodes


def test_nearest_outside_by_more_than_half_a_cell_is_nan():
    g = PGV.read_grid(GRID)
    v = PGV.sample_nearest(g, "PGV", np.array([171.9949, 172.0351, 172.01, np.nan]),
                           np.array([-43.59, -43.59, -43.5749, -43.59]))
    assert np.isnan(v).all()


def test_nearest_on_an_empty_node_is_nan_and_never_filled():
    text = GRID.read_text().replace("172.0100 -43.5900 6.0 21.0000 25.0000", "172.0100 -43.5900 6.0 21.0000 nan")
    g = PGV.parse_grid(text)
    v = PGV.sample_nearest(g, "PGV", np.array([172.011, 172.004]), np.array([-43.591, -43.596]))
    assert np.isnan(v[0]) and np.isfinite(v[1])             # only the empty node itself
    b = PGV.score_baseline("syn-grid-1", ("syn:a", "syn:b"), [172.011, 172.004], [-43.591, -43.596], g,
                           sampling="nearest")
    assert b.entry is None and b.summary["status"].startswith("NOT AVAILABLE: PGV MISSING AT 1 OF 2 SITES")
    assert b.summary["sampling"] == PGV.SAMPLINGS["nearest"] and PGV.kit_built(b)


def test_score_baseline_nearest_row_and_default_unchanged():
    g = PGV.read_grid(GRID)
    ids = ("syn:a", "syn:b")
    lon, lat = [172.014, 172.026], [-43.594, -43.586]
    near = PGV.score_baseline("syn-grid-1", ids, lon, lat, g, sampling="nearest")
    assert np.allclose(near.entry.p, [25.0, 45.0])
    assert "nearest grid node" in near.entry.manifest.statement and PGV.kit_built(near)
    assert near.summary["sampling"] == PGV.SAMPLINGS["nearest"]
    default = PGV.score_baseline("syn-grid-1", ids, lon, lat, g)
    explicit = PGV.score_baseline("syn-grid-1", ids, lon, lat, g, sampling="bilinear")
    assert default.summary == explicit.summary and np.array_equal(default.entry.p, explicit.entry.p)
    assert "bilinear at each site" in default.entry.manifest.statement
    assert default.summary["sampling"] == PGV.SAMPLINGS["bilinear"]
    with pytest.raises(PGV.PgvRefused, match="sampling"):
        PGV.score_baseline("syn-grid-1", ids, lon, lat, g, sampling="cubic")

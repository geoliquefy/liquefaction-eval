"""Tests for evalkit.zhu_grid: the shared Zhu 2017 world grid reader that takes the row order from the grid's own
metadata (G1+G5+G6, eval M4/M5; pillar docs/evals/ERRATUM-ZHU-ROW-ORDER-2026-11-23.md). No em dashes.

A planted asymmetric grid: hazard only in the north band, none in the south band. A site in the north must read the
hazard; a site in the south must read 0. The grid is stored in each order the reader must handle."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from scoreboard.evalkit import world_inputs as W  # noqa: E402
from scoreboard.evalkit import zhu_cells as ZC  # noqa: E402
from scoreboard.evalkit import zhu_grid as ZG  # noqa: E402

X = np.linspace(170.0, 170.9, 10)
Y_ASC = np.linspace(-44.0, -43.3, 8)                       # south first
NORTH_FIRST = np.zeros((8, 10))
NORTH_FIRST[:2] = 0.3                                      # the two north rows carry hazard
NORTH_FIRST[2, 3] = np.nan                                 # one empty cell, so the NaN pattern is not symmetric
NORTH_SITE = (170.45, -43.32)
SOUTH_SITE = (170.45, -43.98)
WRITER = "Created with python MultiHazardGrid.save(/data/gfdata/outputs/x/version.001/x_zhu_2017_general.hdf5)"


def _h5(path: Path, y: np.ndarray, model: np.ndarray, history: str | None = WRITER) -> Path:
    with h5py.File(path, "w") as f:
        f["x"], f["y"], f["model"] = X, y, model.astype(np.float32)
        if history is not None:
            f.attrs["history"] = history
    return path


def _tif(path: Path, data: np.ndarray, top_lat: float, scale_y: float, deflate: bool = True) -> Path:
    """Write a one-strip little-endian float32 GeoTIFF with a tiepoint at raster (0, 0), PixelIsArea."""
    h, w = data.shape
    raw = np.ascontiguousarray(data, "<f4").tobytes()
    body = zlib.compress(raw) if deflate else raw
    dx = float(X[1] - X[0])
    tie = struct.pack("<6d", 0, 0, 0, float(X[0]) - dx / 2, top_lat, 0)
    scale = struct.pack("<3d", dx, scale_y, 0)
    geokeys = struct.pack("<8H", 1, 1, 0, 1, 1025, 0, 1, 1)
    extra = [(33550, 12, 3, scale), (33922, 12, 6, tie), (34735, 3, 8, geokeys)]
    n_tags = 9 + len(extra)
    ifd_off = 8
    data_off = ifd_off + 2 + 12 * n_tags + 4
    blobs, entries, cur = b"", [], data_off
    for tag, typ, cnt, payload in extra:
        entries.append((tag, typ, cnt, cur))
        blobs += payload
        cur += len(payload)
    strip_off = cur
    base = [(256, 3, 1, w), (257, 3, 1, h), (258, 3, 1, 32), (259, 3, 1, 8 if deflate else 1), (262, 3, 1, 1),
            (273, 4, 1, strip_off), (277, 3, 1, 1), (279, 4, 1, len(body)), (339, 3, 1, 3)]
    allt = sorted(base + entries)
    ifd = struct.pack("<H", len(allt))
    for tag, typ, cnt, val in allt:
        if typ == 3 and cnt == 1:
            ifd += struct.pack("<HHIHH", tag, typ, cnt, val, 0)
        else:
            ifd += struct.pack("<HHII", tag, typ, cnt, val)
    ifd += struct.pack("<I", 0)
    path.write_bytes(b"II*\x00" + struct.pack("<I", ifd_off) + ifd + blobs + body)
    return path


def _twin_north_up(h5path: Path, north_first: np.ndarray) -> Path:
    dy = float(Y_ASC[1] - Y_ASC[0])
    return _tif(ZG.twin_path(h5path), north_first, float(Y_ASC.max()) + dy / 2, dy)


def _read(grid: ZG.ZhuGrid):
    r = ZC.read_cells(grid.x, grid.y, grid.model, np.array([NORTH_SITE[0], SOUTH_SITE[0]]),
                      np.array([NORTH_SITE[1], SOUTH_SITE[1]]))
    return r.values


# --- the USGS layout: y ascending, rows north first --------------------------------------------------------------
def test_usgs_layout_writer_convention_reads_north_hazard_in_the_north(tmp_path):
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, NORTH_FIRST)
    g = ZG.load(p, ZG.ROW_ORDER_GEOREF)
    assert g.orientation.row0 == "north" and g.orientation.source == "writer-convention"
    north, south = _read(g)
    assert north == pytest.approx(0.3) and south == 0.0
    assert np.all(np.diff(g.y) > 0)


def test_usgs_layout_as_stored_is_the_mirrored_v2_read(tmp_path):
    """The default read is unchanged: the arrays as stored, which put the north hazard in the south."""
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, NORTH_FIRST)
    g = ZG.load(p)
    assert g.row_order == ZG.ROW_ORDER_AS_STORED == ZG.DEFAULT_ROW_ORDER and g.orientation is None
    with h5py.File(p) as f:
        assert np.array_equal(g.model, f["model"][()], equal_nan=True) and np.array_equal(g.y, f["y"][()])
    north, south = _read(g)
    assert north == 0.0 and south == pytest.approx(0.3)


@pytest.mark.parametrize("deflate", [True, False])
def test_usgs_layout_twin_decides(tmp_path, deflate):
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, NORTH_FIRST, history=None)
    dy = float(Y_ASC[1] - Y_ASC[0])
    _tif(ZG.twin_path(p), NORTH_FIRST, float(Y_ASC.max()) + dy / 2, dy, deflate=deflate)
    g = ZG.load(p, ZG.ROW_ORDER_GEOREF)
    assert g.orientation.source == "tif-twin" and g.orientation.row0 == "north"
    assert g.orientation.detail["match_as_stored"] == 1.0 and g.orientation.detail["match_reversed"] < 0.99
    north, south = _read(g)
    assert north == pytest.approx(0.3) and south == 0.0


def test_twin_wins_over_writer_convention(tmp_path):
    """A file from the known writer whose rows are in fact south first: the twin, not the convention, decides."""
    south_first = NORTH_FIRST[::-1].copy()
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, south_first)
    _twin_north_up(p, NORTH_FIRST)
    g = ZG.load(p, ZG.ROW_ORDER_GEOREF)
    assert g.orientation.source == "tif-twin" and g.orientation.row0 == "south" and not g.orientation.reverse
    north, south = _read(g)
    assert north == pytest.approx(0.3) and south == 0.0


# --- other storage orders -----------------------------------------------------------------------------------------
def test_y_descending_rows_follow_y(tmp_path):
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC[::-1].copy(), NORTH_FIRST, history="something else")
    g = ZG.load(p, ZG.ROW_ORDER_GEOREF)
    assert g.orientation.source == "y-axis"
    assert np.all(np.diff(g.y) > 0)
    north, south = _read(g)
    assert north == pytest.approx(0.3) and south == 0.0


def test_south_first_rows_with_unknown_writer_and_no_twin_are_refused(tmp_path):
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, NORTH_FIRST[::-1].copy(), history="made by hand")
    with pytest.raises(ZG.ZhuRowOrderUnknown):
        ZG.load(p, ZG.ROW_ORDER_GEOREF)
    assert ZG.load(p).row_order == ZG.ROW_ORDER_AS_STORED          # the default read never consults the order


def test_twin_that_matches_neither_reading_is_refused(tmp_path):
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, NORTH_FIRST)
    other = np.full_like(NORTH_FIRST, 0.11)
    _twin_north_up(p, other)
    with pytest.raises(ZG.ZhuRowOrderUnknown):
        ZG.load(p, ZG.ROW_ORDER_GEOREF)


def test_twin_that_does_not_decode_falls_back_to_the_writer(tmp_path):
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, NORTH_FIRST)
    ZG.twin_path(p).write_bytes(b"MM\x00*" + b"\x00" * 16)          # big-endian: not handled
    g = ZG.load(p, ZG.ROW_ORDER_GEOREF)
    assert g.orientation.source == "writer-convention" and "twin_not_used" in g.orientation.detail
    assert _read(g)[0] == pytest.approx(0.3)


def test_twin_with_another_shape_is_refused(tmp_path):
    """A twin that decodes but has another shape contradicts the grid: refused, not a writer fallback (review N4)."""
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, NORTH_FIRST)
    dy = float(Y_ASC[1] - Y_ASC[0])
    _tif(ZG.twin_path(p), NORTH_FIRST[:-1], float(Y_ASC.max()) + dy / 2, dy)
    with pytest.raises(ZG.ZhuRowOrderUnknown, match="twin shape"):
        ZG.load(p, ZG.ROW_ORDER_GEOREF)


def test_twin_with_tiepoint_at_neither_end_is_refused(tmp_path):
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, NORTH_FIRST)
    dy = float(Y_ASC[1] - Y_ASC[0])
    _tif(ZG.twin_path(p), NORTH_FIRST, float(Y_ASC.mean()), dy)         # tiepoint in the middle of the grid
    with pytest.raises(ZG.ZhuRowOrderUnknown, match="neither end"):
        ZG.load(p, ZG.ROW_ORDER_GEOREF)


# --- the Canterbury path: ref_usgs_gf.read_product (review S2; no pillar data needed) -----------------------------
def _gf_product(path: Path, model: np.ndarray, history: str | None = WRITER) -> Path:
    _h5(path, Y_ASC, model, history)
    with h5py.File(path, "a") as f:
        f.create_group("metadata/model/description").attrs["shakemap"] = "planted"
    return path


@pytest.mark.parametrize("twin", [False, True])
def test_read_product_reads_north_hazard_in_the_north(tmp_path, twin):
    """read_product (every Canterbury Zhu read) must put the north band in the north, with or without a twin. A
    read_product that reads the rows as stored (Canterbury mirrored) fails here."""
    from scoreboard import ref_usgs_gf as U
    p = _gf_product(tmp_path / "zhu_2017_general.hdf5", NORTH_FIRST)
    if twin:
        _twin_north_up(p, NORTH_FIRST)
    G = U.read_product(p)
    north, south = G.at(np.array([NORTH_SITE[0], SOUTH_SITE[0]]), np.array([NORTH_SITE[1], SOUTH_SITE[1]]))
    assert north == pytest.approx(0.3) and south == 0.0
    assert np.all(np.diff(G.y) > 0) and G.sha256 == W.sha_file(p) and G.shakemap == "planted"
    with h5py.File(p) as f:                                     # the old fixed flip, for the record
        assert np.array_equal(G.L, np.asarray(f["model"][()], float)[::-1], equal_nan=True)


def test_read_product_follows_the_twin_on_a_south_first_grid(tmp_path):
    from scoreboard import ref_usgs_gf as U
    p = _gf_product(tmp_path / "zhu_2017_general.hdf5", NORTH_FIRST[::-1].copy())
    _twin_north_up(p, NORTH_FIRST)
    G = U.read_product(p)
    north, south = G.at(np.array([NORTH_SITE[0], SOUTH_SITE[0]]), np.array([NORTH_SITE[1], SOUTH_SITE[1]]))
    assert north == pytest.approx(0.3) and south == 0.0


def test_read_product_refuses_a_grid_with_no_known_order(tmp_path):
    from scoreboard import ref_usgs_gf as U
    p = _gf_product(tmp_path / "zhu_2017_general.hdf5", NORTH_FIRST, history="made by hand")
    with pytest.raises(ZG.ZhuRowOrderUnknown):
        U.read_product(p)


def test_bad_row_order_name(tmp_path):
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, NORTH_FIRST)
    with pytest.raises(ValueError):
        ZG.load(p, "flip")


def test_ruling46_reader_on_both_reads(tmp_path):
    """The ruling 46 reader gets the oriented arrays: the north yes-site keeps its hazard only under georef."""
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, NORTH_FIRST)
    lon = np.array([NORTH_SITE[0], SOUTH_SITE[0]])
    lat = np.array([NORTH_SITE[1], SOUTH_SITE[1]])
    lim = np.array([0.5, 0.5]), np.array([30.0, 30.0]), np.array([300.0, 300.0])
    for order, want in ((ZG.ROW_ORDER_AS_STORED, [0.0, 0.3]), (ZG.ROW_ORDER_GEOREF, [0.3, 0.0])):
        g = ZG.load(p, order)
        c = ZC.cell_input_ruled(g.x, g.y, g.model, lon, lat, np.array([1, 0]), *lim)
        assert np.allclose(c.values, want)


# --- D8 wiring ----------------------------------------------------------------------------------------------------
def test_as_dict_names_the_row_order_only_off_default(tmp_path):
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, NORTH_FIRST)
    assert "row_order" not in ZG.load(p).as_dict()
    d = ZG.load(p, ZG.ROW_ORDER_GEOREF).as_dict()
    assert d["row_order"] == ZG.ROW_ORDER_GEOREF and d["orientation"]["row0"] == "north"
    assert d["sha256"] == W.sha_file(p)


def test_run_zhu_records_the_row_order(tmp_path):
    import pandas as pd
    p = _h5(tmp_path / "zhu_2017_general.hdf5", Y_ASC, NORTH_FIRST)
    cases = pd.DataFrame({"earthquake": ["Quake", "Quake"], "lat": [NORTH_SITE[1], SOUTH_SITE[1]],
                          "lon": [NORTH_SITE[0], SOUTH_SITE[0]]})
    v0, r0 = W.run_zhu(cases, {"Quake": ZG.load(p).as_dict()})
    v1, r1 = W.run_zhu(cases, {"Quake": ZG.load(p, ZG.ROW_ORDER_GEOREF).as_dict()})
    assert np.allclose(v0, [0.0, 0.3]) and np.allclose(v1, [0.3, 0.0])
    assert "row_order" not in r0["Quake"] and r1["Quake"]["row_order"] == ZG.ROW_ORDER_GEOREF


# --- one real world grid, data only (skipped when the pillar data are absent) -------------------------------------
HYOGO = (Path(__file__).resolve().parents[1] / ".." / ".." / "data").resolve()
_HY = ("comcat-shakemap-nonus-2026-11-07/raw/usp0006rew/ground-failure-preferred/us-p0006rew-v1-1597402383402/"
       "zhu_2017_general.hdf5")


def _pillar_data():
    for base in (Path(__file__).resolve().parents[i] / "data" for i in range(1, 6)):
        if (base / _HY).exists():
            return base / _HY
    return None


@pytest.mark.skipif(_pillar_data() is None, reason="pillar data not on disk")
def test_hyogoken_nambu_points():
    """Port Island (reclaimed, liquefied in 1995) reads hazard and the Rokko mountains read 0 under georef; the
    stored read gives the reverse (review 2 finding R2-2)."""
    g = ZG.load(_pillar_data(), ZG.ROW_ORDER_GEOREF)
    assert g.orientation.source == "writer-convention" and g.orientation.row0 == "north"
    r = ZC.read_cells(g.x, g.y, g.model, np.array([135.215, 135.25]), np.array([34.665, 34.77]))
    assert r.values[0] > 0.2 and r.values[1] == 0.0
    s = ZG.load(_pillar_data())
    r = ZC.read_cells(s.x, s.y, s.model, np.array([135.215, 135.25]), np.array([34.665, 34.77]))
    assert np.isnan(r.values[0]) and r.values[1] > 0.2


# --- the D8 predict --zhu-row-order flag (review N3) ---------------------------------------------------------------
def _d8_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "d8_world_inputs_t", Path(__file__).resolve().parents[1] / "scripts" / "d8_world_inputs_2026_11_23.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("argv, want", [(["predict"], [0.0, 0.3]),
                                        (["predict", "--zhu-row-order", ZG.ROW_ORDER_AS_STORED], [0.0, 0.3]),
                                        (["predict", "--zhu-row-order", ZG.ROW_ORDER_GEOREF], [0.3, 0.0])])
def test_d8_predict_passes_the_row_order_to_the_zhu_read(tmp_path, monkeypatch, argv, want):
    """The flag reaches the grid read: georef puts the north hazard in the north; no flag keeps the first-slice
    (as stored) read. A predict that ignored the flag fails the georef case."""
    import pandas as pd
    D = _d8_script()
    rel = "g/zhu_2017_general.hdf5"
    (tmp_path / "g").mkdir()
    _h5(tmp_path / rel, Y_ASC, NORTH_FIRST)
    seen = {}
    monkeypatch.setattr(D, "DATA", tmp_path)
    monkeypatch.setattr(D, "OUT", tmp_path / "out")
    monkeypatch.setattr(D, "_load_helpers", lambda: {})
    monkeypatch.setattr(D, "_kit_commit", lambda: "test")
    monkeypatch.setattr(W, "ZHU_GRIDS", {"Quake": rel})
    monkeypatch.setattr(W, "predict", lambda *a, **k: seen.update(k))
    D.main(argv)
    g = seen["zhu_grids"]["Quake"]
    assert g["path"] == "data/" + rel
    assert g.get("row_order", ZG.ROW_ORDER_AS_STORED) == (argv[2] if len(argv) > 2 else ZG.ROW_ORDER_AS_STORED)
    cases = pd.DataFrame({"earthquake": ["Quake", "Quake"], "lat": [NORTH_SITE[1], SOUTH_SITE[1]],
                          "lon": [NORTH_SITE[0], SOUTH_SITE[0]]})
    v, _ = W.run_zhu(cases, seen["zhu_grids"])
    assert np.allclose(v, want)


def test_d8_predict_row_order_choices_are_the_reader_names(capsys):
    D = _d8_script()
    with pytest.raises(SystemExit):
        D.main(["predict", "--zhu-row-order", "flip"])
    err = capsys.readouterr().err
    assert all(name in err for name in ZG.ROW_ORDERS)

"""Ruling 46 Zhu reader (B8, scorecard v3 registration section 4.3). Synthetic grids only. No em dashes."""
import numpy as np
import pytest

from scoreboard.evalkit import zhu_cells as Z

GX = np.array([172.0, 172.1, 172.2])
GY = np.array([-43.6, -43.5])
M = np.array([[0.2, np.nan, 0.4], [np.nan, 0.6, np.nan]])
# sites: on-grid value; empty cell failing PGA; empty failing Vs30; empty passing all three; empty with PGV unknown;
# outside the extent
LON = np.array([172.0, 172.1, 172.0, 172.2, 172.2, 175.0])
LAT = np.array([-43.6, -43.6, -43.5, -43.5, -43.5, -43.5])
Y = np.array([1, 0, 0, 1, 0, 1])
PGA = np.array([0.3, 0.05, 0.3, 0.3, 0.3, 0.3])
PGV = np.array([20.0, 20.0, 20.0, 20.0, np.nan, 20.0])
VS = np.array([200.0, 200.0, 700.0, 200.0, 200.0, 200.0])


def test_split_by_reason():
    c = Z.cell_input_ruled(GX, GY, M, LON, LAT, Y, PGA, PGV, VS)
    assert c.values.tolist() == [0.2, 0.0, 0.0]
    assert c.label.tolist() == [1, 0, 0]
    r = c.rec
    assert (r["n_model_zero"], r["n_off_grid"], r["n_not_checkable"], r["n_outside_extent"]) == (2, 2, 1, 1)
    assert r["n_model_zero_by_limit"] == {"pga": 1, "pgv": 0, "vs30": 1}
    assert (r["n_yes_dropped"], r["n_no_dropped"]) == (2, 1)


def test_a_known_failing_limit_wins_over_a_missing_input():
    pgv = PGV.copy()
    pgv[1] = np.nan          # PGA still fails at site 1, so it stays MODEL ZERO
    c = Z.cell_input_ruled(GX, GY, M, LON, LAT, Y, PGA, pgv, VS)
    assert c.rec["n_model_zero"] == 2


def test_no_inputs_means_every_empty_cell_is_dropped():
    nan = np.full(LON.size, np.nan)
    c = Z.cell_input_ruled(GX, GY, M, LON, LAT, Y, nan, nan, nan)
    assert c.rec["n_model_zero"] == 0 and c.rec["n_not_checkable"] == 4
    assert c.values.tolist() == [0.2]


def test_on_grid_values_equal_the_old_reader():
    old = Z.cell_input(GX, GY, M, LON, LAT, Y)
    new = Z.cell_input_ruled(GX, GY, M, LON, LAT, Y, PGA, PGV, VS)
    assert set(old.values.tolist()) <= set(new.values.tolist())


def test_record_holds_counts_only():
    c = Z.cell_input_ruled(GX, GY, M, LON, LAT, Y, PGA, PGV, VS)
    for k, v in c.rec.items():
        assert isinstance(v, (int, str, dict)), k
    assert "ruling 46" in c.note


def test_shapes_checked():
    with pytest.raises(ValueError):
        Z.cell_input_ruled(GX, GY, M, LON, LAT, Y, PGA[:3], PGV, VS)

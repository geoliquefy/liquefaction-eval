"""Planted tests for the Level 0 recalibration map (ruling 56b). Synthetic data only; no real label is read."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import norm

from scoreboard.evalkit import l0_recal as R
from scoreboard.evalkit import metrics as M


def _rng(seed=20261124):
    return np.random.default_rng(seed)


def test_planted_miscalibrated_model_is_fixed():
    rng = _rng()
    x = rng.normal(0.0, 3.0, 20000)
    p = expit(x)                                     # the model reports expit(x): far too sharp
    y = (rng.random(x.size) < expit(0.7 + 0.35 * x)).astype(int)
    f = R.fit(p, y)
    assert abs(f["a"] - 0.7) < 0.08 and abs(f["b"] - 0.35) < 0.03
    q = R.apply(p, f["a"], f["b"])
    assert M.log_loss(y, q, R.CLIP) < M.log_loss(y, p, R.CLIP) - 0.05
    assert abs(M.recal_slope(y, q, R.CLIP) - 1.0) < 1e-6      # in-sample, the fitted map is calibrated exactly
    assert abs(M.citl(y, q, R.CLIP)) < 1e-6


def test_planted_level0_like_model_with_exact_zeros():
    rng = _rng(7)
    lnk = rng.normal(0.5, 1.2, 8000)
    lnk[:800] = np.inf                               # no liquefiable layer: p exactly 1 in the tail
    lnk[800:1600] = -np.inf                          # p exactly 0
    p = norm.cdf((lnk - 0.2) / 0.2)                  # Level 0 world form, sigma_e = 0
    truth = norm.cdf((np.clip(lnk, -4, 4) - 0.2) / 1.5) * 0.8 + 0.1
    y = (rng.random(lnk.size) < truth).astype(int)
    assert (p == 0).mean() > 0.05 and (p == 1).mean() > 0.05
    f = R.fit(p, y)
    assert 0 < f["b"] < 0.5
    q = R.apply(p, f["a"], f["b"])
    assert M.log_loss(y, q, R.CLIP) < np.log(2) < M.log_loss(y, p, R.CLIP)
    assert np.all((q > 0) & (q < 1))


def test_planted_well_calibrated_model_is_left_near_identity():
    rng = _rng(11)
    x = rng.normal(0.0, 1.5, 20000)
    p = expit(x)
    y = (rng.random(x.size) < p).astype(int)
    f = R.fit(p, y)
    assert abs(f["a"]) < 0.06 and abs(f["b"] - 1.0) < 0.05
    q = R.apply(p, f["a"], f["b"])
    assert np.max(np.abs(q - p)) < 0.03
    assert abs(M.log_loss(y, q, R.CLIP) - M.log_loss(y, p, R.CLIP)) < 0.002


def test_map_keeps_ranking():
    rng = _rng(3)
    p = rng.random(500)
    q = R.apply(p, -0.4, 0.3)
    assert np.array_equal(np.argsort(p, kind="stable"), np.argsort(q, kind="stable"))


def test_negative_slope_stops_s5():
    rng = _rng(5)
    x = rng.normal(0.0, 2.0, 5000)
    y = (rng.random(x.size) < expit(-0.8 * x)).astype(int)
    with pytest.raises(R.RecalStop, match="S5"):
        R.fit(expit(x), y)


def test_separation_stops_s4():
    x = np.linspace(-3, 3, 400)
    y = (x > 0).astype(int)
    with pytest.raises(R.RecalStop, match="S4"):
        R.fit(expit(x), y)


def test_constant_input_stops_s4():
    y = np.r_[np.ones(150, int), np.zeros(150, int)]
    with pytest.raises(R.RecalStop, match="S4"):
        R.fit(np.full(300, 0.3), y)


@pytest.mark.parametrize("n_yes,n_no", [(100, 99), (29, 400), (400, 29)])
def test_count_gate_stops_s3(n_yes, n_no):
    rng = _rng(9)
    y = np.r_[np.ones(n_yes, int), np.zeros(n_no, int)]
    with pytest.raises(R.RecalStop, match="S3"):
        R.fit(rng.random(y.size), y)


@pytest.mark.parametrize("p", [np.array([0.2, np.nan]), np.array([0.2, 1.2]), np.array([-0.1, 0.5])])
def test_bad_inputs_refused(p):
    with pytest.raises(ValueError):
        R.fit(p, np.array([0, 1]))


def test_bad_labels_refused():
    with pytest.raises(ValueError):
        R.fit(np.array([0.2, 0.5]), np.array([0, 2]))


def test_map_file_round_trip_and_sha_check(tmp_path):
    rec = R.map_record(-0.25, 0.4, registration_sha256="0" * 64, kit_commit="abc")
    path = tmp_path / "map.json"
    sha = R.write_map(path, rec)
    got = R.load_map(path, sha)
    assert got == rec
    p = np.array([0.0, 0.1, 0.5, 1.0])
    assert np.allclose(R.apply_map(p, got), R.apply(p, -0.25, 0.4))
    with pytest.raises(FileExistsError):
        R.write_map(path, rec)
    with pytest.raises(ValueError, match="sha256"):
        R.load_map(path, "f" * 64)
    path.write_bytes(path.read_bytes().replace(b'"b":0.4', b'"b":0.5'))
    with pytest.raises(ValueError, match="sha256"):
        R.load_map(path, sha)


def test_map_record_refuses_override_and_missing_keys(tmp_path):
    with pytest.raises(ValueError):
        R.map_record(0.0, 1.0, map_id="other")
    with pytest.raises(ValueError):
        R.write_map(tmp_path / "m.json", {"a": 0.0, "b": 1.0})


def test_summary_keys():
    rng = _rng(13)
    p = rng.random(300)
    y = (rng.random(300) < p).astype(int)
    s = R.summary(y, p)
    assert set(s) >= {"n", "yes_share", "log_loss@1e-06", "brier", "citl", "slope", "share_exact_0", "share_exact_1"}


def _mutate(field, value):
    def f(rec):
        rec[field] = value
        return rec
    return f


@pytest.mark.parametrize("mutate, match", [
    (_mutate("b", 0.0), "b <= 0"),
    (_mutate("b", -0.3), "b <= 0"),
    (_mutate("family", "other-family"), "map id, family or clip"),
    (_mutate("map_id", "other-map"), "map id, family or clip"),
    (_mutate("clip", 1e-9), "map id, family or clip"),
])
def test_load_map_refuses_valid_sha_wrong_content(tmp_path, mutate, match):
    # Review 1 N1: a file whose sha256 matches but whose content is wrong is still refused.
    rec = mutate(R.map_record(-0.25, 0.4, registration_sha256="0" * 64, kit_commit="abc"))
    path = tmp_path / "map.json"
    sha = R.write_map(path, rec)
    with pytest.raises(ValueError, match=match):
        R.load_map(path, sha)


def test_load_map_refuses_valid_sha_non_canonical(tmp_path):
    # Review 1 N1: same record, non-canonical bytes, matching sha256: refused.
    import json
    rec = R.map_record(-0.25, 0.4, registration_sha256="0" * 64, kit_commit="abc")
    data = (json.dumps(rec, indent=2) + "\n").encode("ascii")
    assert data != R.canonical_bytes(rec)
    path = tmp_path / "map.json"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="canonical form"):
        R.load_map(path, R.sha256_bytes(data))

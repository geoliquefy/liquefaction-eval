"""Settings sheets drift guard (G1+G5+G6, eval M3, own-work item A3). No em dashes.

The pillar settings sheets (`docs/evals/settings-sheets/lsn-fragility.md`, `lpi.md`, `bi14-level0.md`) state the
values of the two chain settings ids. Authors read those sheets to confirm or correct our run. If a value in
`scoreboard/evalkit/chain_settings.py` changes, these tests fail, so the sheets get revised in the same change.
Hand values only; no data, no synthetic profiles.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from scoreboard.evalkit import chain_settings as CS  # noqa: E402

# What the sheets print (lsn-fragility.md rows 2, 3, 4, 6; lpi.md rows 1 and 4; bi14-level0.md rows 3 to 5).
SHEET = {
    "default-2026-11": {"ic_cut": 2.6, "fc": "bi14-cfc0", "tip": "qc", "area_ratio": None, "lpi_window_m": 20.0,
                        "lsn_window_m": 15.0},
    "authors-maurer-2026-11-19": {"ic_cut": 2.50, "fc": "maurer2019-christchurch", "tip": "qt", "area_ratio": 0.80,
                                  "lpi_window_m": 20.0, "lsn_window_m": 20.0},
}

# Ids used only by a registered parity check, not by any card, so not on the sheets. Each is declared here.
# peer-global-2026-11-23: item A4, pillar docs/evals/PARITY-M3-ROUND3-REGISTRATION-2026-11-23.md section 3.
PARITY_ONLY = {"peer-global-2026-11-23"}


def test_ids_are_the_two_the_sheets_name():
    assert CS.DEFAULT_ID == "default-2026-11"
    assert CS.AUTHORS_ID == "authors-maurer-2026-11-19"
    assert set(CS.SETTINGS) == set(SHEET) | PARITY_ONLY


@pytest.mark.parametrize("sid", sorted(SHEET))
def test_setting_values_match_the_sheets(sid):
    assert CS.setting(sid) == SHEET[sid], (
        f"chain_settings {sid} differs from the pillar settings sheets; revise docs/evals/settings-sheets/ too")


def test_fines_equations_match_the_sheets():
    # Sheet: default FC = 80 Ic - 137 (B&I eq. 2.30, CFC 0); authors FC = 80.645 Ic - 128.5967; both limited 0 to 100.
    ic = np.array([1.0, 1.8, 2.0, 2.2, 2.4, 3.5])
    assert np.allclose(CS.fines_content(ic, "bi14-cfc0"), np.clip(80.0 * ic - 137.0, 0, 100), atol=0, rtol=0)
    assert np.allclose(CS.fines_content(ic, "maurer2019-christchurch"), np.clip(80.645 * ic - 128.5967, 0, 100),
                       atol=0, rtol=0)
    # PEER 2021/04 PDF p. 53: the Christchurch correlation gives FC "approximately 10% higher for a given Ic".
    gap = CS.fines_content(np.array([2.2]), "maurer2019-christchurch") - CS.fines_content(np.array([2.2]), "bi14-cfc0")
    assert 9.0 < float(gap[0]) < 11.0


def test_tip_rule_matches_the_sheets():
    # Sheet: qt = qc + u2 (1 - a) with a = 0.80 where u2 is recorded, else qc. qc in MPa, u2 in kPa.
    qc = np.array([[5.0, 5.0]])
    u2 = np.array([[100.0, np.nan]])
    qt = CS.tip_resistance(qc, u2, CS.setting(CS.AUTHORS_ID))
    assert qt[0, 0] == pytest.approx(5.0 + 0.1 * 0.2, abs=1e-12)
    assert qt[0, 1] == 5.0
    assert np.array_equal(CS.tip_resistance(qc, u2, CS.setting(CS.DEFAULT_ID)), qc)

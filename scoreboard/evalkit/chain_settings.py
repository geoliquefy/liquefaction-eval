"""Named settings for the B&I 2014 + LPI / LSN chain (eval plan step 3b round 2, G6+G1+G5, M3). No em dashes.

The chain itself is the pinned precedent procedure-comparison code (`procedures.py`, `consequence.py`); this module only
chooses its inputs and screens by a setting id. The default id reproduces the chain the sealed cards used, so no sealed
card changes. Tests: tests/test_chain_settings.py.

Settings
--------
DEFAULT_ID "default-2026-11": what bench v2 and scorecards v0 / v1 use.
    Ic < 2.6 screen; FC = 80 Ic - 137 (B&I 2014 eq. 2.29, CFC = 0); tip = measured qc; LPI 0 to 20 m; LSN window 15 m.

AUTHORS_ID "authors-maurer-2026-11-19": the settings Maurer and co-authors state for Canterbury.
    - Ic < 2.50: engrXiv 121 (Maurer, van Ballegooy, Bradley) section 3.1, "soils with Ic < 2.50 are treated as
      susceptible", for both the CES and the global sets; PEER 2021/04 section 3.4.1, the same for both sets.
    - FC from the Christchurch Ic-FC correlation: engrXiv 121 section 3.1 ("Christchurch-specific Ic - FC correlation
      (Maurer et al., 2016)") for the CES set; PEER 2021/04 section 3.4.1 ("[Maurer et al. 2019]") for the Canterbury
      set. The equation is taken from the PRJ-2937 data paper (Geyin et al., eq. 10): mean FC = 80.645 Ic - 128.5967,
      limited to 0 to 100 %. That the 2016 citation and the 2019 equation are the same correlation is ASSUMED.
      (Their global sets use the general B&I correlation, CFC = 0, which is the default here.)
    - Tip: qt = qc + u2 (1 - a) where u2 is recorded at that depth, else qc. PRJ-2937 carries u2 but no cone area ratio
      a (data paper Table 3), so a = 0.80 is ASSUMED (a typical value). The papers do not say whether they used qc, qt
      or the inverse-filtered qc_inv (NOT CHECKED).
    - LPI 0 to 20 m (Iwasaki); LSN window 20 m (PEER 2021/04 eq. 3.3). engrXiv 121 does not state the LSN window.

PEER_GLOBAL_ID "peer-global-2026-11-23" (item A4, M3 attempt 3): the settings PEER 2021/04 states for its global set.
    Ic < 2.50 and "a general Ic-FC correlation [Boulanger and Idriss 2014]" (section 3.4.1), so CFC = 0; tip = measured
    qc (PRJ-3012 carries no u2); LPI 0 to 20 m (eq. 3.1); LSN window 20 m (eq. 3.3). Registration: pillar
    docs/evals/PARITY-M3-ROUND3-REGISTRATION-2026-11-23.md.

Implementation note: `procedures.bi14_fs` takes Ic only to form FC = 80 Ic - 137 (clipped to 0 to 100). The authors'
FC is passed as Ic_fc = (FC + 137) / 80, the device the step 2c Table 3.1 test uses.
"""
from __future__ import annotations

import numpy as np

DEFAULT_ID = "default-2026-11"
AUTHORS_ID = "authors-maurer-2026-11-19"
PEER_GLOBAL_ID = "peer-global-2026-11-23"

SETTINGS = {
    DEFAULT_ID: {"ic_cut": 2.6, "fc": "bi14-cfc0", "tip": "qc", "area_ratio": None, "lpi_window_m": 20.0,
                 "lsn_window_m": 15.0},
    AUTHORS_ID: {"ic_cut": 2.50, "fc": "maurer2019-christchurch", "tip": "qt", "area_ratio": 0.80,
                 "lpi_window_m": 20.0, "lsn_window_m": 20.0},
    PEER_GLOBAL_ID: {"ic_cut": 2.50, "fc": "bi14-cfc0", "tip": "qc", "area_ratio": None, "lpi_window_m": 20.0,
                     "lsn_window_m": 20.0},
}


def setting(setting_id: str) -> dict:
    if setting_id not in SETTINGS:
        raise KeyError(f"unknown chain setting id {setting_id!r}; known: {sorted(SETTINGS)}")
    return SETTINGS[setting_id]


def fines_content(ic: np.ndarray, kind: str) -> np.ndarray:
    """FC (%) from Ic, clipped to 0 to 100."""
    if kind == "bi14-cfc0":
        return np.clip(80.0 * ic - 137.0, 0.0, 100.0)
    if kind == "maurer2019-christchurch":
        return np.clip(80.645 * ic - 128.5967, 0.0, 100.0)
    raise KeyError(kind)


def tip_resistance(qc_mpa: np.ndarray, u2_kpa: np.ndarray | None, s: dict) -> np.ndarray:
    """qc (MPa) or qt = qc + u2 (1 - a) where u2 is finite."""
    if s["tip"] == "qc" or u2_kpa is None:
        return qc_mpa
    u2 = np.asarray(u2_kpa, float)
    return np.where(np.isfinite(u2), qc_mpa + u2 / 1000.0 * (1.0 - s["area_ratio"]), qc_mpa)


def run(P, C, z, qc, fs, gwt, pga, mw, setting_id: str = DEFAULT_ID, u2=None, predrill_m=None) -> dict:
    """LPI and LSN per profile under one setting. Arrays (n, depths); gwt, pga, mw (n,). qc, fs MPa; u2 kPa.

    P, C: the pinned `procedures` and `consequence` modules. predrill_m: optional pre-drill depths (n,); layers at or
    above them are non-susceptible (the consequence chain's rule). LPI here masks no pre-drill unless it is given.
    """
    s = setting(setting_id)
    g = np.asarray(gwt, float)[:, None]
    a = np.asarray(pga, float)[:, None]
    m = np.asarray(mw, float)[:, None]
    qx = tip_resistance(qc, u2, s)
    sv, svp = P.stresses(z, g)
    ic = P.robertson_ic(qx, fs, sv, svp)[0]
    susc = (z[None, :] > g) & np.isfinite(ic) & (ic < s["ic_cut"]) & (qx * 1000.0 > sv)
    if predrill_m is not None:
        susc = susc & C.predrill_mask(z, np.asarray(predrill_m, float))
    ic_fc = (fines_content(ic, s["fc"]) + 137.0) / 80.0
    b = P.bi14_fs(qx, z, g, a, m, sv, svp, ic_fc)
    fsq = np.where(susc, np.clip(b["fs_liq"], 0.0, P.FS_CAP), np.nan)
    lpi = P.lpi(fsq, z, susc & (z[None, :] <= s["lpi_window_m"]))
    strain = C.zhang2002_strain(fsq, b["qc1ncs"])
    lsn = C.lsn(strain, z, susc, fsq, g, s["lsn_window_m"])
    minfs = np.nanmin(np.where(susc, b["fs_liq"], np.inf), axis=1)
    return {"lpi": lpi, "lsn": lsn, "minfs": minfs, "n_susceptible": susc.sum(axis=1)}

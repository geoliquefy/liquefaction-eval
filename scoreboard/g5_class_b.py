"""G5 class (b) screen: the entry rule of the pooled headline test (G5 amendment 3, ruling 28). Serves GOALS.md G5.

Class (b) (registration 67a52e8, section 4; amendment 3 adopted 2026-11-04): USGS ComCat preferred magnitude
Mw >= 5.0, with 200 or more catalogue CPTs inside the watcher radius R(M) = min(200, 30 x 10^(0.5 (M - 6))) km
(hypocentral distance). All-no earthquakes are allowed. The watcher flags entry only; whether an earthquake becomes a
test earthquake needs a standing grading protocol (census labels or a stated no-ejecta survey), decided later.

"Mw-family" (step 9 of 2026-11-05): the ComCat preferred `magType` is a moment magnitude: mw, mww, mwc, mwb, mwr
(case ignored). mwp (a P-wave proxy, Tsuboi et al. 1995) is NOT counted: it is an early estimate that ComCat
replaces after review, and ruling 11 takes the more conservative option. An mb, ms, ml or md preferred magnitude
does not enter, whatever its value: the class is defined on Mw, and an M >= 5 cut on mb mixes scales at the bar.
No em dashes.
"""
from __future__ import annotations

import math

MIN_MAG = 5.0
MIN_CPT = 200
MW_FAMILY = frozenset({"mw", "mww", "mwc", "mwb", "mwr"})
FLAG = "CLASS B"


def is_mw_family(mag_type) -> bool:
    return isinstance(mag_type, str) and mag_type.strip().lower() in MW_FAMILY


def class_b(mag, mag_type, n_cpt: int) -> dict:
    """{'flag': 'CLASS B' or 'no', 'reason': ...}. Reasons name the first failing part, in a fixed order:
    magnitude, then magnitude type, then CPT count."""
    m = float(mag) if mag is not None else math.nan
    if not (m >= MIN_MAG):
        return {"flag": "no", "reason": f"M {mag} < {MIN_MAG}"}
    if not is_mw_family(mag_type):
        return {"flag": "no", "reason": f"magType {mag_type!s} not Mw-family"}
    if int(n_cpt) < MIN_CPT:
        return {"flag": "no", "reason": f"{int(n_cpt)} CPTs < {MIN_CPT}"}
    return {"flag": FLAG, "reason": f"Mw {m:g} ({str(mag_type).lower()}), {int(n_cpt)} CPTs in R"}

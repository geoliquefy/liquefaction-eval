"""One lab world (protocol section D2): latent logit, true state, observed labels, three entries. No em dashes.

eta_ie = a_e + b_e ln PGA_ie + t T(x_i) + S_i + L_i + lambda D_i
  T  fitted quadratic trend (fixed), t its multiplier;  S  site field (sigma_s, rho_s);  L  optional random
  long-range field;  D  standardised local CPT density (placement stand-in);  ln PGA_ie = ln median + E_e + W_ie.
Observed y from z through Se_i = Se_e (1 - m u_i) and Sp_e, with the error draw spatially correlated (range err_rho;
real residual nuggets near 0 show polygon-level errors). b_e and T are scaled from the marginal GLM to the
conditional latent model by sqrt(1 + 0.346 sigma^2).  Entries see ln median PGA, T, D, S and L, plus
f_k (M_k + N_k): a model-error field and independent noise.  E3 gets + cal_shift on the logit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit, ndtr
from scipy.stats import rankdata

from .fields import FieldSampler

EVENTS = ("sep2010", "feb2011", "jun2011", "dec2011")
# 1a conditional rates of the set B source per earthquake (protocol D2)
SE = {"sep2010": 0.99, "feb2011": 0.96, "jun2011": 0.77, "dec2011": 0.90}
SP = {"sep2010": 0.99, "feb2011": 0.92, "jun2011": 0.88, "dec2011": 0.82}
DENSITY_RADIUS_M = 500.0


@dataclass
class Geometry:
    """Real positions (memory only), labelled masks, median PGA per earthquake, density, fitted trend."""
    x: np.ndarray
    y: np.ndarray
    masks: dict[str, np.ndarray]
    ln_pga: dict[str, np.ndarray]
    sd_lnpga: dict[str, float]
    trend: np.ndarray
    centre: tuple[float, float]
    density: np.ndarray = field(init=False)
    access_u: np.ndarray = field(init=False)

    def __post_init__(self):
        tree = cKDTree(np.column_stack([self.x, self.y]))
        cnt = np.array([len(v) for v in tree.query_ball_point(np.column_stack([self.x, self.y]),
                                                              DENSITY_RADIUS_M)], float)
        lc = np.log(cnt)
        self.density = (lc - lc.mean()) / lc.std()
        self.access_u = 1.0 - (rankdata(cnt) - 0.5) / cnt.size


DEFAULTS = {"sigma_s": 1.0, "rho_s": 1200.0, "aniso": 1.0, "trend_mult": 1.0, "longrand_sd": 0.0,
            "longrand_rho": 20000.0, "lam": 0.3, "miss_m": 0.15, "event_sd": 0.2, "pga_rho": 8500.0,
            "sd_m": 0.7, "rho_m": 2000.0, "sd_n": 0.5, "f1": 1.0, "cal_shift": 0.0,
            "slots": (0, 1, 2, 3), "prevalence": None, "box_sides_m": None, "err_rho": 1000.0}
ATTENUATION_C = 0.346   # logistic-normal: marginal slope = conditional slope / sqrt(1 + c sigma^2) (Zeger et al. 1988)


class World:
    """Samplers and intercepts for one condition; draw(seed) gives one world."""

    def __init__(self, geom: Geometry, params: dict, b: dict[str, float], prevalence: dict[str, float],
                 calib_seed: int = 7):
        self.g, self.p = geom, {**DEFAULTS, **params}
        p = self.p
        # the GLM was fitted on labels (marginal); the latent model is conditional on the site field
        self.scale = float(np.sqrt(1.0 + ATTENUATION_C * (p["sigma_s"] ** 2 + p["longrand_sd"] ** 2)))
        self.b = {e: v * self.scale for e, v in b.items()}
        self.s_err = FieldSampler(geom.x, geom.y, p["err_rho"]) if p["err_rho"] else None
        an = p["aniso"]
        self.s_site = FieldSampler(geom.x, geom.y, p["rho_s"], aniso=an)
        self.s_model = FieldSampler(geom.x, geom.y, p["rho_m"], aniso=an)
        self.s_pga = FieldSampler(geom.x, geom.y, p["pga_rho"])
        self.s_long = FieldSampler(geom.x, geom.y, p["longrand_rho"]) if p["longrand_sd"] > 0 else None
        target = p["prevalence"] or prevalence
        self.a = self._intercepts(target if isinstance(target, dict) else {e: target for e in EVENTS}, calib_seed)

    def _reset(self):
        for s in (self.s_site, self.s_model, self.s_pga, self.s_long, self.s_err):
            if s is not None:
                s.reset()

    def _site_part(self, rng: np.random.Generator) -> np.ndarray:
        p, g = self.p, self.g
        part = p["trend_mult"] * self.scale * g.trend + p["sigma_s"] * self.s_site.draw(rng) + p["lam"] * g.density
        if self.s_long is not None:
            part = part + p["longrand_sd"] * self.s_long.draw(rng)
        return part

    def _intercepts(self, target: dict[str, float], seed: int, n_pilot: int = 100) -> dict[str, float]:
        """Bisection on a_e so the mean observed prevalence over 100 pilot worlds (with event terms) matches the
        target (deviation 4: the protocol said 10 pilots without the event term; that left Sep 0.04 low)."""
        rng = np.random.default_rng(seed)
        g, p = self.g, self.p
        a = {}
        for e in EVENTS:
            m = g.masks[e]
            draws = []
            for _ in range(n_pilot):
                self._reset()
                part = self._site_part(rng)
                w = g.sd_lnpga[e] * self.s_pga.draw(rng) + rng.normal(0.0, p["event_sd"])
                draws.append((part + self.b[e] * (g.ln_pga[e] + w))[m])
            base = np.stack(draws)
            se_i = SE[e] * (1.0 - p["miss_m"] * g.access_u[m])
            lo, hi = -30.0, 30.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                pz = expit(mid + base)
                obs = float(np.mean(se_i * pz + (1.0 - SP[e]) * (1.0 - pz)))
                lo, hi = (mid, hi) if obs < target[e] else (lo, mid)
            a[e] = 0.5 * (lo + hi)
        self._reset()
        return a

    def draw(self, seed: int) -> list[dict]:
        """One world: per earthquake slot, site indices, observed labels, true state, three entries' p."""
        rng = np.random.default_rng(seed)
        self._reset()
        g, p = self.g, self.p
        part = self._site_part(rng)
        k_s = p["sigma_s"]   # deviation 2: model-error sds are relative to the fitted site-field sd
        mod = [k_s * (p["sd_m"] * self.s_model.draw(rng) + rng.normal(0.0, p["sd_n"], g.x.size)) for _ in range(3)]
        f = (p["f1"], 1.0, 1.0)
        keep = np.ones(g.x.size, bool)
        if p["box_sides_m"]:
            side = float(rng.choice(p["box_sides_m"]))
            anyl = np.flatnonzero(np.logical_or.reduce([g.masks[e] for e in EVENTS]))
            c = anyl[rng.integers(anyl.size)]
            keep = (np.abs(g.x - g.x[c]) <= side / 2) & (np.abs(g.y - g.y[c]) <= side / 2)
        out = []
        for slot in p["slots"]:
            e = EVENTS[slot]
            idx = np.flatnonzero(g.masks[e] & keep)
            ev = rng.normal(0.0, p["event_sd"])
            w = g.sd_lnpga[e] * self.s_pga.draw(rng)
            eta = self.a[e] + self.b[e] * (g.ln_pga[e] + ev + w) + part
            z = rng.random(g.x.size) < expit(eta)
            se_i = SE[e] * (1.0 - p["miss_m"] * g.access_u)
            u = ndtr(self.s_err.draw(rng)) if self.s_err is not None else rng.random(g.x.size)
            yobs = np.where(z, u < se_i, u < 1.0 - SP[e])
            known = self.a[e] + self.b[e] * g.ln_pga[e] + part
            # entries integrate the shaking they cannot see (event term, within-event field)
            c = 1.0 / np.sqrt(1.0 + ATTENUATION_C * self.b[e] ** 2 * (g.sd_lnpga[e] ** 2 + p["event_sd"] ** 2))
            ps = [expit(c * (known + f[k] * mod[k])) for k in range(3)]
            if p["cal_shift"]:
                ps[2] = expit(c * (known + f[2] * mod[2]) + p["cal_shift"])
            out.append({"slot": slot, "event": e, "idx": idx, "y": yobs[idx], "z": z[idx],
                        "p": [q[idx] for q in ps], "pga_med": np.exp(g.ln_pga[e][idx])})
        return out

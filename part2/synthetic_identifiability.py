"""Phase B0 - synthetic-data identifiability check for the heat-capacity split.

Question: is the run of `split` to its 0.99 bound on the real calibration data
a property of the model (split not identifiable from T_surf) or a pull from the
real residual (model structural error)?

Method. Synthetic surface-temperature data are generated from the locked
central point (2026-06-18b, Step 4) with the real heat input Q(t) and the real
T_inf, then re-fitted:

  T0   clean (no noise), several starts
  T1   iid Gaussian noise, sigma = RMS of the real residual, N = 200
  T1b  AR(1) noise with the same marginal sigma, generated at 1 Hz and then
       subsampled to stride 10; tau_int = 292 s, N = 200
  T2   the per-cycle mean (offset) of the real residual only
  T3   the real residual minus its per-cycle mean (shape) only
  T4   the full real residual (= the real data; must reproduce G4)

The decision rules R1-R4 are written into the output JSON before any
computation runs.

Simulator. The model is linear and time-invariant (Step 1), so it is integrated
exactly per 1 s interval in modal coordinates instead of with solve_ivp:

  x = T - T_inf,   dx/dt = A x + b Q(t),   b = [1/C_core, 0]
  A = V diag(lam) V^-1;  z = V^-1 x;  beta = V^-1 b
  z_j(k+1) = e^{lam_j} z_j(k) + beta_j * int_0^1 e^{lam_j (1 - tau)} Q(t_k + tau) dtau

The integral uses composite Simpson quadrature on M + 1 points of the true
Q(t) = I(t) * (V(t) - V_ocv(SOC(t))), evaluated with the same interpolants
calibrate.simulate_T uses (calibrate.precompute_cycle). The recurrence is run
with scipy.signal.lfilter. Initial state as in simulate_T: T_core = T_surf =
T_surf_meas[0]. M = 16 for every experiment; G1 checks it against
calibrate.simulate_T at rtol 1e-12, atol 1e-14, max_step 1.

READ-ONLY on everything from the locked run. Writes only
part2/results/phase_b0_synthetic_identifiability.json.

Usage
-----
    python part2/synthetic_identifiability.py [--workers N] [--allow-dirty]

Exit codes: 0 ok; 2 a gate or consistency check failed (the JSON is written
with status "gate_failed" and the evidence).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.special import ndtr

PART2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PART2_DIR.parent
for _p in (str(PROJECT_ROOT), str(PART2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import calibrate as cal                      # noqa: E402  (read-only use)
import identifiability as idt                # noqa: E402  (provenance, JSON writer)
import thermal_model as tm                   # noqa: E402
from thermal_model import ThermalParams      # noqa: E402


# ===========================================================================
# Constants (all fixed before the run; recorded in the JSON)
# ===========================================================================

OUT_JSON = idt.RESULTS_DIR / "phase_b0_synthetic_identifiability.json"
SPLIT_REFIT_JSON = idt.RESULTS_DIR / "phase_a_split_refit_diagnostic.json"

STRIDE = 10
M_EXPERIMENTS = 16

# G1
G1_REFERENCE = {"rtol": 1e-12, "atol": 1e-14, "max_step": 1.0}
G1_STANDARD_TIGHT = {"rtol": 1e-10, "atol": 1e-12, "max_step": 5.0}   # report only
G1_TOL_K = 1e-6
G1_REPORT_M = (8, 32)

# G3
G3_RELS = (0.02, 0.005, 0.001)
G3_TOL = 1e-3

# G4: the Phase A split-refit central fit (diff_step 0.02).
G4_REF_COST = 467.9455299629
G4_COST_RTOL = 1e-4          # 0.01 %
BOUND_TOL = 1e-6             # "at a bound" means within this of it

# Fits
PRIMARY = {"p0": 0.93, "lb": 0.50, "ub": 0.99, "method": "trf", "ftol": 1e-12, "xtol": 1e-12,
           "gtol": 1e-12, "jac": "3-point", "x_scale": "jac", "max_nfev": 1000}
JOINT = {"p0": [0.93, 1.5], "lb": [0.50, 0.1], "ub": [0.99, 30.0], "method": "trf",
         "ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12, "jac": "3-point", "x_scale": "jac",
         "max_nfev": 1000}
T0_STARTS = (0.80, 0.85, 0.93, 0.97, 0.99)
JOINT_T0_RCS_START = 1.5

# Noise
N_MC = 200
SEED_T1 = 20261002
SEED_T1B = 20261003
TAU_INT_S = 292.0
# AR(1) x_k = phi x_{k-1} + e_k at 1 Hz has rho(k) = phi^k, so with the Phase A
# convention tau_int = 1/2 + sum_{k>=1} rho(k) = (1 + phi) / (2 (1 - phi)),
# hence phi = (2 tau_int - 1) / (2 tau_int + 1) = 583/585 for tau_int = 292 s.
AR1_PHI = (2 * TAU_INT_S - 1) / (2 * TAU_INT_S + 1)

CRB_FD_REL = 0.001           # central-difference step for the CRB Jacobian at truth

RULES = {
    "R1": {"tier": "T0", "fit": "primary",
           "condition": "every start in T0_STARTS ends with |split_hat - split_true| <= 1e-4",
           "threshold": 1e-4},
    "R2": {"tier": "T1", "fit": "primary",
           "condition": "sd(split_hat) within [0.7, 1.4] x CRB sd (iid), AND the bound-hit "
                        "fraction within +/- 5 percentage points of the CRB-predicted fraction",
           "sd_ratio_range": [0.7, 1.4], "bound_hit_pp": 5.0},
    "R3": {"tiers": ["T1b", "T1"], "fit": "primary",
           "condition": "|bound-hit fraction(T1b) - bound-hit fraction(T1)| > 10 percentage points",
           "threshold_pp": 10.0},
    "R4": {"tiers": ["T2", "T3"], "fit": "primary",
           "condition": "report which of T2 / T3 lands at split = 0.99 "
                        "(split_hat >= 0.99 - BOUND_TOL); outcome is one of "
                        "'T2 only', 'T3 only', 'both', 'neither'"},
    "definitions": {
        "bound_hit": "split_hat >= ub - BOUND_TOL or split_hat <= lb + BOUND_TOL, BOUND_TOL = 1e-6",
        "predicted_bound_hit": "Gaussian approximation: P(N(split_true, sd^2) >= 0.99) + "
                               "P(N(split_true, sd^2) <= 0.50); sd = iid CRB sd for T1, "
                               "OLS sandwich sd under the AR(1) covariance for T1b",
        "fires": "a rule fires when its condition is true",
    },
}


class GateFailed(RuntimeError):
    def __init__(self, message: str, block: dict):
        super().__init__(message)
        self.block = block


# ===========================================================================
# Context: locked point, caches, Q sub-samples
# ===========================================================================

def simpson_weights(M: int) -> np.ndarray:
    if M % 2:
        raise ValueError("Simpson needs an even number of sub-intervals")
    w = np.ones(M + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w / (3.0 * M)


def q_subsamples(cache: dict, M: int) -> np.ndarray:
    """Q at t_k + m/M, m = 0..M, for every 1 s interval k; shape (n-1, M+1)."""
    t = cache["t_grid"]
    ts = (t[:-1, None] + np.arange(M + 1)[None, :] / M).ravel()
    soc = cache["soc_of_t"](ts)
    q = cache["I_of_t"](ts) * (cache["V_of_t"](ts) - cache["v_ocv"](soc))
    return np.asarray(q, float).reshape(len(t) - 1, M + 1)


def build_context(ms=(M_EXPERIMENTS,)) -> dict:
    locked = idt.load_locked()
    j = locked["json"]
    c4 = j["step4_central_fit"]
    ctx = {
        "files": list(j["cal_cycles"]),
        "C_total": float(j["physical_constants"]["C_total_J_K"]),
        "R_sa": float(j["step2_R_sa_result"]["R_sa_central_K_per_W"]),
        "T_inf": float(locked["T_inf_by_ambient_C"]["25"]),
        "truth": {"split": float(c4["split"]), "R_cs": float(c4["R_cs_K_per_W"])},
        "truth_params": {"C_core": float(c4["C_core_J_K"]), "C_surf": float(c4["C_surf_J_K"]),
                         "R_cs": float(c4["R_cs_K_per_W"]), "R_sa": float(c4["R_sa_K_per_W"])},
        "cycles": [],
    }
    if ctx["truth_params"]["R_sa"] != ctx["R_sa"]:
        raise idt.SpecMismatch("step4_central_fit.R_sa != step2 R_sa_central")
    for f in ctx["files"]:
        cache = cal.precompute_cycle(idt.PROCESSED_DIR / f, T_inf_override=ctx["T_inf"])
        t = cache["t_grid"]
        cyc = {"file": f, "n": len(t), "t": t, "T0": float(cache["T_surf_meas"][0]),
               "y_meas_full": np.asarray(cache["T_surf_meas"], float),
               "y_meas": np.asarray(cache["T_surf_meas"][::STRIDE], float),
               "uniform_1s": bool(np.all(np.diff(t) == 1.0)),
               "T_amb_constant": bool(np.all(cache["T_amb_array"] == ctx["T_inf"])),
               "Qs": {M: q_subsamples(cache, M) for M in ms}}
        ctx["cycles"].append(cyc)
    return ctx


# ===========================================================================
# Exact modal simulator
# ===========================================================================

def modal_ts(C_core, C_surf, R_cs, R_sa, T0, T_inf, Qs, M) -> np.ndarray:
    """T_surf on the 1 Hz grid, exact for the LTI model given Q on the sub-grid."""
    A = np.array([[-1.0 / (R_cs * C_core), 1.0 / (R_cs * C_core)],
                  [1.0 / (R_cs * C_surf), -(1.0 / (R_cs * C_surf) + 1.0 / (R_sa * C_surf))]])
    lam, V = np.linalg.eig(A)
    if np.any(np.iscomplex(lam)) or np.any(np.iscomplex(V)):
        raise RuntimeError("complex modes; the two-state model should have real eigenvalues")
    lam, V = lam.real, V.real
    Vi = np.linalg.inv(V)
    beta = Vi @ np.array([1.0 / C_core, 0.0])
    z0 = Vi @ np.array([T0 - T_inf, T0 - T_inf])
    tau = np.arange(M + 1) / M
    w = simpson_weights(M)
    n = Qs.shape[0] + 1
    out = np.zeros(n)
    k = np.arange(1, n)
    for jm in range(2):
        a = np.exp(lam[jm])
        g = beta[jm] * (Qs @ (w * np.exp(lam[jm] * (1.0 - tau))))
        s = lfilter([1.0], [1.0, -a], g)
        z = np.empty(n)
        z[0] = z0[jm]
        z[1:] = a ** k * z0[jm] + s
        out += V[1, jm] * z
    return T_inf + out


def surf_full(ctx, split, R_cs, M=M_EXPERIMENTS) -> list:
    Ct = ctx["C_total"]
    return [modal_ts(split * Ct, (1 - split) * Ct, R_cs, ctx["R_sa"], c["T0"], ctx["T_inf"],
                     c["Qs"][M], M) for c in ctx["cycles"]]


def surf_stack(ctx, split, R_cs, M=M_EXPERIMENTS) -> np.ndarray:
    return np.concatenate([y[::STRIDE] for y in surf_full(ctx, split, R_cs, M)])


# ===========================================================================
# Fits
# ===========================================================================

def fit_primary(ctx, y, p0=PRIMARY["p0"]) -> dict:
    rc = ctx["truth"]["R_cs"]
    res = least_squares(lambda p: surf_stack(ctx, float(p[0]), rc) - y, np.array([p0], float),
                        bounds=([PRIMARY["lb"]], [PRIMARY["ub"]]), method=PRIMARY["method"],
                        jac=PRIMARY["jac"], x_scale=PRIMARY["x_scale"], ftol=PRIMARY["ftol"],
                        xtol=PRIMARY["xtol"], gtol=PRIMARY["gtol"], max_nfev=PRIMARY["max_nfev"])
    x = float(res.x[0])
    return {"split": x, "cost": float(res.cost), "status": int(res.status), "nfev": int(res.nfev),
            "hit_ub": x >= PRIMARY["ub"] - BOUND_TOL, "hit_lb": x <= PRIMARY["lb"] + BOUND_TOL}


def fit_joint(ctx, y, p0=tuple(JOINT["p0"])) -> dict:
    res = least_squares(lambda p: surf_stack(ctx, float(p[0]), float(p[1])) - y,
                        np.array(p0, float), bounds=(JOINT["lb"], JOINT["ub"]),
                        method=JOINT["method"], jac=JOINT["jac"], x_scale=JOINT["x_scale"],
                        ftol=JOINT["ftol"], xtol=JOINT["xtol"], gtol=JOINT["gtol"],
                        max_nfev=JOINT["max_nfev"])
    s, r = float(res.x[0]), float(res.x[1])
    return {"split": s, "R_cs": r, "cost": float(res.cost), "status": int(res.status),
            "nfev": int(res.nfev),
            "hit_ub": s >= JOINT["ub"][0] - BOUND_TOL, "hit_lb": s <= JOINT["lb"][0] + BOUND_TOL,
            "R_cs_at_bound": bool(r >= JOINT["ub"][1] - BOUND_TOL or r <= JOINT["lb"][1] + BOUND_TOL)}


def summarise(fits: list, truth: dict, keys=("split",)) -> dict:
    out = {"n": len(fits),
           "n_status_0_maxnfev": sum(f["status"] == 0 for f in fits),
           "bound_hit_split": sum(f["hit_ub"] or f["hit_lb"] for f in fits),
           "bound_hit_split_ub": sum(f["hit_ub"] for f in fits),
           "bound_hit_split_lb": sum(f["hit_lb"] for f in fits)}
    out["bound_hit_fraction"] = out["bound_hit_split"] / len(fits)
    for k in keys:
        v = np.array([f[k] for f in fits])
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "bias": float(v.mean() - truth[k]), "median": float(np.median(v)),
                  "min": float(v.min()), "max": float(v.max())}
    if "R_cs" in keys:
        out["R_cs_at_bound"] = sum(f["R_cs_at_bound"] for f in fits)
    return out


# ===========================================================================
# Noise
# ===========================================================================

def iid_draws(ctx, sigma) -> np.ndarray:
    rng = np.random.default_rng(SEED_T1)
    n10 = sum(len(c["y_meas"]) for c in ctx["cycles"])
    return sigma * rng.standard_normal((N_MC, n10))


def ar1_draws(ctx, sigma) -> np.ndarray:
    """Stationary AR(1) at 1 Hz per cycle (independent cycles), then [::STRIDE]."""
    rng = np.random.default_rng(SEED_T1B)
    c = sigma * np.sqrt(1.0 - AR1_PHI ** 2)
    out = []
    for _ in range(N_MC):
        parts = []
        for cyc in ctx["cycles"]:
            e = rng.standard_normal(cyc["n"])
            x = np.empty(cyc["n"])
            x[0] = sigma * e[0]
            x[1:] = lfilter([c], [1.0, -AR1_PHI], e[1:], zi=[AR1_PHI * x[0]])[0]
            parts.append(x[::STRIDE])
        out.append(np.concatenate(parts))
    return np.array(out)


def ar1_cov_blocks(ctx, sigma) -> list:
    phi10 = AR1_PHI ** STRIDE
    blocks = []
    for cyc in ctx["cycles"]:
        m = len(cyc["y_meas"])
        idx = np.arange(m)
        blocks.append(sigma ** 2 * phi10 ** np.abs(idx[:, None] - idx[None, :]))
    return blocks


# ===========================================================================
# Step 1
# ===========================================================================

STEP1A_NEEDLES = {
    "thermal_model.thermal_rhs": (tm.thermal_rhs, [
        "dTc = (Q - (T_core - T_surf) / params.R_cs) / params.C_core",
        "dTs = ((T_core - T_surf) / params.R_cs",
        "- (T_surf - T_amb) / params.R_sa) / params.C_surf"]),
    "calibrate.simulate_T": (cal.simulate_T, [
        'T0 = float(cache["T_surf_meas"][0])', "y0 = np.array([T0, T0])",
        'return I_t * (V_t - float(cache["v_ocv"](soc_t)))']),
    "calibrate.precompute_cycle": (cal.precompute_cycle, [
        'kw = dict(kind="linear"', "amb_series = np.full_like(t, T_inf_override, dtype=float)"]),
}


def step1a(ctx) -> dict:
    found = {}
    for name, (fn, needles) in STEP1A_NEEDLES.items():
        src = inspect.getsource(fn)
        missing = [nd for nd in needles if nd not in src]
        if missing:
            raise idt.SpecMismatch(f"{name} no longer contains {missing}")
        found[name] = needles
    per = {}
    for c in ctx["cycles"]:
        if not (c["uniform_1s"] and c["T_amb_constant"]):
            raise idt.SpecMismatch(f"{c['file']}: grid not uniform 1 s or T_amb not constant T_inf")
        per[c["file"]] = {"n_1Hz": c["n"], "n_stride10": len(c["y_meas"]),
                          "uniform_1s_grid": True, "T_amb_constant_equals_T_inf": True,
                          "T0_C": c["T0"], "T0_minus_T_inf_K": c["T0"] - ctx["T_inf"]}
    return {
        "lti": True,
        "rhs": ["C_core dT_core/dt = Q(t) - (T_core - T_surf)/R_cs",
                "C_surf dT_surf/dt = (T_core - T_surf)/R_cs - (T_surf - T_amb(t))/R_sa"],
        "inputs": "Q(t) and T_amb(t) only; parameters constant; Q does not depend on the state",
        "T_amb": "constant T_inf (interp1d of a constant series, T_inf_override)",
        "initial_condition": "T_core(0) = T_surf(0) = T_surf_meas[0]",
        "Q_interpolation": ("other: Q(t) = I(t) * (V(t) - V_ocv(SOC(t))), with I, V, SOC linear "
                            "interp1d on the 1 Hz grid and V_ocv a linear interp1d of the OCV "
                            "table; Q is piecewise quadratic within each second, with kinks at "
                            "OCV-table nodes, so neither zero- nor first-order hold is exact"),
        "source_needles_verified": found,
        "per_cycle": per,
    }


def step1b() -> dict:
    """Transfer-function coefficient Jacobian ranks (sympy)."""
    import sympy as sp
    s = sp.symbols("s")
    Cc, Cs, Rcs, Rsa, Ct, spl = sp.symbols("C_core C_surf R_cs R_sa C_total split", positive=True)
    A = sp.Matrix([[-1 / (Rcs * Cc), 1 / (Rcs * Cc)],
                   [1 / (Rcs * Cs), -(1 / (Rcs * Cs) + 1 / (Rsa * Cs))]])
    Cm = sp.Matrix([[0, 1]])
    res = (s * sp.eye(2) - A).inv()
    G = {"Q": sp.cancel((Cm * res * sp.Matrix([1 / Cc, 0]))[0]),
         "T_inf": sp.cancel((Cm * res * sp.Matrix([0, 1 / (Rsa * Cs)]))[0])}
    P4 = [Cc, Cs, Rcs, Rsa]
    generic4 = {Cc: sp.Rational(40), Cs: sp.Rational(3), Rcs: sp.Rational(17, 5),
                Rsa: sp.Rational(26, 5)}
    generic2 = {spl: sp.Rational(9, 10), Rcs: sp.Rational(17, 5), Ct: sp.Rational(216, 5),
                Rsa: sp.Rational(26, 5)}
    sub2 = {Cc: spl * Ct, Cs: (1 - spl) * Ct}

    def coeff_vec(g, norm):
        num, den = sp.fraction(sp.cancel(sp.together(g)))
        pn, pd = sp.Poly(num, s), sp.Poly(den, s)
        c = pd.coeff_monomial(1) if norm == "dc" else pd.LC()
        nc = [sp.simplify(x / c) for x in reversed(pn.all_coeffs())]
        dc = [sp.simplify(x / c) for x in reversed(pd.all_coeffs())]
        return nc, dc, [x for x in nc + dc if x.free_symbols]

    xs = sp.Symbol("x")   # unrestricted stand-in for split, so boundary roots are found

    def split_roots(expr):
        roots, unresolved = set(), []
        for fac, _ in sp.factor_list(expr)[1]:
            if spl in fac.free_symbols:
                roots |= set(sp.solve(fac.subs(spl, xs), xs))
            elif fac.is_positive is not True:
                unresolved.append(str(fac))
        return roots, unresolved

    def zero_locus(minors):
        """split values where every nonzero minor vanishes (rank drop), and the
        split values where the minors are undefined (denominator zero)."""
        common, unresolved, poles = None, [], set()
        for m in minors:
            num, den = sp.fraction(sp.together(m))
            roots, unr = split_roots(num)
            unresolved += unr
            poles |= split_roots(den)[0]
            common = roots if common is None else (common & roots)
        common = sorted(common or [], key=str)
        interior = [str(r) for r in common if r.is_number and 0 < r < 1]
        return ([str(r) for r in common], interior, unresolved,
                sorted(str(p) for p in poles))

    out = {"transfer_functions": {k: str(sp.factor(v)) for k, v in G.items()},
           "note": ("The T_inf channel is excited in the LG data only by the initial offset "
                    "T0 - T_inf (T_inf constant); the zero-input response is "
                    "(T0 - T_inf) (1 - G_T(s)) / s."),
           "normalisations": {}}
    for norm in ("dc", "monic"):
        vec = {}
        tf = {}
        for ch, g in G.items():
            nc, dc, v = coeff_vec(g, norm)
            vec[ch] = v
            tf[ch] = {"num_ascending": [str(x) for x in nc], "den_ascending": [str(x) for x in dc]}
        vec["joint"] = list(dict.fromkeys(vec["Q"] + vec["T_inf"]))
        blk = {"coefficients": tf, "case_i_C_core_C_surf_R_cs_R_sa": {},
               "case_ii_split_R_cs": {}}
        for ch in ("Q", "T_inf", "joint"):
            J = sp.Matrix(vec[ch]).jacobian(P4)
            r = int(J.subs(generic4).rank())
            e = {"n_coefficients": len(vec[ch]), "generic_rank": r, "n_params": 4}
            if J.shape[0] == J.shape[1]:
                e["det"] = str(sp.factor(J.det()))
            ns = J.subs(generic4).nullspace()
            if ns:
                e["null_direction_at_generic_point_(dC_core,dC_surf,dR_cs,dR_sa)"] = [str(x) for x in ns[0]]
            blk["case_i_C_core_C_surf_R_cs_R_sa"][ch] = e
            J2 = sp.Matrix([sp.simplify(c.subs(sub2)) for c in vec[ch]]).jacobian([spl, Rcs])
            minors = [sp.factor(J2.extract([a, b], [0, 1]).det())
                      for a in range(J2.shape[0]) for b in range(a + 1, J2.shape[0])]
            nz = [m for m in minors if m != 0]
            locus, interior, unresolved, poles = zero_locus(nz)
            blk["case_ii_split_R_cs"][ch] = {
                "generic_rank": int(J2.subs(generic2).rank()), "n_params": 2,
                "nonzero_2x2_minors": [str(m) for m in nz],
                "split_values_where_all_nonzero_minors_vanish": locus,
                "split_values_where_minors_are_undefined": poles,
                "interior_degenerate_points_split_in_(0,1)": interior,
                "unresolved_factors": unresolved}
        out["normalisations"][norm] = blk
    out["normalisations"]["dc"]["meaning"] = "coefficients divided by the denominator's constant term"
    out["normalisations"]["monic"]["meaning"] = "coefficients divided by the denominator's leading coefficient"
    return out


# ===========================================================================
# Workers (G1 reference integrations, G2 determinism)
# ===========================================================================

def _ref_worker(job):
    fname, t_inf, params, tol = job
    cache = cal.precompute_cycle(idt.PROCESSED_DIR / fname, T_inf_override=t_inf)
    t0 = time.time()
    _, _, ts, _ = cal.simulate_T(cache, ThermalParams(**params), t_eval=cache["t_grid"],
                                 max_step=tol["max_step"], rtol=tol["rtol"], atol=tol["atol"])
    return np.asarray(ts, float), time.time() - t0, os.getpid()


def _g2_worker(_):
    ctx = build_context()
    tr = ctx["truth"]
    y_full = np.concatenate(surf_full(ctx, tr["split"], tr["R_cs"]))
    y_truth = surf_stack(ctx, tr["split"], tr["R_cs"])
    sigma = float(np.sqrt(np.mean((y_truth - np.concatenate([c["y_meas"] for c in ctx["cycles"]])) ** 2)))
    e1 = iid_draws(ctx, sigma)[0]
    e2 = ar1_draws(ctx, sigma)[0]
    f = fit_primary(ctx, y_truth + e1)
    return {"pid": os.getpid(), "y_full": y_full, "iid0": e1, "ar1_0": e2,
            "fit_T1_rep0": np.array([f["split"], f["cost"], f["nfev"]], float)}


# ===========================================================================
# JSON
# ===========================================================================

def _md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def write(payload: dict) -> None:
    idt._atomic_write(OUT_JSON, payload)


def method_block() -> dict:
    return {
        "question": "is split's run to 0.99 a model unidentifiability or a pull from the real residual?",
        "simulator": {
            "kind": "exact modal LTI integration (not solve_ivp, not lsim)",
            "description": ("A = V diag(lam) V^-1 (numpy.linalg.eig); per 1 s interval, "
                            "z_j(k+1) = e^{lam_j} z_j(k) + beta_j * int_0^1 e^{lam_j(1-tau)} Q(t_k+tau) dtau; "
                            "integral by composite Simpson on M+1 points of the true Q from "
                            "calibrate.precompute_cycle's interpolants; recurrence via scipy.signal.lfilter; "
                            "deviation coordinates from T_inf; IC T_core = T_surf = T_surf_meas[0]"),
            "M": M_EXPERIMENTS,
            "why_not_lsim": ("first-order hold at 1 Hz is not exact for this Q (piecewise quadratic "
                             "within each second); max |lsim FOH - simulate_T| on US06 was 0.113 K "
                             "in the pre-run check, and lsim's loop is too slow for N = 200 tiers"),
        },
        "G1_reference": {"function": "calibrate.simulate_T", **G1_REFERENCE,
                         "params": "locked central (step4_central_fit)", "grid": "full 1 Hz, all three cycles",
                         "gate": f"max |modal(M={M_EXPERIMENTS}) - reference| <= {G1_TOL_K} K on every cycle",
                         "report_only": [f"modal M={m} vs reference" for m in G1_REPORT_M]
                         + ["simulate_T at rtol 1e-10, atol 1e-12, max_step 5 vs reference "
                            "(integration error of the standard tight setting)"]},
        "G2": "bitwise determinism: two fresh worker processes and the parent produce identical "
              "modal T_surf (full grid, 3 cycles), identical first iid and AR(1) noise draws, and "
              "an identical T1 replicate-0 primary fit",
        "G3": {"rels": list(G3_RELS), "gate": f"|| dy/dsplit (rel) || within {G3_TOL} of the 0.02 step's, "
                                             "central differences at truth, stride-10 stack"},
        "G4": {"reference": "part2/results/phase_a_split_refit_diagnostic.json fits[central, diff_step 0.02]",
               "reference_cost": G4_REF_COST, "reference_split": 0.99,
               "gate": f"primary fit on the real data lands at 0.99 (>= 0.99 - {BOUND_TOL}) and its cost is "
                       f"within {G4_COST_RTOL * 100} % of the reference"},
        "truth": "locked central point, step4_central_fit (split, R_cs central; C_total, R_sa locked)",
        "data": "three calibration cycles at 25 degC, stride 10, real Q(t) and T_inf",
        "fits": {"primary": PRIMARY, "joint": JOINT,
                 "T0_starts": list(T0_STARTS),
                 "joint_T0_starts": [[s0, JOINT_T0_RCS_START] for s0 in T0_STARTS],
                 "joint_tiers": ["T0", "T1"],
                 "unspecified_choices": ("jac='3-point' and x_scale='jac' (as Part 1); max_nfev=1000; "
                                         "the joint T0 uses every T0 split start with R_cs start 1.5")},
        "noise": {
            "sigma": "RMS of the real residual (modal model at truth minus measured T_surf), stride 10, three cycles",
            "T1": {"model": "iid N(0, sigma^2)", "N": N_MC, "seed": SEED_T1,
                   "generator": "numpy.random.default_rng(seed).standard_normal((N, n_stride10))"},
            "T1b": {"model": "stationary AR(1) at 1 Hz per cycle, cycles independent, then [::10]",
                    "N": N_MC, "seed": SEED_T1B, "tau_int_s": TAU_INT_S,
                    "tau_int_convention": "tau_int = 1/2 + sum_{k>=1} rho(k) (Phase A / Sokal)",
                    "phi_formula": "phi = (2 tau_int - 1) / (2 tau_int + 1)",
                    "phi": AR1_PHI, "phi_fraction": "583/585",
                    "recursion": "x_0 = sigma e_0; x_k = phi x_{k-1} + sigma sqrt(1 - phi^2) e_k",
                    "tau_int_measured_phase_a_s": 292.4928435472976},
            "T2": "y = y_truth - offset, offset = per-cycle mean of the real residual",
            "T3": "y = y_truth - (r_real - offset)",
            "T4": "y = y_truth - r_real (= the measured data); must reproduce G4",
            "initial_condition": "every synthetic fit uses the real T_surf_meas[0] as the IC (known input)",
        },
        "crb": {"jacobian": f"central differences at truth, rel step {CRB_FD_REL}, stride-10 stack",
                "iid": "cov = sigma^2 (J^T J)^-1",
                "ar1_ols_sandwich": "cov = (J^T J)^-1 J^T Sigma J (J^T J)^-1, Sigma = block-diag AR(1) at stride 10",
                "ar1_gls_crb": "cov = (J^T Sigma^-1 J)^-1"},
    }


# ===========================================================================
# Main
# ===========================================================================

def _max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def run(workers: int, allow_dirty: bool) -> int:
    t_start = time.time()
    prov = idt._provenance(allow_dirty)
    prov["md5"]["part2/synthetic_identifiability.py"] = _md5(Path(__file__).resolve())
    prov["md5"]["part2/results/phase_a_split_refit_diagnostic.json"] = _md5(SPLIT_REFIT_JSON)
    import scipy
    import sympy
    prov["sympy"] = sympy.__version__
    payload = {"analysis": "phase_b0_synthetic_identifiability", "status": "preregistered",
               "locked_run_untouched": True, "provenance": prov,
               "rules": RULES,
               "rules_written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "method": method_block()}
    write(payload)                                  # rules on disk before any computation
    print(f"rules pre-registered -> {OUT_JSON}")

    try:
        _run_body(payload, workers)
    except GateFailed as e:
        payload["status"] = "gate_failed"
        payload["failure"] = str(e)
        payload.setdefault("gates", {}).update(e.block)
        payload["wall_s"] = time.time() - t_start
        write(payload)
        print(f"\nGATE FAILED: {e}\nwritten -> {OUT_JSON}")
        return 2
    payload["status"] = "ok"
    payload["wall_s"] = time.time() - t_start
    write(payload)
    print(f"\nwritten -> {OUT_JSON}  ({payload['wall_s']:.0f} s)")
    return 0


def _run_body(payload: dict, workers: int) -> None:
    ctx = build_context(ms=(M_EXPERIMENTS,) + G1_REPORT_M)
    tr, tp = ctx["truth"], ctx["truth_params"]
    gates = payload.setdefault("gates", {})

    # ---- Step 1 ----------------------------------------------------------
    payload["step1"] = {"a": step1a(ctx), "b": step1b()}
    b = payload["step1"]["b"]["normalisations"]["dc"]
    print("Step 1b ranks (dc normalisation):",
          {k: v["generic_rank"] for k, v in b["case_i_C_core_C_surf_R_cs_R_sa"].items()},
          "(split,R_cs):", {k: v["generic_rank"] for k, v in b["case_ii_split_R_cs"].items()})
    write(payload)

    # ---- G1 --------------------------------------------------------------
    jobs = [(c["file"], ctx["T_inf"], tp, tol) for tol in (G1_REFERENCE, G1_STANDARD_TIGHT)
            for c in ctx["cycles"]]
    order = sorted(range(len(jobs)), key=lambda i: -len(ctx["cycles"][i % 3]["t"]))
    print(f"G1: {len(jobs)} reference integrations on {min(workers, len(jobs))} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        futs = {i: ex.submit(_ref_worker, jobs[i]) for i in order}
        refs = {i: futs[i].result() for i in order}
    g1 = {"per_cycle": {}, "tol_K": G1_TOL_K}
    worst = 0.0
    for ci, c in enumerate(ctx["cycles"]):
        ref, ref_s, _ = refs[ci]
        std, std_s, _ = refs[ci + 3]
        row = {"reference_wall_s": ref_s, "standard_tight_wall_s": std_s}
        for M in (M_EXPERIMENTS,) + G1_REPORT_M:
            y = modal_ts(tp["C_core"], tp["C_surf"], tp["R_cs"], tp["R_sa"], c["T0"], ctx["T_inf"],
                         c["Qs"][M], M)
            row[f"modal_M{M}_max_abs_K"] = _max_abs(y, ref)
        row["standard_tight_rtol1e-10_ms5_max_abs_K"] = _max_abs(std, ref)
        worst = max(worst, row[f"modal_M{M_EXPERIMENTS}_max_abs_K"])
        g1["per_cycle"][c["file"]] = row
        print(f"  {c['file'][:24]}: M16 {row['modal_M16_max_abs_K']:.3e}  M8 {row['modal_M8_max_abs_K']:.3e}"
              f"  M32 {row['modal_M32_max_abs_K']:.3e}  rtol1e-10/ms5 {row['standard_tight_rtol1e-10_ms5_max_abs_K']:.3e} K")
    g1["max_abs_K_M16"] = worst
    g1["passed"] = bool(worst <= G1_TOL_K)
    gates["G1"] = g1
    write(payload)
    if not g1["passed"]:
        raise GateFailed(f"G1: max |modal - reference| = {worst:.3e} K > {G1_TOL_K}", {"G1": g1})

    # ---- G2 --------------------------------------------------------------
    print("G2: determinism across processes ...", flush=True)
    res = []
    for _ in range(2):
        with ProcessPoolExecutor(max_workers=1) as ex:
            res.append(ex.submit(_g2_worker, None).result())
    par = _g2_worker(None)
    par["pid"] = os.getpid()
    g2 = {"pids": [r["pid"] for r in res] + [par["pid"]], "compared": {}}
    ok = len(set(g2["pids"])) == 3
    for key in ("y_full", "iid0", "ar1_0", "fit_T1_rep0"):
        same = all(np.array_equal(r[key], par[key]) for r in res)
        g2["compared"][key] = {"bitwise_identical": same,
                               "max_abs_diff": max(_max_abs(r[key], par[key]) for r in res)}
        ok = ok and same
    g2["passed"] = bool(ok)
    gates["G2"] = g2
    write(payload)
    print(f"  pids {g2['pids']}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise GateFailed("G2: results differ across processes", {"G2": g2})

    # ---- G3 --------------------------------------------------------------
    cols = {}
    for rel in G3_RELS:
        d = rel * tr["split"]
        cs = (surf_stack(ctx, tr["split"] + d, tr["R_cs"]) - surf_stack(ctx, tr["split"] - d, tr["R_cs"])) / (2 * d)
        dr = rel * tr["R_cs"]
        cr = (surf_stack(ctx, tr["split"], tr["R_cs"] + dr) - surf_stack(ctx, tr["split"], tr["R_cs"] - dr)) / (2 * dr)
        cols[rel] = (cs, cr)
    n_ref = np.linalg.norm(cols[0.02][0])
    g3 = {"norm_dy_dsplit": {str(r): float(np.linalg.norm(cols[r][0])) for r in G3_RELS},
          "rel_dev_vs_0.02": {str(r): float(np.linalg.norm(cols[r][0]) / n_ref - 1) for r in G3_RELS},
          "report_only_norm_dy_dRcs": {str(r): float(np.linalg.norm(cols[r][1])) for r in G3_RELS},
          "report_only_vector_rel_dist_split_vs_0.001": {
              str(r): float(np.linalg.norm(cols[r][0] - cols[0.001][0]) / np.linalg.norm(cols[0.001][0]))
              for r in G3_RELS},
          "tol": G3_TOL}
    g3["max_abs_rel_dev"] = float(max(abs(v) for v in g3["rel_dev_vs_0.02"].values()))
    g3["passed"] = bool(g3["max_abs_rel_dev"] <= G3_TOL)
    gates["G3"] = g3
    write(payload)
    print(f"G3: ||dy/dsplit|| {g3['norm_dy_dsplit']}  max dev {g3['max_abs_rel_dev']:.2e} -> "
          f"{'PASS' if g3['passed'] else 'FAIL'}")
    if not g3["passed"]:
        raise GateFailed("G3: split sensitivity not step-invariant", {"G3": g3})

    # ---- real residual, sigma, G4 -----------------------------------------
    y_meas = np.concatenate([c["y_meas"] for c in ctx["cycles"]])
    y_truth = surf_stack(ctx, tr["split"], tr["R_cs"])
    r_real = y_truth - y_meas
    sigma = float(np.sqrt(np.mean(r_real ** 2)))
    bounds_idx = np.cumsum([0] + [len(c["y_meas"]) for c in ctx["cycles"]])
    offset = np.empty_like(r_real)
    per_off = {}
    for ci, c in enumerate(ctx["cycles"]):
        sl = slice(bounds_idx[ci], bounds_idx[ci + 1])
        offset[sl] = r_real[sl].mean()
        per_off[c["file"]] = {"mean_K": float(r_real[sl].mean()),
                              "rms_K": float(np.sqrt(np.mean(r_real[sl] ** 2))), "n": int(sl.stop - sl.start)}
    shape = r_real - offset
    payload["real_residual"] = {
        "definition": "r_real = modal T_surf at truth - measured T_surf, stride 10",
        "sigma_rms_K": sigma, "n": int(len(r_real)), "mean_K": float(r_real.mean()),
        "per_cycle": per_off,
        "offset_share_of_mean_square": float(np.sum(offset ** 2) / np.sum(r_real ** 2)),
        "shape_rms_K": float(np.sqrt(np.mean(shape ** 2))),
        "cost_at_truth": float(0.5 * np.sum(r_real ** 2))}

    g4fit = fit_primary(ctx, y_meas)
    with open(SPLIT_REFIT_JSON, encoding="utf-8") as fh:
        refit = json.load(fh)
    ref = [f for f in refit["fits"] if f["name"] == "central" and f["diff_step"] == 0.02]
    if len(ref) != 1 or round(ref[0]["cost"], 10) != G4_REF_COST:
        raise idt.SpecMismatch(f"split-refit central fit not found or cost != {G4_REF_COST}")
    rel = abs(g4fit["cost"] - ref[0]["cost"]) / ref[0]["cost"]
    g4 = {"fit": g4fit, "reference_cost": ref[0]["cost"], "reference_split": ref[0]["split"],
          "cost_rel_diff": rel, "lands_at_0.99": bool(g4fit["hit_ub"]),
          "passed": bool(g4fit["hit_ub"] and rel <= G4_COST_RTOL)}
    gates["G4"] = g4
    write(payload)
    print(f"sigma = {sigma:.6f} K;  G4: split {g4fit['split']:.10f} cost {g4fit['cost']:.10f} "
          f"(ref {ref[0]['cost']:.10f}, rel {rel:.2e}) -> {'PASS' if g4['passed'] else 'FAIL'}")
    if not g4["passed"]:
        raise GateFailed("G4: real-data split-only fit does not reproduce the Phase A central fit", {"G4": g4})

    # ---- CRBs at truth -----------------------------------------------------
    d = CRB_FD_REL * tr["split"]
    Js = (surf_stack(ctx, tr["split"] + d, tr["R_cs"]) - surf_stack(ctx, tr["split"] - d, tr["R_cs"])) / (2 * d)
    dr = CRB_FD_REL * tr["R_cs"]
    Jr = (surf_stack(ctx, tr["split"], tr["R_cs"] + dr) - surf_stack(ctx, tr["split"], tr["R_cs"] - dr)) / (2 * dr)
    sd_iid = sigma / np.linalg.norm(Js)
    blocks = ar1_cov_blocks(ctx, sigma)
    jtsj, jtsij = 0.0, 0.0
    for ci, blk in enumerate(blocks):
        js = Js[bounds_idx[ci]:bounds_idx[ci + 1]]
        jtsj += js @ blk @ js
        jtsij += js @ np.linalg.solve(blk, js)
    sd_ar1_ols = float(np.sqrt(jtsj) / (Js @ Js))
    sd_ar1_gls = float(1 / np.sqrt(jtsij))
    J2 = np.column_stack([Js, Jr])
    cov2 = sigma ** 2 * np.linalg.inv(J2.T @ J2)
    sd2 = np.sqrt(np.diag(cov2))

    def p_hit(sd, x=tr["split"], lb=PRIMARY["lb"], ub=PRIMARY["ub"]):
        return float((1 - ndtr((ub - x) / sd)) + ndtr((lb - x) / sd))

    payload["crb"] = {
        "jacobian_norm_split": float(np.linalg.norm(Js)), "jacobian_norm_R_cs": float(np.linalg.norm(Jr)),
        "primary_iid": {"sd_split": float(sd_iid), "predicted_bound_hit": p_hit(sd_iid)},
        "primary_ar1": {"sd_split_ols_sandwich": sd_ar1_ols, "sd_split_gls_crb": sd_ar1_gls,
                        "predicted_bound_hit_ols": p_hit(sd_ar1_ols)},
        "joint_iid": {"sd_split": float(sd2[0]), "sd_R_cs": float(sd2[1]),
                      "corr": float(cov2[0, 1] / (sd2[0] * sd2[1])),
                      "predicted_split_bound_hit": p_hit(sd2[0]),
                      "predicted_R_cs_bound_hit": p_hit(sd2[1], tr["R_cs"], JOINT["lb"][1], JOINT["ub"][1])},
    }
    write(payload)
    print(f"CRB: primary iid sd {sd_iid:.5f} (pred hit {p_hit(sd_iid):.3f}); AR(1) OLS sd {sd_ar1_ols:.5f} "
          f"(pred hit {p_hit(sd_ar1_ols):.3f}); joint sd split {sd2[0]:.5f}, R_cs {sd2[1]:.4f}")

    # ---- Tiers -------------------------------------------------------------
    tiers = payload.setdefault("tiers", {})
    tiers["T0"] = {"primary": [dict(fit_primary(ctx, y_truth, s0), start=s0) for s0 in T0_STARTS],
                   "joint": [dict(fit_joint(ctx, y_truth, (s0, JOINT_T0_RCS_START)), start=[s0, JOINT_T0_RCS_START])
                             for s0 in T0_STARTS]}
    write(payload)
    print("T0 primary:", [round(f["split"], 8) for f in tiers["T0"]["primary"]])
    print("T0 joint  :", [(round(f["split"], 6), round(f["R_cs"], 4)) for f in tiers["T0"]["joint"]])

    E1 = iid_draws(ctx, sigma)
    t1p = [fit_primary(ctx, y_truth + E1[i]) for i in range(N_MC)]
    print("T1 primary done", flush=True)
    t1j = [fit_joint(ctx, y_truth + E1[i]) for i in range(N_MC)]
    print("T1 joint done", flush=True)
    E2 = ar1_draws(ctx, sigma)
    t1bp = [fit_primary(ctx, y_truth + E2[i]) for i in range(N_MC)]
    print("T1b primary done", flush=True)
    tiers["T1"] = {"primary": {"summary": summarise(t1p, tr), "split_hat": [f["split"] for f in t1p],
                               "status": [f["status"] for f in t1p]},
                   "joint": {"summary": summarise(t1j, tr, ("split", "R_cs")),
                             "split_hat": [f["split"] for f in t1j], "R_cs_hat": [f["R_cs"] for f in t1j],
                             "status": [f["status"] for f in t1j]},
                   "realised_noise_sd_mean": float(E1.std(axis=1, ddof=0).mean())}
    tiers["T1b"] = {"primary": {"summary": summarise(t1bp, tr), "split_hat": [f["split"] for f in t1bp],
                                "status": [f["status"] for f in t1bp]},
                    "realised_noise_sd_mean": float(E2.std(axis=1, ddof=0).mean()),
                    "realised_lag10_autocorr_mean": float(np.mean(
                        [np.corrcoef(e[:-1], e[1:])[0, 1] for e in E2])),
                    "expected_lag10_autocorr": AR1_PHI ** STRIDE}
    tiers["T2"] = {"primary": fit_primary(ctx, y_truth - offset)}
    tiers["T3"] = {"primary": fit_primary(ctx, y_truth - shape)}
    t4 = fit_primary(ctx, y_truth - r_real)
    tiers["T4"] = {"primary": t4, "max_abs_data_diff_vs_measured_K": _max_abs(y_truth - r_real, y_meas),
                   "split_diff_vs_G4": t4["split"] - g4fit["split"],
                   "cost_rel_diff_vs_G4": abs(t4["cost"] - g4fit["cost"]) / g4fit["cost"]}
    t4ok = bool(t4["hit_ub"] and abs(t4["cost"] - ref[0]["cost"]) / ref[0]["cost"] <= G4_COST_RTOL)
    tiers["T4"]["matches_G4"] = t4ok
    write(payload)
    if not t4ok:
        raise GateFailed("T4 does not reproduce G4", {"T4_consistency": tiers["T4"]})

    # ---- Rules -------------------------------------------------------------
    crb = payload["crb"]
    t1s, t1bs = tiers["T1"]["primary"]["summary"], tiers["T1b"]["primary"]["summary"]
    r1_err = [abs(f["split"] - tr["split"]) for f in tiers["T0"]["primary"]]
    sd_ratio = t1s["split"]["sd"] / crb["primary_iid"]["sd_split"]
    hit_diff_pp = 100 * (t1s["bound_hit_fraction"] - crb["primary_iid"]["predicted_bound_hit"])
    r3_pp = 100 * (t1bs["bound_hit_fraction"] - t1s["bound_hit_fraction"])
    t2_hit, t3_hit = tiers["T2"]["primary"]["hit_ub"], tiers["T3"]["primary"]["hit_ub"]
    r4 = {(True, False): "T2 only", (False, True): "T3 only", (True, True): "both",
          (False, False): "neither"}[(t2_hit, t3_hit)]
    payload["rule_outcomes"] = {
        "R1": {"max_abs_error": max(r1_err), "per_start": dict(zip(map(str, T0_STARTS), r1_err)),
               "fires": bool(max(r1_err) <= RULES["R1"]["threshold"]),
               "supplementary_joint_T0_max_abs_split_error": max(abs(f["split"] - tr["split"])
                                                                 for f in tiers["T0"]["joint"]),
               "supplementary_joint_T0_max_abs_R_cs_error": max(abs(f["R_cs"] - tr["R_cs"])
                                                                for f in tiers["T0"]["joint"])},
        "R2": {"sd_ratio": sd_ratio, "bound_hit_observed": t1s["bound_hit_fraction"],
               "bound_hit_predicted": crb["primary_iid"]["predicted_bound_hit"],
               "bound_hit_diff_pp": hit_diff_pp,
               "sd_ratio_in_range": bool(0.7 <= sd_ratio <= 1.4),
               "bound_hit_within_5pp": bool(abs(hit_diff_pp) <= 5.0),
               "fires": bool(0.7 <= sd_ratio <= 1.4 and abs(hit_diff_pp) <= 5.0)},
        "R3": {"bound_hit_T1": t1s["bound_hit_fraction"], "bound_hit_T1b": t1bs["bound_hit_fraction"],
               "diff_pp": r3_pp, "fires": bool(abs(r3_pp) > 10.0)},
        "R4": {"T2_split": tiers["T2"]["primary"]["split"], "T3_split": tiers["T3"]["primary"]["split"],
               "T2_at_0.99": bool(t2_hit), "T3_at_0.99": bool(t3_hit), "outcome": r4},
    }
    write(payload)
    for k, v in payload["rule_outcomes"].items():
        print(f"{k}: {v.get('fires', v.get('outcome'))}  " +
              ", ".join(f"{a}={b:.6g}" if isinstance(b, float) else f"{a}={b}"
                        for a, b in v.items() if a not in ("per_start",)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    return run(a.workers, a.allow_dirty)


if __name__ == "__main__":
    sys.exit(main())

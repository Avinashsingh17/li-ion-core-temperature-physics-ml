"""
calibrate.py — Constrained calibration (Option A) of the two-state thermal model.

Run order (each stage prints its result before moving to the next):

  Step 1  pin total heat capacity from m·c_p (NOT fit).
  Step 2  anchor R_sa from cooling tails (independent of the main fit).
          Fallback to literature natural-convection estimate if no clean tails.
  Step 3  DIAGNOSTIC fit of (split, R_cs) with C_total + R_sa fixed. Reports the
          95 % CI on R_cs from the least_squares Jacobian. **This CI is itself
          a headline result**: it empirically confirms the Lin-2013 / Zheng-2025
          identifiability finding on our own data.
  Step 4  Anchor R_cs at a central value from 18650 jellyroll geometry. Fit
          ONLY the split with everything else fixed — this is the CENTRAL
          model that generates core-temp labels.
  Step 5  Label band — sweep R_cs across its plausible physical range, re-fit
          the split at each R_cs, and capture the spread of T_core. Plot the
          central core line + shaded low/high envelope.
  Step 6  Validation on held-out cycles. Report RMSE / MAE / max for the
          modeled vs measured SURFACE temperature, alongside the naive baseline
          "T_surface = T_ambient." No calibration / validation overlap. Per-cycle
          SOC clamp counts surfaced. Core labels CANNOT be validated directly —
          we plausibility-check the core−surface gap vs literature (~single °C
          in normal drive, growing with C-rate).

Locked decisions used here:
  - constrained, not free 4-param fit  (decisions/calibration-constrained-fit.md).
  - Q = I·(V − V_ocv) with raw current sign  (decisions/sign-convention.md).
  - never random-split time series  (decisions/validation-surface-only.md).
  - calibration and validation cycles never overlap  (this module).

Output: data/calibration/calibration_results.json + sanity plots in data/eda_plots/.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, curve_fit

from build_ocv import AH_NOMINAL, load_ocv
from thermal_model import (
    ThermalParams, cycle_soc, clamp_soc, thermal_rhs,
    SOC_TABLE_MIN, SOC_TABLE_MAX,
    PROCESSED_DIR, PLOTS_DIR, PROJECT_ROOT,
)


# ============================================================================
# Cell physical constants (Step 1 inputs)
# ============================================================================
# Cell mass: not stated as an explicit number in the LG HG2 datasheet. Published
# energy density (240 Wh/kg) × nominal V × Ah implies a lower bound of ~45 g.
# Consensus published mass is 47–48 g; we commit to 48 g per the user's directive.
CELL_MASS_KG = 0.048
# Specific heat of a Li-ion cell: literature mid-range for jellyroll + can.
# 800–1000 J/(kg·K); we anchor at 900.
CP_J_KG_K = 900.0
# Total lumped heat capacity (PINNED, not fit).
C_TOTAL_J_K = CELL_MASS_KG * CP_J_KG_K  # ≈ 43.2 J/K

# 18650 geometry for R_cs anchor (Step 4).
CELL_RADIUS_M = 0.009     # 9 mm
CELL_LENGTH_M = 0.065     # 65 mm
CORE_NODE_RADIUS_M = 0.0045   # effective core-node radius (= r/2)


# ============================================================================
# Cycle selection (LOCKED — calibration / validation must not overlap)
# ============================================================================
# Calibration: well-excited drive cycles at 25 °C with substantial heating
# activity and no SOC clamping expected.
CAL_CYCLE_GLOBS = [
    "LG_25degC_US06_*.parquet",
    "LG_25degC_LA92_*.parquet",
    "LG_25degC_UDDS_*.parquet",
]
# Validation, same ambient (different drive-cycle types, held-out).
VAL_SAME_AMBIENT_GLOBS = [
    "LG_25degC_Mixed1_*.parquet",
    "LG_25degC_Mixed2_*.parquet",
]
# Validation, different ambient (stronger generalization test).
VAL_DIFF_AMBIENT_GLOBS = [
    "LG_10degC_UDDS_*.parquet",
]
# Aggressive cold-ambient validation. Both 0 °C cycles are currently in the
# tail pool — the leakage filter excludes them from tail derivation
# automatically (so the 0 °C tail count drops from 10 to 8, still strong).
# 0 °C also has the largest sample of tails (10) so its T_inf anchor is solid.
VAL_AGGRESSIVE_GLOBS = [
    "LG_0degC_US06_*.parquet",
    "LG_0degC_LA92_*.parquet",
]

# Cooling-tail search pool: scan all LG ambients we'll calibrate / validate on.
# The cell relaxes to whatever the chamber's true rest temperature is at each
# setpoint — we fit T_inf as a free parameter (the user's diagnosis: assuming
# T_inf = nominal setpoint floats the model ~1 °C above measured surface temp).
TAIL_SEARCH_GLOBS = [
    "LG_25degC_*.parquet",
    "LG_10degC_*.parquet",
    "LG_0degC_*.parquet",
    "LG_40degC_*.parquet",
]

# Minimum number of tails at an ambient to use that ambient's own median as
# T_inf. With MIN_TAILS_PER_AMBIENT=1 we use the median of even a single tail
# rather than the cross-ambient mean offset, because the offset depends on
# ambient (it grows with temperature) — so the cross-ambient mean is biased
# when applied to a warm ambient. Ambients with ZERO tails use the fallback.
# Single-tail ambients are flagged "low-confidence" in the printout.
MIN_TAILS_PER_AMBIENT = 1

# Cooling tail filter thresholds.
TAIL_I_THRESHOLD_A = 0.05    # |I| < this counts as "rest"
TAIL_MIN_DURATION_S = 300    # rest must last at least 5 min to capture τ
TAIL_MIN_DT_START_C = 0.3    # cell must be at least 0.3 °C warmer than ambient
TAIL_MIN_DECAY_C = 0.05      # cell must actually cool over the rest period


# ============================================================================
# Output paths
# ============================================================================
OUT_DIR = PROJECT_ROOT / "data" / "calibration"
CAL_RESULTS_JSON = OUT_DIR / "calibration_results.json"
LABEL_BAND_PLOT = PLOTS_DIR / "thermal_label_band.png"
COOLING_TAIL_PLOT = PLOTS_DIR / "cooling_tails.png"
SURFACE_VAL_PLOT = PLOTS_DIR / "surface_validation.png"


# ============================================================================
# Cycle pre-loading + cached forward simulator
# ============================================================================
def expand_glob(pattern: str) -> Path:
    matches = sorted(PROCESSED_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no Parquet matched: {pattern}")
    if len(matches) > 1:
        print(f"  (glob {pattern} matched {len(matches)}, using first)")
    return matches[0]


def precompute_cycle(path: Path, soc_init: float = 1.0,
                     T_inf_override: float | None = None) -> dict:
    """Pre-load a Parquet + build all the static parts used in every fit eval.

    Returned dict can be reused across many `simulate_T(cache, params)` calls
    without re-reading the file or re-building interpolators.

    If `T_inf_override` is given (typically the data-driven T_inf for this
    cycle's ambient group, from cooling-tail fits), the ambient cooling-sink
    series fed to the ODE is replaced by this constant value. The original
    `T_ambient_C` column is preserved as `T_amb_nominal_array` for plotting
    context.
    """
    df = pd.read_parquet(path)
    t = df["time_s"].to_numpy(dtype=float)
    nominal_amb = df["T_ambient_C"].to_numpy(dtype=float)
    ambient_setpoint = int(round(float(df["ambient_setpoint_C"].iloc[0])))

    if T_inf_override is None:
        amb_series = nominal_amb
    else:
        amb_series = np.full_like(t, T_inf_override, dtype=float)

    soc_raw = cycle_soc(df, soc_init=soc_init)
    soc_clamped, n_low, n_high = clamp_soc(soc_raw)

    kw = dict(kind="linear", bounds_error=False,
              fill_value="extrapolate", assume_sorted=True)
    return {
        "t_grid":      t,
        "I_of_t":      interp1d(t, df["current_A"].to_numpy(dtype=float), **kw),
        "V_of_t":      interp1d(t, df["voltage_V"].to_numpy(dtype=float), **kw),
        "T_amb_of_t":  interp1d(t, amb_series, **kw),  # used by the ODE
        "soc_of_t":    interp1d(t, soc_clamped, **kw),
        "v_ocv":       load_ocv(),
        "T_surf_meas": df["T_surface_C"].to_numpy(dtype=float),
        "T_amb_array": amb_series,                      # ODE sink, plotted
        "T_amb_nominal_array": nominal_amb,             # for context
        "I_array":     df["current_A"].to_numpy(dtype=float),
        "df":          df,
        "file_id":     path.stem,
        "ambient_setpoint_C": ambient_setpoint,
        "T_inf_C":     T_inf_override if T_inf_override is not None else ambient_setpoint,
        "soc_clamp_low":  int(n_low),
        "soc_clamp_high": int(n_high),
    }


def simulate_T(cache: dict, params: ThermalParams, t_eval=None,
               max_step: float = 10.0, rtol: float = 1e-4,
               atol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward-simulate the thermal ODE on a pre-loaded cycle cache.

    Returns (t, T_core, T_surf, Q) all sampled at `t_eval` (defaults to the
    cycle's 1 Hz grid).
    """
    t_grid = cache["t_grid"]
    if t_eval is None:
        t_eval = t_grid

    def Q_of_t(t: float) -> float:
        soc_t = float(cache["soc_of_t"](t))
        I_t = float(cache["I_of_t"](t))
        V_t = float(cache["V_of_t"](t))
        return I_t * (V_t - float(cache["v_ocv"](soc_t)))

    # Initial condition: assume thermal equilibrium with measured surface temp
    # at t=0 (cell was at rest before the cycle started). Using `T_amb(0)`
    # would introduce a fixed offset for cycles where the cell actually
    # started somewhat warmer or cooler than the chamber setpoint, and that
    # offset swamps the gradient on the fit parameters.
    T0 = float(cache["T_surf_meas"][0])
    y0 = np.array([T0, T0])

    sol = solve_ivp(
        thermal_rhs,
        t_span=(t_grid[0], t_grid[-1]),
        y0=y0,
        t_eval=t_eval,
        args=(Q_of_t, cache["T_amb_of_t"], params),
        method="RK45",
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed on {cache['file_id']}: {sol.message}")

    I_e = cache["I_of_t"](sol.t)
    V_e = cache["V_of_t"](sol.t)
    soc_e = cache["soc_of_t"](sol.t)
    Q_e = I_e * (V_e - cache["v_ocv"](soc_e))
    return sol.t, sol.y[0], sol.y[1], Q_e


# ============================================================================
# Step 1: Pin C_total
# ============================================================================
def step1_pin_C_total() -> float:
    print("\n" + "=" * 70)
    print("Step 1 — pin C_total from m · c_p (NOT fit)")
    print("=" * 70)
    print(f"  cell mass     m   = {CELL_MASS_KG * 1000:.1f} g  (LG HG2; "
          f"datasheet energy density implies ≥45 g, consensus spec 47–48 g)")
    print(f"  specific heat c_p = {CP_J_KG_K:.0f} J/(kg·K)  (Li-ion literature mid-range)")
    print(f"  C_total = m·c_p   = {C_TOTAL_J_K:.2f} J/K  (pinned)")
    return C_TOTAL_J_K


# ============================================================================
# Step 2: Cooling-tail R_sa anchor
# ============================================================================
def find_cooling_tails(file_paths: list[Path]) -> list[dict]:
    """Scan each file for sustained-rest periods.

    Returns a list of tail dicts including the cycle's ambient setpoint, the
    time slice, and measured T_s / nominal T_amb arrays.
    """
    tails = []
    for path in file_paths:
        df = pd.read_parquet(path)
        abs_I = df["current_A"].abs().to_numpy()
        T_s = df["T_surface_C"].to_numpy()
        T_a = df["T_ambient_C"].to_numpy()
        t = df["time_s"].to_numpy()
        ambient_setpoint = int(round(float(df["ambient_setpoint_C"].iloc[0])))
        rest = abs_I < TAIL_I_THRESHOLD_A

        padded = np.concatenate([[False], rest, [False]]).astype(int)
        boundaries = np.diff(padded)
        starts = np.where(boundaries == 1)[0]
        ends = np.where(boundaries == -1)[0]

        for s, e in zip(starts, ends):
            duration_s = t[e - 1] - t[s]
            if duration_s < TAIL_MIN_DURATION_S:
                continue
            dT0_nominal = T_s[s] - T_a[s]
            if dT0_nominal < TAIL_MIN_DT_START_C:
                continue
            # cell must visibly cool — require the surface to drop by at least
            # TAIL_MIN_DECAY_C from start to end.
            if (T_s[s] - T_s[e - 1]) < TAIL_MIN_DECAY_C:
                continue
            tails.append({
                "file_id": path.stem,
                "ambient_setpoint_C": ambient_setpoint,
                "t": t[s:e] - t[s],
                "T_s": T_s[s:e].copy(),
                "T_a_nominal": T_a[s:e].copy(),
                "duration_s": float(duration_s),
            })
    return tails


def fit_tail_with_free_asymptote(t: np.ndarray, T_s: np.ndarray
                                 ) -> tuple[float, float, float] | None:
    """Fit T_s(t) = T_inf + ΔT₀ · exp(−t/τ) with **all three params free**.

    `T_inf` is the cell's actual rest temperature — NOT assumed to equal the
    chamber setpoint. This is the user's diagnostic correction: LG does not
    provide a measured chamber temperature, so the cooling-tail asymptote is
    our only data-driven estimate of the cooling sink the ODE should target.

    Returns (T_inf, ΔT₀, τ) in (°C, °C, s), or None on failure.
    """
    def model(t_, T_inf, dT0, tau):
        return T_inf + dT0 * np.exp(-t_ / tau)

    T_min, T_max = float(T_s.min()), float(T_s.max())
    # Initial guesses: T_inf ≈ last value, ΔT₀ ≈ swing, τ ≈ a fraction of duration.
    T_inf_0 = float(T_s[-1])
    dT0_0 = max(float(T_s[0] - T_s[-1]), 0.5)
    tau_0 = float(max(60.0, (t[-1] - t[0]) / 3.0))
    p0 = [T_inf_0, dT0_0, tau_0]

    bounds = (
        [T_min - 3.0, 0.05, 10.0],
        [T_max + 3.0, 20.0, 5000.0],
    )
    try:
        popt, _ = curve_fit(model, t, T_s, p0=p0, bounds=bounds, maxfev=10000)
    except Exception:
        return None
    T_inf, dT0, tau = float(popt[0]), float(popt[1]), float(popt[2])
    # Sanity: tau should be in a physically reasonable range.
    if tau < 20 or tau > 3000:
        return None
    return T_inf, dT0, tau


def step2_anchor_R_sa_and_T_inf(C_total: float,
                                exclude_paths: set[Path] | None = None) -> dict:
    """Free-asymptote cooling-tail fits.

    Reports T_inf (the cell's actual rest temperature, NOT the chamber
    setpoint) per ambient group, and re-anchors R_sa from τ of the
    free-asymptote fits.

    `exclude_paths` is the LEAKAGE GUARD: validation-cycle files passed here
    are filtered out of the tail-search pool so that T_inf and R_sa never see
    held-out data. If excluding leaves an ambient with fewer than
    `MIN_TAILS_PER_AMBIENT` tails, that ambient's T_inf falls back to
    (setpoint + cross-ambient mean offset) and the fallback is surfaced.
    """
    print("\n" + "=" * 70)
    print("Step 2 — cooling-tail fits with FREE asymptote (T_inf, R_sa)")
    print("=" * 70)
    candidates = []
    for g in TAIL_SEARCH_GLOBS:
        candidates.extend(sorted(PROCESSED_DIR.glob(g)))
    n_before = len(candidates)
    excluded_names: set[str] = set()
    if exclude_paths:
        excluded_names = {p.name for p in exclude_paths}
        candidates = [c for c in candidates if c.name not in excluded_names]
    n_excluded = n_before - len(candidates)
    print(f"  scanning {len(candidates)} candidate files across "
          f"{len(TAIL_SEARCH_GLOBS)} ambient groups for "
          f"|I|<{TAIL_I_THRESHOLD_A} A sustained ≥{TAIL_MIN_DURATION_S} s "
          f"with the cell visibly cooling "
          f"(excluded {n_excluded} validation file(s) up front)")
    if excluded_names:
        for n in sorted(excluded_names):
            print(f"    EXCLUDED from tail search: {n}")
    tails = find_cooling_tails(candidates)
    print(f"  found {len(tails)} candidate rest-periods")

    fits: list[dict] = []
    plot_payload: list[tuple[dict, dict]] = []
    for tail in tails:
        result = fit_tail_with_free_asymptote(tail["t"], tail["T_s"])
        if result is None:
            continue
        T_inf, dT0, tau = result
        R_sa = tau / C_total
        nominal = tail["ambient_setpoint_C"]
        fits.append({
            "file_id": tail["file_id"],
            "ambient_setpoint_C": nominal,
            "T_inf_C": T_inf,
            "T_inf_offset_C": T_inf - nominal,
            "dT0_C": dT0,
            "tau_s": tau,
            "R_sa_K_per_W": R_sa,
            "duration_s": tail["duration_s"],
        })
        plot_payload.append((tail, fits[-1]))

    if not fits:
        # Literature fallback: gentle convection on an 18650 (h ≈ 5–20 W/(m²·K),
        # A ≈ 3.9e-3 m², so R_sa ≈ 1/(h·A) ≈ 13–51 K/W). T_inf falls back to
        # the nominal setpoint (no correction). Flagged.
        print("  ⚠ no clean tails — falling back to literature R_sa and "
              "nominal setpoints for T_inf")
        return {
            "method": "literature",
            "R_sa_central_K_per_W": 26.0,
            "R_sa_low_K_per_W": 13.0,
            "R_sa_high_K_per_W": 51.0,
            "T_inf_by_ambient_C": {25: 25.0, 10: 10.0, 0: 0.0, -10: -10.0, 40: 40.0},
            "T_inf_offset_C_mean": 0.0,
            "n_tails_used": 0,
            "per_tail": [],
        }

    # Per-tail printout: T_inf vs nominal setpoint.
    print(f"\n  per-tail fits ({len(fits)} usable):")
    print(f"    {'file_id':55s} {'amb_set':>7s} {'T_inf':>7s} "
          f"{'offset':>7s} {'tau':>6s} {'R_sa':>7s}")
    for d in fits:
        print(f"    {d['file_id'][:55]:55s} "
              f"{d['ambient_setpoint_C']:>6d}°C "
              f"{d['T_inf_C']:>6.2f}°C "
              f"{d['T_inf_offset_C']:>+6.2f}°C "
              f"{d['tau_s']:>5.0f}s "
              f"{d['R_sa_K_per_W']:>6.2f}")

    # Aggregate T_inf per ambient (median when ≥ MIN tails; else fall back).
    T_inf_by_amb_raw: dict[int, list[float]] = {}
    for d in fits:
        T_inf_by_amb_raw.setdefault(d["ambient_setpoint_C"], []).append(d["T_inf_C"])

    # Cross-ambient mean offset uses ALL kept tails (including ones from
    # ambients that only had a single observation — that observation is still
    # legitimate signal for the global offset, just not enough on its own to
    # median).
    offsets = [t - amb for amb, ts in T_inf_by_amb_raw.items() for t in ts]
    mean_offset = float(np.mean(offsets))

    T_inf_by_ambient: dict[int, float] = {}
    T_inf_source: dict[int, str] = {}
    fallback_ambients: list[int] = []
    low_confidence_ambients: list[int] = []
    for amb in (-20, -10, 0, 10, 25, 40):
        present = len(T_inf_by_amb_raw.get(amb, []))
        if present >= MIN_TAILS_PER_AMBIENT:
            T_inf_by_ambient[amb] = float(np.median(T_inf_by_amb_raw[amb]))
            T_inf_source[amb] = f"median of {present} tail(s) at this ambient"
            if present == 1:
                low_confidence_ambients.append(amb)
        else:
            T_inf_by_ambient[amb] = float(amb + mean_offset)
            fallback_ambients.append(amb)
            T_inf_source[amb] = (
                f"fallback: setpoint + cross-ambient mean offset "
                f"(0 tails at this ambient)"
            )

    # Offset-vs-ambient trend table — surface the temperature dependence so
    # the reader can see WHY a cross-ambient mean is the wrong fallback to
    # extrapolate to warm ambients.
    print(f"\n  OFFSET-vs-AMBIENT trend (T_inf − nominal_setpoint) — the "
          f"reason we use per-ambient medians, not a global mean:")
    print(f"    {'ambient':>8s}  {'median offset':>15s}  {'n tails':>8s}")
    for amb in sorted(T_inf_by_amb_raw.keys()):
        offset_med = float(np.median([t - amb for t in T_inf_by_amb_raw[amb]]))
        n = len(T_inf_by_amb_raw[amb])
        flag = "  [LOW CONFIDENCE: n=1]" if n == 1 else ""
        print(f"    {amb:>5d} °C  {offset_med:>+13.2f} °C  {n:>8d}{flag}")
    print(f"    cross-ambient mean offset across all {len(offsets)} tails: "
          f"{mean_offset:+.2f} °C  (used ONLY for ambients with 0 tails)")

    print(f"\n  aggregated T_inf per ambient (used as ODE cooling sink):")
    for amb in sorted(T_inf_by_ambient.keys()):
        T_inf = T_inf_by_ambient[amb]
        marker = ""
        if amb in fallback_ambients:
            marker = "  <- FALLBACK (no tails)"
        elif amb in low_confidence_ambients:
            marker = "  <- LOW CONFIDENCE (only 1 tail)"
        print(f"    {amb:>3d} °C nominal  ->  T_inf = {T_inf:6.2f} °C  "
              f"(offset {T_inf - amb:+5.2f} °C)  "
              f"[{T_inf_source[amb]}]{marker}")
    if fallback_ambients:
        print(f"  WARN  cross-ambient mean offset applied at ambient(s): "
              f"{fallback_ambients} — RELIES on extrapolation across an "
              f"offset-vs-ambient trend (see table above)")
    if low_confidence_ambients:
        print(f"  WARN  single-tail ambient(s): {low_confidence_ambients} "
              f"— T_inf there is the one observed value (better than the "
              f"biased cross-ambient mean, but no consistency check)")

    R_sa_values = [d["R_sa_K_per_W"] for d in fits]
    R_sa_central = float(np.median(R_sa_values))
    R_sa_low = float(np.min(R_sa_values))
    R_sa_high = float(np.max(R_sa_values))
    print(f"\n  R_sa (re-anchored from free-asymptote τ): "
          f"median={R_sa_central:.2f}, range=[{R_sa_low:.2f}, {R_sa_high:.2f}] K/W")

    plot_tail_diagnostic(plot_payload, COOLING_TAIL_PLOT)

    return {
        "method": "cooling-tails-free-asymptote",
        "R_sa_central_K_per_W": R_sa_central,
        "R_sa_low_K_per_W": R_sa_low,
        "R_sa_high_K_per_W": R_sa_high,
        "T_inf_by_ambient_C": T_inf_by_ambient,
        "T_inf_offset_C_mean": mean_offset,
        "T_inf_source": T_inf_source,
        "fallback_ambients": fallback_ambients,
        "low_confidence_ambients": low_confidence_ambients,
        "n_validation_files_excluded": n_excluded,
        "validation_files_excluded": sorted(excluded_names),
        "n_tails_used": len(fits),
        "per_tail": fits,
    }


def plot_tail_diagnostic(payload: list[tuple[dict, dict]], out: Path) -> None:
    if not payload:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(payload), 8)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (tail, fit) in zip(axes, payload[:n]):
        t = tail["t"]
        T_s = tail["T_s"]
        T_a_nominal = tail["T_a_nominal"]
        # Reconstruct the model curve.
        T_model = fit["T_inf_C"] + fit["dT0_C"] * np.exp(-t / fit["tau_s"])
        ax.plot(t / 60, T_s, lw=0.8, color="C0", label="measured T_s")
        ax.plot(t / 60, T_model, lw=1.0, ls="--", color="C1",
                label=(f"fit  T_inf={fit['T_inf_C']:.2f}°C "
                       f"(offset {fit['T_inf_offset_C']:+.2f}),  "
                       f"τ={fit['tau_s']:.0f}s,  R_sa={fit['R_sa_K_per_W']:.2f}"))
        ax.axhline(T_a_nominal[0], lw=0.6, ls=":", color="C2",
                   label=f"nominal setpoint = {T_a_nominal[0]:.1f} °C")
        ax.set_xlabel("time within rest period (min)")
        ax.set_ylabel("T (°C)")
        ax.set_title(fit["file_id"][:80], fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ============================================================================
# Step 3: Diagnostic fit (the unidentifiability empirical confirmation)
# ============================================================================
def _residuals_split_Rcs(p: np.ndarray, C_total: float, R_sa: float,
                         cal_caches: list[dict], stride: int) -> np.ndarray:
    split, R_cs = p
    params = ThermalParams(
        C_core=split * C_total,
        C_surf=(1 - split) * C_total,
        R_cs=R_cs,
        R_sa=R_sa,
    )
    parts = []
    for cache in cal_caches:
        t_eval = cache["t_grid"][::stride]
        _, _, T_s_model, _ = simulate_T(cache, params, t_eval=t_eval,
                                        max_step=20, rtol=1e-3, atol=1e-5)
        parts.append(T_s_model - cache["T_surf_meas"][::stride])
    return np.concatenate(parts)


def _residuals_split_only(p: np.ndarray, C_total: float, R_sa: float,
                          R_cs_fixed: float, cal_caches: list[dict],
                          stride: int) -> np.ndarray:
    split = float(p[0])
    params = ThermalParams(
        C_core=split * C_total,
        C_surf=(1 - split) * C_total,
        R_cs=R_cs_fixed,
        R_sa=R_sa,
    )
    parts = []
    for cache in cal_caches:
        t_eval = cache["t_grid"][::stride]
        _, _, T_s_model, _ = simulate_T(cache, params, t_eval=t_eval,
                                        max_step=20, rtol=1e-3, atol=1e-5)
        parts.append(T_s_model - cache["T_surf_meas"][::stride])
    return np.concatenate(parts)


def _jacobian_ci(result, n_params: int) -> np.ndarray:
    """95 % CIs from `least_squares.jac`. Returns NaNs on singular Jacobian."""
    n_resid = len(result.fun)
    if n_resid <= n_params:
        return np.full(n_params, np.nan)
    s2 = float(np.sum(result.fun ** 2) / (n_resid - n_params))
    J = result.jac
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return np.full(n_params, np.inf)
    diag = np.diag(cov)
    diag = np.where(diag > 0, diag, np.nan)
    return 1.96 * np.sqrt(diag)


def step3_diagnostic_fit(C_total: float, R_sa: float,
                         cal_caches: list[dict]) -> dict:
    print("\n" + "=" * 70)
    print("Step 3 — DIAGNOSTIC fit (split, R_cs) free; everything else fixed")
    print("=" * 70)
    print(f"  fixed:  C_total = {C_total:.2f} J/K,  R_sa = {R_sa:.2f} K/W")
    print(f"  free:   split = C_c/C_total ∈ [0.50, 0.99],  R_cs ∈ [0.1, 30.0] K/W")
    print(f"  cal cycles ({len(cal_caches)}):")
    for c in cal_caches:
        print(f"    {c['file_id']}  ({len(c['t_grid'])} samples)")

    stride = 10
    p0 = np.array([0.93, 1.5])
    bounds = ([0.50, 0.1], [0.99, 30.0])

    print("  optimizing (this takes a few minutes)...")
    res = least_squares(
        _residuals_split_Rcs, p0, args=(C_total, R_sa, cal_caches, stride),
        bounds=bounds, method="trf", max_nfev=200, verbose=1,
        x_scale="jac",      # auto-scale params from Jacobian magnitudes
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
        diff_step=0.02,     # FD perturbation; default ~1.5e-8 is too tight
    )
    ci95 = _jacobian_ci(res, n_params=2)
    split_fit, R_cs_fit = float(res.x[0]), float(res.x[1])
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))

    print(f"\n  RESULT:")
    print(f"    split = {split_fit:.4f}  ± {ci95[0]:.4f}  (95 % CI from Jacobian)")
    print(f"    R_cs  = {R_cs_fit:.3f}    ± {ci95[1]:.3f} K/W")
    print(f"    R_cs CI as % of central: ±{100 * ci95[1] / max(R_cs_fit, 1e-9):.1f} %")
    print(f"    cal surface RMSE: {rmse:.3f} °C  over {len(res.fun)} residuals")
    if not res.success:
        print(f"    NOTE: optimizer did not fully converge: {res.message}")

    return {
        "split": split_fit,
        "split_ci95": float(ci95[0]),
        "R_cs_K_per_W": R_cs_fit,
        "R_cs_ci95_K_per_W": float(ci95[1]),
        "R_cs_ci95_pct": float(100 * ci95[1] / max(R_cs_fit, 1e-9)),
        "cal_rmse_C": rmse,
        "n_residual": int(len(res.fun)),
        "optimizer_message": res.message,
    }


# ============================================================================
# Step 4: Anchor R_cs from geometry; fit split only
# ============================================================================
def r_cs_from_geometry(k_radial_W_m_K: float) -> float:
    """Radial-conduction thermal resistance from core node to surface.

    For a hollow cylinder (k uniform, length L), R = ln(r_outer / r_inner) /
    (2π · k · L). Plugging in the 18650's r_outer = 9 mm, r_inner = 4.5 mm
    (the effective core-node radius), L = 65 mm.
    """
    return np.log(CELL_RADIUS_M / CORE_NODE_RADIUS_M) / (2 * np.pi * k_radial_W_m_K * CELL_LENGTH_M)


def step4_anchor_R_cs(R_sa: float) -> dict:
    print("\n" + "=" * 70)
    print("Step 4a — anchor R_cs at central physical value from geometry")
    print("=" * 70)
    k_low, k_central, k_high = 0.2, 0.5, 1.0   # W/(m·K), Li-ion radial anisotropy
    R_low = r_cs_from_geometry(k_high)
    R_central = r_cs_from_geometry(k_central)
    R_high = r_cs_from_geometry(k_low)
    print(f"  18650 dims: r = {CELL_RADIUS_M*1000:.1f} mm, "
          f"L = {CELL_LENGTH_M*1000:.0f} mm; effective core radius "
          f"= {CORE_NODE_RADIUS_M*1000:.1f} mm")
    print(f"  k_radial = 0.2 W/(m·K) → R_cs = {R_high:.2f} K/W  (high)")
    print(f"  k_radial = 0.5 W/(m·K) → R_cs = {R_central:.2f} K/W  (central)")
    print(f"  k_radial = 1.0 W/(m·K) → R_cs = {R_low:.2f} K/W  (low)")

    # Lin 2014 qualitative cross-check: their R_c / R_u ≈ 0.6 (26650 LFP).
    R_cs_lin_ratio = 0.6 * R_sa
    print(f"  Lin 2014 R_c/R_u ≈ 0.6 cross-check on our R_sa = {R_sa:.2f}: "
          f"R_cs ≈ {R_cs_lin_ratio:.2f} K/W")
    if not (R_low <= R_cs_lin_ratio <= R_high):
        print(f"  ⚠ Lin's ratio lies OUTSIDE the geometric range — widening band "
              f"to include both estimates.")
        R_low = min(R_low, R_cs_lin_ratio)
        R_high = max(R_high, R_cs_lin_ratio)
        # Re-centre between low and high
        R_central = float(np.sqrt(R_low * R_high))
        print(f"  widened: R_cs central = {R_central:.2f}, "
              f"range = [{R_low:.2f}, {R_high:.2f}] K/W")
    else:
        print(f"  Lin's ratio lies INSIDE the geometric range — no widening needed.")

    return {
        "method": "geometry + Lin cross-check",
        "k_radial_range_W_m_K": [k_low, k_high],
        "R_cs_low_K_per_W": float(R_low),
        "R_cs_central_K_per_W": float(R_central),
        "R_cs_high_K_per_W": float(R_high),
        "R_cs_lin_cross_check_K_per_W": float(R_cs_lin_ratio),
    }


def step4_central_fit(C_total: float, R_sa: float, R_cs: float,
                      cal_caches: list[dict],
                      stride: int = 10,
                      verbose: bool = True) -> dict:
    if verbose:
        print(f"  fitting split at R_cs = {R_cs:.2f} K/W ...")
    p0 = np.array([0.93])
    bounds = ([0.50], [0.99])
    res = least_squares(
        _residuals_split_only, p0,
        args=(C_total, R_sa, R_cs, cal_caches, stride),
        bounds=bounds, method="trf", max_nfev=80, verbose=0,
        x_scale="jac",
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
        diff_step=0.02,
    )
    ci95 = _jacobian_ci(res, n_params=1)
    split_fit = float(res.x[0])
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    return {
        "split": split_fit,
        "split_ci95": float(ci95[0]),
        "C_core_J_K": split_fit * C_total,
        "C_surf_J_K": (1 - split_fit) * C_total,
        "R_cs_K_per_W": R_cs,
        "R_sa_K_per_W": R_sa,
        "cal_rmse_C": rmse,
    }


# ============================================================================
# Step 5: Label band (the Option A payload)
# ============================================================================
def step5_label_band(C_total: float, R_sa: float, R_cs_range: dict,
                     cal_caches: list[dict],
                     representative: dict) -> tuple[dict, dict]:
    print("\n" + "=" * 70)
    print("Step 5 — label band: sweep R_cs across plausible range, refit split")
    print("=" * 70)
    band: dict[str, dict] = {}
    for name, R_cs in [("low",     R_cs_range["R_cs_low_K_per_W"]),
                       ("central", R_cs_range["R_cs_central_K_per_W"]),
                       ("high",    R_cs_range["R_cs_high_K_per_W"])]:
        print(f"\n  --- band point: {name}  R_cs = {R_cs:.2f} K/W ---")
        fit = step4_central_fit(C_total, R_sa, R_cs, cal_caches,
                                stride=10, verbose=True)
        params = ThermalParams(
            C_core=fit["C_core_J_K"], C_surf=fit["C_surf_J_K"],
            R_cs=fit["R_cs_K_per_W"], R_sa=fit["R_sa_K_per_W"],
        )
        # Tight integration on the representative cycle for the plotted band.
        t, T_core, T_surf, Q = simulate_T(representative, params,
                                          t_eval=representative["t_grid"],
                                          max_step=5, rtol=1e-6, atol=1e-8)
        print(f"    split = {fit['split']:.4f} ± {fit['split_ci95']:.4f}, "
              f"cal surface RMSE = {fit['cal_rmse_C']:.3f} °C")
        band[name] = {
            "params":      params,
            "split":       fit["split"],
            "split_ci95":  fit["split_ci95"],
            "cal_rmse_C":  fit["cal_rmse_C"],
            "t":           t,
            "T_core":      T_core,
            "T_surf":      T_surf,
        }

    central = band["central"]
    gap_central = central["T_core"] - central["T_surf"]
    band_spread_pointwise = band["high"]["T_core"] - band["low"]["T_core"]
    stats = {
        "central_core_surface_gap_max_C": float(gap_central.max()),
        "central_core_surface_gap_mean_C": float(gap_central.mean()),
        "T_core_band_spread_max_C": float(band_spread_pointwise.max()),
        "T_core_band_spread_at_peak_C": float(
            band["high"]["T_core"].max() - band["low"]["T_core"].max()),
    }
    print(f"\n  central core–surface gap: max {stats['central_core_surface_gap_max_C']:.2f} °C, "
          f"mean {stats['central_core_surface_gap_mean_C']:.2f} °C")
    print(f"  T_core band spread: max {stats['T_core_band_spread_max_C']:.2f} °C "
          f"(at-peak: {stats['T_core_band_spread_at_peak_C']:.2f} °C)")
    return band, stats


def plot_label_band(band: dict, cache: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    t_min = band["central"]["t"] / 60.0

    ax.fill_between(t_min, band["low"]["T_core"], band["high"]["T_core"],
                    alpha=0.25, color="C3",
                    label=(f"T_core band  (R_cs ∈ "
                           f"[{band['low']['params'].R_cs:.1f}, "
                           f"{band['high']['params'].R_cs:.1f}] K/W)"))
    ax.plot(t_min, band["central"]["T_core"], lw=1.4, color="C3",
            label=f"T_core central (R_cs={band['central']['params'].R_cs:.2f})")
    ax.plot(t_min, band["central"]["T_surf"], lw=1.2, color="C0",
            label="T_surf modeled (central)")
    ax.plot(t_min, cache["T_surf_meas"], lw=0.7, ls="--", color="k",
            label="T_surf measured")
    ax.plot(t_min, cache["T_amb_array"], lw=0.5, ls=":", color="C2",
            label=f"T_inf = {cache['T_inf_C']:.2f} °C (cooling sink, data-driven)")
    ax.axhline(cache["ambient_setpoint_C"], lw=0.4, ls=":", color="gray",
               label=f"nominal setpoint = {cache['ambient_setpoint_C']} °C")

    ax.set_xlabel("time (min)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"Calibrated two-state thermal model — Option A label band\n"
        f"representative cycle: {cache['file_id']}",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ============================================================================
# Step 6: Held-out validation (surface only)
# ============================================================================
def _metrics(err: np.ndarray) -> dict:
    return {
        "rmse_C": float(np.sqrt(np.mean(err ** 2))),
        "mae_C":  float(np.mean(np.abs(err))),
        "max_abs_err_C": float(np.max(np.abs(err))),
    }


def naive_baseline_metrics(cache: dict) -> dict:
    """Two naive baselines, both reported per the integrity check.

    `nominal`: predict T_surface = nominal chamber setpoint (the value a user
        with no cooling-tail analysis would naturally pick from `T_ambient_C`).
    `T_inf`:   predict T_surface = data-driven T_inf for this cycle's
        ambient (the cooling-tail asymptote — already has the correct level).

    The model-vs-T_inf margin is the real measure of what the two-state
    dynamics contribute beyond knowing the rest temperature; see
    [concept-map: level bias vs dynamics].
    """
    err_nominal = cache["T_amb_nominal_array"] - cache["T_surf_meas"]
    err_T_inf = cache["T_inf_C"] - cache["T_surf_meas"]
    return {
        "nominal_setpoint": _metrics(err_nominal),
        "T_inf":            _metrics(err_T_inf),
    }


def evaluate_on_cycle(cache: dict, params: ThermalParams) -> dict:
    t, T_core, T_surf, Q = simulate_T(cache, params, t_eval=cache["t_grid"],
                                      max_step=5, rtol=1e-6, atol=1e-8)
    err = T_surf - cache["T_surf_meas"]
    # Peak |T_surface_measured - T_inf| excursion — the dynamic signal that
    # was available for the model to capture. Small excursion -> the constant
    # T_inf baseline already gets most of the way there and the model has
    # little to add. Large excursion -> a real dynamics test.
    excursion = float(np.max(np.abs(cache["T_surf_meas"] - cache["T_inf_C"])))
    return {
        "t": t, "T_core": T_core, "T_surf": T_surf, "Q": Q, "error": err,
        "rmse_C": float(np.sqrt(np.mean(err ** 2))),
        "mae_C":  float(np.mean(np.abs(err))),
        "max_abs_err_C": float(np.max(np.abs(err))),
        "core_surface_gap_max_C": float((T_core - T_surf).max()),
        "core_surface_gap_mean_C": float((T_core - T_surf).mean()),
        "peak_excursion_T_s_from_T_inf_C": excursion,
    }


def step6_validate(params: ThermalParams, val_caches: list[dict]) -> list[dict]:
    print("\n" + "=" * 70)
    print("Step 6 — held-out validation against measured surface temp")
    print("=" * 70)
    print(f"  central params: C_core={params.C_core:.2f}, C_surf={params.C_surf:.2f}, "
          f"R_cs={params.R_cs:.2f}, R_sa={params.R_sa:.2f}")
    rows = []
    for cache in val_caches:
        ev = evaluate_on_cycle(cache, params)
        bl = naive_baseline_metrics(cache)
        row = {
            "file_id": cache["file_id"],
            "ambient_setpoint_C": cache["ambient_setpoint_C"],
            "T_inf_used_C": cache["T_inf_C"],
            "n_samples": len(cache["t_grid"]),
            "duration_min": float(cache["t_grid"][-1] / 60.0),
            "soc_clamp_low": cache["soc_clamp_low"],
            "soc_clamp_high": cache["soc_clamp_high"],
            "model_rmse_C": ev["rmse_C"],
            "model_mae_C": ev["mae_C"],
            "model_max_abs_err_C": ev["max_abs_err_C"],
            "baseline_nominal_rmse_C": bl["nominal_setpoint"]["rmse_C"],
            "baseline_nominal_mae_C":  bl["nominal_setpoint"]["mae_C"],
            "baseline_nominal_max_abs_err_C": bl["nominal_setpoint"]["max_abs_err_C"],
            "baseline_T_inf_rmse_C": bl["T_inf"]["rmse_C"],
            "baseline_T_inf_mae_C":  bl["T_inf"]["mae_C"],
            "baseline_T_inf_max_abs_err_C": bl["T_inf"]["max_abs_err_C"],
            "core_surface_gap_max_C": ev["core_surface_gap_max_C"],
            "core_surface_gap_mean_C": ev["core_surface_gap_mean_C"],
            "peak_excursion_T_s_from_T_inf_C": ev["peak_excursion_T_s_from_T_inf_C"],
        }
        # Margin: how much the model beats each baseline.
        row["margin_vs_nominal_C"] = row["baseline_nominal_rmse_C"] - row["model_rmse_C"]
        row["margin_vs_T_inf_C"]   = row["baseline_T_inf_rmse_C"]   - row["model_rmse_C"]
        rows.append(row)
        print(f"\n  {cache['file_id']}")
        print(f"    {row['duration_min']:.1f} min, {row['n_samples']} samples; "
              f"ambient {row['ambient_setpoint_C']} °C -> T_inf used = "
              f"{row['T_inf_used_C']:.2f} °C; "
              f"SOC clamps low/high: {row['soc_clamp_low']}/{row['soc_clamp_high']}")
        print(f"    peak excursion |T_surf_meas - T_inf|: "
              f"{row['peak_excursion_T_s_from_T_inf_C']:.2f} °C "
              f"(the dynamic signal that was there to win)")
        print(f"    model              RMSE = {row['model_rmse_C']:.3f} °C,  "
              f"MAE = {row['model_mae_C']:.3f},  max = {row['model_max_abs_err_C']:.3f}")
        print(f"    baseline (nominal) RMSE = {row['baseline_nominal_rmse_C']:.3f} °C,  "
              f"MAE = {row['baseline_nominal_mae_C']:.3f},  "
              f"max = {row['baseline_nominal_max_abs_err_C']:.3f}  "
              f"(model wins by {row['margin_vs_nominal_C']:+.3f})")
        print(f"    baseline (T_inf)   RMSE = {row['baseline_T_inf_rmse_C']:.3f} °C,  "
              f"MAE = {row['baseline_T_inf_mae_C']:.3f},  "
              f"max = {row['baseline_T_inf_max_abs_err_C']:.3f}  "
              f"(model wins by {row['margin_vs_T_inf_C']:+.3f})  <- real dynamics contribution")
        print(f"    core−surface gap: max {row['core_surface_gap_max_C']:.2f} °C, "
              f"mean {row['core_surface_gap_mean_C']:.2f} °C")
    return rows


def plot_residual_diagnosis(cache: dict, params: ThermalParams, out: Path) -> None:
    """Decompose a held-out cycle's residual: surface error vs current vs time.

    Triggered conditionally when the model loses to the T_inf baseline on an
    aggressive, leakage-clean cycle. The question the plot answers: is the
    error concentrated during active load (a dynamics problem) or a slow drift
    during rests (a residual level / T_inf problem)?
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    ev = evaluate_on_cycle(cache, params)
    t_min = ev["t"] / 60.0
    err = ev["error"]  # modeled - measured

    # Identify "active load" vs "rest" masks for severity windows.
    rest_mask = cache["I_array"][: len(t_min)]
    rest_mask = np.abs(rest_mask) < 0.1   # |I| < 0.1 A → near rest
    rmse_rest = float(np.sqrt(np.mean(err[rest_mask] ** 2))) if rest_mask.any() else float("nan")
    rmse_load = float(np.sqrt(np.mean(err[~rest_mask] ** 2))) if (~rest_mask).any() else float("nan")
    frac_rest = float(rest_mask.mean())

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    # Panel 1: surface error.
    axes[0].plot(t_min, err, lw=0.7, color="C3")
    axes[0].axhline(0, lw=0.4, color="k")
    axes[0].set_ylabel("modeled - measured T_surf (°C)")
    axes[0].set_title(
        f"Residual decomposition — {cache['file_id']}\n"
        f"rest |I|<0.1A occupies {frac_rest*100:.1f}% of samples;  "
        f"err RMSE rest = {rmse_rest:.3f}, load = {rmse_load:.3f}",
        fontsize=9,
    )
    axes[0].grid(alpha=0.3)

    # Panel 2: current (drives load/rest classification).
    axes[1].plot(t_min, cache["I_array"], lw=0.5, color="C0")
    axes[1].axhline(0, lw=0.4, color="k")
    axes[1].set_ylabel("Current (A)\n(<0 = discharge)")
    axes[1].grid(alpha=0.3)

    # Panel 3: temperatures (the input to the eye).
    axes[2].plot(t_min, ev["T_surf"], lw=0.9, color="C0", label="modeled T_surf")
    axes[2].plot(t_min, cache["T_surf_meas"], lw=0.6, ls="--", color="k",
                 label="measured T_surf")
    axes[2].axhline(cache["T_inf_C"], lw=0.6, ls="-.", color="C1",
                    label=f"T_inf = {cache['T_inf_C']:.2f} °C")
    axes[2].set_ylabel("T (°C)")
    axes[2].set_xlabel("time (min)")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  residual decomposition plot: {out.relative_to(PROJECT_ROOT)}")
    print(f"    rest |I|<0.1A:  {frac_rest*100:.1f}% of samples,  "
          f"err RMSE rest = {rmse_rest:.3f}, load = {rmse_load:.3f}")
    if rmse_load > rmse_rest * 1.3:
        print(f"    -> error CONCENTRATED during active load -> dynamics-side issue")
    elif rmse_rest > rmse_load * 1.3:
        print(f"    -> error CONCENTRATED during rest -> residual level / T_inf issue")
    else:
        print(f"    -> error roughly balanced between load and rest "
              f"-> mix of level + dynamics")


def plot_validation(val_caches: list[dict], params: ThermalParams,
                    out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(val_caches)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3 * n), squeeze=False)
    axes = axes.flatten()
    for ax, cache in zip(axes, val_caches):
        ev = evaluate_on_cycle(cache, params)
        t_min = ev["t"] / 60.0
        bl = naive_baseline_metrics(cache)
        ax.plot(t_min, ev["T_surf"], lw=1.0, color="C0",
                label="modeled T_surf")
        ax.plot(t_min, cache["T_surf_meas"], lw=0.7, ls="--", color="k",
                label="measured T_surf")
        ax.plot(t_min, cache["T_amb_nominal_array"], lw=0.5, ls=":", color="C2",
                label=(f"nominal setpoint ({cache['ambient_setpoint_C']}°C)  "
                       f"— baseline RMSE {bl['nominal_setpoint']['rmse_C']:.3f}"))
        ax.axhline(cache["T_inf_C"], lw=0.6, ls="-.", color="C1",
                   label=(f"T_inf = {cache['T_inf_C']:.2f} °C (model sink)  "
                          f"— baseline RMSE {bl['T_inf']['rmse_C']:.3f}"))
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"{cache['file_id']}  "
                     f"model RMSE = {ev['rmse_C']:.3f} °C  "
                     f"vs nominal {bl['nominal_setpoint']['rmse_C']:.3f}, "
                     f"vs T_inf {bl['T_inf']['rmse_C']:.3f}",
                     fontsize=9)
        ax.legend(loc="best", fontsize=7)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (min)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CONSTRAINED CALIBRATION — Option A (anchor R_cs + label band)")
    print("=" * 70)

    cal_paths = [expand_glob(p) for p in CAL_CYCLE_GLOBS]
    val_same = [expand_glob(p) for p in VAL_SAME_AMBIENT_GLOBS]
    val_diff = [expand_glob(p) for p in VAL_DIFF_AMBIENT_GLOBS]
    val_aggressive = [expand_glob(p) for p in VAL_AGGRESSIVE_GLOBS]
    val_paths = val_same + val_diff + val_aggressive

    print("\nCAL cycles (well-excited, 25 °C):")
    for p in cal_paths:
        print(f"  {p.name}")
    print("\nVAL cycles — same ambient as CAL (held out):")
    for p in val_same:
        print(f"  {p.name}")
    print("\nVAL cycles — different ambient (stronger generalization):")
    for p in val_diff:
        print(f"  {p.name}")
    print("\nVAL cycles — AGGRESSIVE cold (severe-thermal generalization):")
    for p in val_aggressive:
        print(f"  {p.name}")

    # Step 1
    C_total = step1_pin_C_total()

    # Step 2 — free-asymptote cooling-tail fits (must run BEFORE building
    # caches so we can use T_inf as the ODE cooling sink).
    # LEAKAGE GUARD: validation files are excluded from the tail-search pool
    # up front so that T_inf and R_sa are derived only from non-held-out data.
    R_sa_result = step2_anchor_R_sa_and_T_inf(
        C_total,
        exclude_paths=set(val_paths),
    )
    R_sa = R_sa_result["R_sa_central_K_per_W"]
    T_inf_by_amb = R_sa_result["T_inf_by_ambient_C"]

    # Look up T_inf per cycle from its ambient setpoint, then pre-load caches
    # with the cooling sink replaced by T_inf (NOT the nominal setpoint).
    def _T_inf_for(path: Path) -> float:
        # parquet name encodes ambient like "LG_25degC_..." or "LG_10degC_...".
        df0 = pd.read_parquet(path, columns=["ambient_setpoint_C"])
        amb = int(round(float(df0["ambient_setpoint_C"].iloc[0])))
        return float(T_inf_by_amb.get(amb, amb))

    print("\nloading cycle caches with data-driven T_inf as cooling sink...")
    cal_T_infs = [_T_inf_for(p) for p in cal_paths]
    val_T_infs = [_T_inf_for(p) for p in val_paths]
    for p, t_inf in zip(cal_paths + val_paths, cal_T_infs + val_T_infs):
        amb = int(p.name.split("degC")[0].split("_")[-1].replace("n", "-"))
        print(f"  {p.name[:60]:60s} ambient={amb:>3d}°C → T_inf={t_inf:6.2f}°C")
    cal_caches = [precompute_cycle(p, T_inf_override=t)
                  for p, t in zip(cal_paths, cal_T_infs)]
    val_caches = [precompute_cycle(p, T_inf_override=t)
                  for p, t in zip(val_paths, val_T_infs)]

    # Step 3 — diagnostic fit (this is itself a headline result)
    diag = step3_diagnostic_fit(C_total, R_sa, cal_caches)

    # Step 4a — anchor R_cs from geometry (+ Lin cross-check)
    R_cs_range = step4_anchor_R_cs(R_sa)

    # Step 4b — central fit
    print("\n" + "=" * 70)
    print("Step 4b — central fit (split only, R_cs anchored at central)")
    print("=" * 70)
    central = step4_central_fit(C_total, R_sa, R_cs_range["R_cs_central_K_per_W"],
                                cal_caches, stride=10, verbose=False)
    central_params = ThermalParams(
        C_core=central["C_core_J_K"], C_surf=central["C_surf_J_K"],
        R_cs=central["R_cs_K_per_W"], R_sa=central["R_sa_K_per_W"],
    )
    print(f"  central: split = {central['split']:.4f} ± {central['split_ci95']:.4f}")
    print(f"           C_core = {central['C_core_J_K']:.2f} J/K, "
          f"C_surf = {central['C_surf_J_K']:.2f} J/K")
    print(f"           R_cs = {central['R_cs_K_per_W']:.2f}, "
          f"R_sa = {central['R_sa_K_per_W']:.2f} K/W")
    print(f"           cal surface RMSE = {central['cal_rmse_C']:.3f} °C")

    # Step 5 — label band (representative cycle = first cal cycle, US06)
    band, band_stats = step5_label_band(
        C_total, R_sa, R_cs_range, cal_caches, representative=cal_caches[0]
    )
    plot_label_band(band, cal_caches[0], LABEL_BAND_PLOT)
    print(f"\n  label band plot: {LABEL_BAND_PLOT.relative_to(PROJECT_ROOT)}")

    # Step 6 — held-out validation
    val_rows = step6_validate(central_params, val_caches)
    plot_validation(val_caches, central_params, SURFACE_VAL_PLOT)
    print(f"\n  surface validation plot: "
          f"{SURFACE_VAL_PLOT.relative_to(PROJECT_ROOT)}")

    # Conditional residual decomposition for aggressive cycles where the
    # model loses to the T_inf baseline. The aggressive cycles are the
    # leakage-clean test of dynamics — if the model loses there, we want a
    # plot showing whether the error is load-driven or rest-driven.
    aggressive_names = {p.name for p in val_aggressive}
    for cache, row in zip(val_caches, val_rows):
        if cache["file_id"] + ".parquet" not in aggressive_names:
            continue
        if row["margin_vs_T_inf_C"] >= 0:
            print(f"\n  aggressive cycle {cache['file_id']} BEATS T_inf "
                  f"baseline by {row['margin_vs_T_inf_C']:+.3f} °C — "
                  f"no diagnosis plot needed")
            continue
        print(f"\n  aggressive cycle {cache['file_id']} loses to T_inf "
              f"baseline by {-row['margin_vs_T_inf_C']:.3f} °C — generating "
              f"residual decomposition")
        out = PLOTS_DIR / f"residual_diag_{cache['file_id']}.png"
        plot_residual_diagnosis(cache, central_params, out)

    summary = {
        "physical_constants": {
            "cell_mass_kg": CELL_MASS_KG, "c_p_J_kg_K": CP_J_KG_K,
            "C_total_J_K": C_total,
            "cell_radius_m": CELL_RADIUS_M, "cell_length_m": CELL_LENGTH_M,
            "core_node_radius_m": CORE_NODE_RADIUS_M,
        },
        "cal_cycles": [p.name for p in cal_paths],
        "val_cycles_same_ambient": [p.name for p in val_same],
        "val_cycles_diff_ambient": [p.name for p in val_diff],
        "step2_R_sa_result": R_sa_result,
        "step3_diagnostic_fit": diag,
        "step4_R_cs_geometry_anchor": R_cs_range,
        "step4_central_fit": central,
        "step5_band_stats": band_stats,
        "step5_band_per_point": {
            name: {"R_cs_K_per_W": d["params"].R_cs,
                   "split": d["split"], "split_ci95": d["split_ci95"],
                   "cal_rmse_C": d["cal_rmse_C"]}
            for name, d in band.items()
        },
        "step6_validation_per_cycle": val_rows,
    }
    CAL_RESULTS_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsummary JSON: {CAL_RESULTS_JSON.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

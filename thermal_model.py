"""
thermal_model.py — Two-state lumped thermal ODE + unfitted-default sanity run.

This module scaffolds the thermal model and runs a forward simulation with
ORDER-OF-MAGNITUDE default parameters. The point of this step is to verify that

    1. the ODE integrates cleanly,
    2. the sign of heat generation is right (`Q ≥ 0` in both charge and discharge),
    3. modeled `T_core ≥ T_surf ≥ T_amb` holds when `Q ≥ 0`,

so we can isolate "the model is wired correctly" from "the parameters are right."
**Calibration is NOT in this file** — that's the next step.

ODE form (Lin two-state, in our sign convention):

    C_core · dT_core/dt = Q − (T_core − T_surf) / R_cs
    C_surf · dT_surf/dt = (T_core − T_surf) / R_cs − (T_surf − T_amb) / R_sa

    state = [T_core, T_surf]

Heat generation:

    Q(t) = I(t) · ( V(t) − V_ocv(SOC(t)) )

using RAW current sign (negative discharge) and the (V − V_ocv) form. See
`knowledge/decisions/heat-generation-irreversible-only.md` and
`knowledge/decisions/sign-convention.md`.

----------------------------------------------------------------------------
SOC tracking conventions — read these before changing anything that touches SOC
----------------------------------------------------------------------------
These three choices barely move Q in the flat mid-SOC plateau but matter near
the OCV-table endpoints. The point is an explicit documented convention, not
perfection.

(a) SOC = 1.0 reference state.
    SOC = 1.0 corresponds to the state at the START of the LG C/20
    characterization file used to build the OCV table — voltage there was
    4.176 V (NOT the cell's true full-charge 4.20 V after CCCV). The cell at
    true full charge actually maps to roughly SOC ≈ 1.01–1.02 in the table's
    own units, but the table only goes up to 1.0. Drive-cycle starts (assumed
    fully CCCV-charged per LG protocol) get treated as SOC = 1.0, accepting
    a ~24 mV underestimation of `V_ocv` near the very top that decays as the
    cell discharges. See knowledge/report-notes.md.

(b) Capacity used to normalize SOC.
    AH_CAPACITY = 3.0 Ah, the datasheet nominal for LG HG2. We pull it
    directly from `build_ocv.AH_NOMINAL` so this module and the OCV table
    are GUARANTEED to use the same value (the user's "use the same value the
    OCV table assumed" rule). The measured 25 °C capacity from the C/20 file
    was 2.778 Ah (~92.6 % of nominal). Switching both modules to the measured
    value is a future refinement and is flagged in knowledge/report-notes.md.

(c) Initial SOC per drive cycle.
    Default soc_init = 1.0, justified by the LG protocol: every drive cycle
    is preceded by a CCCV charge to 4.2 V. For non-LG or contiguous
    multi-cycle files this assumption is wrong; override via the `soc_init`
    kwarg of `simulate_cycle`.

----------------------------------------------------------------------------
SOC clamping rule
----------------------------------------------------------------------------
The OCV table covers SOC ∈ [0.074, 1.0] (the C/20 file ran from full charge
down to 2.8 V cutoff). We clamp the coulomb-counted SOC trajectory to that
range BEFORE looking up V_ocv.

Why clamp SOC, not voltage:
- A hard discharge pulse drops the terminal voltage `V` below 2.8 V via IR
  drop while the true cell SOC may still be well above 7.4 %. Clamping on
  terminal voltage would treat that as "empty cell" and corrupt the V_ocv
  lookup. Clamping on coulomb-counted SOC is consistent with what the OCV
  table actually represents.

Why we MUST clamp at all:
- `build_ocv.load_ocv` returns an `interp1d` with `bounds_error=True`. SOC
  outside [0.074, 1.0] raises, so a silent clamp is safer than a crash.

Silent clamps could hide a problem (cell really did discharge below the table
range — common at cold ambients), so we COUNT and REPORT clamp hits per cycle.

----------------------------------------------------------------------------
Default thermal parameters (UNFITTED)
----------------------------------------------------------------------------
Order-of-magnitude only, just plausible. NOT calibrated.

    C_total ≈ m·c_p ≈ 0.048 kg × 900 J/(kg·K) ≈ 43 J/K
    Split per Lin 2014's ~14:1 core:surface ratio (jellyroll vs casing):
      C_core = 40 J/K
      C_surf = 3 J/K
    R_cs    = 1.5 K/W   (mid-range of Lin 2014's identified R_c = 1.94 K/W
                         for A123 26650, scaled down a touch for our 18650)
    R_sa    = 3.0 K/W   (still-air passive convection; cooling tails will
                         give the actual number once we calibrate)

Lin's absolute values are NOT directly portable to our cell — they're A123
26650 LFP, ours is LG HG2 18650 NMC; only the QUALITATIVE ratios carry.
See knowledge/sources/lin-2014-electro-thermal.md.

Run:  python thermal_model.py [--cycle PATH] [--soc-init FLOAT]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from build_ocv import AH_NOMINAL, load_ocv


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "data" / "eda_plots"

# Default sanity-run cycle: LG UDDS at 25 °C, which Week 1 EDA already used.
DEFAULT_CYCLE = "LG_25degC_UDDS_10-28-18_17.44_551_UDDS_25degC_LGHG2.parquet"


# ---------------------------------------------------------------------------
# OCV-table bounds (must match build_ocv's saved table)
# ---------------------------------------------------------------------------
SOC_TABLE_MIN = 0.074   # 7.4 % — C/20 file cutoff was V = 2.8 V
SOC_TABLE_MAX = 1.000   # 100 % — C/20 file start (V = 4.176 V; see convention (a))


# ---------------------------------------------------------------------------
# Default unfitted parameters
# ---------------------------------------------------------------------------
@dataclass
class ThermalParams:
    """Lumped thermal-model parameters (units in field comments).

    Defaults are plausibility-only — physical order-of-magnitude estimates for
    an LG HG2 18650 (~48 g) in still air. NONE of these have been fitted to
    our data yet. See knowledge/decisions/calibration-constrained-fit.md for
    the calibration plan that will replace them in the next step.
    """
    # Heat capacities. C_total = m·c_p ≈ 0.048 kg × 900 J/(kg·K) ≈ 43 J/K.
    C_core: float = 40.0    # J/K — jellyroll thermal mass (most of total)
    C_surf: float = 3.0     # J/K — thin steel can (small fraction)
    # Thermal resistances. Order-of-magnitude only; calibration will refine.
    R_cs:   float = 1.5     # K/W — internal core↔surface conduction
    R_sa:   float = 3.0     # K/W — passive surface↔ambient convection


# ---------------------------------------------------------------------------
# SOC helpers
# ---------------------------------------------------------------------------
def cycle_soc(df: pd.DataFrame, soc_init: float = 1.0,
              ah_capacity: float = AH_NOMINAL) -> np.ndarray:
    """Coulomb-counted SOC time series for one cycle DataFrame.

    SOC(t) = soc_init + ( Ah(t) − Ah(0) ) / ah_capacity

    Sign convention: `Ah` is signed and accumulates negative during discharge,
    so SOC decreases under discharge and increases under charge. No sign flip.

    Parameters
    ----------
    df          : processed-cycle DataFrame; must include the `Ah` column.
    soc_init    : SOC at row 0 of this cycle. Default 1.0 — see convention (c).
    ah_capacity : capacity used to normalize. Default `build_ocv.AH_NOMINAL` —
                  guarantees consistency with the OCV table.
    """
    ah = df["Ah"].to_numpy(dtype=float)
    return soc_init + (ah - ah[0]) / ah_capacity


def clamp_soc(soc: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Clamp SOC to the OCV-table range; return (clamped, n_below, n_above)."""
    n_below = int((soc < SOC_TABLE_MIN).sum())
    n_above = int((soc > SOC_TABLE_MAX).sum())
    return np.clip(soc, SOC_TABLE_MIN, SOC_TABLE_MAX), n_below, n_above


# ---------------------------------------------------------------------------
# ODE RHS
# ---------------------------------------------------------------------------
def thermal_rhs(t: float, state: np.ndarray,
                Q_of_t, T_amb_of_t, params: ThermalParams) -> list[float]:
    """RHS of the two-state lumped thermal ODE.

    state = [T_core, T_surf], temperatures in °C.

    C_core · dT_core/dt = Q − (T_core − T_surf) / R_cs
    C_surf · dT_surf/dt = (T_core − T_surf) / R_cs − (T_surf − T_amb) / R_sa

    Q_of_t and T_amb_of_t are scalar-in, scalar-out callables (typically
    `scipy.interpolate.interp1d` instances closed over the cycle's 1 Hz grid).
    `solve_ivp` will call this at adaptively chosen times within the cycle.
    """
    T_core, T_surf = state
    Q = float(Q_of_t(t))
    T_amb = float(T_amb_of_t(t))

    dTc = (Q - (T_core - T_surf) / params.R_cs) / params.C_core
    dTs = ((T_core - T_surf) / params.R_cs
           - (T_surf - T_amb) / params.R_sa) / params.C_surf
    return [dTc, dTs]


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    file_id: str
    params: ThermalParams
    t: np.ndarray             # time grid used for sim output, seconds
    T_core: np.ndarray        # modeled core temp, °C
    T_surf: np.ndarray        # modeled surface temp, °C
    T_surf_measured: np.ndarray
    T_amb: np.ndarray         # ambient, interpolated onto t for plotting
    Q: np.ndarray             # heat-gen evaluated on t
    soc: np.ndarray           # post-clamp SOC trajectory on t
    soc_clamp_low: int        # count of pre-clamp samples below SOC_TABLE_MIN
    soc_clamp_high: int       # count above SOC_TABLE_MAX
    soc_init: float
    df: pd.DataFrame          # source DataFrame (for plotting context)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def simulate_cycle(parquet_path: Path,
                   params: ThermalParams | None = None,
                   soc_init: float = 1.0,
                   max_step: float = 5.0) -> SimResult:
    """Load a processed-cycle Parquet and forward-simulate the thermal ODE.

    Parameters
    ----------
    parquet_path : path to a file in `data/processed/`.
    params       : `ThermalParams`. Defaults to the unfitted defaults.
    soc_init     : initial SOC at row 0 of the cycle. See convention (c).
    max_step     : `solve_ivp` `max_step` argument, seconds. Default 5 s, well
                   below the thermal time constants we expect (tens of s+).

    Returns
    -------
    `SimResult` with modeled + measured signals on the cycle's 1 Hz grid.
    """
    params = params if params is not None else ThermalParams()
    df = pd.read_parquet(parquet_path)
    file_id = parquet_path.stem

    # 1 Hz grid produced by data_loading.py.
    t_grid = df["time_s"].to_numpy(dtype=float)
    I = df["current_A"].to_numpy(dtype=float)
    V = df["voltage_V"].to_numpy(dtype=float)
    T_amb = df["T_ambient_C"].to_numpy(dtype=float)
    T_surf_meas = df["T_surface_C"].to_numpy(dtype=float)

    # Coulomb-count SOC over the cycle, then clamp to OCV-table range.
    soc_raw = cycle_soc(df, soc_init=soc_init)
    soc_clamped, n_low, n_high = clamp_soc(soc_raw)

    # Input interpolators on the 1 Hz grid. Linear; extrapolation guarded by
    # the simulation time span never extending beyond the grid (by construction).
    interp_kwargs = dict(kind="linear", bounds_error=False,
                         fill_value="extrapolate", assume_sorted=True)
    I_of_t = interp1d(t_grid, I, **interp_kwargs)
    V_of_t = interp1d(t_grid, V, **interp_kwargs)
    T_amb_of_t = interp1d(t_grid, T_amb, **interp_kwargs)
    soc_of_t = interp1d(t_grid, soc_clamped, **interp_kwargs)

    # OCV table — bounds_error=True; that's why we clamped first.
    v_ocv = load_ocv()

    def Q_of_t(t: float) -> float:
        """Heat-generation rate at time t.

        Q = I·(V − V_ocv(SOC)), with RAW current sign — see
        knowledge/decisions/sign-convention.md.
        """
        soc_t = float(soc_of_t(t))
        return float(I_of_t(t)) * (float(V_of_t(t)) - float(v_ocv(soc_t)))

    # Initial condition: cell in thermal equilibrium with ambient at t=0.
    # The LG cells sat in the chamber long enough for this to be approximately
    # right for a freshly-started drive cycle. If a previous test left the cell
    # warm, the initial T_amb assumption may be off by a degree or two — that
    # shows up as a transient at the start, not a long-term error.
    T_amb_0 = float(T_amb_of_t(t_grid[0]))
    y0 = np.array([T_amb_0, T_amb_0])

    sol = solve_ivp(
        thermal_rhs,
        t_span=(t_grid[0], t_grid[-1]),
        y0=y0,
        t_eval=t_grid,
        args=(Q_of_t, T_amb_of_t, params),
        method="RK45",
        max_step=max_step,
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    # Post-evaluate Q at the same grid for plotting + sanity checks.
    I_eval = I_of_t(sol.t)
    V_eval = V_of_t(sol.t)
    soc_eval = soc_of_t(sol.t)
    Q_eval = I_eval * (V_eval - v_ocv(soc_eval))

    return SimResult(
        file_id=file_id,
        params=params,
        t=sol.t,
        T_core=sol.y[0],
        T_surf=sol.y[1],
        T_surf_measured=T_surf_meas,
        T_amb=T_amb_of_t(sol.t),
        Q=Q_eval,
        soc=soc_of_t(sol.t),
        soc_clamp_low=n_low,
        soc_clamp_high=n_high,
        soc_init=soc_init,
        df=df,
    )


# ---------------------------------------------------------------------------
# Sanity assertions
# ---------------------------------------------------------------------------
def run_sanity_checks(r: SimResult) -> dict:
    """Run the assertions that catch sign bugs and dead simulations.

    The sign assertions are the WHOLE POINT of this scaffold step: if `Q` ever
    goes negative, or `T_core` ever sits below `T_surf`, or `T_surf` ever sits
    below `T_amb`, we have a wiring bug somewhere upstream (almost always in
    the sign of `(V − V_ocv)` or the sign convention of `I`).

    Returns a dict of counts + flag strings; does NOT raise. The caller prints
    the flags loudly so we can't miss them.
    """
    Q = r.Q
    T_c = r.T_core
    T_s = r.T_surf
    T_a = r.T_amb

    # Small numerical tolerances — solve_ivp + interpolation can wiggle a tiny bit.
    eps_Q = 0.05     # W
    eps_T = 0.01     # °C

    n_neg_Q = int((Q < -eps_Q).sum())
    n_core_below_surf = int((T_c < T_s - eps_T).sum())
    n_surf_below_amb = int((T_s < T_a - eps_T).sum())
    n_pos_Q = int((Q > 0.1).sum())
    delta_T_surf = float(T_s.max() - T_s.min())

    flags: list[str] = []
    if n_neg_Q > 0:
        flags.append(
            f"FAIL  Q < 0 in {n_neg_Q} of {len(Q)} samples — LIKELY SIGN BUG "
            "in heat-gen formula (check current sign and V − V_ocv ordering)"
        )
    if n_core_below_surf > 0:
        flags.append(
            f"FAIL  T_core < T_surf in {n_core_below_surf} of {len(T_c)} samples "
            "— heat would have to flow 'uphill'; check ODE signs"
        )
    if n_surf_below_amb > 0:
        flags.append(
            f"FAIL  T_surf < T_amb in {n_surf_below_amb} of {len(T_s)} samples "
            "— cell colder than chamber with no refrigeration; check R_sa sign"
        )
    if n_pos_Q < 10:
        flags.append(
            f"WARN  only {n_pos_Q} samples with Q > 0.1 W — cycle may be too "
            "gentle (or Q computation may be returning ~0) for a real shape check"
        )
    if delta_T_surf < 0.05:
        flags.append(
            f"WARN  modeled T_surf swing only {delta_T_surf:.3f} °C — model "
            "barely moved; parameters may be too inert for this cycle"
        )

    return {
        "n_samples": len(Q),
        "n_neg_Q": n_neg_Q,
        "n_pos_Q": n_pos_Q,
        "n_core_below_surf": n_core_below_surf,
        "n_surf_below_amb": n_surf_below_amb,
        "modeled_T_surf_swing_C": delta_T_surf,
        "soc_clamp_low": r.soc_clamp_low,
        "soc_clamp_high": r.soc_clamp_high,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_sanity(r: SimResult, out_dir: Path = PLOTS_DIR) -> Path:
    """Three-panel sanity plot: temperatures, current, heat generation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    t_min = r.t / 60.0   # minutes is easier to read for multi-hour cycles

    # Panel 1: temperatures.
    axes[0].plot(t_min, r.T_core, label="modeled T_core", lw=1.2, color="C3")
    axes[0].plot(t_min, r.T_surf, label="modeled T_surf", lw=1.2, color="C0")
    axes[0].plot(t_min, r.T_surf_measured, label="measured T_surf",
                 lw=0.8, ls="--", color="k")
    axes[0].plot(t_min, r.T_amb, label="T_amb", lw=0.6, ls=":", color="C2")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title(
        f"{r.file_id}\nUNFITTED thermal model (sanity run): "
        f"C_core={r.params.C_core}, C_surf={r.params.C_surf}, "
        f"R_cs={r.params.R_cs}, R_sa={r.params.R_sa}",
        fontsize=9,
    )
    axes[0].grid(alpha=0.3)

    # Panel 2: current.
    axes[1].plot(t_min, r.df["current_A"], lw=0.7, color="C0")
    axes[1].axhline(0, color="k", lw=0.4)
    axes[1].set_ylabel("Current (A)\n(<0 = discharge)")
    axes[1].grid(alpha=0.3)

    # Panel 3: heat generation.
    axes[2].plot(t_min, r.Q, lw=0.7, color="C3")
    axes[2].axhline(0, color="k", lw=0.4)
    axes[2].set_ylabel("Q (W)")
    axes[2].set_xlabel("time (min)")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"thermal_sanity_{r.file_id}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Two-state thermal ODE sanity scaffold.",
    )
    ap.add_argument("--cycle", type=Path,
                    default=PROCESSED_DIR / DEFAULT_CYCLE,
                    help="Parquet record to simulate.")
    ap.add_argument("--soc-init", type=float, default=1.0,
                    help="initial SOC for the cycle. Default 1.0 (LG protocol).")
    args = ap.parse_args()

    if not args.cycle.exists():
        print(f"FATAL: {args.cycle} not found")
        return 2

    print(f"loading  {args.cycle.name}")
    r = simulate_cycle(args.cycle, soc_init=args.soc_init)

    duration_min = r.t[-1] / 60.0
    print(f"  cycle length:       {len(r.t)} samples  ({duration_min:.1f} min)")
    print(f"  Q range:            [{r.Q.min():+.2f}, {r.Q.max():+.2f}] W "
          f"  median {np.median(r.Q):+.3f} W")
    print(f"  SOC range (final):  [{r.soc.min()*100:.1f} %, {r.soc.max()*100:.1f} %]")
    print(f"  modeled T_core max:  {r.T_core.max():.2f} °C")
    print(f"  modeled T_surf max:  {r.T_surf.max():.2f} °C")
    print(f"  measured T_surf max: {r.T_surf_measured.max():.2f} °C")
    print(f"  T_amb max:           {r.T_amb.max():.2f} °C")

    print("\nsanity checks:")
    sc = run_sanity_checks(r)
    print(f"  Q < 0 (sign bug?):  {sc['n_neg_Q']}")
    print(f"  Q > 0.1 W:          {sc['n_pos_Q']}")
    print(f"  T_core < T_surf:    {sc['n_core_below_surf']}")
    print(f"  T_surf < T_amb:     {sc['n_surf_below_amb']}")
    print(f"  modeled T_surf swing: {sc['modeled_T_surf_swing_C']:.2f} °C")
    print(f"  SOC clamp hits (low/high): {sc['soc_clamp_low']} / {sc['soc_clamp_high']}")

    if sc["flags"]:
        print("\nFLAGS:")
        for f in sc["flags"]:
            print(f"  {f}")
    else:
        print("  all assertions passed — model is wired correctly")

    plot = plot_sanity(r)
    print(f"\nplot:  {plot.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

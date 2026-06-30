"""
build_ocv.py — Build V_ocv(SOC) from the LG C/20 characterization file.

The LG dataset includes a near-equilibrium C/20 discharge-and-charge cycle at
25 °C. We segment the *discharge* half, integrate `Ah` into SOC, resample onto
a regular SOC grid, and save the resulting V_ocv(SOC) lookup.

Outputs:
  data/processed/ocv_lg_25degC.parquet     # columns: SOC, V_ocv
  data/processed/ocv_lg_25degC.meta.json   # provenance + sanity-check summary
  data/eda_plots/ocv_lg_25degC.png         # diagnostic plot

Downstream code can import `load_ocv()` to get a callable `V_ocv(soc)`.
See knowledge/decisions/ocv-characterization.md for the design call.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
SOURCE_MAT = (
    PROJECT_ROOT / "data" / "LG"
    / "LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020" / "25degC"
    / "10-26-18_09.29 549_C20DisCh_25degC_LGHG2.mat"
)
OUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "ocv_lg_25degC.parquet"
OUT_META    = PROJECT_ROOT / "data" / "processed" / "ocv_lg_25degC.meta.json"
OUT_PLOT    = PROJECT_ROOT / "data" / "eda_plots"   / "ocv_lg_25degC.png"

AH_NOMINAL = 3.0           # LG HG2 rated capacity (Ah)
DSOC = 0.001               # SOC grid step (0.1 %)
I_DISCHARGE_THRESHOLD = -0.05   # A — start of discharge half
V_EMPTY = 2.8              # V — minimum voltage that marks end of discharge

# Expected OCV endpoints for an Li-ion 18650 (Kollmeyer LG HG2). Used for
# sanity warnings, NOT enforced — the actual curve drives downstream code.
V_OCV_FULL_NOMINAL  = 4.20
V_OCV_EMPTY_NOMINAL = 2.80


# ---------------------------------------------------------------------------
# Load & segment
# ---------------------------------------------------------------------------
def load_c20(path: Path = SOURCE_MAT) -> pd.DataFrame:
    """Read the C/20 characterization .mat into a tidy DataFrame.

    Mirrors data_loading.py's LG schema but reads directly from raw — this file
    is intentionally NOT in data/processed/ because the main loader excludes
    characterization tests.
    """
    md = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    m = md["meas"]
    return pd.DataFrame({
        "time_s":      np.asarray(m.Time, dtype=float),
        "voltage_V":   np.asarray(m.Voltage, dtype=float),
        "current_A":   np.asarray(m.Current, dtype=float),
        "Ah":          np.asarray(m.Ah, dtype=float),
        "T_surface_C": np.asarray(m.Battery_Temp_degC, dtype=float),
    })


def segment_discharge(df: pd.DataFrame,
                      i_thresh: float = I_DISCHARGE_THRESHOLD,
                      v_empty: float = V_EMPTY) -> pd.DataFrame:
    """Return the rows belonging to the C/20 discharge half-cycle.

    Start: first sample with current below the discharge threshold (negative,
    well above noise).
    End:   first sample after start where voltage falls to v_empty OR the
    current changes sign (charge half begins).
    """
    i = df["current_A"].to_numpy()
    v = df["voltage_V"].to_numpy()

    start_candidates = np.where(i < i_thresh)[0]
    if len(start_candidates) == 0:
        raise ValueError("no discharge current found in file")
    start = int(start_candidates[0])

    # End: voltage hits empty OR current goes positive again.
    after = slice(start + 1, None)
    v_hit_empty = np.where(v[after] <= v_empty)[0]
    i_pos = np.where(i[after] >= 0)[0]
    end_idx = []
    if len(v_hit_empty):
        end_idx.append(start + 1 + int(v_hit_empty[0]))
    if len(i_pos):
        end_idx.append(start + 1 + int(i_pos[0]))
    if not end_idx:
        # No clean end — take to end of file.
        end = len(df)
    else:
        end = min(end_idx)

    return df.iloc[start:end + 1].reset_index(drop=True)


# ---------------------------------------------------------------------------
# SOC + OCV
# ---------------------------------------------------------------------------
def compute_soc(df: pd.DataFrame, ah_nominal: float = AH_NOMINAL) -> pd.DataFrame:
    """Add an SOC column via coulomb counting against `ah_nominal`.

    Convention: SOC = 1 at the start of the discharge segment; decreases
    monotonically toward 0 as charge is removed. `Ah` is negative for
    discharge, so we add it (subtracting a negative) -> SOC decreases.
    """
    df = df.copy()
    delta_ah = df["Ah"] - df["Ah"].iloc[0]   # negative as discharge proceeds
    df["SOC"] = 1.0 + delta_ah / ah_nominal
    return df


def resample_to_soc_grid(df: pd.DataFrame,
                         dsoc: float = DSOC) -> pd.DataFrame:
    """Linear-interp V onto a regular SOC grid.

    SOC during discharge is monotone-decreasing, so we sort ascending in SOC
    before interpolating. The grid covers the observed SOC range — we do NOT
    extrapolate above/below it. Downstream code that needs OCV outside this
    range must handle it explicitly (clamp or warn).
    """
    soc_raw = df["SOC"].to_numpy()
    v_raw   = df["voltage_V"].to_numpy()
    order   = np.argsort(soc_raw)
    soc_sorted = soc_raw[order]
    v_sorted   = v_raw[order]

    # Drop near-duplicate SOC values that would break interp.
    keep = np.concatenate(([True], np.diff(soc_sorted) > 1e-7))
    soc_sorted = soc_sorted[keep]
    v_sorted   = v_sorted[keep]

    soc_grid = np.arange(np.ceil(soc_sorted[0] / dsoc) * dsoc,
                         np.floor(soc_sorted[-1] / dsoc) * dsoc + 1e-9, dsoc)
    v_grid = np.interp(soc_grid, soc_sorted, v_sorted)
    return pd.DataFrame({"SOC": soc_grid, "V_ocv": v_grid})


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_check(ocv: pd.DataFrame, segment: pd.DataFrame,
                 ah_nominal: float = AH_NOMINAL) -> dict:
    """Return a dict of human-readable flags and key numbers."""
    soc = ocv["SOC"].to_numpy()
    v   = ocv["V_ocv"].to_numpy()

    flags: list[str] = []
    # V_ocv should be monotonically increasing with SOC. A meaningful downward
    # jump (V dropping as SOC rises) would signal noise or a bad segment.
    dv = np.diff(v)
    monotone_violations = int((dv < -1e-3).sum())
    if monotone_violations:
        flags.append(
            f"V_ocv non-monotone in {monotone_violations} step(s) "
            f"(>1 mV downward jumps as SOC increases)"
        )

    v_low, v_high = v[0], v[-1]
    if abs(v_high - V_OCV_FULL_NOMINAL) > 0.1:
        flags.append(
            f"V_ocv at SOC={soc[-1]:.3f} = {v_high:.3f} V — "
            f"differs from nominal full ({V_OCV_FULL_NOMINAL} V) by >100 mV"
        )
    if abs(v_low - V_OCV_EMPTY_NOMINAL) > 0.1:
        flags.append(
            f"V_ocv at SOC={soc[0]:.3f} = {v_low:.3f} V — "
            f"differs from nominal empty ({V_OCV_EMPTY_NOMINAL} V) by >100 mV"
        )

    soc_low_gap = soc[0]
    soc_high_gap = 1.0 - soc[-1]
    if soc_high_gap > 0.005:
        flags.append(
            f"top SOC sliver missing: curve covers up to {soc[-1]*100:.1f}% — "
            f"V_ocv near 100% SOC will need extrapolation or a clip"
        )
    if soc_low_gap > 0.005:
        flags.append(
            f"bottom SOC sliver missing: curve starts at {soc[0]*100:.1f}%"
        )

    ah_used = abs(segment["Ah"].iloc[-1] - segment["Ah"].iloc[0])
    if ah_used < 0.9 * ah_nominal:
        flags.append(
            f"only {ah_used:.2f} Ah extracted vs {ah_nominal} Ah nominal — "
            f"possible incomplete discharge"
        )

    return {
        "flags": flags,
        "n_rows_segment": len(segment),
        "n_rows_grid": len(ocv),
        "soc_range_grid": [float(soc[0]), float(soc[-1])],
        "v_ocv_at_low_soc":  float(v_low),
        "v_ocv_at_high_soc": float(v_high),
        "ah_extracted": float(ah_used),
        "duration_h": float(segment["time_s"].iloc[-1] - segment["time_s"].iloc[0]) / 3600.0,
        "T_surface_C_range": [float(segment["T_surface_C"].min()),
                              float(segment["T_surface_C"].max())],
        "ah_nominal_used": ah_nominal,
        "source_file": str(SOURCE_MAT.name),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_ocv(ocv: pd.DataFrame, segment: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7))

    ax = axes[0]
    ax.plot(ocv["SOC"], ocv["V_ocv"], lw=1.2)
    ax.set_xlabel("SOC")
    ax.set_ylabel("V_ocv (V)")
    ax.set_title("LG HG2 — V_ocv(SOC) from C/20 discharge at 25 °C")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

    # Time-domain trace of the segment for provenance.
    ax = axes[1]
    t = segment["time_s"] / 3600.0
    ax.plot(t, segment["voltage_V"], lw=0.7, label="V")
    ax2 = ax.twinx()
    ax2.plot(t, segment["current_A"], lw=0.7, color="C1", label="I")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("V (V)")
    ax2.set_ylabel("I (A)", color="C1")
    ax.set_title("Source discharge segment (raw)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API for downstream consumers
# ---------------------------------------------------------------------------
def load_ocv(path: Path = OUT_PARQUET):
    """Return a callable V_ocv(soc) that linearly interpolates the saved curve.

    Outside the saved SOC range the callable raises rather than silently
    extrapolating — heat-gen code should clamp SOC explicitly.
    """
    df = pd.read_parquet(path)
    return interp1d(df["SOC"].to_numpy(), df["V_ocv"].to_numpy(),
                    kind="linear", bounds_error=True, assume_sorted=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"loading {SOURCE_MAT.name}")
    raw = load_c20()
    print(f"  raw: {len(raw)} rows, {raw['time_s'].iloc[-1]/3600:.2f} h")

    seg = segment_discharge(raw)
    print(f"  segmented discharge half: {len(seg)} rows, "
          f"{(seg['time_s'].iloc[-1] - seg['time_s'].iloc[0])/3600:.2f} h, "
          f"V {seg['voltage_V'].iloc[0]:.3f} -> {seg['voltage_V'].iloc[-1]:.3f}, "
          f"Ah extracted = {abs(seg['Ah'].iloc[-1] - seg['Ah'].iloc[0]):.3f}")

    seg = compute_soc(seg)
    ocv = resample_to_soc_grid(seg)
    info = sanity_check(ocv, seg)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    ocv.to_parquet(OUT_PARQUET, index=False)
    OUT_META.write_text(json.dumps(info, indent=2))
    plot_ocv(ocv, seg, OUT_PLOT)

    print(f"\nwrote {OUT_PARQUET.name}  ({len(ocv)} rows on SOC grid)")
    print(f"wrote {OUT_META.name}")
    print(f"wrote {OUT_PLOT.relative_to(PROJECT_ROOT)}")
    print(f"\nendpoints:  V_ocv(SOC={info['soc_range_grid'][0]:.3f}) = "
          f"{info['v_ocv_at_low_soc']:.4f} V,  "
          f"V_ocv(SOC={info['soc_range_grid'][1]:.3f}) = "
          f"{info['v_ocv_at_high_soc']:.4f} V")
    print(f"SOC coverage:  [{info['soc_range_grid'][0]*100:.1f}%, "
          f"{info['soc_range_grid'][1]*100:.1f}%]")
    print(f"surface temp over sweep: {info['T_surface_C_range'][0]:.2f} -> "
          f"{info['T_surface_C_range'][1]:.2f} °C  "
          f"(delta {info['T_surface_C_range'][1]-info['T_surface_C_range'][0]:.2f} °C)")

    if info["flags"]:
        print("\nFLAGS:")
        for f in info["flags"]:
            print(f"  - {f}")
    else:
        print("\nno sanity flags")


if __name__ == "__main__":
    main()
